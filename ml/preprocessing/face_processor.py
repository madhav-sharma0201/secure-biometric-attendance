"""Face detection, alignment and cropping.

Used by BOTH dataset preprocessing and live inference. Sharing this code path is
deliberate: if training crops and serving crops are produced differently, the model
sees a different input distribution in production than it was trained on, and the
reported metrics stop predicting real behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ImageNet statistics — the backbones we use are ImageNet-pretrained.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_preprocessing_config() -> dict:
    """Read configs/preprocessing.yaml if present, else fall back to the shipped values.

    Defaults live in one file rather than in two call sites, because the failure mode
    of disagreeing detection settings is silent: crops differ subtly, metrics stay
    plausible, and production behaviour drifts from what was measured.
    """
    import os
    defaults = {"det_size": 640, "image_size": 112,
                "liveness_image_size": 224, "context_scale": 1.8}
    for candidate in ("configs/preprocessing.yaml",
                      os.path.join(os.path.dirname(__file__), "..", "..",
                                   "configs", "preprocessing.yaml")):
        if os.path.exists(candidate):
            try:
                import yaml
                cfg = yaml.safe_load(open(candidate)) or {}
                det = cfg.get("detection", {})
                liv = cfg.get("liveness", {})
                return {
                    "det_size": int(det.get("det_size", defaults["det_size"])),
                    "image_size": int(det.get("image_size", defaults["image_size"])),
                    "liveness_image_size": int(
                        liv.get("image_size", defaults["liveness_image_size"])),
                    "context_scale": float(
                        liv.get("context_scale", defaults["context_scale"])),
                }
            except Exception:
                return defaults
    return defaults


PREPROCESSING = _load_preprocessing_config()


@dataclass
class DetectedFace:
    """One detected face.

    `crop` is the tight ArcFace-aligned face used for RECOGNITION.
    `context_crop` is a wider, unaligned region used for LIVENESS — it deliberately
    includes what surrounds the face, because a screen attack is given away by the
    bezel, the hand, the screen edge and reflections, none of which survive the tight
    aligned crop.
    """

    crop: np.ndarray          # (112, 112, 3) uint8 RGB — aligned, for recognition
    bbox: np.ndarray          # (4,) x1, y1, x2, y2 in original image coordinates
    det_score: float
    context_crop: np.ndarray | None = None   # (N, N, 3) uint8 RGB — for liveness


class FaceProcessor:
    """Detects and aligns faces using InsightFace's SCRFD detector.

    The detector is loaded lazily so that importing this module stays cheap — matters
    for the API process, which imports it at startup but may not need it immediately.
    """

    def __init__(self, image_size: int | None = None, det_size: int | None = None,
                 ctx_id: int = -1):
        """ctx_id: -1 for CPU, >=0 selects a GPU. Kaggle training passes 0.

        image_size and det_size default to configs/preprocessing.yaml so that the
        training and serving paths cannot be configured differently by accident.
        """
        self.image_size = image_size if image_size is not None else PREPROCESSING["image_size"]
        self.det_size = det_size if det_size is not None else PREPROCESSING["det_size"]
        self.liveness_image_size = PREPROCESSING["liveness_image_size"]
        self.context_scale = PREPROCESSING["context_scale"]
        self.ctx_id = ctx_id
        self._app = None

    def _ensure_loaded(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            # Detection only. The recognition model is loaded separately by the
            # recognition service so the two concerns stay independent.
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
            app.prepare(ctx_id=self.ctx_id, det_size=(self.det_size, self.det_size))
            self._app = app
        return self._app

    def detect_all(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect every face. Returns aligned crops sorted by area, largest first.

        The caller decides what to do with multiple faces. The decision engine
        rejects them (MULTIPLE_FACES); preprocessing keeps the largest.
        """
        from insightface.utils import face_align

        app = self._ensure_loaded()
        faces = app.get(image_bgr)

        out: list[DetectedFace] = []
        for f in faces:
            # norm_crop applies the canonical ArcFace 5-point similarity transform.
            crop_bgr = face_align.norm_crop(image_bgr, f.kps, image_size=self.image_size)
            ctx_bgr = self._context_crop(image_bgr, f.bbox)
            out.append(
                DetectedFace(
                    crop=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                    bbox=f.bbox,
                    det_score=float(f.det_score),
                    context_crop=cv2.cvtColor(ctx_bgr, cv2.COLOR_BGR2RGB),
                )
            )

        out.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        return out

    def _context_crop(self, image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Square crop of `context_scale` x the face box, clamped to the frame.

        Not aligned: rotating the region would move the bezel and screen edges around
        inconsistently, and those are the cues this crop exists to capture. Edges are
        replicated rather than zero-padded so a face near the frame border does not
        introduce a hard black rectangle the model could mistake for a device edge.
        """
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * self.context_scale

        half = side / 2.0
        left, top = int(round(cx - half)), int(round(cy - half))
        right, bottom = int(round(cx + half)), int(round(cy + half))

        pad_l, pad_t = max(0, -left), max(0, -top)
        pad_r, pad_b = max(0, right - w), max(0, bottom - h)
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)

        region = image_bgr[top:bottom, left:right]
        if region.size == 0:
            region = image_bgr
        if pad_l or pad_t or pad_r or pad_b:
            region = cv2.copyMakeBorder(region, pad_t, pad_b, pad_l, pad_r,
                                        cv2.BORDER_REPLICATE)
        return cv2.resize(region, (self.liveness_image_size, self.liveness_image_size),
                          interpolation=cv2.INTER_AREA)

    def detect_primary(self, image_bgr: np.ndarray) -> DetectedFace | None:
        """Largest detected face, or None. Convenience wrapper for preprocessing."""
        faces = self.detect_all(image_bgr)
        return faces[0] if faces else None


def normalize(crop_rgb: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB -> float32 CHW, ImageNet-normalised.

    Kept as a free function rather than a method so it can be applied to cached
    crops without constructing a detector.
    """
    x = crop_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))
