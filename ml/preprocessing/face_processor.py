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


@dataclass
class DetectedFace:
    """One detected face: the aligned crop plus the metadata the caller needs."""

    crop: np.ndarray  # (size, size, 3), uint8, RGB
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 in original image coordinates
    det_score: float


class FaceProcessor:
    """Detects and aligns faces using InsightFace's SCRFD detector.

    The detector is loaded lazily so that importing this module stays cheap — matters
    for the API process, which imports it at startup but may not need it immediately.
    """

    def __init__(self, image_size: int = 112, det_size: int = 640, ctx_id: int = -1):
        """ctx_id: -1 for CPU, >=0 selects a GPU. Kaggle training passes 0."""
        self.image_size = image_size
        self.det_size = det_size
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
            out.append(
                DetectedFace(
                    crop=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                    bbox=f.bbox,
                    det_score=float(f.det_score),
                )
            )

        out.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        return out

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
