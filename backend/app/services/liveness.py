"""Liveness inference over the exported ONNX model.

Runs on onnxruntime rather than torch: the serving container is ~50 MB instead of
~2.5 GB and is faster on CPU. The trained model is an artifact the backend consumes;
the backend never imports the training stack.
"""
from __future__ import annotations

import json
import os

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    The naive 1/(1+exp(-x)) overflows for large negative logits and warns or produces
    inf. A confident spoof prediction is exactly where large negative logits occur, so
    the naive form breaks precisely on the inputs that matter most.
    """
    out = np.empty_like(x, dtype=np.float64)
    pos, neg = x >= 0, x < 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


class LivenessService:
    """Scores a sequence of aligned face crops as live or spoof."""

    def __init__(self, model_path: str, meta_path: str | None = None,
                 sequence_length: int = 8, image_size: int = 112):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.model_version = "unknown"
        self.trained_threshold: float | None = None
        self._sess = None

        meta_path = meta_path or os.path.splitext(model_path)[0] + "_meta.json"
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            self.model_version = meta.get("model_id", "unknown")
            self.trained_threshold = meta.get("threshold")
            self.sequence_length = meta.get("sequence_length", sequence_length)
            self.image_size = meta.get("image_size", image_size)

    def _ensure_loaded(self):
        if self._sess is None:
            import onnxruntime as ort
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"liveness model not found at {self.model_path!r}. Train it first "
                    "and place liveness.onnx + liveness_meta.json in models/."
                )
            self._sess = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"])
        return self._sess

    def score(self, crops: list[np.ndarray]) -> float:
        """P(live) for a sequence of normalised CHW float32 crops.

        Fewer frames than the model expects are padded by repeating the last one,
        matching the training-time convention exactly. Looping back to the start would
        fabricate reverse motion the model never saw during training.
        """
        if not crops:
            raise ValueError("no frames supplied")

        frames = list(crops)
        if len(frames) < self.sequence_length:
            frames += [frames[-1]] * (self.sequence_length - len(frames))
        frames = frames[:self.sequence_length]

        x = np.stack(frames)[None, ...].astype(np.float32)   # (1, N, 3, H, W)
        sess = self._ensure_loaded()

        # The baseline export takes a single frame; the temporal export takes a
        # sequence. Adapt to whatever was actually exported rather than assuming.
        expected = sess.get_inputs()[0].shape
        if len(expected) == 4:
            x = x[0]

        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        return float(sigmoid(np.asarray(logits, dtype=np.float64)).mean())
