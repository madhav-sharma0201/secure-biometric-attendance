"""Face recognition: enrollment templates and cosine matching over ArcFace embeddings.

The backbone is pretrained (InsightFace `buffalo_l`, ArcFace R50). We do not train it —
training a face-recognition foundation model needs millions of identities and is not
the contribution of this project. What we build is the pipeline around it: enrollment,
template aggregation, matching, and threshold calibration.

The embedding maths is deliberately separated from model loading so the matching logic
can be tested exhaustively without a 300 MB model present.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    """Project onto the unit sphere so dot product == cosine similarity."""
    return v / np.clip(np.linalg.norm(v, axis=axis, keepdims=True), eps, None)


def build_template(embeddings: np.ndarray) -> np.ndarray:
    """Aggregate several enrollment embeddings into one template.

    Normalise, average, then renormalise. Averaging raw (unnormalised) vectors would
    let a single high-magnitude embedding — often a poor-quality capture — dominate
    the template. Normalising first gives every enrollment image equal weight.
    """
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError(f"expected a non-empty 2-D array, got shape {embeddings.shape}")
    return l2_normalize(l2_normalize(embeddings).mean(axis=0))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of `a` (D,) against `b` (N, D). Returns (N,)."""
    return l2_normalize(b) @ l2_normalize(a)


@dataclass(frozen=True)
class Match:
    user_id: str | None
    similarity: float
    margin: float          # gap to the runner-up; low margin means an ambiguous match
    n_candidates: int


def best_match(probe: np.ndarray, templates: dict[str, np.ndarray],
               threshold: float, min_margin: float = 0.0) -> Match:
    """Nearest registered template above `threshold`.

    `min_margin` guards against ambiguity: if the top two candidates are nearly equal
    (identical twins, siblings, or simply a weak probe), accepting the top one is a
    coin flip. Requiring a margin turns that coin flip into a rejection, which is the
    correct behaviour for a fail-closed system.
    """
    if not templates:
        return Match(None, 0.0, 0.0, 0)

    ids = list(templates)
    mat = np.stack([templates[i] for i in ids])
    sims = cosine_similarity(probe, mat)

    order = np.argsort(-sims)
    top = float(sims[order[0]])
    runner_up = float(sims[order[1]]) if len(order) > 1 else -1.0
    margin = top - runner_up

    if top < threshold or margin < min_margin:
        return Match(None, top, margin, len(ids))
    return Match(ids[order[0]], top, margin, len(ids))


def serialize(v: np.ndarray) -> bytes:
    """float32 little-endian bytes, for the BYTEA column."""
    return np.asarray(v, dtype="<f4").tobytes()


def deserialize(b: bytes, dim: int) -> np.ndarray:
    v = np.frombuffer(b, dtype="<f4")
    if v.size != dim:
        raise ValueError(f"embedding has {v.size} values, expected {dim}")
    return v.astype(np.float32)


class RecognitionService:
    """Wraps InsightFace. Loaded lazily so importing this module stays cheap."""

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1, det_size: int = 640):
        self.model_name, self.ctx_id, self.det_size = model_name, ctx_id, det_size
        self._app = None

    @property
    def model_version(self) -> str:
        return self.model_name

    def _ensure_loaded(self):
        if self._app is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name=self.model_name)
            app.prepare(ctx_id=self.ctx_id, det_size=(self.det_size, self.det_size))
            self._app = app
        return self._app

    def embed(self, image_bgr: np.ndarray) -> tuple[np.ndarray | None, int]:
        """Returns (embedding, n_faces). Embedding is None unless exactly one face.

        Refusing to embed when several faces are present is intentional: choosing 'the
        biggest face' here would silently let a bystander be enrolled or matched. The
        decision engine rejects MULTIPLE_FACES; this service does not second-guess it.
        """
        app = self._ensure_loaded()
        faces = app.get(image_bgr)
        if len(faces) != 1:
            return None, len(faces)
        return l2_normalize(faces[0].normed_embedding.astype(np.float32)), 1
