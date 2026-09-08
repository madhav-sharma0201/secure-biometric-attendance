"""Datasets for liveness training and evaluation.

Two modes over the same cached face crops:

  mode='frame'     one crop  -> (3, H, W)      for the single-frame baseline
  mode='sequence'  N crops   -> (N, 3, H, W)   for the temporal model

Both are scored at CLIP level at evaluation time (see `ml/evaluation/`): the baseline
averages its per-frame scores across the clip, the temporal model consumes the sequence
directly. Comparing a per-frame metric against a per-clip metric would not be a
comparison at all, and clip-level matches deployment, where one decision is made per
capture rather than per frame.
"""
from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.preprocessing.face_processor import PREPROCESSING, normalize


class SequenceAugment:
    """Augmentation applied IDENTICALLY to every frame in a sequence.

    This is the part that is easy to get wrong. Standard per-image augmentation would
    flip frame 2 and not frame 3, injecting motion that never happened and destroying
    exactly the temporal signal the LSTM is meant to read. Augmentation parameters are
    therefore drawn once per sample and reused across its frames.
    """

    def __init__(self, flip=0.5, brightness=0.3, contrast=0.3, jpeg=0.2, blur=0.2):
        self.flip, self.brightness = flip, brightness
        self.contrast, self.jpeg, self.blur = contrast, jpeg, blur

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        do_flip = random.random() < self.flip
        b = random.uniform(-self.brightness, self.brightness) if self.brightness else 0.0
        c = random.uniform(1 - self.contrast, 1 + self.contrast) if self.contrast else 1.0
        do_jpeg = random.random() < self.jpeg
        q = random.randint(30, 90)
        do_blur = random.random() < self.blur
        k = random.choice([3, 5])

        out = []
        for f in frames:
            if do_flip:
                f = f[:, ::-1]
            if b or c != 1.0:
                f = np.clip(f.astype(np.float32) * c + b * 255.0, 0, 255).astype(np.uint8)
            if do_jpeg:
                # Recompression is a realistic corruption: replayed video has been
                # encoded twice. Teaching the model to survive it discourages reliance
                # on compression artifacts as a shortcut for 'spoof'.
                ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    f = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if do_blur:
                f = cv2.GaussianBlur(f, (k, k), 0)
            out.append(np.ascontiguousarray(f))
        return out


class LivenessDataset(Dataset):
    """Reads cached face crops produced by `ml/preprocessing/extract.py`.

    `rows` are manifest records (clip_id, label, attack_type, ...). `crops_dir` holds
    one directory of jpgs per clip.
    """

    def __init__(
        self,
        rows: list[dict],
        crops_dir: str,
        mode: str = "sequence",
        seq_len: int = 8,
        augment: SequenceAugment | None = None,
        samples_per_clip: int = 1,
    ):
        assert mode in ("frame", "sequence")
        self.mode, self.seq_len = mode, seq_len
        self.augment = augment
        self.crops_dir = crops_dir

        self.items: list[dict] = []
        self.skipped: list[str] = []
        for r in rows:
            # A pooled corpus draws clips from several crop directories, so each row
            # may carry its own. Falling back to the shared directory keeps
            # single-source use unchanged.
            base = r.get("crops_dir") or crops_dir
            clip_dir = os.path.join(base, r["clip_id"])
            if not os.path.isdir(clip_dir):
                self.skipped.append(r["clip_id"])
                continue
            frames = sorted(f for f in os.listdir(clip_dir) if f.endswith(".jpg"))
            if not frames:
                self.skipped.append(r["clip_id"])
                continue
            rec = {
                "clip_id": r["clip_id"],
                "label": 1 if r["label"] == "live" else 0,
                "attack_type": r.get("attack_type", "unknown"),
                "frames": [os.path.join(clip_dir, f) for f in frames],
            }
            for _ in range(samples_per_clip):
                self.items.append(rec)

    def __len__(self) -> int:
        return len(self.items)

    def _pick_sequence(self, frames: list[str]) -> list[str]:
        """Choose seq_len frames, preserving temporal order.

        Short clips are padded by repeating the last frame rather than by looping back
        to the start: looping would fabricate reverse motion, which is not a thing a
        real face does and would be a spurious cue.
        """
        n = len(frames)
        if n >= self.seq_len:
            start = random.randint(0, n - self.seq_len)
            return frames[start:start + self.seq_len]
        return frames + [frames[-1]] * (self.seq_len - n)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        if self.mode == "frame":
            paths = [random.choice(rec["frames"])]
        else:
            paths = self._pick_sequence(rec["frames"])

        imgs = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                # Match whatever size the cached crops actually are; hardcoding 112
                # silently breaks once the liveness input is decoupled from ArcFace.
                size = imgs[0].shape[0] if imgs else PREPROCESSING["liveness_image_size"]
                img = np.zeros((size, size, 3), dtype=np.uint8)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.augment is not None:
            imgs = self.augment(imgs)

        tensors = [torch.from_numpy(normalize(i)) for i in imgs]
        x = tensors[0] if self.mode == "frame" else torch.stack(tensors)
        return x, torch.tensor(float(rec["label"])), rec["clip_id"]


def clip_eval_batches(rows, crops_dir, mode, seq_len=8, max_seqs=4):
    """Yield every clip with all of its evaluation views, for clip-level scoring.

    frame mode    -> all frames of the clip, scored and averaged
    sequence mode -> up to `max_seqs` evenly spaced windows, scored and averaged

    Deterministic: no randomness at evaluation time, so a rerun reproduces the metric.
    """
    for r in rows:
        base = r.get("crops_dir") or crops_dir
        clip_dir = os.path.join(base, r["clip_id"])
        if not os.path.isdir(clip_dir):
            continue
        frames = sorted(os.path.join(clip_dir, f)
                        for f in os.listdir(clip_dir) if f.endswith(".jpg"))
        if not frames:
            continue

        views: list[list[str]] = []
        if mode == "frame":
            views = [[f] for f in frames]
        else:
            n = len(frames)
            if n <= seq_len:
                views = [frames + [frames[-1]] * (seq_len - n)]
            else:
                starts = np.linspace(0, n - seq_len, min(max_seqs, n - seq_len + 1))
                views = [frames[int(s):int(s) + seq_len] for s in starts]

        batch = []
        for view in views:
            imgs = []
            for p in view:
                img = cv2.imread(p)
                if img is None:
                    img = np.zeros((PREPROCESSING["liveness_image_size"],
                                    PREPROCESSING["liveness_image_size"], 3), dtype=np.uint8)
                imgs.append(torch.from_numpy(normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            batch.append(imgs[0] if mode == "frame" else torch.stack(imgs))

        yield {
            "clip_id": r["clip_id"],
            "label": 1 if r["label"] == "live" else 0,
            "attack_type": r.get("attack_type", "unknown"),
            "views": torch.stack(batch),
        }
