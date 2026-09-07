"""Turn a dataset manifest into cached face-crop sequences on disk.

Design note: this module knows nothing about any specific dataset. It consumes a
manifest CSV with a fixed schema, so adapting to a new dataset means writing a small
manifest builder, not rewriting preprocessing. That separation is what lets us swap
datasets on day 1 without touching the training code.

Input manifest columns:
    path         absolute path to a video or image
    label        'live' or 'spoof'
    subject      subject/person identifier  <- REQUIRED, splits group on this
    attack_type  e.g. 'live', 'print', 'phone', 'laptop', 'replay'
    session      optional recording session id

Output:
    <out_dir>/<clip_id>/frame_000.jpg ...      aligned 112x112 crops
    <out_dir>/manifest.csv                     one row per clip, with n_frames
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import cv2

from .face_processor import FaceProcessor

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class ClipRecord:
    clip_id: str
    label: str
    subject: str
    attack_type: str
    session: str
    n_frames: int
    n_detect_fail: int


def _sample_video_frames(path: str, stride: int, max_frames: int) -> list:
    """Read every `stride`-th frame, up to `max_frames`.

    Uniform stride rather than random sampling: the temporal model needs frames at a
    consistent time spacing, otherwise 'motion between adjacent frames' means something
    different for every sample.
    """
    cap = cv2.VideoCapture(path)
    frames, idx = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def process_manifest(
    manifest_path: str,
    out_dir: str,
    processor: FaceProcessor,
    stride: int = 3,
    max_frames: int = 16,
    limit: int | None = None,
) -> list[ClipRecord]:
    """Preprocess every clip in the manifest. Returns per-clip records.

    `limit` processes only the first N rows — use it to smoke-test the pipeline on a
    laptop before launching the full run on a GPU box.
    """
    os.makedirs(out_dir, exist_ok=True)
    records: list[ClipRecord] = []

    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit is not None:
        rows = rows[:limit]

    for i, row in enumerate(rows):
        src = row["path"]
        clip_id = f"{row['subject']}__{row['attack_type']}__{i:05d}"
        clip_dir = os.path.join(out_dir, clip_id)

        ext = os.path.splitext(src)[1].lower()
        if ext in VIDEO_EXT:
            frames = _sample_video_frames(src, stride, max_frames)
        else:
            img = cv2.imread(src)
            frames = [img] if img is not None else []

        os.makedirs(clip_dir, exist_ok=True)
        kept, failed = 0, 0
        for f_idx, frame in enumerate(frames):
            face = processor.detect_primary(frame)
            if face is None:
                # A missed detection is dropped, not padded. Padding would teach the
                # model that duplicated frames mean 'live', which is an artifact of
                # our pipeline rather than a property of the face.
                failed += 1
                continue
            crop_bgr = cv2.cvtColor(face.crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(clip_dir, f"frame_{f_idx:03d}.jpg"), crop_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept += 1

        records.append(
            ClipRecord(
                clip_id=clip_id,
                label=row["label"],
                subject=row["subject"],
                attack_type=row.get("attack_type", "unknown"),
                session=row.get("session", ""),
                n_frames=kept,
                n_detect_fail=failed,
            )
        )

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} clips", flush=True)

    out_manifest = os.path.join(out_dir, "manifest.csv")
    with open(out_manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "label", "subject", "attack_type", "session",
                    "n_frames", "n_detect_fail"])
        for r in records:
            w.writerow([r.clip_id, r.label, r.subject, r.attack_type, r.session,
                        r.n_frames, r.n_detect_fail])

    total_fail = sum(r.n_detect_fail for r in records)
    empty = [r.clip_id for r in records if r.n_frames == 0]
    print(f"\nwrote {out_manifest}")
    print(f"clips: {len(records)}   frames kept: {sum(r.n_frames for r in records)}   "
          f"detection failures: {total_fail}")
    if empty:
        # Worth surfacing loudly: if detection fails disproportionately on spoof clips,
        # the detector is doing part of the anti-spoofing job and the model's measured
        # performance will be optimistic.
        print(f"WARNING: {len(empty)} clips yielded zero faces, e.g. {empty[:5]}")
    return records
