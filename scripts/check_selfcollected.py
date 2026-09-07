"""Validate the self-collected recordings before they are relied on as a test set.

Checks each clip is readable, long enough, and reasonably sized, and reports the
per-attack counts that the evaluation table will be built from. Run this BEFORE
uploading — a clip OpenCV cannot decode is worth finding now, not on Kaggle.

Usage:
    python scripts/check_selfcollected.py data/raw/self
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import cv2

EXPECTED = {"live", "phone", "laptop", "print"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MIN_SECONDS = 2.0
MIN_FRAMES = 24


def main(root: str) -> None:
    if not os.path.isdir(root):
        raise SystemExit(f"{root!r} does not exist. Create it and add your clips.")

    counts: Counter = Counter()
    problems: list[str] = []
    durations: list[float] = []

    for folder in sorted(os.listdir(root)):
        fdir = os.path.join(root, folder)
        if not os.path.isdir(fdir):
            continue
        if folder.lower() not in EXPECTED:
            problems.append(f"unexpected folder {folder!r} — expected one of {sorted(EXPECTED)}")
            continue

        for fn in sorted(os.listdir(fdir)):
            ext = os.path.splitext(fn)[1].lower()
            path = os.path.join(fdir, fn)
            if ext not in VIDEO_EXT:
                if not fn.startswith("."):
                    problems.append(f"{folder}/{fn}: not a video file")
                continue

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                problems.append(f"{folder}/{fn}: OpenCV cannot open this file "
                                "(likely HEVC — see the ffmpeg command in the docs)")
                cap.release()
                continue

            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, _frame = cap.read()
            cap.release()

            if not ok:
                problems.append(f"{folder}/{fn}: opens but cannot decode frames")
                continue
            if n < MIN_FRAMES:
                problems.append(f"{folder}/{fn}: only {n} frames — too short for a sequence")
                continue

            dur = n / fps if fps else 0.0
            durations.append(dur)
            if dur and dur < MIN_SECONDS:
                problems.append(f"{folder}/{fn}: {dur:.1f}s — shorter than {MIN_SECONDS}s")
            if min(w, h) < 240:
                problems.append(f"{folder}/{fn}: {w}x{h} is very low resolution")

            counts[folder.lower()] += 1

    print("clips found")
    for k in sorted(EXPECTED):
        n = counts.get(k, 0)
        flag = "" if n >= 3 else "   <-- need at least 3"
        print(f"  {k:<8} {n:>3}{flag}")

    total = sum(counts.values())
    n_live = counts.get("live", 0)
    print(f"\ntotal {total}   live {n_live}   spoof {total - n_live}")
    if durations:
        print(f"duration: min {min(durations):.1f}s  mean {sum(durations)/len(durations):.1f}s  "
              f"max {max(durations):.1f}s")

    missing = EXPECTED - set(counts)
    if missing:
        print(f"\nmissing attack folders: {sorted(missing)} "
              "(those rows will be absent from the per-attack table)")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:25]:
            print(f"  - {p}")

    # An empty run must not report success. A validator that passes on no input is
    # the same class of mistake as a leakage check that never fires.
    if total == 0:
        raise SystemExit("\nFAIL: no clips found. Add recordings before continuing.")
    if n_live == 0 or total - n_live == 0:
        raise SystemExit("\nFAIL: need both live and spoof clips to compute APCER/BPCER.")
    if problems:
        raise SystemExit(f"\nFAIL: fix the {len(problems)} problem(s) above.")
    print("\nPASS: clips are usable")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/self")
