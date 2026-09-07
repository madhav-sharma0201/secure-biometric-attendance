"""Phase 2 helper: inspect a candidate dataset before committing to it.

Run this inside a Kaggle notebook after attaching a dataset. It answers the
questions that decide whether the dataset is usable:
  - video or images?
  - how many subjects, and can we group splits by subject? (MLR-3)
  - live/spoof balance
  - what attack types are labelled
  - resolution and duration

Usage (Kaggle cell):
    !python inspect_dataset.py /kaggle/input/<dataset-slug>
"""
import sys
import os
from collections import Counter

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main(root: str) -> None:
    ext_counts: Counter = Counter()
    top_dirs: Counter = Counter()
    samples: dict[str, str] = {}
    total = 0

    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = rel.split(os.sep)
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            ext_counts[ext] += 1
            total += 1
            if parts[0] != ".":
                top_dirs[os.sep.join(parts[:2])] += 1
            if ext not in samples:
                samples[ext] = os.path.join(rel, fn)

    print(f"root: {root}")
    print(f"total files: {total}\n")

    print("file types:")
    for ext, n in ext_counts.most_common(15):
        print(f"  {ext or '<none>':<10} {n}")

    n_video = sum(n for e, n in ext_counts.items() if e in VIDEO_EXT)
    n_image = sum(n for e, n in ext_counts.items() if e in IMAGE_EXT)
    print(f"\nvideos: {n_video}   images: {n_image}")
    print("=> TEMPORAL MODEL VIABLE" if n_video > 100 else "=> image-only: CNN-LSTM not trainable on this data")

    print("\ndirectory structure (first two levels, top 25 by file count):")
    for d, n in top_dirs.most_common(25):
        print(f"  {d:<60} {n}")

    print("\nexample paths per type:")
    for ext, p in list(samples.items())[:10]:
        print(f"  {ext:<10} {p}")

    print("\nWhat to look for next:")
    print("  1. Is there a subject/person identifier in the path or a metadata file?")
    print("     Without it we cannot split by subject and the results are not credible.")
    print("  2. Are attack types (print / phone / laptop / replay) distinguishable?")
    print("  3. Is there an official train/test protocol to follow?")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input")
