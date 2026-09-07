"""Build a preprocessing manifest from a raw dataset directory.

Dataset-specific adapters live here and nowhere else. Everything downstream
(preprocessing, splits, training) consumes the manifest schema, so supporting a new
dataset means adding one function here.

Schema produced:
    path, label, subject, clip, attack_type, session

Usage:
    python scripts/build_manifest.py trainingdatapro /kaggle/input/<slug> out.csv
    python scripts/build_manifest.py selfcollected  data/raw/self          out.csv
"""
from __future__ import annotations

import csv
import os
import sys

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _rows_from_tree(root: str, label_of, attack_of, subject_of=None) -> list[dict]:
    rows = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in VIDEO_EXT:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            label = label_of(rel)
            if label is None:
                continue
            clip = os.path.splitext(rel)[0].replace(os.sep, "_")
            rows.append({
                "path": full,
                "label": label,
                # No person identifier is available in these datasets, so `subject` is
                # set to the clip id and grouping is done on `clip`. Recorded honestly
                # rather than inventing IDs that would imply a guarantee we don't have.
                "subject": subject_of(rel) if subject_of else clip,
                "clip": clip,
                "attack_type": attack_of(rel),
                "session": "",
            })
    return rows


def trainingdatapro(root: str) -> list[dict]:
    """`trainingdatapro/real-vs-fake-anti-spoofing-video-classification`.

    Layout: train/ and test/ each containing real_video/ and attack/.
    We ignore the vendor's train/test split and re-split ourselves — their split
    basis is undocumented, and an unverifiable split is not a usable one.
    """
    def label_of(rel: str):
        low = rel.lower()
        if "real" in low:
            return "live"
        if "attack" in low:
            return "spoof"
        return None

    def attack_of(rel: str):
        return "live" if label_of(rel) == "live" else "replay_phone"

    return _rows_from_tree(root, label_of, attack_of)


def selfcollected(root: str) -> list[dict]:
    """Self-collected set. Expected layout:

        data/raw/self/live/<anything>.mp4
        data/raw/self/print/...
        data/raw/self/phone/...
        data/raw/self/laptop/...

    Attack type is taken from the top-level folder, which is what makes the
    per-attack evaluation table possible.
    """
    known = {"live", "print", "phone", "laptop", "replay"}

    def top(rel: str) -> str:
        return rel.split(os.sep)[0].lower()

    def label_of(rel: str):
        t = top(rel)
        if t not in known:
            return None
        return "live" if t == "live" else "spoof"

    return _rows_from_tree(root, label_of, lambda rel: top(rel))


ADAPTERS = {"trainingdatapro": trainingdatapro, "selfcollected": selfcollected}


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ADAPTERS:
        print(f"usage: build_manifest.py [{'|'.join(ADAPTERS)}] <raw_dir> <out.csv>")
        raise SystemExit(2)

    _, name, root, out = sys.argv
    rows = ADAPTERS[name](root)
    if not rows:
        raise SystemExit(f"no videos found under {root!r} — check the path and layout")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "subject", "clip",
                                           "attack_type", "session"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"wrote {out}: {len(rows)} clips")
    print("  labels:  ", dict(Counter(r["label"] for r in rows)))
    print("  attacks: ", dict(Counter(r["attack_type"] for r in rows)))


if __name__ == "__main__":
    main()
