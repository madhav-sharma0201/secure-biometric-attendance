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


# Folder index -> (label, attack_type) for the printed-mask dataset.
# Source: dataset card, "Inside the attacks folder there are 10 sub-folders".
PRINTED_MASK_TYPES = {
    "1":  ("live",  "live_no_glasses"),
    "2":  ("live",  "live_glasses"),
    "3":  ("spoof", "mask_static"),
    "4":  ("spoof", "mask_static_realglasses"),
    "5":  ("spoof", "mask_handheld"),
    "6":  ("spoof", "mask_handheld_realglasses"),
    "7":  ("spoof", "mask_static_printedglasses"),
    "8":  ("spoof", "mask_static_bothglasses"),
    "9":  ("spoof", "mask_handheld_printedglasses"),
    "10": ("spoof", "mask_handheld_bothglasses"),
}


def printed_masks(root: str) -> list[dict]:
    """`trainingdatapro/attacks-with-2d-printed-masks-of-indian-people`.

    Layout: attacks/<type 1..10>/<person>.mp4, where the FILENAME is the person
    index and is consistent across all ten type folders. That filename is therefore a
    genuine subject identifier, which is what makes subject-grouped splitting possible
    for this dataset (unlike the others evaluated — see docs/02-dataset.md).

    The handheld/static distinction is retained in `attack_type` because it is the
    variable the temporal model is expected to exploit: a hand-held mask jitters,
    a mounted one does not.
    """
    rows = []
    for dirpath, _dirs, files in os.walk(root):
        type_dir = os.path.basename(dirpath)
        if type_dir not in PRINTED_MASK_TYPES:
            continue
        label, attack = PRINTED_MASK_TYPES[type_dir]
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in VIDEO_EXT:
                continue
            person = os.path.splitext(fn)[0]
            rows.append({
                "path": os.path.join(dirpath, fn),
                "label": label,
                "subject": f"person_{person}",   # real subject id
                "clip": f"person_{person}_type_{type_dir}",
                "attack_type": attack,
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

    def subject_of(rel: str):
        """train/real_video/7.mp4 and train/attack/7.mp4 are the SAME person.

        The attack clip is a replay of that person's genuine clip, so the numeric
        filename is an implicit subject id. Treating each file as its own group would
        let a person appear as live in train and as spoof in test.
        """
        parts = rel.split(os.sep)
        split_dir = parts[0] if len(parts) > 1 else "x"
        stem = os.path.splitext(os.path.basename(rel))[0].strip()
        return f"rvf_{split_dir}_{stem}"

    return _rows_from_tree(root, label_of, attack_of, subject_of)


def printout_masks(root: str) -> list[dict]:
    """`trainingdatapro/cut-out-printout-attacks` (Printed 2D Masks Attacks Video).

    Layout: live_selfie/N.mp4, live_video/N.mp4, 2d_masks/N.mp4 for N in 0..8.
    The index is a subject id shared across all three folders (9 subjects).

    Note: one file is named "2 .mp4" with an embedded space. Walking the tree handles
    that; parsing the CSV paths would not.
    """
    folder_map = {
        "live_selfie": ("live", "live_selfie"),
        "live_video": ("live", "live_video"),
        "2d_masks": ("spoof", "print_mask"),
    }
    rows = []
    for dirpath, _dirs, files in os.walk(root):
        folder = os.path.basename(dirpath)
        if folder not in folder_map:
            continue
        label, attack = folder_map[folder]
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in VIDEO_EXT:
                continue
            person = os.path.splitext(fn)[0].strip()   # tolerate "2 .mp4"
            rows.append({
                "path": os.path.join(dirpath, fn),
                "label": label,
                "subject": f"pm_person_{person}",
                "clip": f"pm_person_{person}_{folder}",
                "attack_type": attack,
                "session": "",
            })
    return rows


# Type -> (label, attack_type) for the lighting dataset. Taken verbatim from the
# dataset card, NOT inferred from the names: "<x>_video" reads like a replay attack
# but is in fact a genuine recording of a person moving their head under that
# lighting. Guessing here would have mislabelled every live sample as spoof.
LIGHTING_TYPES = {
    "darkroom_video":   ("live",  "live_darkroom"),
    "daylight_video":   ("live",  "live_daylight"),
    "lightroom_video":  ("live",  "live_lightroom"),
    "nightlight_video": ("live",  "live_nightlight"),
    "darkroom_photo":   ("spoof", "display_photo_darkroom"),
    "daylight_photo":   ("spoof", "display_photo_daylight"),
    "lightroom_photo":  ("spoof", "display_photo_lightroom"),
    "nightlight_photo": ("spoof", "display_photo_nightlight"),
    "monitor_video":    ("spoof", "display_replay"),
    "outline":          ("spoof", "print_mask"),
    "mask":             ("spoof", "print_mask_cutout"),
}


def lighting(root: str) -> list[dict]:
    """`trainingdatapro/biometric-attacks-in-different-lighting`.

    Flat `files/N.mp4` with labels only in dataset_info.csv, so this adapter is
    CSV-driven rather than tree-driven. No subject identifier exists — acceptable
    because this set is used only for evaluation and is never split.
    """
    csv_path = None
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower() == "dataset_info.csv":
                csv_path = os.path.join(dirpath, fn)
                break
    if csv_path is None:
        raise SystemExit(f"dataset_info.csv not found under {root!r}")

    by_name = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXT:
                by_name.setdefault(fn, os.path.join(dirpath, fn))

    rows, unknown = [], set()
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            fname = os.path.basename((r.get("file") or "").strip())
            vtype = (r.get("type") or "").strip()
            if vtype not in LIGHTING_TYPES:
                unknown.add(vtype)
                continue
            path = by_name.get(fname)
            if path is None:
                continue
            label, attack = LIGHTING_TYPES[vtype]
            stem = os.path.splitext(fname)[0]
            rows.append({
                "path": path,
                "label": label,
                "subject": f"lt_{stem}",
                "clip": f"lt_{stem}",
                "attack_type": attack,
                "session": "",
            })

    if unknown:
        # Loud rather than silent: an unrecognised type means the vendor changed the
        # schema, and quietly dropping those clips would shrink the test set unnoticed.
        print(f"WARNING: unrecognised types skipped: {sorted(unknown)}")
    return rows


# Folder-name keywords -> (label, attack_type) for vendor datasets whose layout is a
# flat set of category folders.
#
# This table is EXPLICIT rather than inferred. The lighting dataset taught the lesson:
# "<condition>_video" reads like a replay attack but is a genuine recording, and
# guessing would have inverted every live sample. Anything not listed here is skipped
# loudly rather than assigned a guess.
FOLDER_LABELS = {
    # live
    "real":            ("live",  "live"),
    "live":            ("live",  "live"),
    "live_videos":     ("live",  "live"),
    "live_video":      ("live",  "live"),
    "live_selfie":     ("live",  "live_selfie"),
    "genuine":         ("live",  "live"),
    # display / screen attacks — the failure mode this retrain targets
    "screen":          ("spoof", "display_screen"),
    "monitor":         ("spoof", "display_monitor"),
    "display":         ("spoof", "display_screen"),
    "replay":          ("spoof", "display_replay"),
    "phone":           ("spoof", "display_phone"),
    "laptop":          ("spoof", "display_laptop"),
    "tablet":          ("spoof", "display_tablet"),
    "monitor_video":   ("spoof", "display_monitor"),
    # print / mask attacks
    "mask":            ("spoof", "print_mask"),
    "print":           ("spoof", "print_photo"),
    "printed":         ("spoof", "print_photo"),
    "photo":           ("spoof", "print_photo"),
    "outline":         ("spoof", "print_mask"),
    "cutout":          ("spoof", "print_mask_cutout"),
    "paper":           ("spoof", "print_photo"),
}

MEDIA_EXT = VIDEO_EXT | {".jpg", ".jpeg", ".png"}


def folder_categorised(root: str, source_tag: str = "webcam") -> list[dict]:
    """Generic adapter for datasets laid out as <root>/**/<category>/<file>.

    Reports every folder it sees and how it labelled it, so a wrong mapping is visible
    before training rather than after.
    """
    rows: list[dict] = []
    seen: dict[str, int] = {}
    skipped: dict[str, int] = {}

    for dirpath, _dirs, files in os.walk(root):
        folder = os.path.basename(dirpath).lower().replace("-", "_").replace(" ", "_")
        media = [f for f in sorted(files)
                 if os.path.splitext(f)[1].lower() in MEDIA_EXT]
        if not media:
            continue

        match = None
        for key, val in FOLDER_LABELS.items():
            if key == folder or folder.startswith(key + "_") or folder.endswith("_" + key):
                match = val
                break
        if match is None:
            for key, val in FOLDER_LABELS.items():
                if key in folder:
                    match = val
                    break
        if match is None:
            skipped[folder] = skipped.get(folder, 0) + len(media)
            continue

        label, attack = match
        seen[f"{folder} -> {label}/{attack}"] = seen.get(f"{folder} -> {label}/{attack}", 0) + len(media)
        for fn in media:
            stem = os.path.splitext(fn)[0].strip()
            rows.append({
                "path": os.path.join(dirpath, fn),
                "label": label,
                # No person identifier in these releases. Subject == file, so grouping
                # is clip-level here; recorded honestly rather than invented.
                "subject": f"{source_tag}_{folder}_{stem}",
                "clip": f"{source_tag}_{folder}_{stem}",
                "attack_type": attack,
                "session": "",
            })

    print(f"  [{source_tag}] folder -> label mapping:")
    for k, n in sorted(seen.items()):
        print(f"      {k:<46} {n} files")
    if skipped:
        print(f"  [{source_tag}] SKIPPED unrecognised folders (not guessed):")
        for k, n in sorted(skipped.items()):
            print(f"      {k:<46} {n} files")
    return rows


def webcam_attacks(root: str) -> list[dict]:
    """`trainingdatapro/full-hd-webcam-live-attacks` and `low-webcam-live-attacks`."""
    return folder_categorised(root, source_tag="webcam")


def web_camera_liveness(root: str) -> list[dict]:
    """`trainingdatapro/web-camera-face-liveness-detection` — webcam-domain monitor
    and mask attacks, the closest public match to a laptop-kiosk deployment."""
    return folder_categorised(root, source_tag="webcamlive")


def display_attacks(root: str) -> list[dict]:
    """`axondata/liveness-detection-real-and-display-attacks-5k` — Real/ vs Screen/.

    NOTE: its live samples are stills and its attacks are videos. That asymmetry is a
    source-bias risk on its own, so this set is used ALONGSIDE matched-capture sources,
    never as the only display-attack data.
    """
    return folder_categorised(root, source_tag="display")


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


ADAPTERS = {
    # Primary: trained and tested on. Has subject IDs, matched live/spoof capture.
    "printed_masks": printed_masks,
    # Webcam- and display-domain sources added to fix the display-attack failure.
    "webcam_attacks": webcam_attacks,
    "web_camera_liveness": web_camera_liveness,
    "display_attacks": display_attacks,
    # External test sets: NEVER trained on. Each contributes unseen attack types.
    "real_vs_fake": trainingdatapro,   # phone replay
    "printout_masks": printout_masks,  # printed mask, 9 subjects
    "lighting": lighting,              # display replay + 4 lighting conditions
    # Optional demo footage.
    "selfcollected": selfcollected,
}


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
    print("  subjects:", len({r["subject"] for r in rows}))
    print("  labels:  ", dict(Counter(r["label"] for r in rows)))
    print("  attacks: ", dict(Counter(r["attack_type"] for r in rows)))


if __name__ == "__main__":
    main()
