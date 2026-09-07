"""Subject-grouped train/val/test splits, with a hard leakage check.

The most common way a face anti-spoofing project produces impressive-looking and
completely meaningless results is splitting frames randomly. Frames from one video are
near-duplicates, so a random split puts near-copies of test samples into training and
the model scores ~99% by memorising faces rather than learning spoof cues.

We therefore split on SUBJECT: every clip belonging to a person lands entirely in one
split, and the test set contains only people the model has never seen. `assert_no_leakage`
makes this a checked invariant rather than a good intention.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict


class LeakageError(AssertionError):
    """Raised when a subject appears in more than one split."""


def load_manifest(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def split_by_subject(
    rows: list[dict],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Partition clips into train/val/test by subject.

    Subjects are shuffled with a fixed seed, then assigned by proportion of subjects
    (not of clips). Clip counts per split will therefore be uneven — that is correct
    and preferable to balancing clips at the cost of splitting a subject.
    """
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)

    subjects = sorted(by_subject)
    random.Random(seed).shuffle(subjects)

    n = len(subjects)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if n_train == 0 or n_val == 0 or n - n_train - n_val == 0:
        raise ValueError(
            f"only {n} subjects — too few to split without an empty partition. "
            "Use a dataset with more subjects, or group by recording session instead."
        )

    groups = {
        "train": subjects[:n_train],
        "val": subjects[n_train:n_train + n_val],
        "test": subjects[n_train + n_val:],
    }
    splits = {k: [r for s in subs for r in by_subject[s]] for k, subs in groups.items()}
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    """Fail loudly if any subject appears in two splits."""
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            subj = r["subject"]
            prev = seen.setdefault(subj, name)
            if prev != name:
                raise LeakageError(
                    f"subject {subj!r} appears in both {prev!r} and {name!r} — "
                    "splits are leaking and any metrics computed from them are invalid"
                )


def summarise(splits: dict[str, list[dict]]) -> str:
    """Human-readable split summary. Print this into the report; reviewers look for it."""
    lines = [f"{'split':<8}{'subjects':>10}{'clips':>8}{'live':>8}{'spoof':>8}"]
    for name in ("train", "val", "test"):
        rows = splits.get(name, [])
        subs = {r["subject"] for r in rows}
        live = sum(1 for r in rows if r["label"] == "live")
        lines.append(
            f"{name:<8}{len(subs):>10}{len(rows):>8}{live:>8}{len(rows) - live:>8}"
        )
    return "\n".join(lines)


def write_splits(splits: dict[str, list[dict]], out_dir: str) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)
    for name, rows in splits.items():
        if not rows:
            continue
        with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
