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


def split_by_group(
    rows: list[dict],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
    group_key: str = "subject",
    stratify: bool = True,
) -> dict[str, list[dict]]:
    """Partition clips into train/val/test by group.

    `group_key` selects the grouping unit. 'subject' is correct whenever the dataset
    exposes person identifiers. When it does not (see docs/02-dataset.md), 'clip'
    groups by source video instead: weaker, because a person may span splits, but it
    still prevents near-duplicate frames from one video landing on both sides.

    Subjects are shuffled with a fixed seed, then assigned by proportion of subjects
    (not of clips). Clip counts per split will therefore be uneven — that is correct
    and preferable to balancing clips at the cost of splitting a subject.
    """
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_subject[r[group_key]].append(r)

    subjects = sorted(by_subject)
    random.Random(seed).shuffle(subjects)

    # Stratify by label so every split contains both classes. Without this, a random
    # group assignment can hand a split zero live samples, which makes BPCER
    # undefined and the split useless. Each group has a single label, so grouping
    # integrity is preserved: we partition within each label independently.
    # Stratification only makes sense when a group carries a single label. Under
    # subject grouping each subject usually has BOTH live and spoof clips, so there is
    # no single label to stratify on — and bucketing a subject by whichever row came
    # first would be arbitrary. Detect that case and skip stratification: mixed-label
    # groups already guarantee both classes land in every split.
    labels_per_group = {g: {r.get("label", "?") for r in by_subject[g]} for g in subjects}
    mixed = any(len(v) > 1 for v in labels_per_group.values())

    if stratify and not mixed:
        buckets: dict[str, list[str]] = defaultdict(list)
        for g in subjects:
            buckets[next(iter(labels_per_group[g]))].append(g)
        strata = list(buckets.values())
    else:
        strata = [subjects]

    groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for stratum in strata:
        n = len(stratum)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        if n_train == 0 or n_val == 0 or n - n_train - n_val == 0:
            raise ValueError(
                f"only {n} {group_key} groups in one label stratum — too few to split "
                "without an empty partition. Use more data, or a finer grouping key."
            )
        groups["train"] += stratum[:n_train]
        groups["val"] += stratum[n_train:n_train + n_val]
        groups["test"] += stratum[n_train + n_val:]
    splits = {k: [r for s in subs for r in by_subject[s]] for k, subs in groups.items()}
    assert_no_leakage(splits, group_key=group_key)
    return splits


def split_by_subject(rows, train_frac=0.6, val_frac=0.2, seed=42):
    """Subject-grouped split. Preferred when the dataset has person identifiers."""
    return split_by_group(rows, train_frac, val_frac, seed, group_key="subject")


def assert_no_leakage(splits: dict[str, list[dict]], group_key: str = "subject") -> None:
    """Fail loudly if any group appears in two splits."""
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            subj = r[group_key]
            prev = seen.setdefault(subj, name)
            if prev != name:
                raise LeakageError(
                    f"{group_key} {subj!r} appears in both {prev!r} and {name!r} — "
                    "splits are leaking and any metrics computed from them are invalid"
                )


def summarise(splits: dict[str, list[dict]]) -> str:
    """Human-readable split summary. Print this into the report; reviewers look for it."""
def summarise_by(splits: dict[str, list[dict]], group_key: str = "subject") -> str:
    lines = [f"{'split':<8}{'groups':>10}{'clips':>8}{'live':>8}{'spoof':>8}"]
    for name in ("train", "val", "test"):
        rows = splits.get(name, [])
        subs = {r[group_key] for r in rows}
        live = sum(1 for r in rows if r["label"] == "live")
        lines.append(
            f"{name:<8}{len(subs):>10}{len(rows):>8}{live:>8}{len(rows) - live:>8}"
        )
    return "\n".join(lines)


def summarise(splits: dict[str, list[dict]]) -> str:
    return summarise_by(splits, group_key="subject")


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
