"""Tests for the split invariant. These run without any dataset or ML dependency."""
import pytest

from ml.preprocessing.splits import (
    LeakageError,
    assert_no_leakage,
    split_by_subject,
    summarise,
)


def make_rows(n_subjects=10, clips_per_subject=4):
    rows = []
    for s in range(n_subjects):
        for c in range(clips_per_subject):
            rows.append({
                "clip_id": f"s{s}_c{c}",
                "subject": f"subj{s}",
                "label": "live" if c % 2 == 0 else "spoof",
                "attack_type": "live" if c % 2 == 0 else "print",
            })
    return rows


def test_no_subject_spans_two_splits():
    splits = split_by_subject(make_rows())
    members = {name: {r["subject"] for r in rows} for name, rows in splits.items()}
    assert members["train"] & members["val"] == set()
    assert members["train"] & members["test"] == set()
    assert members["val"] & members["test"] == set()


def test_every_clip_is_assigned_exactly_once():
    rows = make_rows()
    splits = split_by_subject(rows)
    ids = [r["clip_id"] for s in splits.values() for r in s]
    assert len(ids) == len(rows)
    assert len(set(ids)) == len(rows)


def test_split_is_deterministic_under_seed():
    a = split_by_subject(make_rows(), seed=7)
    b = split_by_subject(make_rows(), seed=7)
    assert [r["clip_id"] for r in a["test"]] == [r["clip_id"] for r in b["test"]]


def test_leakage_is_detected():
    """The guard must actually fire — a check that never fails protects nothing."""
    leaking = {
        "train": [{"subject": "alice", "label": "live"}],
        "test": [{"subject": "alice", "label": "spoof"}],
    }
    with pytest.raises(LeakageError, match="alice"):
        assert_no_leakage(leaking)


def test_too_few_subjects_raises_rather_than_silently_emptying_a_split():
    with pytest.raises(ValueError, match="too few"):
        split_by_subject(make_rows(n_subjects=2))


def test_summarise_reports_all_splits():
    out = summarise(split_by_subject(make_rows()))
    assert "train" in out and "val" in out and "test" in out


# --- clip-level grouping (used when the dataset has no subject IDs) ---

def test_clip_grouping_keeps_each_source_video_in_one_split():
    from ml.preprocessing.splits import split_by_group
    rows = [
        {"clip_id": f"v{v}_seq{s}", "clip": f"v{v}", "label": "live", "subject": "unknown"}
        for v in range(12) for s in range(5)
    ]
    splits = split_by_group(rows, group_key="clip")
    members = {n: {r["clip"] for r in rs} for n, rs in splits.items()}
    assert members["train"] & members["test"] == set()
    assert members["train"] & members["val"] == set()
    # every sequence from a video travels with its video
    assert sum(len(rs) for rs in splits.values()) == len(rows)


def test_clip_grouping_detects_leakage_on_the_clip_key():
    from ml.preprocessing.splits import LeakageError, assert_no_leakage
    leaking = {
        "train": [{"clip": "v1", "subject": "unknown"}],
        "test": [{"clip": "v1", "subject": "unknown"}],
    }
    with pytest.raises(LeakageError, match="v1"):
        assert_no_leakage(leaking, group_key="clip")


def test_every_split_contains_both_classes():
    """A split with no live samples makes BPCER undefined; stratification prevents it."""
    from ml.preprocessing.splits import split_by_group
    rows = [
        {"clip_id": f"c{i}", "clip": f"c{i}", "subject": f"c{i}",
         "label": "live" if i < 8 else "spoof"}
        for i in range(16)
    ]
    splits = split_by_group(rows, group_key="clip")
    for name, rs in splits.items():
        labels = {r["label"] for r in rs}
        assert labels == {"live", "spoof"}, f"{name} split has only {labels}"


def test_stratification_still_respects_group_boundaries():
    from ml.preprocessing.splits import split_by_group
    rows = [
        {"clip": f"v{v}", "subject": f"v{v}", "label": "live" if v < 8 else "spoof",
         "seq": s}
        for v in range(16) for s in range(3)
    ]
    splits = split_by_group(rows, group_key="clip")
    members = {n: {r["clip"] for r in rs} for n, rs in splits.items()}
    assert members["train"] & members["test"] == set()
    assert members["train"] & members["val"] == set()
    assert members["val"] & members["test"] == set()
