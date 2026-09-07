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
