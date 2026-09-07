"""Adapter tests. These guard the label mappings, which are the highest-risk part:
a wrong live/spoof mapping silently inverts an entire evaluation set.
"""
import csv
import os

import pytest

from scripts.build_manifest import (
    LIGHTING_TYPES,
    PRINTED_MASK_TYPES,
    lighting,
    printed_masks,
    printout_masks,
)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_printed_masks_uses_filename_as_subject_across_type_folders(tmp_path):
    for t in range(1, 11):
        for person in range(5):
            _touch(str(tmp_path / "attacks" / str(t) / f"{person}.mp4"))
    rows = printed_masks(str(tmp_path))
    assert len(rows) == 50
    assert len({r["subject"] for r in rows}) == 5
    # the same person index must map to the same subject in every type folder
    p0 = [r for r in rows if r["subject"] == "person_0"]
    assert len(p0) == 10
    assert {r["label"] for r in p0} == {"live", "spoof"}


def test_printed_masks_label_mapping_matches_the_dataset_card():
    """Folders 1-2 are genuine; 3-10 are attacks."""
    assert PRINTED_MASK_TYPES["1"][0] == "live"
    assert PRINTED_MASK_TYPES["2"][0] == "live"
    for t in map(str, range(3, 11)):
        assert PRINTED_MASK_TYPES[t][0] == "spoof", t


def test_lighting_video_types_are_live_not_spoof():
    """Regression guard for the trap in this dataset.

    '<condition>_video' reads like a replay attack but the dataset card defines it as
    a genuine recording of a person moving their head. 'monitor_video' IS a replay.
    Inverting these would corrupt every external metric.
    """
    for t in ("darkroom_video", "daylight_video", "lightroom_video", "nightlight_video"):
        assert LIGHTING_TYPES[t][0] == "live", t
    for t in ("darkroom_photo", "daylight_photo", "lightroom_photo", "nightlight_photo",
              "monitor_video", "mask", "outline"):
        assert LIGHTING_TYPES[t][0] == "spoof", t


def test_lighting_is_csv_driven_and_warns_on_unknown_types(tmp_path, capsys):
    files = tmp_path / "files"
    rows_in = [("0.mp4", "daylight_video"), ("1.mp4", "monitor_video"),
               ("2.mp4", "brand_new_type")]
    for fn, _t in rows_in:
        _touch(str(files / fn))
    with open(tmp_path / "dataset_info.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "type"])
        w.writeheader()
        w.writerows({"file": f, "type": t} for f, t in rows_in)

    rows = lighting(str(tmp_path))
    assert len(rows) == 2                     # unknown type dropped
    assert {r["label"] for r in rows} == {"live", "spoof"}
    assert "brand_new_type" in capsys.readouterr().out   # and it said so


def test_printout_masks_tolerates_spaces_in_filenames(tmp_path):
    """The real dataset contains a file literally named '2 .mp4'."""
    for folder in ("live_selfie", "live_video", "2d_masks"):
        _touch(str(tmp_path / folder / "0.mp4"))
    _touch(str(tmp_path / "live_video" / "2 .mp4"))

    rows = printout_masks(str(tmp_path))
    subjects = {r["subject"] for r in rows}
    assert "pm_person_2" in subjects          # space stripped, not a separate subject
    assert "pm_person_0" in subjects


def test_missing_lighting_csv_fails_loudly(tmp_path):
    _touch(str(tmp_path / "files" / "0.mp4"))
    with pytest.raises(SystemExit, match="dataset_info.csv"):
        lighting(str(tmp_path))
