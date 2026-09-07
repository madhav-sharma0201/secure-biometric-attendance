"""Tests for PAD metrics, using hand-computable examples."""
import numpy as np
import pytest

from ml.evaluation.metrics import compute_metrics, roc_auc, select_threshold


def test_perfect_separation_gives_zero_error():
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.95, 0.1, 0.2, 0.05])
    m = compute_metrics(labels, scores, threshold=0.5)
    assert m.apcer == 0.0
    assert m.bpcer == 0.0
    assert m.acer == 0.0
    assert m.roc_auc == 1.0


def test_apcer_counts_attacks_accepted_as_live():
    # 4 spoofs, 2 of which score above threshold -> APCER = 0.5
    labels = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.9, 0.8, 0.7, 0.1, 0.1])
    m = compute_metrics(labels, scores, threshold=0.5)
    assert m.apcer == pytest.approx(0.5)
    assert m.bpcer == pytest.approx(0.0)
    assert m.acer == pytest.approx(0.25)


def test_bpcer_counts_live_rejected():
    # 4 live, 1 below threshold -> BPCER = 0.25
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1])
    m = compute_metrics(labels, scores, threshold=0.5)
    assert m.bpcer == pytest.approx(0.25)
    assert m.apcer == pytest.approx(0.0)


def test_worst_case_apcer_exceeds_pooled_when_one_attack_fails():
    """The ISO worst-case rule must surface a single badly-handled attack type."""
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    # 'replay' attacks both accepted (APCER 1.0); 'print' attacks both rejected (0.0)
    scores = np.array([0.9, 0.9, 0.99, 0.98, 0.01, 0.02, 0.01, 0.02])
    attacks = ["live", "live", "replay", "replay", "print", "print", "print", "print"]
    m = compute_metrics(labels, scores, threshold=0.5, attack_types=attacks)
    assert m.apcer_per_attack["replay"] == pytest.approx(1.0)
    assert m.apcer_per_attack["print"] == pytest.approx(0.0)
    assert m.apcer == pytest.approx(2 / 6)          # pooled looks acceptable
    assert m.apcer_worst_case == pytest.approx(1.0)  # worst case exposes the failure


def test_accuracy_can_look_good_while_apcer_is_terrible():
    """The reason accuracy is not the headline metric."""
    labels = np.array([1] * 18 + [0] * 2)
    scores = np.array([0.9] * 18 + [0.9, 0.9])   # every attack accepted
    m = compute_metrics(labels, scores, threshold=0.5)
    assert m.accuracy == pytest.approx(0.9)
    assert m.apcer == pytest.approx(1.0)


def test_roc_auc_handles_ties():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])   # all identical -> uninformative
    assert roc_auc(labels, scores) == pytest.approx(0.5)


def test_select_threshold_min_acer_separates_classes():
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.8, 0.9, 0.85, 0.2, 0.1, 0.15])
    t = select_threshold(labels, scores, criterion="min_acer")
    m = compute_metrics(labels, scores, t)
    assert m.acer == pytest.approx(0.0)


def test_apcer_target_criterion_is_strict():
    labels = np.array([1] * 5 + [0] * 5)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1])
    t = select_threshold(labels, scores, criterion="apcer_target", target_apcer=0.0)
    m = compute_metrics(labels, scores, t)
    assert m.apcer == 0.0


def test_degenerate_reject_everything_threshold_is_rejected():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])   # inseparable: APCER=0 only by rejecting all
    with pytest.raises(ValueError, match="rejects every bona fide"):
        select_threshold(labels, scores, criterion="apcer_target", target_apcer=0.0)


def test_threshold_sits_in_the_middle_of_a_perfectly_separated_gap():
    """Perfect validation separation makes many thresholds tie at zero cost.

    Choosing the first leaves the operating point flush against the gap edge, so a
    test score drifting slightly lands on the wrong side. This was the failure mode
    on the first training run: validation ACER 0%, test BPCER 60%.
    """
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.90, 0.92, 0.95, 0.10, 0.12, 0.08])
    t = select_threshold(labels, scores, criterion="min_acer")
    assert 0.12 < t < 0.90, f"threshold {t} is not inside the separating gap"
    assert 0.4 < t < 0.6, f"threshold {t} is not near the middle of the gap"
    assert compute_metrics(labels, scores, t).acer == 0.0
