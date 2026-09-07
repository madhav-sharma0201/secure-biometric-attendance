"""Presentation-attack detection metrics, following ISO/IEC 30107-3 conventions.

Convention used throughout: label 1 = live (bona fide), 0 = spoof (attack).
`score` is the model's P(live). A sample is predicted live when score >= threshold.

    APCER  Attack Presentation Classification Error Rate
           = fraction of ATTACKS wrongly accepted as live.  Security failure.
    BPCER  Bona fide Presentation Classification Error Rate
           = fraction of REAL PEOPLE wrongly rejected as spoof.  Usability failure.
    ACER   = (APCER + BPCER) / 2

Accuracy is deliberately not the headline metric: on an imbalanced set a model can
score 95% accuracy while accepting most attacks, which is the only error that matters
for a security system.

ISO note: APCER is defined PER attack type (PAI species), and the figure reported for a
system is the WORST species, not the average. `apcer_worst_case` implements this. A
mean-over-attacks APCER flatters a model that fails badly on one attack type — which is
precisely the situation a real attacker exploits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PADMetrics:
    threshold: float
    apcer: float                      # pooled over all attacks
    apcer_worst_case: float           # ISO-style: worst single attack type
    apcer_per_attack: dict[str, float] = field(default_factory=dict)
    bpcer: float = 0.0
    acer: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    confusion: dict[str, int] = field(default_factory=dict)
    n_live: int = 0
    n_spoof: int = 0

    def summary(self) -> str:
        lines = [
            f"threshold           {self.threshold:.4f}",
            f"APCER (pooled)      {self.apcer * 100:.2f}%",
            f"APCER (worst case)  {self.apcer_worst_case * 100:.2f}%",
            f"BPCER               {self.bpcer * 100:.2f}%",
            f"ACER                {self.acer * 100:.2f}%",
            f"ROC-AUC             {self.roc_auc:.4f}",
            f"accuracy            {self.accuracy * 100:.2f}%",
            f"precision / recall  {self.precision:.4f} / {self.recall:.4f}",
            f"F1                  {self.f1:.4f}",
            f"samples             {self.n_live} live, {self.n_spoof} spoof",
            f"confusion           {self.confusion}",
        ]
        if self.apcer_per_attack:
            lines.append("per-attack APCER:")
            for k, v in sorted(self.apcer_per_attack.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {k:<16} {v * 100:6.2f}%")
        return "\n".join(lines)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney U) identity, with correct tie handling."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # average rank for ties
        i = j + 1

    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    attack_types: list[str] | None = None,
) -> PADMetrics:
    """Full metric set at a fixed threshold.

    `attack_types` aligns with labels/scores; entries for live samples are ignored.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred_live = scores >= threshold

    is_live = labels == 1
    is_spoof = ~is_live
    n_live, n_spoof = int(is_live.sum()), int(is_spoof.sum())

    tp = int((pred_live & is_live).sum())        # live accepted
    fn = int((~pred_live & is_live).sum())       # live rejected  -> BPCER
    fp = int((pred_live & is_spoof).sum())       # attack accepted -> APCER
    tn = int((~pred_live & is_spoof).sum())

    apcer = fp / n_spoof if n_spoof else 0.0
    bpcer = fn / n_live if n_live else 0.0

    per_attack: dict[str, float] = {}
    if attack_types is not None:
        at = np.asarray(attack_types, dtype=object)
        for kind in sorted({a for a, s in zip(at, is_spoof) if s}):
            mask = is_spoof & (at == kind)
            if mask.sum():
                per_attack[str(kind)] = float((pred_live & mask).sum() / mask.sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return PADMetrics(
        threshold=float(threshold),
        apcer=apcer,
        apcer_worst_case=max(per_attack.values()) if per_attack else apcer,
        apcer_per_attack=per_attack,
        bpcer=bpcer,
        acer=(apcer + bpcer) / 2.0,
        accuracy=(tp + tn) / len(labels) if len(labels) else 0.0,
        precision=precision,
        recall=recall,
        f1=(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        roc_auc=roc_auc(labels, scores),
        confusion={"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        n_live=n_live,
        n_spoof=n_spoof,
    )


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    criterion: str = "min_acer",
    target_apcer: float = 0.01,
) -> float:
    """Choose an operating threshold. MUST be called on VALIDATION data only.

    criterion:
      'min_acer'      threshold minimising (APCER + BPCER) / 2
      'eer'           threshold where APCER ~= BPCER
      'apcer_target'  strictest threshold holding APCER <= target_apcer.
                      Appropriate for attendance: wrongly accepting a spoof (proxy
                      attendance) is worse than wrongly rejecting a real student,
                      who can simply try again.

    Selecting a threshold on the test set would make the reported test metrics
    optimistically biased — the test set would have been used for tuning.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))

    best_t, best_cost = 0.5, float("inf")
    for t in candidates:
        m = compute_metrics(labels, scores, float(t))
        if criterion == "min_acer":
            cost = m.acer
        elif criterion == "eer":
            cost = abs(m.apcer - m.bpcer)
        elif criterion == "apcer_target":
            cost = m.bpcer if m.apcer <= target_apcer else float("inf")
        else:
            raise ValueError(f"unknown criterion {criterion!r}")
        if cost < best_cost:
            best_t, best_cost = float(t), cost

    if best_cost == float("inf"):
        raise ValueError(
            f"no threshold achieves APCER <= {target_apcer}; the model is not strong "
            "enough for this operating point"
        )

    # Guard against the degenerate solution. Rejecting every sample trivially gives
    # APCER = 0, which satisfies the criterion while making the system unusable. This
    # is the failure mode where a metric looks perfect and the product is broken.
    final = compute_metrics(labels, scores, best_t)
    if final.bpcer >= 1.0:
        raise ValueError(
            f"the only threshold meeting APCER <= {target_apcer} rejects every bona "
            "fide sample (BPCER = 100%); the model cannot separate the classes"
        )
    return best_t
