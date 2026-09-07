"""Final evaluation: fix the threshold on validation, then measure once.

Protocol enforced here:
  1. Threshold is selected on the VALIDATION split only.
  2. The primary test split is scored once at that threshold.
  3. Every external set is scored at the SAME threshold — no re-tuning per set,
     which is what makes them honest measurements of unseen conditions.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from ml.evaluation.metrics import compute_metrics, select_threshold
from ml.training.train import score_clips


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a rate of k errors in n trials.

    Reported alongside every error rate because the test splits here are small
    (50 clips on the primary test set). A point estimate of '4% ACER' from 50 clips
    implies a precision the data cannot support; the interval makes that visible
    instead of hiding it.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def evaluate_all(model, splits, external, crops_dirs, mode, seq_len, device,
                 threshold_criterion="min_acer", target_apcer=0.01) -> dict:
    """Returns a report dict; also safe to json.dump."""
    primary_dir = crops_dirs["primary"]

    y_val, s_val, a_val, _ = score_clips(model, splits["val"], primary_dir, mode,
                                         seq_len, device)
    threshold = select_threshold(y_val, s_val, criterion=threshold_criterion,
                                 target_apcer=target_apcer)
    val_m = compute_metrics(y_val, s_val, threshold, a_val)

    y_te, s_te, a_te, id_te = score_clips(model, splits["test"], primary_dir, mode,
                                          seq_len, device)
    test_m = compute_metrics(y_te, s_te, threshold, a_te)

    n_spoof = int((y_te == 0).sum())
    n_live = int((y_te == 1).sum())
    report = {
        "threshold": threshold,
        "threshold_criterion": threshold_criterion,
        "threshold_selected_on": "validation split",
        "validation": _pack(val_m),
        "test": _pack(test_m),
        "test_confidence_intervals_95": {
            "apcer": wilson_interval(test_m.confusion["fp"], n_spoof),
            "bpcer": wilson_interval(test_m.confusion["fn"], n_live),
        },
        "test_misclassified": [
            {"clip_id": c, "label": int(l), "score": float(s), "attack": a}
            for c, l, s, a in zip(id_te, y_te, s_te, a_te)
            if (s >= threshold) != (l == 1)
        ],
        "external": {},
    }

    for name, rows in external.items():
        d = crops_dirs.get(name, crops_dirs["primary"])
        y, s, a, ids = score_clips(model, rows, d, mode, seq_len, device)
        if len(y) == 0:
            continue
        m = compute_metrics(y, s, threshold, a)
        report["external"][name] = {
            **_pack(m),
            "n_clips": int(len(y)),
            "note": "unseen during training; scored at the validation-selected threshold",
        }
    return report


def _pack(m) -> dict:
    return {
        "apcer": m.apcer, "apcer_worst_case": m.apcer_worst_case,
        "apcer_per_attack": m.apcer_per_attack,
        "bpcer": m.bpcer, "acer": m.acer, "roc_auc": m.roc_auc,
        "accuracy": m.accuracy, "precision": m.precision, "recall": m.recall,
        "f1": m.f1, "confusion": m.confusion,
        "n_live": m.n_live, "n_spoof": m.n_spoof,
    }


def format_report(report: dict) -> str:
    L = []
    t = report["test"]
    ci = report["test_confidence_intervals_95"]
    L.append(f"threshold {report['threshold']:.4f} "
             f"(criterion: {report['threshold_criterion']}, selected on validation)\n")
    L.append("PRIMARY TEST SET (unseen subjects)")
    L.append(f"  ACER   {t['acer']*100:6.2f}%")
    L.append(f"  APCER  {t['apcer']*100:6.2f}%   95% CI [{ci['apcer'][0]*100:.1f}, {ci['apcer'][1]*100:.1f}]")
    L.append(f"  BPCER  {t['bpcer']*100:6.2f}%   95% CI [{ci['bpcer'][0]*100:.1f}, {ci['bpcer'][1]*100:.1f}]")
    L.append(f"  AUC    {t['roc_auc']:.4f}")
    L.append(f"  clips  {t['n_live']} live / {t['n_spoof']} spoof")

    if t["apcer_per_attack"]:
        L.append("\n  per-attack APCER (higher = more attacks accepted as live)")
        for k, v in sorted(t["apcer_per_attack"].items(), key=lambda kv: -kv[1]):
            L.append(f"    {k:<32} {v*100:6.2f}%")

    if report["external"]:
        L.append("\nEXTERNAL SETS (unseen attack types, same threshold)")
        L.append(f"  {'set':<18}{'clips':>7}{'ACER':>9}{'APCER':>9}{'BPCER':>9}{'AUC':>8}")
        for name, e in report["external"].items():
            L.append(f"  {name:<18}{e['n_clips']:>7}{e['acer']*100:>8.2f}%"
                     f"{e['apcer']*100:>8.2f}%{e['bpcer']*100:>8.2f}%{e['roc_auc']:>8.3f}")
        for name, e in report["external"].items():
            if e["apcer_per_attack"]:
                L.append(f"\n  {name} per-attack APCER")
                for k, v in sorted(e["apcer_per_attack"].items(), key=lambda kv: -kv[1]):
                    L.append(f"    {k:<32} {v*100:6.2f}%")

    n_err = len(report["test_misclassified"])
    L.append(f"\n{n_err} misclassified test clip(s)")
    for e in report["test_misclassified"][:10]:
        kind = "spoof accepted as live" if e["label"] == 0 else "live rejected as spoof"
        L.append(f"  {e['clip_id']:<40} {kind:<24} score {e['score']:.3f}")
    return "\n".join(L)


def save_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
