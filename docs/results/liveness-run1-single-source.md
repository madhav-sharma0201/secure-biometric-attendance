# Liveness Run 1 — Single-Source Training (FAILED, and why it matters)

**Kaggle kernel version 6.** Trained on the printed-masks dataset alone.
Kept because the failure is the most instructive result the project produced.

## Setup

- Train / val / test: 12 / 4 / 5 **subjects**, subject-grouped (no leakage)
- 120 / 40 / 50 clips; **24 live** and 96 spoof clips in training
- MobileNetV3-Small backbone, ImageNet-pretrained
- `pos_weight = 4.0` for the 4:1 spoof:live imbalance
- Threshold selected on validation, test scored once

## Results

| Model | Val ACER | Test ACER | Test APCER | Test BPCER | Test AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| A single-frame CNN | 0.00% | 31.25% | 2.50% | **60.00%** | 0.920 | 0.3392 |
| B CNN + BiLSTM | 1.56% | 26.25% | 2.50% | **50.00%** | 0.802 | 0.0979 |

95% CI on Model A: APCER [0.4, 12.9], BPCER [31.3, 83.2].

External sets, scored at the same threshold:

| Set | A ACER | B ACER |
|---|---:|---:|
| real_vs_fake (phone replay) | 43.12% | 38.75% |
| printout_masks | 61.11% | 50.00% |
| lighting (display replay) | 46.13% | 37.42% |

Per-attack APCER was 0.00% on every mask variant for both models, except
`mask_static_printedglasses` at 20.00%.

## Diagnosis

**Validation ACER 0.00%, test ACER 26–31%.** The model did not learn to detect
presentation attacks. It learned the four validation subjects.

**The errors are almost entirely bona fide rejections.** APCER 2.5% against BPCER
50–60%: it blocks attacks nearly perfectly and turns away half of genuine users. As a
product that is unusable. Misclassified test clips make it concrete —
`person_16__live_no_glasses` scored 0.056 and `person_11__live_glasses` 0.022. Real
faces, confidently called spoofs.

Two independent causes, both fixed in run 2:

**1. Too few live subjects.** 24 live clips from 12 people is not enough to learn what
a live face looks like in general. With 96 spoof clips and `pos_weight = 4.0`, the
cheapest hypothesis available to the model is "faces I recognise are live, everything
else is spoof" — and it took it.

**2. The threshold had zero margin.** `select_threshold` only considered *observed*
scores as candidates. On well-separated validation data the lowest-cost threshold is
therefore the lowest live score itself, leaving no margin below it. Any unseen live
face scoring even slightly lower is rejected. Model B's threshold was 0.0979 and test
live faces scored 0.022–0.056 — just underneath. Candidates are now the midpoints
between consecutive scores, which places the operating point in the middle of the
separating gap.

The two compound: cause 1 pushes unseen live faces to low scores, cause 2 puts the
threshold exactly where those scores land.

## Why this run is kept

Had the frames been split randomly instead of by subject, this model would have
reported **~0% ACER** and the project would have shipped a number that was entirely
fake. Three methodology choices turned a flattering result into a true one:

- **Subject-grouped splits** — exposed that validation performance did not transfer
- **BPCER reported separately from APCER** — a single ACER of 31% hides that the system
  rejects 60% of real users; the split shows exactly which error dominates
- **Held-out external sets** — confirmed the failure was general, not specific to one
  test split

The temporal model beat the single-frame baseline on **every** external set
(38.75 vs 43.12, 50.00 vs 37.42... see table), which is the ablation's clearest signal:
CNN-LSTM generalises better to unseen sources even where its in-domain AUC is lower.

The predicted mechanism — temporal cues helping specifically on hand-held masks — did
**not** appear: APCER was 0% on hand-held and static variants alike. Attack detection
was never the failure mode. That prediction is recorded as falsified.
