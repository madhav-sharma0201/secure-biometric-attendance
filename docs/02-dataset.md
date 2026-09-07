# Phase 2 — Dataset Selection

## Candidates evaluated

| Dataset | Format | Attacks | Subject IDs | License | Verdict |
|---|---|---|---|---|---|
| CASIA-FASD | video | print, cut-photo, replay | yes (per-subject dirs) | EULA, multi-day | rejected: 3-day timeline |
| Replay-Attack (Idiap) | video | print, photo, replay | yes | EULA | rejected: same |
| OULU-NPU / SiW | video | print, replay | yes | EULA, ~100 GB | rejected: EULA + size |
| CelebA-Spoof | **images only** | rich, annotated | yes (10k ids) | CC BY-NC | rejected as primary: no temporal signal |
| `trainingdatapro/real-vs-fake` | video | **replay only** | **no** | CC BY-NC-ND 4.0 | **selected (primary)** |
| `axondata/face-anti-spoofing-dataset` | video | mostly 3D masks | **no** | CC BY-NC 4.0 | rejected: wrong threat model |

## Why the axondata set was rejected despite looking largest

Its headline ("100,000+ videos, 11 attack types") describes the vendor's commercial
product; the free Kaggle sample is a small excerpt. More importantly its attack mix is
dominated by 3D masks — silicone, latex, resin, cloth. That is the wrong threat model
for attendance. The realistic proxy-attendance attack is a phone screen or a printed
photo, not a custom silicone mask of a classmate.

Choosing a dataset because it is large, rather than because its attacks match the
deployment threat, is a common and expensive mistake.

## Data splitting strategy

**Constraint discovered:** neither Kaggle dataset exposes a per-person identifier. The
vendor's `worker_id` field exists only in their paid full release. Subject-grouped
splitting is therefore impossible.

**Fallback: clip-level grouping.** All frames and all sequences derived from one source
video are assigned to exactly one split. This eliminates the dominant leakage risk —
near-duplicate frames appearing in both train and test.

**Residual risk and why it is limited here.** Clip grouping does not prevent the same
person appearing in both train and test. In this dataset that matters less than usual:
every attack video is a replay of that same person's genuine video, so each individual
appears on *both* sides of the label. Memorising an identity therefore provides no
signal for the live/spoof decision. The leakage is real but largely defanged by the
dataset's construction.

**This is a limitation, not a solved problem, and the README states it as such.**

## Consequence: the self-collected test set is now mandatory

Because no unseen-subject split is available from the public data, the self-collected
set is the project's only measurement of true generalisation to new people, and its
only source of print and laptop-screen attacks. It is promoted from optional (see
`01-scope.md`) to required.

Target: ~15 live clips (varied lighting, pose, distance, camera) and ~15 spoofs
(phone screen, laptop screen, printed photo). Roughly 20 minutes to record. Used
exclusively as a held-out external test set — never for training or threshold selection.

## Honest framing for the report

- Intra-dataset results are computed on clip-grouped splits of a single-vendor,
  single-attack-type dataset. They measure replay detection, not anti-spoofing broadly.
- The self-collected set is small, and collected by one person in one location. It
  indicates generalisation direction; it does not establish it.
- No claim is made about print, cut-photo, or 3D mask attacks beyond what the
  self-collected set contains.
