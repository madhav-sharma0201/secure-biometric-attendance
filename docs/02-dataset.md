# Phase 2 — Dataset Selection

## Final selection

| Role | Dataset | Why |
|---|---|---|
| **Primary (train + test)** | `trainingdatapro/attacks-with-2d-printed-masks-of-indian-people` | Only free option with real subject IDs; live and spoof from the same people and session |
| **Cross-attack eval** | `trainingdatapro/real-vs-fake-anti-spoofing-video-classification` | Phone replay — an attack type absent from training |
| **External test** | self-collected | Unseen people, unseen cameras, unseen conditions |

## Primary dataset structure

1.82 GB, 210 videos, 3–4 s each, indoor and outdoor, varied lighting.

```
attacks/1/<person>.mp4   real, no glasses              LIVE
attacks/2/<person>.mp4   real, with glasses            LIVE
attacks/3/<person>.mp4   mask, static                  SPOOF
attacks/4/<person>.mp4   mask + real glasses, static   SPOOF
attacks/5/<person>.mp4   mask, hand-held               SPOOF
attacks/6/<person>.mp4   mask + real glasses, handheld SPOOF
attacks/7..10/           printed-glasses variants      SPOOF
```

21 subjects x 10 videos. **The filename is the person index and is consistent across
all ten folders**, so `attacks/5/7.mp4` is the same person as `attacks/1/7.mp4`.

## Why this dataset over the alternatives

**It has subject identifiers.** Every other free candidate (`real-vs-fake`,
`axondata`, the vendor's other releases) names files by a bare counter with no person
mapping, forcing clip-level grouping. Here, subject-grouped splitting is possible:
the test set contains people the model has never seen. This was listed as requirement
MLR-3 and was nearly abandoned.

**Live and spoof come from the same subjects and the same capture session.** This
matters more than it appears. Combining a live-only dataset with a spoof-only dataset
lets a model separate the classes by camera signature, compression artifacts or
lighting rather than by spoofing cues — scoring near-perfectly while learning nothing
about presentation attacks. Matched capture removes that shortcut entirely.

**The static/hand-held split is a testable hypothesis.** Four attack variants are
mounted and four are hand-held. A hand-held mask jitters; a mounted one does not. If
the temporal model beats the single-frame baseline, this is where the gain should
appear, and the per-attack table will show whether it does. That converts the
CNN-LSTM from an architectural assertion into an experiment.

**Demographic relevance.** Subjects are Indian, matching the intended deployment
population. Public FAS datasets are demographically narrow; this partially addresses
the bias limitation recorded in Phase 0. It does not eliminate it.

## Split strategy

Subject-grouped, 60/20/20 by subject: **12 train / 4 val / 5 test subjects**
(120 / 40 / 50 clips). Verified: zero subject overlap between any two splits.

Stratification is disabled under subject grouping because each subject carries both
labels; mixed-label groups already guarantee both classes appear in every split.

## Known weaknesses — stated in the README, not hidden

- **21 subjects is small.** The test set is 5 people, 50 clips. ACER estimates from
  50 clips carry wide confidence intervals. Report the interval, not just the point
  estimate, and do not present a single ACER figure as precise.
- **Class imbalance is 4:1 spoof:live** (168 / 42). Handled with class weighting in
  the loss. APCER and BPCER are computed per class and are unaffected by imbalance,
  which is a further reason not to headline accuracy.
- **Primary attacks are print/mask only.** No replay attack appears in training.
  Replay is therefore evaluated strictly as an *unseen attack type* against the
  secondary dataset — a harder and more honest test than training on it.
- **Single vendor, single capture pipeline.** Generalisation beyond it is measured
  only by the secondary and self-collected sets.

## Licensing

Both datasets are CC BY-NC / CC BY-NC-ND — non-commercial use only. Acceptable for a
portfolio and academic project; attribution is included in the README. Any commercial
deployment would require different data.
