# Phase 2 — Dataset Selection

## Final selection

| Role | Dataset | Why |
|---|---|---|
| **Primary (train + test)** | `trainingdatapro/attacks-with-2d-printed-masks-of-indian-people` | Only free option with real subject IDs; live and spoof from the same people and session |
| **External test 1** | `trainingdatapro/real-vs-fake-anti-spoofing-video-classification` | Phone replay — unseen attack type |
| **External test 2** | `trainingdatapro/cut-out-printout-attacks` | Printed mask, 9 further subjects |
| **External test 3** | `trainingdatapro/biometric-attacks-in-different-lighting` | Display replay + 4 lighting conditions |
| **Demo only** | self-collected (~6 clips) | Demo video and qualitative failure examples; no metrics claimed |

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


## External test sets (added after primary selection)

None of these are trained on. Each is evaluated once, at the threshold already fixed on
the primary validation split, to answer: **does the model detect attack types it never
saw during training?** This is a harder and more informative question than training on
every attack type, and it is closer to reality — a deployed system meets attacks its
training data did not contain.

| Adapter | Dataset | Size | Contributes |
|---|---|---|---|
| `real_vs_fake` | real-vs-fake-anti-spoofing | 3.3 GB, 160 clips | phone replay |
| `printout_masks` | cut-out-printout-attacks | 706 MB, 27 clips, 9 subjects | printed mask, live selfie/video |
| `lighting` | biometric-attacks-in-different-lighting | 1.5 GB, 54 clips | display replay, print mask with cut-outs, and live footage in dark / daylight / lit / nightlight |

### A label trap in the lighting dataset

Its `type` values are not self-explanatory and guessing from the names inverts them:

| Type | Actually is |
|---|---|
| `darkroom_video`, `daylight_video`, `lightroom_video`, `nightlight_video` | **LIVE** — a real person moving their head under that lighting |
| `darkroom_photo`, `daylight_photo`, `lightroom_photo`, `nightlight_photo` | **SPOOF** — a photo displayed on a monitor and filmed |
| `monitor_video` | **SPOOF** — a video replayed on a monitor and filmed |
| `mask`, `outline` | **SPOOF** — printed 2D mask, with and without cut-out eye holes |

`"<condition>_video"` reads like a replay attack and is not. Labelling it as one would
have inverted every live sample in this set. The mapping is taken verbatim from the
dataset card and is pinned by a regression test
(`tests/test_manifest.py::test_lighting_video_types_are_live_not_spoof`).

### Why the self-collected set was downgraded

It was called mandatory when clip-level grouping was the only option and the project
had no unseen-subject test at all. With subject-grouped splits on the primary dataset
plus three external sets covering print, phone replay and display replay across many
subjects and lighting conditions, that gap is closed by public data — with more
subjects and more cameras than one person could film.

What self-collected footage still uniquely provides is the deployment camera and a
demo. Roughly six clips are recorded on day 3 for the demo video and qualitative
failure examples only. **No metric is computed from them**, because six clips from one
person cannot support one.
