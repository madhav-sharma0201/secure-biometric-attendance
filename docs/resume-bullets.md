# Resume Bullets

Every figure below is a measured result from this repository. Sources are named so each
can be defended in an interview.

---

**Trained and evaluated a CNN-based face anti-spoofing model on 439 video clips from
162 subjects, achieving 9.74% ACER (10.39% APCER / 9.09% BPCER, ROC-AUC 0.951) on a
subject-disjoint test set under ISO/IEC 30107-3 metrics.**
<sub>docs/results/liveness_modelA_cnn.json — test set is 33 subjects never seen in training.</sub>

**Diagnosed and fixed a generalisation failure that cut bona fide rejection (BPCER) from
60% to 9.1%, tracing it to a threshold-selection bug that left zero margin and to
training on only 24 live clips from 12 subjects.**
<sub>docs/results/liveness-run1-single-source.md — run 1 vs run 2, both retained.</sub>

**Ran a controlled ablation of single-frame CNN against CNN-BiLSTM, showing the temporal
model lost on aggregate ACER (15.15% vs 9.74%) while halving APCER on phone-replay
attacks (14.3% vs 35.7%), and shipped the model the evidence supported.**
<sub>Both architectures share one backbone so the comparison isolates temporal modelling.</sub>

**Built a fail-closed biometric verification service (FastAPI, PostgreSQL, ONNX Runtime,
ArcFace) with the decision engine as a pure function, verified by a brute-force sweep of
1,152 input combinations plus database-level duplicate-attendance constraints.**
<sub>backend/app/core/decision.py, backend/tests/ — 91 tests.</sub>

**Deployed to Kubernetes and measured replica scaling over 3 trials per configuration:
55.6 / 171.6 / 194.6 req/s at 1 / 2 / 4 replicas, identifying node CPU saturation beyond
2 replicas and a +/-31% variance that made single-trial numbers unreportable.**
<sub>docs/results/load-test.md — includes the port-forward confound that first inverted the result.</sub>

**Cut end-to-end verification latency 2.3x (7203 ms to 3146 ms) by pinning ONNX Runtime
thread counts to the container CPU limit and baking model weights into the image,
after profiling showed face detection — not the trained model (5.45 ms) — dominated.**
<sub>docs/results/inference-latency.md.</sub>

---

## Shorter variant (4 bullets)

- Trained a CNN face anti-spoofing model on 439 clips / 162 subjects: **9.74% ACER,
  0.951 ROC-AUC** on a subject-disjoint test set, evaluated under ISO/IEC 30107-3
  (APCER/BPCER/ACER) with 95% confidence intervals.
- Diagnosed a generalisation failure and **cut BPCER from 60% to 9.1%**, tracing it to a
  zero-margin threshold-selection bug and insufficient live-subject diversity.
- Built a fail-closed FastAPI + PostgreSQL + ONNX verification service with the decision
  engine as a pure function, **91 tests** including a 1,152-combination safety sweep.
- Deployed on Kubernetes and measured scaling across 3 trials/config
  (**55.6 / 171.6 / 194.6 req/s** at 1/2/4 replicas), plus a **2.3x** latency reduction
  from fixing container thread contention.

---

## What NOT to claim

The following would be false or unsupportable:

- **Not** "99% accuracy" or similar. Accuracy is not reported because on a 4:1 imbalanced
  set a model can hit 90% accuracy while accepting every attack — there is a test in the
  repo demonstrating exactly that.
- **Not** "production-ready" or "deployed in production". It runs on a single-node local
  cluster and has never served real users.
- **Not** "detects all presentation attacks". Worst-case APCER is **50%** on photos
  displayed on a lit screen.
- **Not** "real-time". P95 is 3.6 s, against a 2 s target that was **not met**.
- **Not** any claim of cross-vendor generalisation. All four datasets come from one
  vendor; run 1's cross-dataset ACER of 38-61% suggests generalisation beyond it is poor.

If asked "what went wrong?", the honest and strongest answer is run 1: validation ACER
of 0.00% alongside test BPCER of 60%, why subject-grouped splitting exposed it, and how
it was fixed.
