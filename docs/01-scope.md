# Revised Scope (3-day build)

Supersedes the roadmap in `00-architecture.md` §9. The architecture itself is unchanged
except where noted below.

## Timeline

| Day | Deliverable |
|---|---|
| 1 | Dataset chosen; preprocessing pipeline; subject-wise splits; **baseline CNN trained** |
| 2 | **CNN-LSTM trained**; APCER/BPCER/ACER + per-attack eval; threshold calibration; ONNX export |
| 3 | FastAPI backend + ArcFace + decision engine + Postgres + minimal UI; Docker Compose; **kind deployment + replica test**; README |

## Architecture change: single service, not three

`00-architecture.md` §4 proposed separate liveness and recognition services. **Revised: one
FastAPI backend loads both ONNX models in-process.**

Rationale: the two models sit on the same request path, scale together, and are never
called independently. Splitting them adds two network hops and two deployments to serve
a system whose peak load is a classroom. The monolith is the correct design at this
scale; distribution here would be over-engineering, not architecture.

Kubernetes value is unaffected — the cluster still runs three Deployments (backend,
postgres, frontend) exercising Services, ConfigMap, Secret, PVC, probes, HPA and Ingress.

## In scope

- Subject-wise grouped splits (no frame leakage)
- Baseline single-frame CNN **and** CNN-LSTM — baseline retained as the control that
  shows whether temporal modelling helps at all; doubles as the ablation
- APCER / BPCER / ACER, per-attack breakdown, ROC-AUC, confusion matrix
- Thresholds calibrated on validation; test set evaluated once
- ArcFace (InsightFace `buffalo_l`) enrollment + cosine matching; FAR/FRR measured
- Fail-closed decision engine as a pure, unit-tested function
- Postgres with `UNIQUE(user_id, session_id)` enforced at the DB level
- Docker Compose; measured end-to-end latency (P50/P95/P99)
- Kubernetes on kind, actually deployed, with a 1-vs-2 replica measurement
- README stating measured numbers and honest limitations

## Out of scope — and declared as such in the README

| Dropped | Reason |
|---|---|
| Augmentation ablation arm | Time; baseline-vs-LSTM comparison retained |
| Deep failure analysis (Phase 29) | Time; a handful of failure examples included instead |
| Active liveness / challenge-response (Phase 18) | Additive security layer, not core |
| MLflow (Phase 8) | Replaced by JSON run logs — same information, no infrastructure |
| Separate ML microservices (Phase 21) | See architecture change above |
| Built frontend toolchain | Single HTML page with `getUserMedia` |
| Full auth/authz, rate limiting (Phase 25) | Reduced to API key + password hashing |
| Integration and load test suites (Phase 26) | Unit tests on the decision engine only |
| 4-replica scaling test | Single-node kind on 8 shared cores cannot support it honestly; 1-vs-2 only |

## Optional if time permits (day 3)

A 20-minute self-collected test set — ~15 live clips across lighting conditions, ~15
spoofs (phone screen, laptop screen, printout). Used purely as an external generalisation
test. Highest value-per-minute item remaining: it is the only measurement of real-world
generalisation and the most convincing part of a live demo.

## Local development environment note

PyTorch publishes no wheels for Intel macOS on Python 3.13 (x86-64 Mac builds stopped
after torch 2.2.2, which predates 3.13). Two virtualenvs are therefore used locally:

- `.venv`   Python 3.13 — tests, backend, preprocessing utilities
- `.venv311` Python 3.11 + torch 2.2.2 + numpy<2 — running the ML pipeline locally on
  synthetic fixtures before committing a Kaggle GPU session to it

Training itself always runs on Kaggle. `.venv311` exists only so that shape errors,
config mismatches and export bugs are caught on a laptop in seconds rather than
after a preprocessing run on a GPU box.
