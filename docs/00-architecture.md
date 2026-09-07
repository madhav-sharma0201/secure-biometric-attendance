# Phase 0 — Requirements & Architecture

**Project:** Secure Biometric Attendance System
**Core ML problem:** Given a camera stream, decide whether the presented face is a *bona fide live person* or a *presentation attack* (print / screen / replay), then verify identity before marking attendance.
**Honest one-line description:** A trained CNN-LSTM face anti-spoofing model integrated with a pretrained ArcFace identity-verification pipeline, deployed as containerised microservices on Kubernetes.

---

## 0. Hardware reality (drives everything below)

| Resource | Available | Implication |
|---|---|---|
| CPU | Intel i9-9880H, 8C/16T @ 2.3 GHz | Fine for inference, unusable for training |
| GPU | None (no CUDA, no MPS) | **All training happens off-machine** |
| RAM | 16 GB | Caps K8s local cluster size and model size |
| Disk | ~95 GB free | Rules out CelebA-Spoof (76 GB) and OULU-NPU (~100 GB) as *local* datasets |

**Decision: split the workload.**

- **Training / heavy preprocessing → Google Colab or Kaggle** (free T4/P100). Kaggle gives 30 GPU-hours/week and 20 GB persistent datasets; Colab gives ~12 h sessions. Same repo, run via a thin notebook that clones the repo and calls `ml/training/train.py`.
- **Everything else → local.** Preprocessing dev on a small subset, evaluation, inference, backend, Postgres, Docker, Kubernetes (kind), load testing.

This is not a compromise — it's how ML work is actually organised. Code is written and tested locally on a 50-sample subset; the full run is launched on a GPU box.

---

## 1. Functional requirements

**FR-1 Enrollment.** An admin registers a user (student_id, name, email) and enrolls their face from N≥3 captured images. The system stores face *embeddings*, not raw images.
**FR-2 Verification.** Given a short burst of camera frames, the system returns a decision with liveness confidence, identity confidence, and matched user.
**FR-3 Attendance marking.** On an approved verification within an open session, attendance is recorded exactly once per (user, session).
**FR-4 Sessions.** An admin creates attendance sessions with a start/end time. Verification outside an open session is rejected.
**FR-5 Reporting.** Query attendance by session and by user.
**FR-6 Rejection transparency.** Every rejection carries a machine-readable reason code.
**FR-7 Fail-closed.** Any uncertainty, error, timeout, or partial failure results in *no* attendance being marked.
**FR-8 Data rights.** A user's biometric data can be deleted on request without destroying the attendance audit trail.

**Rejection reason codes:** `NO_FACE`, `MULTIPLE_FACES`, `LIVENESS_FAILED`, `LOW_LIVENESS_CONFIDENCE`, `UNKNOWN_PERSON`, `LOW_IDENTITY_CONFIDENCE`, `ALREADY_MARKED`, `NO_OPEN_SESSION`, `SYSTEM_ERROR`.

*(`LIVENESS_FAILED` = classified spoof with confidence; `LOW_LIVENESS_CONFIDENCE` = score sits in the uncertain band between thresholds. Same distinction for identity.)*

---

## 2. ML requirements

**MLR-1** The liveness model is trained by us. Not an API call, not an off-the-shelf checkpoint. This is the project's ML contribution.
**MLR-2** The face recognition backbone is pretrained (ArcFace/InsightFace). We build enrollment, embedding, matching and threshold calibration around it. Training a face-recognition foundation model from scratch is neither feasible nor sensible.
**MLR-3** Splits are grouped by **subject**, never by frame. Test subjects are never seen in training.
**MLR-4** Decision thresholds are selected on validation data only. The test set is touched once.
**MLR-5** Liveness is reported with APCER / BPCER / ACER, per-attack breakdown, and ROC-AUC — not accuracy alone.
**MLR-6** A **cross-dataset evaluation** is mandatory (train on dataset A, test on dataset B + our own collected data). Intra-dataset FAS numbers are near-saturated and misleading on their own.
**MLR-7** Every claim in the README traces to a logged experiment run. No fabricated numbers.
**MLR-8** An ablation (single-frame CNN vs CNN-LSTM vs CNN-LSTM+aug) justifies the architecture experimentally.

---

## 3. Non-functional requirements

| ID | Requirement | How it's verified |
|---|---|---|
| NFR-1 | End-to-end verification latency budget: target P95 < 2 s on CPU | Phase 19 measurement |
| NFR-2 | Liveness model < 20 MB, runs on CPU | Model size check |
| NFR-3 | Services are stateless and horizontally scalable | Phase 23 load test |
| NFR-4 | Biometric data encrypted at rest, never logged | Phase 25 |
| NFR-5 | Reproducible: fixed seeds, pinned deps, config-driven training | Re-run produces same metrics ± tolerance |
| NFR-6 | Full stack runs locally via `docker compose up` | Manual check |
| NFR-7 | All ML services expose `/health` and `/ready` | K8s probes |

Note NFR-1 is a *target*, not a claim. It gets replaced by a measured number in Phase 19.

---

## 4. System architecture

```
Browser (camera, getUserMedia)
        │  captures N-frame burst, sends as multipart
        ▼
   Backend API (FastAPI)  ── orchestrator, fail-closed
        │
        ├──► Liveness Service   (our trained CNN-LSTM, ONNX)
        │
        ├──► Recognition Service (InsightFace/ArcFace, ONNX)
        │
        └──► PostgreSQL  (users, embeddings, sessions, attendance)
```

**Why the backend orchestrates rather than the frontend calling ML services directly:**
the decision logic must be server-side and untrusted-client-proof. A frontend that calls the liveness service itself can simply skip it. The ML services are internal-only (ClusterIP), never exposed via Ingress.

**Why three services rather than one monolith:** they have different resource profiles and scaling behaviour (recognition is embedding + a vector search; liveness is a sequence model). It also gives a real reason for the Kubernetes work rather than a contrived one. Honest tradeoff: two extra network hops add latency (~5–15 ms each locally) and operational complexity for a system that will never see high traffic. This is justified as a *learning and demonstration* objective, and I'll measure the hop cost so the README can state it truthfully.

---

## 5. ML architecture

### Liveness (the contribution)

```
N sampled frames  ──► face detect + align (per frame, 112×112)
                          │
                          ▼
              shared CNN backbone (MobileNetV3-Small / EfficientNet-B0)
                          │  N × D feature vectors
                          ▼
                     BiGRU / LSTM  (hidden 128)
                          │
                          ▼
                    Dense → 1 logit → LIVE / SPOOF
```

**Why temporal helps against replay:** a replayed video on a screen carries artefacts that are *temporally coherent in the wrong way* — screen refresh/moiré banding that moves with the display rather than the face, absent micro-motion (involuntary head tremor, blink irregularity, pulse-driven skin colour shift), and reflection highlights that stay fixed while the face moves. A single frame can be sharp and convincing; a sequence exposes that the "person" doesn't move like a person. A print attack additionally shows rigid planar motion — the whole face translates as a flat surface with no parallax.

**Why MobileNetV3-Small, not ResNet-50:** it must run on CPU in the deployed system and train inside a free Colab session. ~2.5M params vs 25M. Alternative considered: EfficientNet-B0 (5.3M) — I'll try both if time allows, it's a cheap ablation axis.

**Sequence length:** start at N=8 frames sampled at stride 3 from a ~1 s window. Longer sequences capture more motion but cost linearly in both training and inference latency. This is a config value, tuned once.

**What could go wrong:** the model learns dataset-specific artefacts (a particular camera's compression signature) rather than spoofing cues. This is *the* known failure mode of FAS research and is why MLR-6 (cross-dataset eval) is non-negotiable. Expect intra-dataset ACER of 1–5% and cross-dataset ACER of 15–30% — and expect to report both honestly. A project that reports the gap is more credible than one that hides it.

### Recognition

Pretrained InsightFace `buffalo_l` (RetinaFace detector + ArcFace R50 recogniser, ONNX, CPU-capable). Enrollment averages L2-normalised embeddings from ≥3 images into a template. Matching = cosine similarity against registered templates, threshold calibrated on validation pairs. `model_version` stored alongside every embedding so a backbone change forces re-enrollment rather than silently corrupting matches.

---

## 6. Database architecture

PostgreSQL. Tables per spec, with these additions:

- `face_embeddings.embedding` as `BYTEA` (or `pgvector` if we add the extension) + `dim` + `model_version`. At classroom scale (<10k users) brute-force cosine in NumPy is faster than a vector index and far simpler. pgvector only if scale demands it.
- `attendance` gets `UNIQUE (user_id, session_id)` — duplicate prevention enforced at the **database level**, not just in application code. Application-level checks lose to concurrent requests.
- `attendance.status` enum: `present` / `flagged`.
- `users.status`: `active` / `inactive` / `biometrics_deleted`.
- Rejected attempts go to a separate `verification_attempts` audit table (reason code, confidences, timestamp — **no images, no embeddings**).

---

## 7. API architecture

Routes as specified, plus `/sessions/{id}/close`. Layering: `api/` (HTTP + validation only) → `services/` (business logic, decision engine) → `repositories/` (DB access) → `models/` (SQLAlchemy). Pydantic schemas at the boundary. The decision engine is a **pure function** of its inputs — no DB, no HTTP — so it is trivially unit-testable, which matters because it is the security-critical component.

---

## 8. Kubernetes architecture

Local `kind` cluster. Namespace `attendance`. Deployments for backend / liveness / recognition / frontend, StatefulSet + PVC for Postgres. ConfigMap for thresholds and model versions, Secret for DB credentials. Resource requests/limits on every pod (essential — without limits, the load test measures nothing meaningful). Readiness probes gate traffic until models are loaded into memory; liveness probes restart hung pods. HPA on CPU for the two ML services.

**Honest prediction:** with 8 physical cores shared across the whole cluster, scaling from 2→4 replicas of a CPU-bound inference service will likely show *diminishing or zero* throughput gain, because there is no spare CPU. That is a real, reportable finding — "scaling improved throughput from 1→2 replicas by X% and plateaued at 4 due to CPU saturation on a single-node cluster" is a far stronger sentence than a fabricated linear-scaling claim.

---

## 9. Development roadmap

Grouped into milestones with a critical path. Dataset access requests are **day-1 blocking items** — some require signed EULAs with multi-day turnaround.

| Milestone | Phases | Gate to pass |
|---|---|---|
| M0 Foundations | 1, 2 | Repo scaffolded; dataset access secured; split strategy written |
| M1 Data | 3, 4 | Preprocessing reproducible from a script; EDA documented |
| M2 Baseline | 5, 7, 8 | Single-frame CNN trained; metrics logged; tracker working |
| M3 Main model | 6, 9, 10 | CNN-LSTM beats baseline on ACER; per-attack table produced |
| M4 Generalisation | 11, 28, 29 | Custom data collected; ablation table; failure analysis |
| M5 Identity | 12, 13 | Enrollment + matching working; FAR/FRR measured |
| M6 System | 14, 15, 16, 17 | End-to-end verification works locally |
| M7 Deployment | 19, 20, 21, 22, 23 | Runs on kind; load test measured |
| M8 Hardening | 24, 25, 26 | Auth, logging, tests |
| M9 Report | 27, 30, 31 | README + resume bullets from measured numbers only |

Phase 18 (active liveness) is optional and deferred to the end.

---

## 10. Risks and limitations

| Risk | Severity | Mitigation |
|---|---|---|
| No local GPU | **High** | Colab/Kaggle for training; local for everything else |
| Dataset EULA delays or refusals | **High** | Request access to 2–3 datasets on day 1, in parallel |
| Model overfits to dataset artefacts | **High** | Cross-dataset eval + custom collected test set; report the gap |
| Intra-dataset metrics look implausibly good | Medium | Always report cross-dataset alongside; never quote intra-dataset alone |
| 16 GB RAM insufficient for full local K8s stack | Medium | Small models, tight resource limits, scale test on 1/2/4 only |
| Scope: 31 phases is a lot | Medium | Milestone gates; M0–M4 is already a strong project by itself |
| Custom data collection privacy | Medium | Written consent, no raw image retention, documented deletion |

**Stated limitations for the README (write these now, be honest later):**
- Evaluated on academic datasets and a small self-collected set; not validated on a deployed population.
- Not tested against 3D mask or deepfake attacks — out of scope, and the training data contains no such attacks.
- Single-node Kubernetes cluster; scalability findings do not extrapolate to a multi-node production cluster.
- No formal fairness audit across skin tone / demographics; public FAS datasets are demographically narrow. Any deployment claim would require this.
