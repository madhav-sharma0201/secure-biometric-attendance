# Secure Biometric Attendance System

A trained CNN-LSTM face anti-spoofing model integrated with ArcFace identity
verification, deployed as containerised services on Kubernetes.

> **Status: in progress.** Every number in this README is measured. Sections awaiting a
> completed training run are marked `PENDING` rather than filled with plausible
> placeholders.

---

## 1. Problem

Face-recognition attendance is trivially defeated. Hold up a classmate's photo, or
replay a video of them on a phone, and a recognition-only system marks them present.
Proxy attendance is the actual threat, and identity matching does not address it.

The ML problem is therefore **not** "who is this person" — pretrained models solve that
well. It is:

> Given a camera stream, is this a live person or a presentation attack?

That question is the project's ML contribution. Identity verification is built on a
pretrained backbone around it.

## 2. What is and is not claimed

**Trained here:** the liveness model (CNN and CNN-LSTM), from an ImageNet-pretrained
backbone, on subject-grouped splits, evaluated with ISO/IEC 30107-3 metrics.

**Not trained here:** the face-recognition backbone. ArcFace (InsightFace `buffalo_l`)
is used pretrained. Training a face-recognition foundation model needs millions of
identities; what is built here is the enrollment, template, matching and threshold
pipeline around it.

## 3. Architecture

```mermaid
flowchart TD
    C[Browser camera<br/>8-frame burst] --> B[FastAPI backend]
    B --> D[Face detect + align<br/>SCRFD, ArcFace 5-point]
    D --> L[Liveness model<br/>CNN-LSTM, ONNX]
    L --> R[Recognition<br/>ArcFace embedding]
    R --> E[Decision engine<br/>pure, fail-closed]
    E --> P[(PostgreSQL)]
```

A single backend loads both ONNX models in-process. Separate liveness and recognition
microservices were considered and rejected: the models sit on the same request path,
scale together, and are never called independently, so splitting them would add two
network hops and two deployments to a system whose peak load is a classroom.
See `docs/01-scope.md`.

### Decision rule

Attendance is marked only if **all** of:

- exactly one face detected
- liveness confidence ≥ threshold
- identity similarity ≥ threshold, with a margin over the runner-up
- user registered and active
- an attendance session is currently open
- attendance not already marked for this (user, session)

Anything else rejects with a reason code. The engine is a pure function with a single
approval path, so any check added above it rejects by default. Rejections never return
the matched identity — otherwise `/verify` becomes an oracle for "who does this photo
most resemble".

Reason codes: `NO_FACE`, `MULTIPLE_FACES`, `LIVENESS_FAILED`,
`LOW_LIVENESS_CONFIDENCE`, `UNKNOWN_PERSON`, `LOW_IDENTITY_CONFIDENCE`,
`ALREADY_MARKED`, `NO_OPEN_SESSION`, `USER_INACTIVE`, `SYSTEM_ERROR`.

## 4. Dataset

| Role | Dataset | Contributes |
|---|---|---|
| Train + test | `trainingdatapro/attacks-with-2d-printed-masks-of-indian-people` | 21 subjects, 8 printed-mask attack variants, matched live capture |
| External test | `trainingdatapro/real-vs-fake-anti-spoofing-video-classification` | phone replay (unseen attack) |
| External test | `trainingdatapro/cut-out-printout-attacks` | printed mask, 9 further subjects |
| External test | `trainingdatapro/biometric-attacks-in-different-lighting` | display replay + live footage in 4 lighting conditions |

The primary dataset was chosen over larger alternatives because it is the only free
option with **real subject identifiers**, and because its live and spoof clips come
from the same people in the same session. That second property matters: combining a
live-only dataset with a spoof-only dataset lets a model separate the classes by camera
signature or compression artifacts rather than by spoofing cues, scoring near-perfectly
while learning nothing. Full reasoning in `docs/02-dataset.md`.

**Splitting is subject-grouped**, so the test set contains people the model has never
seen. Enforced by a checked invariant (`LeakageError`), not by convention.

Measured split:

```
split       groups   clips    live   spoof
train           12     120      24      96
val              4      40       8      32
test             5      50      10      40
test subjects: person_0, person_9, person_11, person_15, person_16
```

## 5. Preprocessing

```
video -> sampled frames -> SCRFD detection -> ArcFace 5-point alignment
      -> 112x112 crop -> ImageNet normalisation -> tensors
```

The **same detector and alignment code runs at training and serving time**. If crops
were produced differently in the two settings, the model would see a different input
distribution in production than it trained on, and the measured metrics would stop
predicting real behaviour.

Measured preprocessing yield:

| Dataset | Clips | Frames kept | Detection failures |
|---|---:|---:|---:|
| primary | 210 | 3,357 | 3 |
| real_vs_fake | 160 | 2,526 | 34 |
| printout_masks | 18 | 288 | 0 |
| lighting | 54 | 816 | 48 |

Detection failures are logged and surfaced, because a detector that fails
disproportionately on spoof clips is silently doing part of the anti-spoofing job and
would make the model's APCER look better than it is.

## 6. Models

Both architectures share one backbone (`build_backbone`), so the ablation isolates a
single variable — whether temporal modelling helps.

```
Model A (control)   frame  -> MobileNetV3-Small -> pool -> FC -> logit
Model B (main)      N=8    -> MobileNetV3-Small -> BiLSTM(128) -> FC -> logit
```

**Why temporal information should help.** A replayed video carries temporal artifacts a
single frame cannot show: screen refresh banding that moves with the display rather
than the face, absent micro-motion (blink irregularity, involuntary tremor), and
reflections fixed while the face moves. A print attack shows rigid planar motion with
no parallax.

**This dataset makes that hypothesis testable.** Four attack variants are hand-held and
four are mounted. A hand-held mask jitters; a mounted one does not. If the temporal
model helps, the gain should concentrate in the hand-held variants — the per-attack
table shows whether it does. That converts an architectural assertion into an experiment
with a predicted outcome.

**Why MobileNetV3-Small:** it must train in a free Kaggle session and run on CPU in the
deployed container (~2.5M params vs ResNet-50's ~25M). Anti-spoofing cues are largely
local texture, which does not need a high-capacity backbone; capacity is better spent
on the temporal component.

**Augmentation is applied identically across a sequence.** Per-frame augmentation would
flip frame 2 and not frame 3, injecting motion that never happened and destroying the
exact signal the LSTM exists to read.

## 7. Evaluation protocol

Per ISO/IEC 30107-3:

- **APCER** — fraction of attacks wrongly accepted as live (security failure)
- **BPCER** — fraction of real people wrongly rejected (usability failure)
- **ACER** — their mean

Accuracy is deliberately not the headline: on a 4:1 imbalanced set a model can score 90%
accuracy while accepting **every** attack. There is a test demonstrating exactly that.

**APCER is reported worst-case per attack type**, not pooled. A model that blocks every
print attack but accepts every replay shows a pooled APCER of 33% and a worst-case of
100%. An attacker uses the attack that works.

**Thresholds are selected on validation only; the test set is scored once.** External
sets are scored at that same threshold — re-tuning per set would turn them into
validation data and make the "unseen" claim false.

**Error rates are reported with 95% Wilson confidence intervals.** The test set is 5
subjects and 50 clips; a bare point estimate would imply a precision the data cannot
support.

### Results

`PENDING` — training run in progress.

| Model | Test ACER | APCER | BPCER | ROC-AUC |
|---|---|---|---|---|
| A single-frame CNN | PENDING | PENDING | PENDING | PENDING |
| B CNN + BiLSTM | PENDING | PENDING | PENDING | PENDING |

Per-attack and external-set results: `PENDING`.

## 8. Backend

FastAPI + SQLAlchemy + PostgreSQL, layered `api -> services -> repositories -> models`.

**Schema.** `users`, `face_embeddings`, `sessions`, `attendance`,
`verification_attempts`.

Two constraints are enforced by the **database**, not application code:

- `UNIQUE(user_id, session_id)` on attendance. An application-level "already marked?"
  check loses to concurrency: two simultaneous requests can both read *not marked*
  before either writes. The constraint is what actually holds.
- `ON DELETE CASCADE` from users to embeddings, so deleting biometrics leaves no
  orphans. Attendance is deliberately **not** cascaded — biometric data is deletable on
  request, attendance is an institutional record.

`verification_attempts` stores outcomes, scores and reasons — **never images or
embeddings**. A test fails if anyone adds such a column.

## 9. Deployment

Docker: multi-stage builds, non-root users, healthchecks, `cap_drop: ALL`. Model
weights are **mounted, not baked in** — retraining swaps a file rather than rebuilding
a 2.2 GB image.

Kubernetes (`kind`): Deployments, StatefulSet + PVCs, ConfigMap, Secret, Ingress, HPA,
resource requests and limits on every pod.

**`/health` and `/ready` are different endpoints on purpose.** `/health` reports process
liveness and never touches the database. `/ready` checks the database and returns 503 if
it is unreachable — a pod whose DB is down is alive but must not receive traffic. A
`startupProbe` with `failureThreshold: 30` covers model loading, which otherwise gets
the pod killed mid-load in a crash loop that looks like a broken image.

### Measured replica scaling

kind, single node, Intel i9-9880H (8C/16T), concurrency 16, in-cluster load generator:

| Replicas | Throughput | P50 | P95 | P99 | Errors |
|---:|---:|---:|---:|---:|---:|
| 1 | 65.5 req/s | 222 ms | 395 ms | 473 ms | 0 |
| 2 | 162.3 req/s (2.48x) | 44 ms | 291 ms | 383 ms | 0 |
| 4 | 232.8 req/s (3.55x) | 33 ms | 217 ms | 336 ms | 0 |

1→2 is superlinear because a single replica is capped by its own CPU limit and its
latency includes queueing. 2→4 is sublinear because the node's shared cores become the
bottleneck. An initial measurement through `kubectl port-forward` showed scaling making
things *slower*; that was the load generator, not the system. Documented in
`docs/results/load-test.md`.

This measures the API and database path, **not** ML inference. End-to-end verification
throughput will be dominated by inference latency, measured separately.

## 10. Security and privacy

- Only embeddings are stored, never raw face images.
- `model_version` is stored with every embedding; embeddings from different backbones
  are not comparable, so a backbone change forces re-enrollment rather than silently
  producing meaningless similarities.
- Biometric deletion (`DELETE /users/{id}/biometrics`) removes embeddings while
  preserving the attendance audit trail.
- Audit log records outcomes and reasons, never biometric data.
- Secrets via Kubernetes Secrets; none committed.
- Datasets are CC BY-NC / CC BY-NC-ND — non-commercial use only.

## 11. Limitations

- **21 training subjects, 5 test subjects.** Small. Confidence intervals are wide and
  are reported as such.
- **Single vendor, single capture pipeline** for the primary data. Generalisation is
  measured only by the external sets.
- **No replay attack in training** — replay is evaluated strictly as an unseen attack
  type, which is harder and more honest than training on it.
- **No 3D mask or deepfake coverage.** Out of scope, and absent from the training data.
- **Single-node Kubernetes.** Scaling figures do not extrapolate to a multi-node cluster.
- **No demographic fairness audit.** Public FAS datasets are demographically narrow.
  The primary dataset's Indian subjects partially match the intended deployment
  population but do not constitute a fairness evaluation.
- Trained and evaluated on academic datasets; **not validated on a deployed population**.

## 12. Setup

```bash
git clone https://github.com/madhav-sharma0201/secure-biometric-attendance.git
cd secure-biometric-attendance
cp .env.example .env          # then edit thresholds and credentials

python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest    # full test suite
```

Run the stack:

```bash
docker compose up --build     # frontend :8080, backend :8000, postgres :5432
```

Kubernetes:

```bash
kind create cluster --config kubernetes/kind-cluster.yaml
kubectl apply -f kubernetes/namespace.yaml
kubectl -n attendance create secret generic attendance-secrets \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 16)" \
  --from-literal=API_KEY="$(openssl rand -hex 16)"
docker build -t attendance-backend:latest -f backend/Dockerfile .
docker build -t attendance-frontend:latest ./frontend
kind load docker-image attendance-backend:latest attendance-frontend:latest --name attendance
kubectl apply -f kubernetes/
```

Training runs on Kaggle (see `notebooks/train_liveness.ipynb`); this machine has no
CUDA GPU.

## 13. Documents

- `docs/00-architecture.md` — requirements and architecture
- `docs/01-scope.md` — scope decisions and what was cut
- `docs/02-dataset.md` — dataset selection and split strategy
- `docs/03-recording-protocol.md` — self-collected data protocol
- `docs/results/load-test.md` — Kubernetes scaling measurements
