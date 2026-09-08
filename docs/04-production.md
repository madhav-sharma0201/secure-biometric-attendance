# Production Readiness

What is genuinely production-grade here, what is not, and why.

## Not production-ready: the model

**Worst-case APCER is 50%** on photographs displayed on a lit screen. Someone holding a
phone up to the camera is admitted roughly half the time. Pooled APCER (10.39%) hides
this, which is why ISO/IEC 30107-3 reports the worst attack species.

**P95 verification latency is 3.6 s** against a 2 s target.

Neither is fixable by hardening. Both need data and hardware:

| Gap | What it needs |
|---|---|
| 50% APCER on display attacks | Substantially more display/replay training data; the current corpus is print-mask dominated |
| 3.6 s P95 | GPU inference, or re-preprocessing and retraining at `det_size=320` on the serving path |
| Single-vendor data | Datasets from other capture pipelines to establish generalisation |

**Do not deploy this against real attendance fraud in its current state.** It is a
working, honestly-measured system, not a fielded product.

## Production-grade: the engineering

### Security

- **API-key auth** with constant-time comparison; refuses to serve (503) rather than
  serving openly when the key is unset or left at its default.
- **CORS is deny-by-default.** No wildcard. Same-origin through nginx in the deployed
  configuration; `CORS_ORIGINS` opts specific origins in.
- **Upload bounds** — per-file size cap enforced while streaming, plus caps on frame
  and enrollment-image counts. Without these, a shaped request pins CPU by forcing
  hundreds of face-detection passes.
- **Malformed input is a rejection, not a crash.** `cv2.imdecode` raises on empty or
  corrupt bytes; uploads are attacker-controlled, so decode failures return a clean
  rejection instead of a 500.
- **Rate limiting** on verification endpoints (per-pod; see Limitations).
- **NetworkPolicy** default-denies the backend: only the frontend and in-namespace
  scrapers may reach it; it may only reach Postgres and DNS.
- **Non-root containers** (uid 10001 / nginx 101), `cap_drop: ALL`,
  `allowPrivilegeEscalation: false`, `seccompProfile: RuntimeDefault`.
- **No secrets in the image or repo.** Kubernetes Secrets only; `secrets.example.yaml`
  is a template and the real file is gitignored.
- **CI fails the build on committed credentials** (gitleaks over full history).

### Data integrity

- **Alembic migrations** run in an init container before any app container starts, so
  a rollout cannot serve traffic against a schema it does not match. `create_all()` is
  local/test only — it silently skips existing tables, so a later column never appears
  and the failure surfaces at runtime.
- **`UNIQUE(user_id, session_id)`** enforced by the database. An application-level
  check loses to concurrency; the constraint is what actually holds.
- **Nightly `pg_dump` CronJob** with 14-day retention that verifies the dump is
  non-trivial before pruning — a zero-byte backup nobody checks is worse than none.

### Operations

- **`/health` vs `/ready` are distinct.** `/health` is process liveness and never
  touches the database. `/ready` returns 503 unless the database is reachable *and*
  both models loaded — a pod that cannot verify anyone must not receive traffic.
- **`startupProbe`** (`failureThreshold: 30`) covers model loading, which otherwise
  gets the pod killed mid-load in a crash loop that looks like a broken image.
- **Graceful shutdown**: `terminationGracePeriodSeconds: 40` with a preStop drain,
  because a verification takes seconds and a rollout would otherwise SIGKILL mid-request.
- **PodDisruptionBudgets** keep at least one replica during node drains.
- **Prometheus metrics** at `/metrics`: request counts and durations, verification
  outcomes by reason, liveness-score distribution, rate-limit hits, model-loaded gauge.
- **Structured JSON logs** with request IDs. No images, embeddings, bodies or headers.
- **Model weights baked into the image** — InsightFace otherwise downloads ~300 MB from
  GitHub on every pod start, which exceeded the probe budget and restarted pods.
- **ONNX thread counts pinned to the CPU limit.** ONNX Runtime sizes its pool from the
  *node's* CPU count, not the cgroup quota, so an unpinned 1-CPU pod ran 8 threads over
  one core. Pinning cut end-to-end latency from 7,203 ms to 3,146 ms.

### Privacy

- Only embeddings are stored, never raw images; enrollment images live in memory for
  the duration of the request.
- `model_version` travels with every embedding — embeddings from different backbones
  are not comparable, so a backbone change forces re-enrollment rather than silently
  producing meaningless similarity scores.
- `DELETE /users/{id}/biometrics` removes embeddings while preserving the attendance
  audit trail: biometric data is deletable on request, attendance is an institutional
  record.
- The audit table stores outcomes, scores and reasons — never images or embeddings, and
  a test fails if such a column is ever added.
- **Metrics carry no user labels.** A per-user counter would let scrape history
  reconstruct who attended what and when.

## Known limitations

| Limitation | Impact |
|---|---|
| Rate limiter is in-process | Effective global limit is N x the configured value with N replicas. Needs Redis for a true shared limit. |
| Single shared API key | No per-device identity or rotation. Real deployment needs per-kiosk credentials and OIDC for admins. |
| No TLS inside the cluster | Ingress terminates TLS; pod-to-pod traffic is plaintext. Needs a service mesh or per-service certs. |
| `readOnlyRootFilesystem: false` | InsightFace and ONNX Runtime write scratch files at load. |
| No log aggregation or tracing | Logs are structured but not shipped anywhere. |
| Backups are in-cluster | Same failure domain as the database. Real deployment ships them off-cluster. |
| Single-node cluster measurements | Scaling figures do not extrapolate to multi-node. |

## Deploy checklist

```bash
# 1. Secrets (never committed)
kubectl -n attendance create secret generic attendance-secrets \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)" \
  --from-literal=API_KEY="$(openssl rand -hex 24)"

# 2. Model artifact
gh release download v1.0.0 --dir models/

# 3. Images
docker build -t attendance-backend:latest -f backend/Dockerfile .
docker build -t attendance-frontend:latest ./frontend

# 4. Apply (migrations run automatically in the init container)
kubectl apply -f kubernetes/

# 5. Verify before sending traffic
kubectl -n attendance rollout status deploy/backend
kubectl -n attendance exec deploy/backend -- curl -fsS localhost:8000/ready
```

`/ready` must return `models_loaded: true`. If it does not, the pod cannot verify
anyone and must not receive traffic.

### Adopting migrations on an existing database

A database created by an earlier `create_all()` already has the tables but no Alembic
version row, so the init container fails with `relation "sessions" already exists`.
Stamp it once:

```bash
kubectl -n attendance run alembic-stamp --rm -i --restart=Never \
  --image=attendance-backend:latest --image-pull-policy=IfNotPresent \
  --command -- sh -c 'alembic stamp head'
```

A fresh database needs no stamp — `alembic upgrade head` creates everything.
