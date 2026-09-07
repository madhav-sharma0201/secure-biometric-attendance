# Kubernetes Replica Scaling — Measured Results

**Date:** 2026-09-08
**Cluster:** kind (single node), Docker Desktop, Intel i9-9880H 8C/16T, 16 GB RAM
**Endpoint:** `GET /users` (backend → Postgres round trip)
**Load generator:** in-cluster pod, concurrency 16, 600 requests, 60 discarded warmup requests
**Backend resources:** requests 250m CPU / 512Mi, limits 1000m CPU / 1536Mi

## Results

| Replicas | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Errors |
|---:|---:|---:|---:|---:|---:|
| 1 | 65.54 | 222.45 | 394.64 | 473.47 | 0 |
| 2 | 162.33 | 44.27 | 291.22 | 383.37 | 0 |
| 4 | 232.78 | 32.51 | 216.90 | 336.04 | 0 |

Scaling relative to 1 replica: **2 replicas = 2.48x**, **4 replicas = 3.55x**.

## Reading the numbers honestly

**1 → 2 replicas is superlinear (2.48x for 2x the pods).** That is not magic and not a
measurement error: a single replica is capped by its own 1000m CPU limit and is
saturated, so its latency includes substantial queueing. Adding the second replica
removes that queue as well as adding capacity, which is why P50 drops 5x (222 ms → 44 ms)
while throughput more than doubles. Superlinear scaling from 1 to 2 is the signature of
a single instance being resource-capped rather than of unusually good parallelism.

**2 → 4 replicas is sublinear (1.43x for 2x the pods).** By 4 replicas the CPU limits
sum to 4000m against 8 physical cores that are also running Postgres, the kubelet, the
load generator and the host OS. The bottleneck has moved from the pod's own limit to
the node's total CPU. Further replicas would flatten further.

**Latency improves monotonically while throughput gains shrink.** P95 falls 394 → 291 →
217 ms. Extra replicas keep reducing queueing even once they stop adding much
throughput.

## A confound that was found and removed

The first attempt measured through `kubectl port-forward` from the host and produced:

| Replicas | Throughput (req/s) |
|---:|---:|
| 1 | 64.63 |
| 2 | 54.52 |

That reads as "scaling makes it slower". It was wrong. `kubectl port-forward` funnels
all traffic through a single proxied connection on the host, so the measurement was
bounded by the tunnel rather than by the backend, and adding replicas only added
scheduling overhead behind a fixed-width pipe.

Re-running the identical test from a pod inside the cluster — hitting the ClusterIP
service directly, exactly as the frontend does — produced the results in the table
above. **The load generator, not the system under test, was the bottleneck.**

This is recorded rather than quietly deleted because the failure mode is common and the
wrong numbers were plausible: a 15% slowdown is exactly the sort of result someone
would rationalise as "coordination overhead" and publish.

## Limitations

- Single-node cluster. These figures do not extrapolate to a multi-node deployment
  where pods do not contend for the same CPUs.
- `/users` exercises the API and database path but **not** ML inference. Liveness and
  recognition latency are measured separately (Phase 19) and are far larger; the
  end-to-end verification throughput will be dominated by them, not by this figure.
- Concurrency was fixed at 16. Throughput at saturation depends on offered load.
- One run per configuration; no repeat trials, so run-to-run variance is unquantified.
