# Kubernetes Replica Scaling — Measured Results

**Date:** 2026-09-08
**Cluster:** kind, single node, Docker Desktop. Intel i9-9880H 8C/16T, 16 GB RAM.
**Endpoint:** `GET /users` — API + Postgres round trip, with API-key auth enabled
**Load generator:** in-cluster pod (ClusterIP, as the frontend uses)
**Parameters:** concurrency 16, 500 requests per trial, 50 discarded warmup requests
**Backend resources:** requests 250m CPU / 512Mi, limits 1000m CPU / 1536Mi
**Trials:** 3 per configuration

## Raw trials (throughput, req/s)

| Replicas | Trial 1 | Trial 2 | Trial 3 | Median | Range |
|---:|---:|---:|---:|---:|---|
| 1 | 53.36 | 55.57 | 56.73 | **55.57** | 53.4 – 56.7 (±3%) |
| 2 | 177.71 | 102.19 | 171.59 | **171.59** | 102.2 – 177.7 (**±31%**) |
| 4 | 194.61 | 165.96 | 201.90 | **194.61** | 166.0 – 201.9 (±10%) |

## Median results

| Replicas | Throughput | P50 (ms) | P95 (ms) | Scaling vs 1 |
|---:|---:|---:|---:|---:|
| 1 | 55.57 req/s | 284 | 412 | 1.00x |
| 2 | 171.59 req/s | 38 | 317 | **3.09x** |
| 4 | 194.61 req/s | 33 | 315 | **3.50x** |

Zero errors across all nine trials.

## Reading these numbers honestly

**Variance is the headline, not the medians.** A single trial at 2 replicas produced
102 req/s; another produced 178 req/s on identical configuration. Reporting one run
would have been reporting noise. The single-trial figures collected earlier
(65.5 / 162.3 / 232.8) fall inside this spread and should not be quoted as precise.

**1 replica is stable; multi-replica is not.** The single-replica case varies by ±3%
because it is cleanly saturated at its own 1000m CPU limit — the bottleneck is fixed
and predictable. At 2 and 4 replicas the pods contend for the same 8 physical cores
alongside Postgres, the kubelet, the load generator and the host OS, so throughput
depends on how the scheduler happens to place work. **On a single-node cluster,
multi-replica throughput is inherently noisy.**

**1 → 2 is superlinear (3.09x for 2x the pods).** Not an error: one replica is CPU-capped
and its latency is dominated by queueing (P50 284 ms). The second replica removes that
queue as well as adding capacity, which is why P50 collapses 7.5x to 38 ms while
throughput more than triples. Superlinear 1→2 is the signature of a single instance
being resource-capped, not of unusually good parallelism.

**2 → 4 is nearly flat (1.13x).** Four replicas request 4000m CPU on a node with 8 cores
that are also doing everything else. The bottleneck has moved from the pod's own limit
to the node. P95 is unchanged (317 → 315 ms). **Adding replicas past 2 does not help on
this hardware**, and the HPA's `maxReplicas: 4` is therefore generous rather than useful
here.

## A confound that was found and removed

The first attempt measured through `kubectl port-forward` from the host:

| Replicas | Throughput |
|---:|---:|
| 1 | 64.63 req/s |
| 2 | 54.52 req/s |

That reads as "scaling makes it slower", and it was wrong. `kubectl port-forward` funnels
all traffic through a single proxied connection on the host, so the measurement was
bounded by the tunnel, and extra replicas only added scheduling overhead behind a
fixed-width pipe. Re-running identically from a pod inside the cluster produced the
results above. **The load generator was the bottleneck, not the system under test.**

Recorded rather than deleted because the wrong numbers were plausible: a 15% slowdown is
exactly the sort of result that gets rationalised as "coordination overhead" and
published.

## Limitations

- **Single-node cluster.** These figures do not extrapolate to a multi-node deployment
  where pods do not contend for the same CPUs. The flat 2→4 result in particular is a
  property of this node, not of the application.
- **`/users` does not exercise ML inference.** It measures the API and database path
  only. End-to-end verification is dominated by face detection (~107 ms per frame — see
  `inference-latency.md`), so verification throughput will be far lower and bounded by
  inference, not by this figure.
- **Three trials is few.** Enough to establish that variance is large; not enough for a
  confident distribution.
- Concurrency fixed at 16; throughput at saturation depends on offered load.
