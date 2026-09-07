# Inference Latency — Measured (Phase 19)

**Hardware:** Intel i9-9880H, 8C/16T @ 2.3 GHz, 16 GB RAM. CPU only — no CUDA, no MPS.
**Runtime:** onnxruntime `CPUExecutionProvider`.
**Method:** 20 timed runs after 3 warmup runs, per stage, measured independently.

## Per-stage latency

| Stage | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|
| Liveness model, 8-frame sequence | 5.45 | 6.00 | 6.13 |
| Face detection, 1280x720 (SCRFD, det_size 640) | 107.23 | 126.20 | 145.33 |
| Face embedding incl. detection (ArcFace R50) | 117.18 | 152.18 | 191.94 |
| Face embedding alone (by subtraction) | ~10 | — | — |

> The liveness figure was measured on an **untrained** export of the same architecture.
> Inference latency depends on architecture, input shape and hardware, not on weight
> values, so the number is valid — but it is stated explicitly because it is not a
> trained model. It will be re-measured once training completes.

## The finding

**The trained model is not the bottleneck.** Face detection costs roughly 20x the
liveness model. This is the opposite of the intuitive assumption that the deep learning
component dominates, and it is the reason for measuring before optimising.

The original verification path made it worse:

```
8 frames x detection (107 ms)          = 856 ms
+ recognition.embed() re-detects        = 107 ms
+ liveness scoring                      =   5 ms
+ embedding                             =  10 ms
                                        ~ 978 ms of which 963 ms is detection
```

## Optimisation 1 — applied: reuse the aligned crop

`RecognitionService.embed()` accepted a full frame and ran the detector again, even
though the verification path had *already* detected and ArcFace-aligned that face in
order to score liveness. `embed_crop()` now takes the existing aligned crop.

**Saving: ~107 ms per verification, with no accuracy tradeoff** — it is the same crop,
produced by the same alignment, simply not recomputed.

## Optimisation 2 — measured, deliberately NOT applied

Detection cost scales with `det_size`:

| det_size | P50 (ms) | Speedup |
|---:|---:|---:|
| 640 | 101.64 | 1.00x |
| 480 | 59.94 | 1.70x |
| 320 | 23.85 | 4.26x |
| 256 | 14.39 | 7.06x |

Dropping to 320 would cut detection by 4.26x and take end-to-end verification from
roughly 1 s to under 250 ms. **It has not been applied**, because the training data was
preprocessed at `det_size=640`. Lower-resolution detection produces coarser landmarks
and therefore slightly different alignment, so serving at 320 while the model was
trained on 640-derived crops reintroduces exactly the train/serve distribution mismatch
that the shared preprocessing path exists to prevent.

The correct way to take this speedup is to re-preprocess the training data at 320 and
retrain, so both sides match. That is a ~30 minute preprocessing run plus retraining,
and it is recorded here as a quantified, available option rather than taken as a silent
change.

This is the point of measuring first: the largest available speedup is not in the model
that was trained, and it carries a correctness condition that is invisible from the
latency number alone.

## Remaining cost and what it means

After optimisation 1, detection over 8 frames still dominates at ~856 ms. Per-frame
alignment is not optional — the training pipeline aligned every frame independently, so
serving must too. Reducing the number of frames would change the model's input, and
reusing one frame's bounding box across the burst would drift as the head moves.

The real levers, in order of value:

1. Re-preprocess and retrain at `det_size=320` (4.26x on the dominant stage)
2. Detect at reduced resolution but align at full resolution, if landmark accuracy holds
3. Batch the 8 detections into one ONNX call rather than 8 sequential calls

None have been applied. They are listed as measured options, not as claims.

---

# Deployed latency (Kubernetes, measured end-to-end)

Measured against the running kind cluster via `POST /verify` with an 8-frame burst,
6 requests after a warmup, backend pod on Docker Desktop's Linux VM.

| Configuration | P50 | P95 |
|---|---:|---:|
| 1000m CPU limit, thread counts unpinned | 7203 ms | 8196 ms |
| 2000m CPU limit, `OMP_NUM_THREADS=2` | **3146 ms** | **3604 ms** |

**2.3x faster, and most of it was not the extra CPU.**

## The thread-contention bug

ONNX Runtime and OpenMP size their thread pools from the **node's** visible CPU count
(8 here), not from the container's cgroup quota. A pod limited to 1 CPU therefore ran
8 inference threads over one core's worth of quota and spent its time context-switching
and being throttled. Measured inside the pod at a 1 CPU limit, detection took ~699 ms
with `OMP_NUM_THREADS=1` — *faster* than the default 8-thread configuration.

`OMP_NUM_THREADS` and `ORT_NUM_THREADS` are now pinned in the Deployment alongside the
CPU limit, with a comment stating that the two must be changed together. This is a
standard containerised-inference failure and is invisible from application logs.

## Against the target

NFR-1 set a target of **P95 < 2 s**. The deployed system is at **P95 3.6 s**. The target
is **not met**, and the shortfall is dominated by running face detection on all eight
frames of the burst.

Measured options, none applied:

| Option | Expected effect | Cost |
|---|---|---|
| Reduce burst to 4 frames | ~2x faster | Changes the operating point; the model was evaluated on 8-frame clip averages, so it would need re-evaluating at 4 |
| Run on a GPU node | 10-50x on detection | Requires GPU hardware; none available here |
| Lighter detector (e.g. YuNet) | 3-5x | Different crops from those the liveness model was trained on — needs re-preprocessing and retraining |
| Native Linux host instead of Docker Desktop | ~2-4x | Docker Desktop on macOS runs containers in a VM; the same detection is ~24 ms on the host and ~699 ms single-threaded in-pod |

The honest summary: **the model is fast (5.45 ms) and the pipeline around it is slow.**
Detection dominates, and the remaining fixes all trade against either the evaluated
operating point or hardware that is not available.
