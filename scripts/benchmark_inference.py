"""Per-stage inference latency (Phase 19).

Measures face detection, liveness scoring and face embedding separately, because
"end-to-end latency" alone does not tell you what to optimise.

On weights: inference latency depends on architecture, input shape and hardware, not on
the VALUES of the weights. If no trained checkpoint is present, a randomly initialised
model of the same architecture is exported and used, and the report says so. That gives
a valid latency figure while being explicit that it is not a trained model.

Usage:
    python scripts/benchmark_inference.py --runs 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarise(name: str, times_ms: list[float], note: str = "") -> dict:
    return {
        "stage": name,
        "n": len(times_ms),
        "mean_ms": round(statistics.fmean(times_ms), 2),
        "p50_ms": round(percentile(times_ms, 0.50), 2),
        "p95_ms": round(percentile(times_ms, 0.95), 2),
        "p99_ms": round(percentile(times_ms, 0.99), 2),
        "min_ms": round(min(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "note": note,
    }


def bench_liveness(runs: int, warmup: int, seq_len: int, image_size: int) -> dict:
    import os
    import onnxruntime as ort

    model_path, note = "models/liveness.onnx", "trained model"
    if not os.path.exists(model_path):
        import torch
        from ml.liveness.models import build_model

        os.makedirs("models", exist_ok=True)
        model_path = "models/_bench_untrained.onnx"
        note = ("UNTRAINED model of the same architecture — latency is weight-"
                "independent, but this is not a trained model")
        model = build_model({"arch": "cnn_lstm", "backbone": "mobilenet_v3_small",
                             "pretrained": False, "hidden_size": 128})
        model.eval()
        dummy = torch.randn(1, seq_len, 3, image_size, image_size)
        torch.onnx.export(model, dummy, model_path, opset_version=17,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}})

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    x = (np.random.randn(1, seq_len, 3, image_size, image_size).astype(np.float32)
         if len(shape) == 5 else
         np.random.randn(1, 3, image_size, image_size).astype(np.float32))

    for _ in range(warmup):
        sess.run(None, {name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return summarise(f"liveness ({seq_len} frames)", times, note)


def bench_detection_and_embedding(runs: int, warmup: int) -> list[dict]:
    try:
        import cv2
        from ml.preprocessing.face_processor import FaceProcessor
        from backend.app.services.recognition import RecognitionService
    except ImportError as e:
        return [{"stage": "detection/embedding", "error": f"unavailable: {e}"}]

    # A synthetic frame. Detection cost is dominated by input resolution, which is
    # fixed, so a synthetic frame gives a representative timing even though it
    # contains no real face.
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

    out = []
    try:
        fp = FaceProcessor(image_size=112, ctx_id=-1)
        for _ in range(warmup):
            fp.detect_all(frame)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fp.detect_all(frame)
            times.append((time.perf_counter() - t0) * 1000)
        out.append(summarise("face detection (1280x720)", times,
                             "SCRFD via InsightFace, CPU"))
    except Exception as e:
        out.append({"stage": "face detection", "error": str(e)})

    try:
        rs = RecognitionService(ctx_id=-1)
        for _ in range(warmup):
            rs.embed(frame)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            rs.embed(frame)
            times.append((time.perf_counter() - t0) * 1000)
        out.append(summarise("face embedding (ArcFace)", times,
                             "includes detection; subtract the detection row"))
    except Exception as e:
        out.append({"stage": "face embedding", "error": str(e)})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=112)
    ap.add_argument("--out", default="docs/results/inference_latency.json")
    ap.add_argument("--skip-face", action="store_true")
    args = ap.parse_args()

    import platform
    results = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
        "stages": [bench_liveness(args.runs, args.warmup, args.seq_len, args.image_size)],
    }
    if not args.skip_face:
        results["stages"] += bench_detection_and_embedding(args.runs, args.warmup)

    print(f"{'stage':<34}{'P50 ms':>10}{'P95 ms':>10}{'P99 ms':>10}")
    print("-" * 64)
    for s in results["stages"]:
        if "error" in s:
            print(f"{s['stage']:<34}  ERROR: {s['error'][:60]}")
        else:
            print(f"{s['stage']:<34}{s['p50_ms']:>10.2f}{s['p95_ms']:>10.2f}{s['p99_ms']:>10.2f}")
    print()
    for s in results["stages"]:
        if s.get("note"):
            print(f"note ({s['stage']}): {s['note']}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
