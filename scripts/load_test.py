"""Load test for the Kubernetes replica-scaling experiment (Phase 23).

Measures throughput and latency percentiles at a fixed concurrency, so results at
1, 2 and 4 replicas are comparable.

Honest measurement notes:
  - Percentiles come from the FULL sample, not a running estimate.
  - A warmup phase runs first and is discarded. The first request to a pod pays model
    loading and JIT costs; including it would understate steady-state performance and
    make replica counts incomparable (more replicas = more cold starts).
  - Errors are reported separately and never counted as fast successes. A server that
    fails instantly under load would otherwise look like the fastest configuration.

Usage:
    python scripts/load_test.py --url http://localhost:18000 --endpoint /health \
        --concurrency 16 --requests 500 --label "1 replica"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, asdict

import httpx


@dataclass
class Result:
    label: str
    endpoint: str
    concurrency: int
    n_requests: int
    n_success: int
    n_error: int
    duration_sec: float
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    status_codes: dict[str, int]


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


async def _worker(client, url, payload, queue, latencies, codes):
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        t0 = time.perf_counter()
        try:
            if payload is None:
                r = await client.get(url)
            else:
                r = await client.post(url, files=payload)
            code = str(r.status_code)
        except Exception as e:
            code = f"exc:{type(e).__name__}"
        dt = (time.perf_counter() - t0) * 1000.0
        codes[code] = codes.get(code, 0) + 1
        if code.startswith("2"):
            latencies.append(dt)


async def run(url: str, endpoint: str, concurrency: int, n: int, warmup: int,
              label: str, payload=None) -> Result:
    target = url.rstrip("/") + endpoint
    limits = httpx.Limits(max_connections=concurrency * 2,
                          max_keepalive_connections=concurrency * 2)

    async with httpx.AsyncClient(timeout=60.0, limits=limits) as client:
        if warmup:
            q = asyncio.Queue()
            for _ in range(warmup):
                q.put_nowait(1)
            await asyncio.gather(*[_worker(client, target, payload, q, [], {})
                                   for _ in range(concurrency)])

        latencies: list[float] = []
        codes: dict[str, int] = {}
        q = asyncio.Queue()
        for _ in range(n):
            q.put_nowait(1)

        t0 = time.perf_counter()
        await asyncio.gather(*[_worker(client, target, payload, q, latencies, codes)
                               for _ in range(concurrency)])
        duration = time.perf_counter() - t0

    lat = sorted(latencies)
    n_ok = len(lat)
    return Result(
        label=label, endpoint=endpoint, concurrency=concurrency, n_requests=n,
        n_success=n_ok, n_error=n - n_ok, duration_sec=round(duration, 3),
        throughput_rps=round(n_ok / duration, 2) if duration else 0.0,
        p50_ms=round(percentile(lat, 0.50), 2), p95_ms=round(percentile(lat, 0.95), 2),
        p99_ms=round(percentile(lat, 0.99), 2),
        mean_ms=round(statistics.fmean(lat), 2) if lat else float("nan"),
        max_ms=round(max(lat), 2) if lat else float("nan"),
        status_codes=codes,
    )


def format_table(results: list[Result]) -> str:
    L = [f"{'config':<14}{'RPS':>9}{'P50 ms':>9}{'P95 ms':>9}{'P99 ms':>9}{'errors':>8}"]
    L.append("-" * 58)
    for r in results:
        L.append(f"{r.label:<14}{r.throughput_rps:>9.2f}{r.p50_ms:>9.2f}"
                 f"{r.p95_ms:>9.2f}{r.p99_ms:>9.2f}{r.n_error:>8}")
    if len(results) > 1:
        base = results[0]
        L.append("")
        L.append("scaling relative to the first configuration:")
        for r in results[1:]:
            factor = r.throughput_rps / base.throughput_rps if base.throughput_rps else 0
            L.append(f"  {r.label:<14} {factor:.2f}x throughput")
        L.append("")
        L.append("A factor well below the replica ratio means the bottleneck is not")
        L.append("replica count — on a single-node cluster it is usually shared CPU.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:18000")
    ap.add_argument("--endpoint", default="/health")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = asyncio.run(run(args.url, args.endpoint, args.concurrency,
                          args.requests, args.warmup, args.label))
    print(format_table([res]))
    print()
    print(json.dumps(asdict(res), indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(asdict(res), fh, indent=2)


if __name__ == "__main__":
    main()
