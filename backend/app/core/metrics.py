"""Prometheus metrics.

Deliberately excludes anything identifying. Decision outcomes are counted by REASON,
never by user: a per-user counter in a metrics endpoint is a slow-motion privacy leak,
because scrape history reconstructs who attended what and when.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

http_requests = Counter(
    "attendance_http_requests_total", "HTTP requests",
    ["method", "path", "status"], registry=REGISTRY,
)
http_duration = Histogram(
    "attendance_http_request_seconds", "HTTP request duration",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    registry=REGISTRY,
)
verifications = Counter(
    "attendance_verifications_total", "Verification outcomes",
    ["decision", "reason"], registry=REGISTRY,
)
verification_duration = Histogram(
    "attendance_verification_seconds", "End-to-end verification duration",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 12), registry=REGISTRY,
)
liveness_score = Histogram(
    "attendance_liveness_score", "Distribution of liveness scores",
    buckets=(0.05, 0.1, 0.25, 0.4, 0.4535, 0.5, 0.7, 0.9, 0.99), registry=REGISTRY,
)
models_loaded = Gauge(
    "attendance_models_loaded", "1 when both models are loaded", registry=REGISTRY,
)
rate_limited = Counter(
    "attendance_rate_limited_total", "Requests rejected by the rate limiter",
    registry=REGISTRY,
)


def render() -> bytes:
    return generate_latest(REGISTRY)


def normalise_path(path: str) -> str:
    """Collapse identifiers so /users/<uuid> does not create unbounded label cardinality."""
    parts = path.strip("/").split("/")
    out = []
    for p in parts:
        if len(p) >= 16 and ("-" in p or p.isalnum()):
            out.append("{id}")
        else:
            out.append(p)
    return "/" + "/".join(out) if out != [""] else "/"
