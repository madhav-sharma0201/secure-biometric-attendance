"""FastAPI application.

Models load once at startup, not per request: loading ONNX and ArcFace on every call
would dominate latency and make the Phase 19 measurements meaningless.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

import time
import uuid

from backend.app.api.routes import router
from backend.app.core.config import settings
from backend.app.core import metrics
from backend.app.core.security import require_api_key, verification_limiter
from backend.app.core.db import get_engine, init_db
from backend.app.core.decision import Thresholds
from backend.app.schemas.api import HealthOut
from backend.app.services.liveness import LivenessService
from backend.app.services.recognition import RecognitionService
from backend.app.services.verification import VerificationService

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)),
    processors=[structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer()],
)
log = structlog.get_logger()


@dataclass
class Services:
    faces: object
    liveness: LivenessService
    recognition: RecognitionService
    verification: VerificationService


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ml.preprocessing.face_processor import FaceProcessor

    init_db()

    faces = FaceProcessor(image_size=settings.image_size, ctx_id=-1)
    liveness = LivenessService(settings.liveness_model_path,
                               sequence_length=settings.sequence_length,
                               image_size=settings.image_size)
    recognition = RecognitionService(settings.recognition_model_version, ctx_id=-1)

    thresholds = Thresholds(liveness=settings.liveness_threshold,
                            identity=settings.face_match_threshold,
                            liveness_uncertain_band=settings.liveness_uncertain_band)

    # Load models eagerly rather than on first request. Lazy loading pushes the cost
    # onto a real user and lets /ready report success before the model can actually
    # serve. The startupProbe (failureThreshold 30) exists to cover this window.
    model_errors = []
    try:
        liveness._ensure_loaded()
        log.info("liveness_model_loaded", version=liveness.model_version,
                 threshold=liveness.trained_threshold)
    except Exception as e:
        model_errors.append(f"liveness: {e}")
        log.error("liveness_model_load_failed", error=str(e))
    try:
        faces._ensure_loaded()
        recognition._ensure_loaded()
        log.info("recognition_model_loaded", version=recognition.model_version)
    except Exception as e:
        model_errors.append(f"recognition: {e}")
        log.error("recognition_model_load_failed", error=str(e))

    app.state.model_errors = model_errors
    metrics.models_loaded.set(0 if model_errors else 1)
    app.state.services = Services(
        faces=faces, liveness=liveness, recognition=recognition,
        verification=VerificationService(faces, liveness, recognition, thresholds),
    )
    log.info("startup", liveness_threshold=thresholds.liveness,
             identity_threshold=thresholds.identity,
             liveness_model=settings.liveness_model_path)

    yield

    # Uvicorn stops accepting new connections on SIGTERM and waits for in-flight
    # requests. A verification can take seconds, so terminationGracePeriodSeconds in
    # the Deployment must exceed that or Kubernetes SIGKILLs mid-request.
    log.info("shutdown", grace_seconds=settings.shutdown_grace_seconds)


app = FastAPI(title="Secure Biometric Attendance", version="0.1.0", lifespan=lifespan)
# CORS is deny-by-default. The frontend is served from the same origin through nginx,
# so no cross-origin access is needed in the deployed configuration. Wildcard origins
# would let any site drive this API using a logged-in visitor's browser.
if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


@app.middleware("http")
async def observability_and_auth(request, call_next):
    """Request logging, auth and rate limiting in one pass.

    Logged: method, path, status, duration, request id, client. NOT logged: request
    bodies, images, embeddings or headers — a verification request body is biometric
    data, and an access log is exactly the wrong place for it.
    """
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    client = request.client.host if request.client else "unknown"

    try:
        require_api_key(request)
    except Exception as exc:
        status_code = getattr(exc, "status_code", 500)
        log.warning("request_rejected", request_id=request_id, path=request.url.path,
                    status=status_code, reason="auth", client=client)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=status_code,
                            content={"detail": getattr(exc, "detail", "unauthorized")})

    if request.url.path in ("/verify", "/attendance/mark"):
        allowed, remaining = verification_limiter.check(client)
        if not allowed:
            metrics.rate_limited.inc()
            log.warning("rate_limited", request_id=request_id, client=client,
                        path=request.url.path)
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=429,
                                content={"detail": "too many verification attempts"})

    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000

    label_path = metrics.normalise_path(request.url.path)
    metrics.http_requests.labels(request.method, label_path, str(response.status_code)).inc()
    metrics.http_duration.labels(request.method, label_path).observe(duration_ms / 1000)

    log.info("request", request_id=request_id, method=request.method,
             path=request.url.path, status=response.status_code,
             duration_ms=round(duration_ms, 2), client=client)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(router)


@app.get("/health", response_model=HealthOut)
def health():
    """Liveness probe: is the process up? Deliberately does not touch the database."""
    svc = getattr(app.state, "services", None)
    return HealthOut(
        status="ok",
        liveness_model_loaded=bool(svc and svc.liveness._sess is not None),
        recognition_model_loaded=bool(svc and svc.recognition._app is not None),
        database_reachable=True,
        liveness_model_version=svc.liveness.model_version if svc else "unloaded",
        recognition_model_version=svc.recognition.model_version if svc else "unloaded",
    )


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus scrape endpoint. Internal only — not routed through the Ingress."""
    from fastapi.responses import Response
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/ready")
def ready():
    """Readiness probe: can this pod actually serve traffic?

    Distinct from /health on purpose. A pod whose database is unreachable is alive but
    must not receive requests — returning ok here would route traffic into failures.
    """
    from fastapi.responses import JSONResponse

    svc = getattr(app.state, "services", None)
    db_ok = True
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    # A pod whose models failed to load is alive but cannot verify anyone. Reporting
    # ready would route verification traffic into guaranteed SYSTEM_ERROR responses.
    models_ok = bool(svc) and svc.liveness._sess is not None
    ready = bool(svc) and db_ok and models_ok
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "database": db_ok, "services": bool(svc),
                 "models_loaded": models_ok,
                 "model_errors": getattr(app.state, "model_errors", [])},
    )
