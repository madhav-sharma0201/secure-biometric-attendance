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

    app.state.services = Services(
        faces=faces, liveness=liveness, recognition=recognition,
        verification=VerificationService(faces, liveness, recognition, thresholds),
    )
    log.info("startup", liveness_threshold=thresholds.liveness,
             identity_threshold=thresholds.identity,
             liveness_model=settings.liveness_model_path)
    yield


app = FastAPI(title="Secure Biometric Attendance", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


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
            log.warning("rate_limited", request_id=request_id, client=client,
                        path=request.url.path)
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=429,
                                content={"detail": "too many verification attempts"})

    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000

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

    ready = bool(svc) and db_ok
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "database": db_ok, "services": bool(svc)},
    )
