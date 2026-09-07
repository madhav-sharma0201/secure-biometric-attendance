# Multi-stage: wheels are built in a throwaway stage so the runtime image carries no
# compiler toolchain. Smaller image, smaller attack surface.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ cmake && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-backend.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-backend.txt


FROM python:3.11-slim AS runtime

# libGL and libglib are needed by OpenCV even in headless builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

# Non-root. A container that serves biometric verification should not run as root.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements-backend.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-backend.txt \
    && rm -rf /wheels

COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser ml/ ./ml/
COPY --chown=appuser:appuser configs/ ./configs/

# Model weights are mounted at runtime, not baked in: they are large, they change
# independently of the code, and a trained model is data rather than source.
RUN mkdir -p /app/models && chown appuser:appuser /app/models

USER appuser
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 INSIGHTFACE_HOME=/app/models/.insightface

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
