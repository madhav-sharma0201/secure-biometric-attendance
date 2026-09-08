"""Runtime configuration, loaded from the environment.

Thresholds live here rather than in code because they are experimental results, not
constants: they are recalibrated whenever the model is retrained, and the deployed
value must be traceable to the run that produced it.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://attendance:change_me@localhost:5432/attendance"

    liveness_model_path: str = "models/liveness.onnx"
    liveness_model_version: str = "modelA_cnn"
    recognition_model_version: str = "buffalo_l"

    # Calibrated on validation data. See docs and the run report they came from.
    liveness_threshold: float = 0.4535
    face_match_threshold: float = 0.5
    liveness_uncertain_band: float = 0.10

    sequence_length: int = 8
    image_size: int = 112

    api_key: str = "change_me"
    log_level: str = "INFO"

    # CORS. Empty means same-origin only, which is correct behind the nginx proxy.
    # A wildcard here would let any website on the internet drive this API using a
    # visitor's browser, so it is never the default.
    cors_origins: str = ""

    # Upload bounds. A verification burst is a handful of small JPEGs; without a cap
    # an unauthenticated-shaped request can pin CPU and memory by sending hundreds of
    # large images that each cost a face-detection pass.
    max_upload_bytes: int = 8 * 1024 * 1024      # per file
    max_frames_per_request: int = 16
    max_enrollment_images: int = 10

    # Graceful shutdown: how long to let in-flight verifications finish.
    shutdown_grace_seconds: int = 20

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
