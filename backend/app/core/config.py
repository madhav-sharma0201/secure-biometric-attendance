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
    liveness_model_version: str = "dev"
    recognition_model_version: str = "buffalo_l"

    # Calibrated on validation data. See docs and the run report they came from.
    liveness_threshold: float = 0.5
    face_match_threshold: float = 0.5
    liveness_uncertain_band: float = 0.10

    sequence_length: int = 8
    image_size: int = 112

    api_key: str = "change_me"
    log_level: str = "INFO"


settings = Settings()
