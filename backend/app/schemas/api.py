"""Request/response schemas. Validation happens here, at the boundary."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    student_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr


class UserOut(BaseModel):
    id: str
    student_id: str
    name: str
    email: str
    status: str
    created_at: dt.datetime
    enrolled: bool = False

    model_config = {"from_attributes": True}


class SessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    start_time: dt.datetime
    end_time: dt.datetime

    @field_validator("end_time")
    @classmethod
    def _end_after_start(cls, v, info):
        start = info.data.get("start_time")
        if start and v <= start:
            raise ValueError("end_time must be after start_time")
        return v


class SessionOut(BaseModel):
    id: str
    name: str
    start_time: dt.datetime
    end_time: dt.datetime
    is_open: bool = False

    model_config = {"from_attributes": True}


class EnrollmentOut(BaseModel):
    user_id: str
    n_images_accepted: int
    n_images_rejected: int
    model_version: str
    embedding_id: str


class VerifyOut(BaseModel):
    """Deliberately loose: a rejection carries only a reason.

    Returning identity fields on a rejected attempt would turn this endpoint into an
    oracle for 'who does this photo most resemble'.
    """
    decision: str
    reason: str | None = None
    user_id: str | None = None
    identity_confidence: float | None = None
    liveness_confidence: float | None = None
    attendance_id: str | None = None


class AttendanceOut(BaseModel):
    id: str
    user_id: str
    session_id: str
    timestamp: dt.datetime
    identity_confidence: float
    liveness_confidence: float
    status: str

    model_config = {"from_attributes": True}


class HealthOut(BaseModel):
    status: str
    liveness_model_loaded: bool
    recognition_model_loaded: bool
    database_reachable: bool
    liveness_model_version: str
    recognition_model_version: str
