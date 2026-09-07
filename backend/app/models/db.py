"""SQLAlchemy models.

Two constraints here are load-bearing and are enforced by the DATABASE, not by
application code:

  - UNIQUE(user_id, session_id) on attendance. An application-level "have they already
    been marked?" check loses to concurrent requests: two simultaneous verifications
    can both read 'not marked' before either writes. The database constraint is the
    only thing that actually holds under concurrency.
  - ON DELETE CASCADE from users to face_embeddings, so deleting a user's biometrics
    cannot leave orphaned embeddings behind.

Attendance rows are deliberately NOT cascaded from users: the attendance record is an
audit trail and must survive the deletion of biometric data (see Phase 25 — a user can
withdraw their biometrics without erasing the fact that they attended).
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    server_default=func.now())

    embeddings: Mapped[list["FaceEmbedding"]] = relationship(
        back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("status in ('active','inactive','biometrics_deleted')",
                        name="ck_users_status"),
    )


class FaceEmbedding(Base):
    """One enrolled face template.

    Stored as raw float32 bytes plus its dimension and the model version that produced
    it. `model_version` is not decoration: embeddings from different backbones are not
    comparable, so a backbone change must force re-enrollment rather than silently
    producing meaningless similarities.
    """
    __tablename__ = "face_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    dim: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    server_default=func.now())

    user: Mapped[User] = relationship(back_populates="embeddings")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    start_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    server_default=func.now())


class Attendance(Base):
    __tablename__ = "attendance"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                   server_default=func.now())
    identity_confidence: Mapped[float] = mapped_column(Float)
    liveness_confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="present")

    __table_args__ = (
        # The real duplicate-attendance guard.
        UniqueConstraint("user_id", "session_id", name="uq_attendance_user_session"),
        CheckConstraint("status in ('present','flagged')", name="ck_attendance_status"),
    )


class VerificationAttempt(Base):
    """Audit trail of every verification, successful or not.

    Deliberately stores NO images and NO embeddings — only the outcome, the scores and
    the reason. Enough to investigate an incident or spot a spoofing campaign, without
    creating a biometric database that did not need to exist.
    """
    __tablename__ = "verification_attempts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    matched_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    approved: Mapped[bool] = mapped_column(default=False)
    reason: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    liveness_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    identity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_faces: Mapped[int] = mapped_column(Integer, default=0)
    liveness_model_version: Mapped[str] = mapped_column(String(64), default="unknown")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    server_default=func.now())
