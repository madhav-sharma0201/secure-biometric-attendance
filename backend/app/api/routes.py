"""HTTP layer. Validation and wiring only — no business logic lives here."""
from __future__ import annotations

import datetime as dt

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from backend.app.core.config import settings
from backend.app.core.db import get_db
from backend.app.models.db import Attendance, FaceEmbedding, Session as ClassSession, User
from backend.app.schemas.api import (
    AttendanceOut,
    EnrollmentOut,
    SessionCreate,
    SessionOut,
    UserCreate,
    UserOut,
    VerifyOut,
)
from backend.app.services.recognition import build_template, serialize

router = APIRouter()


def _decode(data: bytes) -> np.ndarray | None:
    """Decode image bytes, returning None for anything unusable.

    cv2.imdecode raises on empty or malformed input rather than returning None, which
    turns a junk upload into a 500. Uploads are attacker-controlled, so decoding
    failures must be an ordinary rejection, not an unhandled exception.
    """
    if not data:
        return None
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if img is None or img.size == 0:
        return None
    return img


async def _read_bounded(upload: UploadFile) -> bytes:
    """Read an upload, refusing anything over the configured size.

    Reading the whole body first and checking afterwards still buys the attacker the
    memory; the cap is enforced while streaming so an oversized file is rejected
    before it is fully buffered.
    """
    limit = settings.max_upload_bytes
    chunks, total = [], 0
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                413, f"file exceeds the {limit // (1024 * 1024)} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _enforce_count(items: list, limit: int, what: str) -> None:
    if len(items) > limit:
        raise HTTPException(413, f"too many {what}: {len(items)} sent, limit is {limit}")
    if not items:
        raise HTTPException(422, f"no {what} supplied")


def _services(request):
    return request.app.state.services


# ---------- users ----------

@router.post("/users", response_model=UserOut, status_code=201)
def create_user(payload: UserCreate, db: OrmSession = Depends(get_db)):
    exists = db.execute(
        select(User).where((User.student_id == payload.student_id) |
                           (User.email == payload.email))
    ).scalars().first()
    if exists:
        raise HTTPException(409, "student_id or email already registered")

    user = User(**payload.model_dump())
    db.add(user)
    db.commit()
    return UserOut.model_validate(user, from_attributes=True)


@router.get("/users", response_model=list[UserOut])
def list_users(db: OrmSession = Depends(get_db)):
    users = db.execute(select(User)).scalars().all()
    enrolled = {e.user_id for e in db.execute(select(FaceEmbedding)).scalars().all()}
    out = []
    for u in users:
        item = UserOut.model_validate(u, from_attributes=True)
        item.enrolled = u.id in enrolled
        out.append(item)
    return out


@router.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: str, db: OrmSession = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "user not found")
    return UserOut.model_validate(user, from_attributes=True)


@router.delete("/users/{user_id}/biometrics", status_code=200)
def delete_biometrics(user_id: str, db: OrmSession = Depends(get_db)):
    """Delete a user's biometric data while preserving their attendance record.

    Biometric data is deletable on request; attendance is an institutional record.
    Those are different retention policies, so they are different operations.
    """
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "user not found")
    n = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user_id).delete()
    user.status = "biometrics_deleted"
    db.commit()
    return {"user_id": user_id, "embeddings_deleted": n,
            "attendance_records_preserved": True}


# ---------- enrollment ----------

@router.post("/enrollment", response_model=EnrollmentOut)
async def enroll(request: Request, user_id: str = Form(...),
                 images: list[UploadFile] = File(...),
                 db: OrmSession = Depends(get_db)):
    """Enroll a user from several face images.

    Images with zero or several faces are rejected rather than silently using the
    largest: enrolling a bystander's face against a student's identity is a
    catastrophic and completely silent failure.
    """
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "user not found")

    _enforce_count(images, settings.max_enrollment_images, "enrollment images")

    svc = _services(request)
    accepted, rejected = [], 0
    for f in images:
        img = _decode(await _read_bounded(f))
        if img is None:
            rejected += 1
            continue
        emb, n_faces = svc.recognition.embed(img)
        if emb is None:
            rejected += 1
            continue
        accepted.append(emb)

    if len(accepted) < 3:
        raise HTTPException(
            422,
            f"need at least 3 usable images with exactly one face; "
            f"got {len(accepted)} usable, {rejected} rejected",
        )

    template = build_template(np.stack(accepted))
    db.query(FaceEmbedding).filter(
        FaceEmbedding.user_id == user_id,
        FaceEmbedding.model_version == svc.recognition.model_version,
    ).delete()

    row = FaceEmbedding(user_id=user_id, embedding=serialize(template),
                        dim=int(template.shape[0]),
                        model_version=svc.recognition.model_version)
    db.add(row)
    db.commit()

    return EnrollmentOut(user_id=user_id, n_images_accepted=len(accepted),
                         n_images_rejected=rejected,
                         model_version=svc.recognition.model_version,
                         embedding_id=row.id)


# ---------- verification ----------

@router.post("/verify", response_model=VerifyOut)
async def verify(request: Request, frames: list[UploadFile] = File(...),
                 session_id: str | None = Form(None),
                 mark_attendance: bool = Form(True),
                 db: OrmSession = Depends(get_db)):
    """Verify a capture burst and, if approved, mark attendance."""
    _enforce_count(frames, settings.max_frames_per_request, "frames")

    svc = _services(request)
    decoded = []
    for f in frames:
        img = _decode(await _read_bounded(f))
        if img is not None:
            decoded.append(img)

    if not decoded:
        return VerifyOut(decision="rejected", reason="NO_FACE")

    result = svc.verification.verify(db, decoded, session_id, mark_attendance)
    return VerifyOut(**result)


@router.post("/attendance/mark", response_model=VerifyOut)
async def mark(request: Request, frames: list[UploadFile] = File(...),
               session_id: str | None = Form(None),
               db: OrmSession = Depends(get_db)):
    return await verify(request, frames, session_id, True, db)


# ---------- sessions ----------

@router.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(payload: SessionCreate, db: OrmSession = Depends(get_db)):
    row = ClassSession(**payload.model_dump())
    db.add(row)
    db.commit()
    return _session_out(row)


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(db: OrmSession = Depends(get_db)):
    return [_session_out(s) for s in db.execute(select(ClassSession)).scalars().all()]


def _session_out(s: ClassSession) -> SessionOut:
    now = dt.datetime.now(dt.timezone.utc)
    start, end = s.start_time, s.end_time
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    out = SessionOut.model_validate(s, from_attributes=True)
    out.is_open = start <= now <= end
    return out


# ---------- attendance ----------

@router.get("/attendance", response_model=list[AttendanceOut])
def list_attendance(session_id: str | None = None, db: OrmSession = Depends(get_db)):
    stmt = select(Attendance)
    if session_id:
        stmt = stmt.where(Attendance.session_id == session_id)
    return [AttendanceOut.model_validate(a, from_attributes=True)
            for a in db.execute(stmt).scalars().all()]


@router.get("/attendance/{user_id}", response_model=list[AttendanceOut])
def user_attendance(user_id: str, db: OrmSession = Depends(get_db)):
    rows = db.execute(select(Attendance).where(Attendance.user_id == user_id)).scalars().all()
    return [AttendanceOut.model_validate(a, from_attributes=True) for a in rows]
