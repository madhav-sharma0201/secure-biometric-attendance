"""Constraint tests against a real (SQLite) database.

The duplicate-attendance guard is the one business rule that application code cannot
enforce correctly on its own, so it is tested at the database level.
"""
import datetime as dt

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession

from backend.app.models.db import (
    Attendance,
    Base,
    FaceEmbedding,
    Session as ClassSession,
    User,
    VerificationAttempt,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite://")

    # SQLite ignores foreign keys unless explicitly told not to, which would let the
    # cascade tests pass vacuously.
    @event.listens_for(engine, "connect")
    def _fk(conn, _rec):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with OrmSession(engine) as s:
        yield s


def _user(s, sid="S1", email="a@b.c"):
    u = User(student_id=sid, name="Test", email=email)
    s.add(u); s.commit()
    return u


def _session(s):
    now = dt.datetime.now(dt.timezone.utc)
    c = ClassSession(name="Lecture", start_time=now, end_time=now + dt.timedelta(hours=1))
    s.add(c); s.commit()
    return c


def test_duplicate_attendance_is_rejected_by_the_database(db):
    u, c = _user(db), _session(db)
    db.add(Attendance(user_id=u.id, session_id=c.id,
                      identity_confidence=0.9, liveness_confidence=0.95))
    db.commit()

    db.add(Attendance(user_id=u.id, session_id=c.id,
                      identity_confidence=0.91, liveness_confidence=0.96))
    with pytest.raises(IntegrityError):
        db.commit()


def test_same_user_can_attend_two_different_sessions(db):
    u, c1, c2 = _user(db), _session(db), _session(db)
    for c in (c1, c2):
        db.add(Attendance(user_id=u.id, session_id=c.id,
                          identity_confidence=0.9, liveness_confidence=0.9))
    db.commit()
    assert db.query(Attendance).count() == 2


def test_deleting_a_user_removes_their_embeddings(db):
    u = _user(db)
    db.add(FaceEmbedding(user_id=u.id, embedding=b"\x00" * 8, dim=2,
                         model_version="buffalo_l"))
    db.commit()
    assert db.query(FaceEmbedding).count() == 1

    db.delete(u); db.commit()
    assert db.query(FaceEmbedding).count() == 0


def test_invalid_status_values_are_rejected(db):
    db.add(User(student_id="S9", name="X", email="x@y.z", status="nonsense"))
    with pytest.raises(IntegrityError):
        db.commit()


def test_student_id_and_email_are_unique(db):
    _user(db, "S1", "a@b.c")
    db.add(User(student_id="S1", name="Other", email="other@b.c"))
    with pytest.raises(IntegrityError):
        db.commit()


def test_verification_attempts_store_no_biometric_data(db):
    """Guard against a future change adding an image or embedding column here."""
    cols = set(VerificationAttempt.__table__.columns.keys())
    forbidden = {"image", "image_data", "frame", "embedding", "face_image", "raw"}
    assert cols & forbidden == set(), f"biometric data leaked into audit log: {cols & forbidden}"


def test_embedding_records_its_model_version(db):
    """Embeddings from different backbones are not comparable."""
    u = _user(db)
    db.add(FaceEmbedding(user_id=u.id, embedding=b"\x00" * 8, dim=2,
                         model_version="buffalo_l"))
    db.commit()
    assert db.query(FaceEmbedding).one().model_version == "buffalo_l"
