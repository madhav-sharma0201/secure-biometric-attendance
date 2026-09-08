"""API integration tests against a real SQLite database with stubbed ML services.

Stubbing the models rather than mocking the services keeps the ACTUAL orchestration,
decision engine, persistence and constraint behaviour under test — only the two
learned components are replaced, and they are the only parts these tests are not
about.
"""
import datetime as dt

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.core.decision import Thresholds
from backend.app.models.db import Base
from backend.app.services.recognition import l2_normalize
from backend.app.services.verification import VerificationService

RNG = np.random.default_rng(7)
DIM = 128


class StubFace:
    def __init__(self, crop):
        self.crop = crop
        self.bbox = np.array([0, 0, 112, 112])
        self.det_score = 0.99


class StubFaceProcessor:
    """Reads the number of faces and the identity from pixel values in the fixture.

    Blue channel mean encodes the face count; red channel mean encodes which person.
    That keeps the tests deterministic and free of real imagery.
    """
    image_size = 112

    def detect_all(self, img):
        n = int(round(img[:, :, 0].mean() / 40.0))
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        crop[:, :, 2] = int(img[:, :, 2].mean())
        return [StubFace(crop) for _ in range(n)]


class StubLiveness:
    model_version = "stub-v1"

    def __init__(self, score=0.99):
        self.score_value = score
        self._sess = object()

    def score(self, crops):
        return self.score_value


class StubRecognition:
    model_version = "stub-arcface"

    def __init__(self):
        self._app = object()

    def _vec(self, person_id: int):
        return l2_normalize(np.default_rng if False else
                            np.random.default_rng(person_id).normal(size=DIM).astype(np.float32))

    def embed(self, img):
        n = int(round(img[:, :, 0].mean() / 40.0))
        if n != 1:
            return None, n
        return self._vec(int(round(img[:, :, 2].mean()))), 1

    def embed_crop(self, crop_rgb):
        return self._vec(int(round(crop_rgb[:, :, 2].mean())))


def make_image(n_faces=1, person=5):
    img = np.zeros((240, 240, 3), dtype=np.uint8)
    img[:, :, 0] = n_faces * 40
    img[:, :, 2] = person
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture()
def client(tmp_path):
    from backend.app import main as main_mod
    from backend.app.core import db as db_mod

    engine = create_engine(f"sqlite:///{tmp_path}/t.db")

    @event.listens_for(engine, "connect")
    def _fk(conn, _r):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)

    db_mod.set_engine(engine)

    def override_db():
        s = TestSession()
        try:
            yield s
        finally:
            s.close()

    app = main_mod.app
    app.dependency_overrides[db_mod.get_db] = override_db

    from backend.app.core.config import settings as app_settings
    app_settings.api_key = "test-key"

    with TestClient(app, headers={"X-API-Key": "test-key"}) as c:
        # Installed AFTER startup: the lifespan builds the real services, so stubs
        # set before entering the context would simply be overwritten.
        faces, liveness, recog = StubFaceProcessor(), StubLiveness(), StubRecognition()
        app.state.services = main_mod.Services(
            faces=faces, liveness=liveness, recognition=recog,
            verification=VerificationService(faces, liveness, recog,
                                             Thresholds(liveness=0.8, identity=0.5)),
        )
        c.app_services = app.state.services
        yield c
    app.dependency_overrides.clear()


def _enroll(client, person=5, student_id="S1", email="a@b.c"):
    u = client.post("/users", json={"student_id": student_id, "name": "T",
                                    "email": email}).json()
    files = [("images", (f"{i}.jpg", make_image(1, person), "image/jpeg")) for i in range(3)]
    r = client.post("/enrollment", data={"user_id": u["id"]}, files=files)
    assert r.status_code == 200, r.text
    return u


def _open_session(client):
    now = dt.datetime.now(dt.timezone.utc)
    r = client.post("/sessions", json={
        "name": "Lecture",
        "start_time": (now - dt.timedelta(minutes=5)).isoformat(),
        "end_time": (now + dt.timedelta(hours=1)).isoformat()})
    assert r.status_code == 201, r.text
    return r.json()


def test_authenticated_routes_reject_a_missing_key(client):
    from fastapi.testclient import TestClient
    from backend.app import main as main_mod
    with TestClient(main_mod.app) as bare:      # no X-API-Key header
        assert bare.get("/users").status_code == 401
        assert bare.get("/health").status_code == 200    # probes stay public


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is True


def test_user_creation_rejects_duplicates(client):
    body = {"student_id": "S1", "name": "T", "email": "a@b.c"}
    assert client.post("/users", json=body).status_code == 201
    assert client.post("/users", json=body).status_code == 409


def test_enrollment_requires_at_least_three_usable_images(client):
    u = client.post("/users", json={"student_id": "S2", "name": "T",
                                    "email": "b@b.c"}).json()
    files = [("images", ("0.jpg", make_image(1, 5), "image/jpeg"))]
    r = client.post("/enrollment", data={"user_id": u["id"]}, files=files)
    assert r.status_code == 422 and "at least 3" in r.text


def test_enrollment_rejects_images_with_multiple_faces(client):
    u = client.post("/users", json={"student_id": "S3", "name": "T",
                                    "email": "c@b.c"}).json()
    files = [("images", (f"{i}.jpg", make_image(2, 5), "image/jpeg")) for i in range(4)]
    r = client.post("/enrollment", data={"user_id": u["id"]}, files=files)
    assert r.status_code == 422


def test_full_approval_path_marks_attendance(client):
    u = _enroll(client)
    s = _open_session(client)
    frames = [("frames", (f"{i}.jpg", make_image(1, 5), "image/jpeg")) for i in range(4)]
    r = client.post("/verify", data={"session_id": s["id"]}, files=frames).json()
    assert r["decision"] == "approved", r
    assert r["user_id"] == u["id"]
    assert r["attendance_id"]
    assert len(client.get("/attendance").json()) == 1


def test_second_attempt_is_rejected_as_already_marked(client):
    _enroll(client)
    s = _open_session(client)
    frames = lambda: [("frames", (f"{i}.jpg", make_image(1, 5), "image/jpeg")) for i in range(4)]
    assert client.post("/verify", data={"session_id": s["id"]}, files=frames()).json()["decision"] == "approved"
    second = client.post("/verify", data={"session_id": s["id"]}, files=frames()).json()
    assert second["decision"] == "rejected"
    assert second["reason"] == "ALREADY_MARKED"
    assert len(client.get("/attendance").json()) == 1


def test_spoof_is_rejected_and_no_attendance_recorded(client):
    _enroll(client)
    s = _open_session(client)
    client.app_services.liveness.score_value = 0.05
    frames = [("frames", (f"{i}.jpg", make_image(1, 5), "image/jpeg")) for i in range(4)]
    r = client.post("/verify", data={"session_id": s["id"]}, files=frames).json()
    assert r["decision"] == "rejected" and r["reason"] == "LIVENESS_FAILED"
    assert r.get("user_id") is None
    assert client.get("/attendance").json() == []


def test_multiple_faces_rejected(client):
    _enroll(client)
    s = _open_session(client)
    frames = [("frames", (f"{i}.jpg", make_image(2, 5), "image/jpeg")) for i in range(4)]
    r = client.post("/verify", data={"session_id": s["id"]}, files=frames).json()
    assert r["reason"] == "MULTIPLE_FACES"


def test_unenrolled_person_is_rejected(client):
    _enroll(client, person=5)
    s = _open_session(client)
    frames = [("frames", (f"{i}.jpg", make_image(1, 200), "image/jpeg")) for i in range(4)]
    r = client.post("/verify", data={"session_id": s["id"]}, files=frames).json()
    assert r["decision"] == "rejected"
    assert r["reason"] in ("UNKNOWN_PERSON", "LOW_IDENTITY_CONFIDENCE")


def test_verification_outside_an_open_session_is_rejected(client):
    _enroll(client)
    now = dt.datetime.now(dt.timezone.utc)
    client.post("/sessions", json={"name": "Past",
                                   "start_time": (now - dt.timedelta(days=1)).isoformat(),
                                   "end_time": (now - dt.timedelta(hours=23)).isoformat()})
    frames = [("frames", (f"{i}.jpg", make_image(1, 5), "image/jpeg")) for i in range(4)]
    r = client.post("/verify", files=frames).json()
    assert r["reason"] == "NO_OPEN_SESSION"


def test_biometric_deletion_preserves_attendance(client):
    u = _enroll(client)
    s = _open_session(client)
    frames = [("frames", (f"{i}.jpg", make_image(1, 5), "image/jpeg")) for i in range(4)]
    client.post("/verify", data={"session_id": s["id"]}, files=frames)

    r = client.delete(f"/users/{u['id']}/biometrics").json()
    assert r["embeddings_deleted"] == 1
    assert len(client.get("/attendance").json()) == 1     # audit trail survives


# --- production hardening ---

def test_too_many_frames_is_rejected(client):
    from backend.app.core.config import settings
    frames = [("frames", (f"{i}.jpg", make_image(1, 5), "image/jpeg"))
              for i in range(settings.max_frames_per_request + 5)]
    r = client.post("/verify", files=frames)
    assert r.status_code == 413, r.text
    assert "too many" in r.text


def test_empty_upload_is_rejected(client):
    r = client.post("/verify", files=[("frames", ("x.jpg", b"", "image/jpeg"))])
    # an empty file decodes to nothing -> NO_FACE, never an approval
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert r.json()["decision"] == "rejected"


def test_oversized_upload_is_rejected(client, monkeypatch):
    from backend.app.core.config import settings
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)
    big = b"\xff\xd8\xff" + b"\x00" * 4096
    r = client.post("/verify", files=[("frames", ("big.jpg", big, "image/jpeg"))])
    assert r.status_code == 413
    assert "exceeds" in r.text


def test_too_many_enrollment_images_is_rejected(client):
    from backend.app.core.config import settings
    u = client.post("/users", json={"student_id": "SBIG", "name": "T",
                                    "email": "big@b.c"}).json()
    imgs = [("images", (f"{i}.jpg", make_image(1, 5), "image/jpeg"))
            for i in range(settings.max_enrollment_images + 3)]
    r = client.post("/enrollment", data={"user_id": u["id"]}, files=imgs)
    assert r.status_code == 413


def test_cors_is_not_wildcarded_by_default():
    """A wildcard origin would let any website drive this API via a visitor's browser."""
    from backend.app.core.config import Settings
    assert Settings().cors_origin_list == []


def test_metrics_endpoint_exposes_counters(client):
    client.get("/users")
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "attendance_http_requests_total" in body
    assert "attendance_models_loaded" in body


def test_metrics_do_not_leak_user_identifiers(client):
    """Per-user metric labels would let scrape history reconstruct attendance."""
    u = client.post("/users", json={"student_id": "SMET", "name": "T",
                                    "email": "met@b.c"}).json()
    client.get(f"/users/{u['id']}")
    body = client.get("/metrics").text
    assert u["id"] not in body, "user id leaked into metrics labels"
    assert "SMET" not in body


def test_metric_paths_are_normalised(client):
    """Raw ids in path labels would blow up cardinality unboundedly."""
    from backend.app.core.metrics import normalise_path
    assert normalise_path("/users/4eb5583f-99c1-4b16-8369-bb52737f1941") == "/users/{id}"
    assert normalise_path("/health") == "/health"
    assert normalise_path("/attendance") == "/attendance"
