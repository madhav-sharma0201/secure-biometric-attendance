import time

import pytest
from fastapi import HTTPException

from backend.app.core.security import RateLimiter, require_api_key
from backend.app.core.config import settings


class FakeRequest:
    def __init__(self, path="/verify", key=None):
        self.url = type("U", (), {"path": path})()
        self.headers = {"x-api-key": key} if key else {}


def test_probe_endpoints_do_not_require_a_key():
    """Kubernetes probes cannot present credentials."""
    for p in ("/health", "/ready"):
        require_api_key(FakeRequest(p))


def test_unset_key_refuses_to_serve_rather_than_serving_openly(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "change_me")
    with pytest.raises(HTTPException) as e:
        require_api_key(FakeRequest("/verify", "anything"))
    assert e.value.status_code == 503


def test_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "correct-key")
    with pytest.raises(HTTPException) as e:
        require_api_key(FakeRequest("/verify", "wrong-key"))
    assert e.value.status_code == 401


def test_missing_key_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "correct-key")
    with pytest.raises(HTTPException) as e:
        require_api_key(FakeRequest("/verify"))
    assert e.value.status_code == 401


def test_correct_key_passes(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "correct-key")
    require_api_key(FakeRequest("/verify", "correct-key"))


def test_rate_limiter_blocks_after_the_limit():
    rl = RateLimiter(max_requests=3, window_sec=60)
    assert all(rl.check("1.2.3.4")[0] for _ in range(3))
    allowed, remaining = rl.check("1.2.3.4")
    assert not allowed and remaining == 0


def test_rate_limiter_is_per_client():
    rl = RateLimiter(max_requests=2, window_sec=60)
    for _ in range(2):
        rl.check("a")
    assert rl.check("a")[0] is False
    assert rl.check("b")[0] is True     # a different client is unaffected


def test_rate_limiter_window_expires():
    rl = RateLimiter(max_requests=1, window_sec=0.15)
    assert rl.check("x")[0] is True
    assert rl.check("x")[0] is False
    time.sleep(0.2)
    assert rl.check("x")[0] is True
