"""Authentication, rate limiting and request context.

Scope note: this is API-key authentication with an in-process rate limiter, which is
appropriate for a single-tenant deployment on one cluster. It is NOT a substitute for
per-user identity and OAuth in a real institutional deployment, and the README says so.
Stating the limit is better than implying coverage the code does not have.
"""
from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from backend.app.core.config import settings

# Endpoints reachable without a key. Verification is intentionally NOT here: a kiosk
# gets a key. Probes are, because Kubernetes cannot present credentials.
PUBLIC_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}


def require_api_key(request: Request) -> None:
    """Constant-time API key check.

    hmac.compare_digest rather than ==: a plain comparison short-circuits on the first
    differing byte, leaking key material through response timing. The window is small
    over a network but the correct comparison costs nothing.
    """
    if request.url.path in PUBLIC_PATHS:
        return

    provided = request.headers.get("x-api-key", "")
    expected = settings.api_key

    if expected in ("", "change_me"):
        # Refuse to run unauthenticated rather than silently accepting everything.
        # A deployment that forgot to set a key should fail loudly, not serve openly.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "API_KEY is unset or left at its default; refusing to serve authenticated routes",
        )

    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


class RateLimiter:
    """Sliding-window limiter, per client IP.

    In-process and therefore per-pod: with N replicas the effective global limit is
    N times the configured value. That is a real limitation of not using shared state
    (Redis), and it is documented rather than hidden. For brute-force resistance on a
    verification endpoint it is still worth having.
    """

    def __init__(self, max_requests: int = 30, window_sec: float = 60.0):
        self.max_requests = max_requests
        self.window = window_sec
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.max_requests:
            return False, 0
        q.append(now)
        return True, self.max_requests - len(q)

    def reset(self) -> None:
        self._hits.clear()


verification_limiter = RateLimiter(max_requests=30, window_sec=60.0)
