"""The decision engine.

A PURE function: no database, no HTTP, no clock, no model. Everything it needs is
passed in, and it returns a verdict. That makes the security-critical logic of the
whole system exhaustively unit-testable without spinning up infrastructure — which is
the point, because this is the component an attacker is trying to defeat.

Fail-closed is enforced structurally rather than by convention: the function returns
APPROVED from exactly one place, reached only after every check has passed. Any new
check added above that point rejects by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Reason(str, Enum):
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LIVENESS_FAILED = "LIVENESS_FAILED"
    LOW_LIVENESS_CONFIDENCE = "LOW_LIVENESS_CONFIDENCE"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    LOW_IDENTITY_CONFIDENCE = "LOW_IDENTITY_CONFIDENCE"
    ALREADY_MARKED = "ALREADY_MARKED"
    NO_OPEN_SESSION = "NO_OPEN_SESSION"
    USER_INACTIVE = "USER_INACTIVE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass(frozen=True)
class Thresholds:
    """Operating point. Both values are calibrated on validation data, never on test.

    `liveness_uncertain_band` separates a confident spoof call from an inconclusive
    one. Both reject; they are distinguished only so the user gets a useful message
    and the logs can tell the two situations apart.
    """
    liveness: float
    identity: float
    liveness_uncertain_band: float = 0.10

    def __post_init__(self):
        for name in ("liveness", "identity"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} threshold must be in [0, 1], got {v}")


@dataclass(frozen=True)
class VerificationInput:
    n_faces: int
    liveness_score: float | None = None
    identity_score: float | None = None
    matched_user_id: str | None = None
    user_is_active: bool = True
    session_is_open: bool = False
    already_marked: bool = False


@dataclass(frozen=True)
class Decision:
    approved: bool
    reason: Reason | None = None
    user_id: str | None = None
    liveness_confidence: float | None = None
    identity_confidence: float | None = None

    def to_dict(self) -> dict:
        if self.approved:
            return {
                "decision": "approved",
                "user_id": self.user_id,
                "identity_confidence": round(self.identity_confidence or 0.0, 4),
                "liveness_confidence": round(self.liveness_confidence or 0.0, 4),
            }
        return {"decision": "rejected", "reason": self.reason.value if self.reason else None}


def decide(inp: VerificationInput, thr: Thresholds) -> Decision:
    """Evaluate a verification attempt. Returns approved only if every check passes.

    Order matters: cheap and unambiguous checks first, so a rejected attempt reports
    the most specific useful reason. Liveness is checked BEFORE identity deliberately —
    if the presentation is an attack, which registered user it happens to resemble is
    not information we want to compute, log, or leak back to the caller.
    """
    def reject(r: Reason) -> Decision:
        return Decision(approved=False, reason=r)

    if inp.n_faces == 0:
        return reject(Reason.NO_FACE)
    if inp.n_faces > 1:
        return reject(Reason.MULTIPLE_FACES)

    # Liveness gate.
    if inp.liveness_score is None:
        return reject(Reason.SYSTEM_ERROR)
    if inp.liveness_score < thr.liveness - thr.liveness_uncertain_band:
        return reject(Reason.LIVENESS_FAILED)
    if inp.liveness_score < thr.liveness:
        return reject(Reason.LOW_LIVENESS_CONFIDENCE)

    # Identity gate.
    if inp.identity_score is None:
        return reject(Reason.SYSTEM_ERROR)
    if inp.matched_user_id is None:
        return reject(Reason.UNKNOWN_PERSON)
    if inp.identity_score < thr.identity:
        return reject(Reason.LOW_IDENTITY_CONFIDENCE)

    # Business rules.
    if not inp.user_is_active:
        return reject(Reason.USER_INACTIVE)
    if not inp.session_is_open:
        return reject(Reason.NO_OPEN_SESSION)
    if inp.already_marked:
        return reject(Reason.ALREADY_MARKED)

    return Decision(
        approved=True,
        user_id=inp.matched_user_id,
        liveness_confidence=inp.liveness_score,
        identity_confidence=inp.identity_score,
    )
