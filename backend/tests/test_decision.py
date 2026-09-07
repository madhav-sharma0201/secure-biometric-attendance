"""Decision engine tests.

This is the component an attacker targets, so the tests are exhaustive rather than
representative: every rejection reason, every boundary, and the fail-closed property
itself.
"""
import pytest

from backend.app.core.decision import (
    Decision,
    Reason,
    Thresholds,
    VerificationInput,
    decide,
)

THR = Thresholds(liveness=0.8, identity=0.7, liveness_uncertain_band=0.1)


def ok(**kw):
    """A fully valid attempt; override one field per test to isolate each rule."""
    base = dict(n_faces=1, liveness_score=0.95, identity_score=0.9,
                matched_user_id="u1", user_is_active=True,
                session_is_open=True, already_marked=False)
    base.update(kw)
    return VerificationInput(**base)


def test_all_checks_passing_is_approved():
    d = decide(ok(), THR)
    assert d.approved and d.user_id == "u1"
    assert d.to_dict()["decision"] == "approved"


@pytest.mark.parametrize("kw,reason", [
    (dict(n_faces=0), Reason.NO_FACE),
    (dict(n_faces=2), Reason.MULTIPLE_FACES),
    (dict(liveness_score=0.10), Reason.LIVENESS_FAILED),
    (dict(liveness_score=0.75), Reason.LOW_LIVENESS_CONFIDENCE),
    (dict(matched_user_id=None), Reason.UNKNOWN_PERSON),
    (dict(identity_score=0.50), Reason.LOW_IDENTITY_CONFIDENCE),
    (dict(user_is_active=False), Reason.USER_INACTIVE),
    (dict(session_is_open=False), Reason.NO_OPEN_SESSION),
    (dict(already_marked=True), Reason.ALREADY_MARKED),
    (dict(liveness_score=None), Reason.SYSTEM_ERROR),
    (dict(identity_score=None), Reason.SYSTEM_ERROR),
])
def test_each_failure_mode_rejects_with_its_reason(kw, reason):
    d = decide(ok(**kw), THR)
    assert not d.approved
    assert d.reason is reason


def test_rejection_never_leaks_the_matched_identity():
    """A rejected attempt must not tell the caller who they resembled.

    Otherwise the endpoint becomes an oracle: present a photo, read back the name of
    the closest registered user.
    """
    for kw in (dict(liveness_score=0.1), dict(identity_score=0.5), dict(already_marked=True)):
        payload = decide(ok(**kw), THR).to_dict()
        assert payload["decision"] == "rejected"
        assert "user_id" not in payload
        assert "identity_confidence" not in payload


def test_liveness_is_evaluated_before_identity():
    """A spoof must be rejected as a spoof, not as an unknown person.

    Identity of an attack presentation is not something we want to compute or log.
    """
    d = decide(ok(liveness_score=0.1, matched_user_id=None, identity_score=0.0), THR)
    assert d.reason is Reason.LIVENESS_FAILED


def test_threshold_boundaries_are_inclusive_at_the_threshold():
    assert decide(ok(liveness_score=0.8), THR).approved        # exactly at threshold
    assert not decide(ok(liveness_score=0.7999), THR).approved
    assert decide(ok(identity_score=0.7), THR).approved
    assert not decide(ok(identity_score=0.6999), THR).approved


def test_uncertain_band_separates_confident_spoof_from_inconclusive():
    assert decide(ok(liveness_score=0.69), THR).reason is Reason.LIVENESS_FAILED
    assert decide(ok(liveness_score=0.71), THR).reason is Reason.LOW_LIVENESS_CONFIDENCE


def test_invalid_thresholds_are_rejected_at_construction():
    with pytest.raises(ValueError, match="liveness"):
        Thresholds(liveness=1.5, identity=0.7)
    with pytest.raises(ValueError, match="identity"):
        Thresholds(liveness=0.8, identity=-0.1)


def test_multiple_simultaneous_failures_still_reject():
    """Fail-closed: nothing about combining failures can produce an approval."""
    d = decide(VerificationInput(n_faces=3, liveness_score=0.0, identity_score=0.0,
                                 matched_user_id=None, user_is_active=False,
                                 session_is_open=False, already_marked=True), THR)
    assert not d.approved


def test_decision_is_immutable():
    """A verdict must not be mutable after the fact by downstream code."""
    d = decide(ok(), THR)
    with pytest.raises(Exception):
        d.approved = False


def test_no_input_combination_approves_without_all_gates():
    """Brute-force the fail-closed property across the whole input space."""
    from itertools import product
    approvals = []
    for n, live, ident, uid, act, sess, marked in product(
        [0, 1, 2], [None, 0.1, 0.79, 0.95], [None, 0.5, 0.95],
        [None, "u1"], [True, False], [True, False], [True, False]
    ):
        d = decide(VerificationInput(n, live, ident, uid, act, sess, marked), THR)
        if d.approved:
            approvals.append((n, live, ident, uid, act, sess, marked))

    for n, live, ident, uid, act, sess, marked in approvals:
        assert n == 1 and live is not None and live >= THR.liveness
        assert ident is not None and ident >= THR.identity
        assert uid is not None and act and sess and not marked
    assert approvals, "sanity: at least one combination should approve"
