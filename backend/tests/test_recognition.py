"""Matching-logic tests. No model required — the embedding maths is separable."""
import numpy as np
import pytest

from backend.app.services.recognition import (
    best_match,
    build_template,
    cosine_similarity,
    deserialize,
    l2_normalize,
    serialize,
)

RNG = np.random.default_rng(0)


def unit(d=512):
    return l2_normalize(RNG.normal(size=d).astype(np.float32))


def test_l2_normalize_produces_unit_vectors():
    v = RNG.normal(size=(5, 512))
    assert np.allclose(np.linalg.norm(l2_normalize(v), axis=-1), 1.0)


def test_l2_normalize_survives_a_zero_vector():
    """A zero embedding must not produce NaNs that silently poison every comparison."""
    out = l2_normalize(np.zeros(512, dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_template_normalises_before_averaging():
    """A high-magnitude embedding must not dominate the template."""
    base = unit()
    good = np.stack([base, base, base])
    # same directions, but one scaled 100x
    skewed = np.stack([base * 100, base, base])
    assert np.allclose(build_template(good), build_template(skewed), atol=1e-5)


def test_template_of_one_embedding_is_that_embedding():
    e = unit()
    assert np.allclose(build_template(e[None, :]), e, atol=1e-6)


def test_empty_enrollment_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        build_template(np.zeros((0, 512), dtype=np.float32))


def test_cosine_similarity_is_one_for_identical_vectors():
    e = unit()
    assert cosine_similarity(e, e[None, :])[0] == pytest.approx(1.0, abs=1e-5)


def test_best_match_finds_the_right_user():
    a, b, c = unit(), unit(), unit()
    templates = {"alice": a, "bob": b, "carol": c}
    m = best_match(a, templates, threshold=0.5)
    assert m.user_id == "alice" and m.similarity > 0.99


def test_below_threshold_returns_no_match():
    templates = {"alice": unit(), "bob": unit()}
    m = best_match(unit(), templates, threshold=0.9)
    assert m.user_id is None


def test_empty_registry_returns_no_match_rather_than_crashing():
    m = best_match(unit(), {}, threshold=0.5)
    assert m.user_id is None and m.n_candidates == 0


def test_ambiguous_match_is_rejected_when_a_margin_is_required():
    """Two near-identical templates must not be resolved by a coin flip."""
    base = unit()
    twin = l2_normalize(base + 0.001 * RNG.normal(size=512).astype(np.float32))
    templates = {"alice": base, "alice_twin": twin}

    assert best_match(base, templates, threshold=0.5, min_margin=0.0).user_id is not None
    ambiguous = best_match(base, templates, threshold=0.5, min_margin=0.05)
    assert ambiguous.user_id is None
    assert ambiguous.margin < 0.05


def test_serialize_round_trips_exactly():
    e = unit()
    assert np.allclose(deserialize(serialize(e), 512), e, atol=1e-7)


def test_deserialize_rejects_a_wrong_dimension():
    """Guards against comparing embeddings from different backbones."""
    with pytest.raises(ValueError, match="expected 512"):
        deserialize(serialize(unit(128)), 512)
