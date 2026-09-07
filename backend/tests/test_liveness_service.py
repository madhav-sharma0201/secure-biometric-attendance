import numpy as np
import pytest

from backend.app.services.liveness import LivenessService, sigmoid


def test_sigmoid_matches_reference_in_the_normal_range():
    x = np.linspace(-10, 10, 50)
    assert np.allclose(sigmoid(x), 1 / (1 + np.exp(-x)), atol=1e-12)


def test_sigmoid_is_stable_at_extreme_logits():
    """Large negative logits are exactly what a confident spoof produces."""
    x = np.array([-1000.0, -800.0, 0.0, 800.0, 1000.0])
    out = sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[-1] == pytest.approx(1.0, abs=1e-12)


def test_missing_model_fails_with_an_actionable_message():
    svc = LivenessService("models/does_not_exist.onnx")
    with pytest.raises(FileNotFoundError, match="Train it first"):
        svc.score([np.zeros((3, 112, 112), dtype=np.float32)])


def test_empty_frame_list_is_rejected():
    svc = LivenessService("models/does_not_exist.onnx")
    with pytest.raises(ValueError, match="no frames"):
        svc.score([])
