"""Tests for forge.models.neural_policy module."""
from __future__ import annotations

import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.models.neural_policy import (  # noqa: E402
    DEFAULT_POLICY_HIDDEN_SIZES,
    DEFAULT_POLICY_WEIGHT_FILE,
    NeuralMCTSPolicy,
    NeuralPolicyConfig,
)
from forge.models.policy_network import PolicyNetwork  # noqa: E402

# Test dimensions
OBS_DIM = 16
ACTION_DIM = 5
HIDDEN_SIZES = [24, 24]
BATCH_SIZE = 8


def _make_config(**overrides: object) -> NeuralPolicyConfig:
    """Create a small test config."""
    defaults: dict = {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "hidden_sizes": list(HIDDEN_SIZES),
        "device": "cpu",
    }
    defaults.update(overrides)
    return NeuralPolicyConfig(**defaults)


# ---------------------------------------------------------------------------
# NeuralPolicyConfig
# ---------------------------------------------------------------------------


class TestNeuralPolicyConfig:
    """Tests for NeuralPolicyConfig defaults and customisation."""

    def test_defaults(self) -> None:
        """Default config should use module-level constants."""
        cfg = NeuralPolicyConfig()
        assert cfg.hidden_sizes == DEFAULT_POLICY_HIDDEN_SIZES
        assert cfg.device == "cpu"
        assert cfg.learning_rate == pytest.approx(3e-4)

    def test_custom_values(self) -> None:
        """All fields should accept custom values."""
        cfg = NeuralPolicyConfig(obs_dim=32, action_dim=10, hidden_sizes=[64])
        assert cfg.obs_dim == 32
        assert cfg.action_dim == 10
        assert cfg.hidden_sizes == [64]

    def test_is_dataclass(self) -> None:
        """NeuralPolicyConfig should be a proper dataclass."""
        field_names = {f.name for f in fields(NeuralPolicyConfig)}
        expected = {"obs_dim", "action_dim", "hidden_sizes", "learning_rate", "device"}
        assert field_names == expected


# ---------------------------------------------------------------------------
# ABC compliance
# ---------------------------------------------------------------------------


class TestNeuralMCTSPolicyABC:
    """Tests for PolicyNetwork ABC compliance."""

    def test_isinstance_policy_network(self) -> None:
        """NeuralMCTSPolicy should be an instance of PolicyNetwork."""
        policy = NeuralMCTSPolicy(_make_config())
        assert isinstance(policy, PolicyNetwork)

    def test_config_property(self) -> None:
        """config property should return the stored config."""
        cfg = _make_config()
        policy = NeuralMCTSPolicy(cfg)
        assert policy.config is cfg


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestNeuralMCTSPolicyForward:
    """Tests for NeuralMCTSPolicy.forward()."""

    def test_forward_single_obs(self) -> None:
        """forward() with 1-D input should return (action_dim,) probabilities."""
        policy = NeuralMCTSPolicy(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        probs = policy.forward(obs)
        assert probs.shape == (ACTION_DIM,)

    def test_forward_batch_obs(self) -> None:
        """forward() with 2-D input should return (batch, action_dim) probabilities."""
        policy = NeuralMCTSPolicy(_make_config())
        obs = np.random.randn(BATCH_SIZE, OBS_DIM).astype(np.float32)
        probs = policy.forward(obs)
        assert probs.shape == (BATCH_SIZE, ACTION_DIM)

    def test_forward_sums_to_one(self) -> None:
        """forward() output should sum to approximately 1.0 (softmax)."""
        policy = NeuralMCTSPolicy(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        probs = policy.forward(obs)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_forward_all_positive(self) -> None:
        """forward() should return non-negative probabilities."""
        policy = NeuralMCTSPolicy(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        probs = policy.forward(obs)
        assert np.all(probs >= 0)

    def test_forward_dtype(self) -> None:
        """forward() should return float32 array."""
        policy = NeuralMCTSPolicy(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        probs = policy.forward(obs)
        assert probs.dtype == np.float32


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestNeuralMCTSPolicyEvaluate:
    """Tests for NeuralMCTSPolicy.evaluate()."""

    def test_evaluate_returns_tuple(self) -> None:
        """evaluate() should return (priors, value) tuple."""
        policy = NeuralMCTSPolicy(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        priors, value = policy.evaluate(obs)
        assert priors.shape == (ACTION_DIM,)
        assert isinstance(value, float)

    def test_evaluate_priors_sum_to_one(self) -> None:
        """evaluate() priors should sum to ~1.0."""
        policy = NeuralMCTSPolicy(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        priors, _ = policy.evaluate(obs)
        assert priors.sum() == pytest.approx(1.0, abs=1e-5)

    def test_evaluate_value_is_finite(self) -> None:
        """evaluate() value should be a finite number."""
        policy = NeuralMCTSPolicy(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        _, value = policy.evaluate(obs)
        assert np.isfinite(value)


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


class TestNeuralMCTSPolicyTrainStep:
    """Tests for NeuralMCTSPolicy.train_step()."""

    def test_train_step_returns_metrics(self) -> None:
        """train_step() should return loss metrics."""
        policy = NeuralMCTSPolicy(_make_config())
        N = 16
        batch = {
            "observations": np.random.randn(N, OBS_DIM).astype(np.float32),
            "target_priors": np.full((N, ACTION_DIM), 1 / ACTION_DIM, dtype=np.float32),
            "target_values": np.random.randn(N).astype(np.float32),
        }
        metrics = policy.train_step(batch)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "loss" in metrics
        assert np.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestNeuralMCTSPolicySaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self) -> None:
        """Saving and loading should preserve forward output."""
        cfg = _make_config()
        policy1 = NeuralMCTSPolicy(cfg)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        probs1 = policy1.forward(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            policy1.save(path)
            policy2 = NeuralMCTSPolicy(cfg)
            policy2.load(path)
            probs2 = policy2.forward(obs)
            np.testing.assert_allclose(probs1, probs2, atol=1e-6)
        finally:
            Path(path).unlink()

    def test_load_validates_obs_dim(self) -> None:
        """load() should raise ValueError on obs_dim mismatch."""
        policy1 = NeuralMCTSPolicy(_make_config(obs_dim=16))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            policy1.save(path)
            policy2 = NeuralMCTSPolicy(_make_config(obs_dim=32))
            with pytest.raises(ValueError, match="obs_dim"):
                policy2.load(path)
        finally:
            Path(path).unlink()

    def test_load_validates_action_dim(self) -> None:
        """load() should raise ValueError on action_dim mismatch."""
        policy1 = NeuralMCTSPolicy(_make_config(action_dim=5))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            policy1.save(path)
            policy2 = NeuralMCTSPolicy(_make_config(action_dim=10))
            with pytest.raises(ValueError, match="action_dim"):
                policy2.load(path)
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# load_from_npz
# ---------------------------------------------------------------------------


class TestNeuralMCTSPolicyLoadFromNpz:
    """Tests for load_from_npz() with mocked WeightLoader."""

    def test_load_from_npz_calls_loader(self) -> None:
        """load_from_npz() should call loader.load_npz with the filename."""
        policy = NeuralMCTSPolicy(_make_config())

        # Create fake npz data matching param shapes
        all_params = (
            list(policy.encoder.parameters())
            + list(policy.policy_head.parameters())
            + list(policy.value_head.parameters())
        )
        fake_data = {
            f"param_{i:03d}": p.detach().cpu().numpy() for i, p in enumerate(all_params)
        }

        mock_loader = MagicMock()
        mock_loader.load_npz.return_value = fake_data

        policy.load_from_npz(mock_loader, "custom/path.npz")
        mock_loader.load_npz.assert_called_once_with("custom/path.npz")

    def test_load_from_npz_default_filename(self) -> None:
        """load_from_npz() should use default filename when not specified."""
        policy = NeuralMCTSPolicy(_make_config())

        mock_loader = MagicMock()
        mock_loader.load_npz.return_value = {}

        policy.load_from_npz(mock_loader)
        mock_loader.load_npz.assert_called_once_with(DEFAULT_POLICY_WEIGHT_FILE)
