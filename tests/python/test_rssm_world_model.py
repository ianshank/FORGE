"""Tests for forge.models.rssm_world_model module."""
from __future__ import annotations

import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.models.rssm_world_model import (  # noqa: E402
    DEFAULT_DETERMINISTIC_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_RSSM_WEIGHT_FILE,
    DEFAULT_STATE_DIM,
    DEFAULT_STOCHASTIC_DIM,
    RSSMConfig,
    RSSMWorldModel,
)
from forge.models.world_model import WorldModel  # noqa: E402

# Test dimensions (small for speed)
OBS_DIM = 16
ACTION_DIM = 5
STOCHASTIC_DIM = 8
DETERMINISTIC_DIM = 12
HIDDEN_DIM = 24
STATE_DIM = 16


def _make_config(**overrides: int | str) -> RSSMConfig:
    """Create a small test config."""
    defaults = {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "state_dim": STATE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "stochastic_dim": STOCHASTIC_DIM,
        "deterministic_dim": DETERMINISTIC_DIM,
        "device": "cpu",
    }
    defaults.update(overrides)
    return RSSMConfig(**defaults)


# ---------------------------------------------------------------------------
# RSSMConfig
# ---------------------------------------------------------------------------


class TestRSSMConfig:
    """Tests for RSSMConfig defaults and customisation."""

    def test_defaults(self) -> None:
        """Default config should use module-level constants."""
        cfg = RSSMConfig()
        assert cfg.state_dim == DEFAULT_STATE_DIM
        assert cfg.hidden_dim == DEFAULT_HIDDEN_DIM
        assert cfg.stochastic_dim == DEFAULT_STOCHASTIC_DIM
        assert cfg.deterministic_dim == DEFAULT_DETERMINISTIC_DIM
        assert cfg.device == "cpu"

    def test_custom_values(self) -> None:
        """All fields should accept custom values."""
        cfg = RSSMConfig(obs_dim=32, action_dim=4, hidden_dim=64, device="cuda")
        assert cfg.obs_dim == 32
        assert cfg.action_dim == 4
        assert cfg.hidden_dim == 64
        assert cfg.device == "cuda"

    def test_is_dataclass(self) -> None:
        """RSSMConfig should be a proper dataclass."""
        field_names = {f.name for f in fields(RSSMConfig)}
        expected = {
            "obs_dim",
            "action_dim",
            "state_dim",
            "hidden_dim",
            "stochastic_dim",
            "deterministic_dim",
            "device",
        }
        assert field_names == expected


# ---------------------------------------------------------------------------
# RSSMWorldModel — ABC compliance
# ---------------------------------------------------------------------------


class TestRSSMWorldModelABC:
    """Tests for WorldModel ABC compliance."""

    def test_isinstance_world_model(self) -> None:
        """RSSMWorldModel should be an instance of WorldModel."""
        model = RSSMWorldModel(_make_config())
        assert isinstance(model, WorldModel)

    def test_config_property(self) -> None:
        """config property should return the stored config."""
        cfg = _make_config()
        model = RSSMWorldModel(cfg)
        assert model.config is cfg


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


class TestRSSMWorldModelEncode:
    """Tests for RSSMWorldModel.encode()."""

    def test_encode_output_shape(self) -> None:
        """encode() should return array of shape (stochastic_dim,)."""
        model = RSSMWorldModel(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        result = model.encode(obs)
        assert result.shape == (STOCHASTIC_DIM,)

    def test_encode_output_dtype(self) -> None:
        """encode() should return float32."""
        model = RSSMWorldModel(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        result = model.encode(obs)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestRSSMWorldModelPredict:
    """Tests for RSSMWorldModel.predict()."""

    def test_predict_from_stochastic_state(self) -> None:
        """predict() with stochastic-only state should return obs_dim."""
        model = RSSMWorldModel(_make_config())
        state = np.random.randn(STOCHASTIC_DIM).astype(np.float32)
        result = model.predict(state, action=0)
        assert result.shape == (OBS_DIM,)

    def test_predict_from_full_state(self) -> None:
        """predict() with full state should return obs_dim."""
        model = RSSMWorldModel(_make_config())
        full_dim = DETERMINISTIC_DIM + STOCHASTIC_DIM
        state = np.random.randn(full_dim).astype(np.float32)
        result = model.predict(state, action=2)
        assert result.shape == (OBS_DIM,)

    def test_predict_action_bounds(self) -> None:
        """predict() should handle actions at boundaries."""
        model = RSSMWorldModel(_make_config())
        state = np.random.randn(STOCHASTIC_DIM).astype(np.float32)
        # Action at upper bound
        result = model.predict(state, action=ACTION_DIM - 1)
        assert result.shape == (OBS_DIM,)

    def test_predict_returns_finite(self) -> None:
        """predict() should return finite values."""
        model = RSSMWorldModel(_make_config())
        state = np.random.randn(STOCHASTIC_DIM).astype(np.float32)
        result = model.predict(state, action=1)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# imagine
# ---------------------------------------------------------------------------


class TestRSSMWorldModelImagine:
    """Tests for RSSMWorldModel.imagine()."""

    def test_imagine_output_shape(self) -> None:
        """imagine() should return (T, obs_dim) predictions."""
        model = RSSMWorldModel(_make_config())
        state = np.random.randn(STOCHASTIC_DIM).astype(np.float32)
        actions = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        result = model.imagine(state, actions)
        assert result.shape == (5, OBS_DIM)

    def test_imagine_single_step(self) -> None:
        """imagine() with a single action should return (1, obs_dim)."""
        model = RSSMWorldModel(_make_config())
        state = np.random.randn(STOCHASTIC_DIM).astype(np.float32)
        actions = np.array([0], dtype=np.int64)
        result = model.imagine(state, actions)
        assert result.shape == (1, OBS_DIM)

    def test_imagine_returns_finite(self) -> None:
        """imagine() should return finite values."""
        model = RSSMWorldModel(_make_config())
        state = np.random.randn(STOCHASTIC_DIM).astype(np.float32)
        actions = np.array([1, 2], dtype=np.int64)
        result = model.imagine(state, actions)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


class TestRSSMWorldModelTrainStep:
    """Tests for RSSMWorldModel.train_step()."""

    def test_train_step_returns_metrics(self) -> None:
        """train_step() should return loss metrics."""
        model = RSSMWorldModel(_make_config())
        N = 8
        batch = {
            "observations": np.random.randn(N, OBS_DIM).astype(np.float32),
            "actions": np.random.randint(0, ACTION_DIM, N).astype(np.int64),
            "next_observations": np.random.randn(N, OBS_DIM).astype(np.float32),
        }
        metrics = model.train_step(batch)
        assert "loss" in metrics
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics
        assert np.isfinite(metrics["loss"])
        assert np.isfinite(metrics["recon_loss"])
        assert np.isfinite(metrics["kl_loss"])


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestRSSMWorldModelSaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self) -> None:
        """Saving and loading should preserve predictions."""
        cfg = _make_config()
        model1 = RSSMWorldModel(cfg)
        state = np.random.randn(STOCHASTIC_DIM).astype(np.float32)

        # Fix random state for reproducible comparison
        torch.manual_seed(42)
        pred1 = model1.predict(state, action=0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model1.save(path)
            model2 = RSSMWorldModel(cfg)
            model2.load(path)
            torch.manual_seed(42)
            pred2 = model2.predict(state, action=0)
            np.testing.assert_allclose(pred1, pred2, atol=1e-5)
        finally:
            Path(path).unlink()

    def test_load_validates_obs_dim(self) -> None:
        """load() should raise ValueError on dimension mismatch."""
        model1 = RSSMWorldModel(_make_config(obs_dim=16))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model1.save(path)
            model2 = RSSMWorldModel(_make_config(obs_dim=32))
            with pytest.raises(ValueError, match="obs_dim"):
                model2.load(path)
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# load_from_hub
# ---------------------------------------------------------------------------


class TestRSSMWorldModelLoadFromHub:
    """Tests for load_from_hub() with mocked WeightLoader."""

    def test_load_from_hub_calls_resolve_and_load(self) -> None:
        """load_from_hub() should resolve the path then load."""
        model = RSSMWorldModel(_make_config())

        # Save real weights to a temp file so load() succeeds
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        model.save(path)

        try:
            mock_loader = MagicMock()
            mock_loader.resolve_path.return_value = Path(path)

            model.load_from_hub(mock_loader, filename="rssm/custom.pt")
            mock_loader.resolve_path.assert_called_once_with("rssm/custom.pt")
        finally:
            Path(path).unlink()

    def test_load_from_hub_default_filename(self) -> None:
        """load_from_hub() should use default filename when not specified."""
        model = RSSMWorldModel(_make_config())

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        model.save(path)

        try:
            mock_loader = MagicMock()
            mock_loader.resolve_path.return_value = Path(path)

            model.load_from_hub(mock_loader)
            mock_loader.resolve_path.assert_called_once_with(DEFAULT_RSSM_WEIGHT_FILE)
        finally:
            Path(path).unlink()
