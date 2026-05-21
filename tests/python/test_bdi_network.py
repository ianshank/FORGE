"""Tests for forge.models.bdi_network module."""

from __future__ import annotations

import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.models.bdi_network import (  # noqa: E402
    DEFAULT_AFFECT_DIM,
    DEFAULT_AFFECT_WEIGHT_FILE,
    DEFAULT_BDI_HIDDEN_SIZES,
    DEFAULT_BELIEF_DIM,
    DEFAULT_BELIEF_WEIGHT_FILE,
    DEFAULT_DESIRE_DIM,
    DEFAULT_DESIRE_WEIGHT_FILE,
    DEFAULT_INTENTION_DIM,
    DEFAULT_INTENTION_WEIGHT_FILE,
    BDIConfig,
    BDINetwork,
    BDIState,
)

# Test dimensions
OBS_DIM = 16
BELIEF_DIM = 8
DESIRE_DIM = 6
INTENTION_DIM = 6
AFFECT_DIM = 4
HIDDEN_SIZES = [12, 12]


def _make_config(**overrides: object) -> BDIConfig:
    """Create a small test config."""
    defaults: dict = {
        "obs_dim": OBS_DIM,
        "belief_dim": BELIEF_DIM,
        "desire_dim": DESIRE_DIM,
        "intention_dim": INTENTION_DIM,
        "affect_dim": AFFECT_DIM,
        "hidden_sizes": list(HIDDEN_SIZES),
        "device": "cpu",
    }
    defaults.update(overrides)
    return BDIConfig(**defaults)


# ---------------------------------------------------------------------------
# BDIConfig
# ---------------------------------------------------------------------------


class TestBDIConfig:
    """Tests for BDIConfig defaults and customisation."""

    def test_defaults(self) -> None:
        """Default config should use module-level constants."""
        cfg = BDIConfig()
        assert cfg.belief_dim == DEFAULT_BELIEF_DIM
        assert cfg.desire_dim == DEFAULT_DESIRE_DIM
        assert cfg.intention_dim == DEFAULT_INTENTION_DIM
        assert cfg.affect_dim == DEFAULT_AFFECT_DIM
        assert cfg.hidden_sizes == DEFAULT_BDI_HIDDEN_SIZES
        assert cfg.device == "cpu"

    def test_custom_values(self) -> None:
        """All fields should accept custom values."""
        cfg = BDIConfig(obs_dim=32, belief_dim=64, hidden_sizes=[128])
        assert cfg.obs_dim == 32
        assert cfg.belief_dim == 64
        assert cfg.hidden_sizes == [128]

    def test_is_dataclass(self) -> None:
        """BDIConfig should be a proper dataclass."""
        field_names = {f.name for f in fields(BDIConfig)}
        expected = {
            "obs_dim",
            "belief_dim",
            "desire_dim",
            "intention_dim",
            "affect_dim",
            "hidden_sizes",
            "device",
        }
        assert field_names == expected


# ---------------------------------------------------------------------------
# BDIState
# ---------------------------------------------------------------------------


class TestBDIState:
    """Tests for BDIState dataclass."""

    def test_fields(self) -> None:
        """BDIState should have belief, desire, intention, affect fields."""
        state = BDIState(
            belief=np.zeros(4),
            desire=np.zeros(3),
            intention=np.zeros(3),
            affect=np.zeros(2),
        )
        assert state.belief.shape == (4,)
        assert state.desire.shape == (3,)
        assert state.intention.shape == (3,)
        assert state.affect.shape == (2,)


# ---------------------------------------------------------------------------
# BDINetwork — construction
# ---------------------------------------------------------------------------


class TestBDINetworkInit:
    """Tests for BDINetwork construction."""

    def test_default_config(self) -> None:
        """BDINetwork with no args should use default config."""
        bdi = BDINetwork()
        assert bdi.config.belief_dim == DEFAULT_BELIEF_DIM

    def test_custom_config(self) -> None:
        """BDINetwork should accept custom config."""
        cfg = _make_config()
        bdi = BDINetwork(cfg)
        assert bdi.config is cfg


# ---------------------------------------------------------------------------
# encode_belief
# ---------------------------------------------------------------------------


class TestBDINetworkEncodeBelief:
    """Tests for BDINetwork.encode_belief()."""

    def test_output_shape(self) -> None:
        """encode_belief() should return (belief_dim,)."""
        bdi = BDINetwork(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        belief = bdi.encode_belief(obs)
        assert belief.shape == (BELIEF_DIM,)

    def test_output_dtype(self) -> None:
        """encode_belief() should return float32."""
        bdi = BDINetwork(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        belief = bdi.encode_belief(obs)
        assert belief.dtype == np.float32

    def test_output_finite(self) -> None:
        """encode_belief() should return finite values."""
        bdi = BDINetwork(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        belief = bdi.encode_belief(obs)
        assert np.all(np.isfinite(belief))


# ---------------------------------------------------------------------------
# encode_desire
# ---------------------------------------------------------------------------


class TestBDINetworkEncodeDesire:
    """Tests for BDINetwork.encode_desire()."""

    def test_output_shape(self) -> None:
        """encode_desire() should return (desire_dim,)."""
        bdi = BDINetwork(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        belief = np.random.randn(BELIEF_DIM).astype(np.float32)
        desire = bdi.encode_desire(obs, belief)
        assert desire.shape == (DESIRE_DIM,)


# ---------------------------------------------------------------------------
# predict_intention
# ---------------------------------------------------------------------------


class TestBDINetworkPredictIntention:
    """Tests for BDINetwork.predict_intention()."""

    def test_output_shape(self) -> None:
        """predict_intention() should return (intention_dim,)."""
        bdi = BDINetwork(_make_config())
        belief = np.random.randn(BELIEF_DIM).astype(np.float32)
        desire = np.random.randn(DESIRE_DIM).astype(np.float32)
        intention = bdi.predict_intention(belief, desire)
        assert intention.shape == (INTENTION_DIM,)


# ---------------------------------------------------------------------------
# estimate_affect
# ---------------------------------------------------------------------------


class TestBDINetworkEstimateAffect:
    """Tests for BDINetwork.estimate_affect()."""

    def test_output_shape(self) -> None:
        """estimate_affect() should return (affect_dim,)."""
        bdi = BDINetwork(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        belief = np.random.randn(BELIEF_DIM).astype(np.float32)
        affect = bdi.estimate_affect(obs, belief)
        assert affect.shape == (AFFECT_DIM,)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestBDINetworkForward:
    """Tests for BDINetwork.forward()."""

    def test_returns_bdi_state(self) -> None:
        """forward() should return a BDIState with correct dimensions."""
        bdi = BDINetwork(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        state = bdi.forward(obs)

        assert isinstance(state, BDIState)
        assert state.belief.shape == (BELIEF_DIM,)
        assert state.desire.shape == (DESIRE_DIM,)
        assert state.intention.shape == (INTENTION_DIM,)
        assert state.affect.shape == (AFFECT_DIM,)

    def test_all_outputs_finite(self) -> None:
        """forward() should produce all-finite outputs."""
        bdi = BDINetwork(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        state = bdi.forward(obs)

        assert np.all(np.isfinite(state.belief))
        assert np.all(np.isfinite(state.desire))
        assert np.all(np.isfinite(state.intention))
        assert np.all(np.isfinite(state.affect))


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestBDINetworkSaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self) -> None:
        """Saving and loading should preserve forward outputs."""
        cfg = _make_config()
        bdi1 = BDINetwork(cfg)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        state1 = bdi1.forward(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            bdi1.save(path)
            bdi2 = BDINetwork(cfg)
            bdi2.load(path)
            state2 = bdi2.forward(obs)

            np.testing.assert_allclose(state1.belief, state2.belief, atol=1e-6)
            np.testing.assert_allclose(state1.desire, state2.desire, atol=1e-6)
            np.testing.assert_allclose(state1.intention, state2.intention, atol=1e-6)
            np.testing.assert_allclose(state1.affect, state2.affect, atol=1e-6)
        finally:
            Path(path).unlink()

    def test_load_validates_obs_dim(self) -> None:
        """load() should raise ValueError on obs_dim mismatch."""
        bdi1 = BDINetwork(_make_config(obs_dim=16))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            bdi1.save(path)
            bdi2 = BDINetwork(_make_config(obs_dim=32))
            with pytest.raises(ValueError, match="obs_dim"):
                bdi2.load(path)
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# load_from_hub
# ---------------------------------------------------------------------------


class TestBDINetworkLoadFromHub:
    """Tests for load_from_hub() with mocked WeightLoader."""

    def test_load_from_hub_calls_all_four_files(self) -> None:
        """load_from_hub() should load all four sub-network weight files."""
        bdi = BDINetwork(_make_config())

        mock_loader = MagicMock()
        # Return empty dicts so no actual loading happens
        mock_loader.load_npz.return_value = {}

        bdi.load_from_hub(mock_loader)

        calls = [c.args[0] for c in mock_loader.load_npz.call_args_list]
        assert DEFAULT_BELIEF_WEIGHT_FILE in calls
        assert DEFAULT_DESIRE_WEIGHT_FILE in calls
        assert DEFAULT_INTENTION_WEIGHT_FILE in calls
        assert DEFAULT_AFFECT_WEIGHT_FILE in calls

    def test_load_from_hub_custom_filenames(self) -> None:
        """load_from_hub() should accept custom filenames."""
        bdi = BDINetwork(_make_config())

        mock_loader = MagicMock()
        mock_loader.load_npz.return_value = {}

        bdi.load_from_hub(
            mock_loader,
            belief_file="custom/b.npz",
            desire_file="custom/d.npz",
            intention_file="custom/i.npz",
            affect_file="custom/a.npz",
        )

        calls = [c.args[0] for c in mock_loader.load_npz.call_args_list]
        assert "custom/b.npz" in calls
        assert "custom/d.npz" in calls
        assert "custom/i.npz" in calls
        assert "custom/a.npz" in calls
