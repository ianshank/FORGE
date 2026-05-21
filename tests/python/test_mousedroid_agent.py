"""Tests for forge.agents.mousedroid_agent module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.agents.base_agent import BaseAgent  # noqa: E402
from forge.agents.mousedroid_agent import (  # noqa: E402
    DEFAULT_CONSTITUTIONAL_HIDDEN_SIZES,
    MouseDroidAgent,
    MouseDroidConfig,
)
from forge.models.bdi_network import BDINetwork  # noqa: E402
from forge.models.neural_policy import NeuralMCTSPolicy  # noqa: E402
from forge.models.policy_network import ActorCriticNetwork  # noqa: E402
from forge.models.rssm_world_model import RSSMWorldModel  # noqa: E402
from forge.utils.weight_loader import DEFAULT_REPO_ID  # noqa: E402

# Test dimensions (small for speed)
OBS_DIM = 16
ACTION_DIM = 5
BATCH_SIZE = 8


def _make_config(**overrides: object) -> MouseDroidConfig:
    """Create a small test config."""
    defaults: dict = {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "device": "cpu",
        "rssm_state_dim": 8,
        "rssm_hidden_dim": 16,
        "rssm_stochastic_dim": 6,
        "rssm_deterministic_dim": 8,
        "belief_dim": 6,
        "desire_dim": 4,
        "intention_dim": 4,
        "affect_dim": 3,
        "bdi_hidden_sizes": [8, 8],
        "policy_hidden_sizes": [12, 12],
        "constitutional_hidden_sizes": [12, 12],
        "auto_download": False,
    }
    defaults.update(overrides)
    return MouseDroidConfig(**defaults)


# ---------------------------------------------------------------------------
# MouseDroidConfig
# ---------------------------------------------------------------------------


class TestMouseDroidConfig:
    """Tests for MouseDroidConfig defaults and customisation."""

    def test_defaults(self) -> None:
        """Default config should use module-level constants."""
        cfg = MouseDroidConfig()
        assert cfg.name == "mousedroid"
        assert cfg.repo_id == DEFAULT_REPO_ID
        assert cfg.constitutional_hidden_sizes == DEFAULT_CONSTITUTIONAL_HIDDEN_SIZES

    def test_inherits_agent_config(self) -> None:
        """MouseDroidConfig should extend AgentConfig."""
        cfg = MouseDroidConfig()
        # AgentConfig fields should be present
        assert hasattr(cfg, "learning_rate")
        assert hasattr(cfg, "gamma")
        assert hasattr(cfg, "hidden_sizes")

    def test_custom_values(self) -> None:
        """All fields should accept custom values."""
        cfg = MouseDroidConfig(
            obs_dim=32,
            action_dim=10,
            repo_id="custom/repo",
            belief_dim=64,
        )
        assert cfg.obs_dim == 32
        assert cfg.action_dim == 10
        assert cfg.repo_id == "custom/repo"
        assert cfg.belief_dim == 64


# ---------------------------------------------------------------------------
# MouseDroidAgent — construction
# ---------------------------------------------------------------------------


class TestMouseDroidAgentInit:
    """Tests for MouseDroidAgent construction."""

    def test_creation(self) -> None:
        """MouseDroidAgent should construct without errors."""
        agent = MouseDroidAgent(_make_config())
        assert agent.device == "cpu"

    def test_isinstance_base_agent(self) -> None:
        """MouseDroidAgent should be a BaseAgent."""
        agent = MouseDroidAgent(_make_config())
        assert isinstance(agent, BaseAgent)

    def test_device_auto(self) -> None:
        """device='auto' should resolve via get_device()."""
        with patch("forge.utils.device.get_device", return_value="cpu"):
            agent = MouseDroidAgent(_make_config(device="auto"))
            assert agent.device == "cpu"


# ---------------------------------------------------------------------------
# Sub-component access
# ---------------------------------------------------------------------------


class TestMouseDroidAgentComponents:
    """Tests for sub-component property access."""

    def test_world_model(self) -> None:
        """world_model should be an RSSMWorldModel."""
        agent = MouseDroidAgent(_make_config())
        assert isinstance(agent.world_model, RSSMWorldModel)

    def test_bdi(self) -> None:
        """bdi should be a BDINetwork."""
        agent = MouseDroidAgent(_make_config())
        assert isinstance(agent.bdi, BDINetwork)

    def test_policy(self) -> None:
        """policy should be a NeuralMCTSPolicy."""
        agent = MouseDroidAgent(_make_config())
        assert isinstance(agent.policy, NeuralMCTSPolicy)

    def test_constitutional_policy(self) -> None:
        """constitutional_policy should be an ActorCriticNetwork."""
        agent = MouseDroidAgent(_make_config())
        assert isinstance(agent.constitutional_policy, ActorCriticNetwork)

    def test_mousedroid_config(self) -> None:
        """mousedroid_config should return the MouseDroidConfig."""
        cfg = _make_config()
        agent = MouseDroidAgent(cfg)
        assert agent.mousedroid_config is cfg


# ---------------------------------------------------------------------------
# act
# ---------------------------------------------------------------------------


class TestMouseDroidAgentAct:
    """Tests for MouseDroidAgent.act()."""

    def test_returns_valid_action(self) -> None:
        """act() should return an action within bounds."""
        agent = MouseDroidAgent(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _info = agent.act(obs)
        assert 0 <= action < ACTION_DIM

    def test_returns_info_dict(self) -> None:
        """act() should return info with expected keys."""
        agent = MouseDroidAgent(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        _, info = agent.act(obs)
        assert "value" in info
        assert "priors" in info
        assert "belief_norm" in info
        assert "desire_norm" in info
        assert "intention_norm" in info
        assert "affect_norm" in info

    def test_increments_step_count(self) -> None:
        """act() should increment step_count."""
        agent = MouseDroidAgent(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        assert agent.step_count == 0
        agent.act(obs)
        assert agent.step_count == 1
        agent.act(obs)
        assert agent.step_count == 2

    def test_info_values_are_finite(self) -> None:
        """act() info values should all be finite numbers."""
        agent = MouseDroidAgent(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        _, info = agent.act(obs)
        assert np.isfinite(info["value"])
        assert np.isfinite(info["belief_norm"])
        assert np.isfinite(info["desire_norm"])
        assert np.isfinite(info["intention_norm"])
        assert np.isfinite(info["affect_norm"])


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------


class TestMouseDroidAgentLearn:
    """Tests for MouseDroidAgent.learn()."""

    def test_learn_returns_metrics(self) -> None:
        """learn() should return training metrics."""
        agent = MouseDroidAgent(_make_config())
        N = 16
        batch = {
            "observations": np.random.randn(N, OBS_DIM).astype(np.float32),
            "actions": np.random.randint(0, ACTION_DIM, N).astype(np.int64),
            "old_log_probs": np.full(N, -np.log(ACTION_DIM), dtype=np.float32),
            "advantages": np.random.randn(N).astype(np.float32),
            "returns": np.random.randn(N).astype(np.float32),
        }
        metrics = agent.learn(batch)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "loss" in metrics
        assert np.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestMouseDroidAgentSaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self) -> None:
        """Saving and loading should preserve agent state."""
        cfg = _make_config()
        agent1 = MouseDroidAgent(cfg)
        agent1._step_count = 42

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/agent"
            agent1.save(path)

            # Verify files were created
            base = Path(path)
            assert base.with_suffix(".rssm.pt").exists()
            assert base.with_suffix(".bdi.pt").exists()
            assert base.with_suffix(".policy.pt").exists()
            assert base.with_suffix(".constitutional.pt").exists()
            assert base.with_suffix(".json").exists()

            agent2 = MouseDroidAgent(cfg)
            agent2.load(path)
            assert agent2.step_count == 42

    def test_save_creates_directories(self) -> None:
        """save() should create parent directories."""
        agent = MouseDroidAgent(_make_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/deep/nested/agent"
            agent.save(path)
            assert Path(path).with_suffix(".json").exists()


# ---------------------------------------------------------------------------
# load_from_hub
# ---------------------------------------------------------------------------


class TestMouseDroidAgentLoadFromHub:
    """Tests for load_from_hub() with mocked dependencies."""

    def test_load_from_hub_creates_loader(self) -> None:
        """load_from_hub() without loader should create one from config."""
        agent = MouseDroidAgent(_make_config())

        mock_loader_cls = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load_npz.return_value = {}
        mock_loader.resolve_path.return_value = Path("/fake.pt")
        mock_loader_cls.return_value = mock_loader

        with (
            patch("forge.agents.mousedroid_agent.WeightLoader", mock_loader_cls),
            patch.object(agent.world_model, "load_from_hub"),
            patch.object(agent.bdi, "load_from_hub"),
            patch.object(agent.policy, "load_from_npz"),
        ):
            agent.load_from_hub()
            mock_loader_cls.assert_called_once()

    def test_load_from_hub_uses_provided_loader(self) -> None:
        """load_from_hub() with a loader should use it directly."""
        agent = MouseDroidAgent(_make_config())

        mock_loader = MagicMock()
        mock_loader.load_npz.return_value = {}

        mock_wm = MagicMock()
        mock_bdi = MagicMock()
        mock_pol = MagicMock()
        agent._world_model = mock_wm
        agent._bdi = mock_bdi
        agent._policy = mock_pol

        agent.load_from_hub(mock_loader)

        mock_wm.load_from_hub.assert_called_once_with(mock_loader)
        mock_bdi.load_from_hub.assert_called_once_with(mock_loader)
        mock_pol.load_from_npz.assert_called_once_with(mock_loader)
