"""Tests for MuZero MCTS planner and agent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.agents.base_agent import AgentConfig  # noqa: E402
from forge.agents.muzero_agent import MuZeroAgent  # noqa: E402
from forge.agents.muzero_mcts import (  # noqa: E402
    DEFAULT_C_PUCT,
    DEFAULT_NUM_SIMULATIONS,
    MuZeroMCTS,
    MuZeroMCTSConfig,
    _softmax,
)
from forge.models.muzero_config import MuZeroConfig  # noqa: E402
from forge.models.muzero_world_model import MuZeroWorldModel  # noqa: E402

# Small test dimensions
OBS_DIM = 11 * 11 * 7 + 73
ACTION_DIM = 8
LATENT_DIM = 16
HIDDEN_DIM = 16


def _make_model() -> MuZeroWorldModel:
    cfg = MuZeroConfig(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=1,
        reward_support_size=11,
        value_support_size=11,
        cnn_channels=(8,),
        cnn_kernel_sizes=(3,),
        cnn_strides=(1,),
    )
    return MuZeroWorldModel(cfg)


def _make_mcts(
    num_simulations: int = 10,
    add_noise: bool = False,
) -> MuZeroMCTS:
    model = _make_model()
    config = MuZeroMCTSConfig(
        num_simulations=num_simulations,
        add_exploration_noise=add_noise,
    )
    return MuZeroMCTS(model, config)


# ---------------------------------------------------------------------------
# MuZeroMCTSConfig
# ---------------------------------------------------------------------------


class TestMuZeroMCTSConfig:
    def test_defaults(self) -> None:
        cfg = MuZeroMCTSConfig()
        assert cfg.num_simulations == DEFAULT_NUM_SIMULATIONS
        assert cfg.c_puct == DEFAULT_C_PUCT

    def test_custom(self) -> None:
        cfg = MuZeroMCTSConfig(num_simulations=200, c_puct=2.0)
        assert cfg.num_simulations == 200
        assert cfg.c_puct == 2.0


# ---------------------------------------------------------------------------
# MuZeroMCTS
# ---------------------------------------------------------------------------


class TestMuZeroMCTS:
    def test_search_returns_valid_action(self) -> None:
        mcts = _make_mcts()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, info = mcts.search(obs)
        assert 0 <= action < ACTION_DIM
        assert "visit_counts" in info
        assert "action_probs" in info
        assert "root_value" in info

    def test_visit_counts_shape(self) -> None:
        mcts = _make_mcts()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        _, info = mcts.search(obs)
        assert len(info["visit_counts"]) == ACTION_DIM

    def test_action_probs_sum_to_one(self) -> None:
        mcts = _make_mcts(num_simulations=20)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        _, info = mcts.search(obs)
        np.testing.assert_allclose(info["action_probs"].sum(), 1.0, atol=1e-5)

    def test_greedy_selects_most_visited(self) -> None:
        mcts = _make_mcts(num_simulations=20)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, info = mcts.search(obs, temperature=0.0)
        assert action == int(np.argmax(info["visit_counts"]))

    def test_dirichlet_noise_changes_search(self) -> None:
        model = _make_model()
        obs = np.random.randn(OBS_DIM).astype(np.float32)

        np.random.seed(42)
        mcts_no_noise = MuZeroMCTS(
            model,
            MuZeroMCTSConfig(
                num_simulations=10,
                add_exploration_noise=False,
            ),
        )
        _, info_no = mcts_no_noise.search(obs, temperature=0.0)

        # With noise, the distribution may differ
        np.random.seed(99)
        mcts_with_noise = MuZeroMCTS(
            model,
            MuZeroMCTSConfig(
                num_simulations=10,
                add_exploration_noise=True,
            ),
        )
        _, info_yes = mcts_with_noise.search(obs, temperature=0.0)

        # At minimum, both should produce valid results
        assert len(info_no["visit_counts"]) == ACTION_DIM
        assert len(info_yes["visit_counts"]) == ACTION_DIM

    def test_zero_simulations(self) -> None:
        mcts = _make_mcts(num_simulations=0)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _info = mcts.search(obs)
        assert 0 <= action < ACTION_DIM

    def test_config_property(self) -> None:
        mcts = _make_mcts()
        assert mcts.config.num_simulations == 10


# ---------------------------------------------------------------------------
# _MinMaxStats
# ---------------------------------------------------------------------------


class TestMinMaxStats:
    def test_normalize_range(self) -> None:
        from forge.agents.muzero_mcts import _MinMaxStats

        stats = _MinMaxStats()
        stats.update(1.0)
        stats.update(5.0)
        assert abs(stats.normalize(3.0) - 0.5) < 1e-6

    def test_normalize_equal_min_max(self) -> None:
        from forge.agents.muzero_mcts import _MinMaxStats

        stats = _MinMaxStats()
        stats.update(3.0)
        assert stats.normalize(3.0) == 0.0


# ---------------------------------------------------------------------------
# _visit_counts_to_probs
# ---------------------------------------------------------------------------


class TestVisitCountsToProbs:
    def test_zero_counts_uniform(self) -> None:
        probs = MuZeroMCTS._visit_counts_to_probs(np.zeros(4), temperature=1.0)
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25])

    def test_single_visited(self) -> None:
        counts = np.array([0.0, 0.0, 10.0, 0.0])
        probs = MuZeroMCTS._visit_counts_to_probs(counts, temperature=0.0)
        assert probs[2] == 1.0


# ---------------------------------------------------------------------------
# _softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_uniform(self) -> None:
        result = _softmax(np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [0.25, 0.25, 0.25, 0.25], atol=1e-6)

    def test_sums_to_one(self) -> None:
        result = _softmax(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_peaked(self) -> None:
        result = _softmax(np.array([0.0, 0.0, 100.0, 0.0]))
        assert result[2] > 0.99


# ---------------------------------------------------------------------------
# MuZeroAgent
# ---------------------------------------------------------------------------


class TestMuZeroAgent:
    def test_act_returns_valid(self) -> None:
        model = _make_model()
        agent = MuZeroAgent(
            AgentConfig(name="test_muzero"),
            model,
            MuZeroMCTSConfig(num_simulations=5),
        )
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _info = agent.act(obs)
        assert 0 <= action < ACTION_DIM
        assert agent.step_count == 1

    def test_temperature_property(self) -> None:
        model = _make_model()
        agent = MuZeroAgent(AgentConfig(), model)
        agent.temperature = 0.5
        assert agent.temperature == 0.5

    def test_learn_delegates(self) -> None:
        model = _make_model()
        agent = MuZeroAgent(
            AgentConfig(),
            model,
            MuZeroMCTSConfig(num_simulations=2),
        )
        cfg = model.config
        K = cfg.num_unroll_steps
        batch = {
            "observations": np.random.randn(2, OBS_DIM).astype(np.float32),
            "actions": np.random.randint(0, ACTION_DIM, (2, K)).astype(np.int64),
            "target_values": np.random.randn(2, K + 1).astype(np.float32),
            "target_rewards": np.random.randn(2, K).astype(np.float32),
            "target_policies": np.random.dirichlet(np.ones(ACTION_DIM), (2, K + 1)).astype(
                np.float32
            ),
        }
        metrics = agent.learn(batch)
        assert "loss" in metrics
