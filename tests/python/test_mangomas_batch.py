"""Tests for MangoMAS batch episode collector."""

from __future__ import annotations

import numpy as np
import pytest

from forge.mangomas.batch import BatchCollector, BatchResult, EpisodeData
from forge.mangomas.config import BatchCollectorConfig


def _mock_reset(seed: int) -> np.ndarray:
    """Mock environment reset returning a fixed observation."""
    return np.ones(18, dtype=np.float32) * 0.5


def _mock_step(obs: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
    """Mock environment step."""
    next_obs = obs + 0.01
    reward = 1.0
    done = False
    return next_obs, reward, done


def _mock_step_terminates(obs: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
    """Mock environment step that terminates after a few steps."""
    next_obs = obs + 0.01
    reward = 1.0
    done = float(next_obs[0]) > 0.55  # terminates after ~5 steps
    return next_obs, reward, done


class TestEpisodeData:
    """Tests for EpisodeData dataclass."""

    def test_fields(self) -> None:
        ep = EpisodeData(
            observations=np.zeros((10, 18)),
            actions=np.zeros(9, dtype=np.int64),
            rewards=np.ones(9),
            dones=np.zeros(9),
            total_reward=9.0,
            length=9,
            seed=42,
        )
        assert ep.length == 9
        assert ep.total_reward == 9.0
        assert ep.seed == 42


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_num_episodes(self) -> None:
        result = BatchResult(episodes=[], total_steps=0)
        assert result.num_episodes == 0

    def test_steps_per_second(self) -> None:
        result = BatchResult(episodes=[], total_steps=1000, total_time_secs=2.0)
        assert result.steps_per_second == pytest.approx(500.0)

    def test_steps_per_second_zero_time(self) -> None:
        result = BatchResult(episodes=[], total_steps=100, total_time_secs=0.0)
        assert result.steps_per_second == 0.0


class TestBatchCollector:
    """Tests for BatchCollector."""

    def test_init_defaults(self) -> None:
        collector = BatchCollector()
        assert collector.config.max_steps == 1000
        assert collector.config.num_envs == 8

    def test_init_custom_config(self) -> None:
        config = BatchCollectorConfig(max_steps=50, num_envs=2, seed=99)
        collector = BatchCollector(config=config)
        assert collector.config.max_steps == 50

    def test_collect_with_random_policy(self) -> None:
        config = BatchCollectorConfig(max_steps=10, action_space_size=5)
        collector = BatchCollector(config=config)
        result = collector.collect(
            num_episodes=3,
            step_fn=_mock_step,
            reset_fn=_mock_reset,
        )
        assert result.num_episodes == 3
        assert result.total_steps == 30  # 3 episodes * 10 max steps

    def test_collect_with_policy_fn(self) -> None:
        config = BatchCollectorConfig(max_steps=5)
        collector = BatchCollector(config=config)

        def always_zero(obs: np.ndarray) -> int:
            return 0

        result = collector.collect(
            num_episodes=2,
            step_fn=_mock_step,
            reset_fn=_mock_reset,
            policy_fn=always_zero,
        )
        assert result.num_episodes == 2
        for ep in result.episodes:
            assert np.all(ep.actions == 0)

    def test_collect_early_termination(self) -> None:
        config = BatchCollectorConfig(max_steps=100)
        collector = BatchCollector(config=config)
        result = collector.collect(
            num_episodes=1,
            step_fn=_mock_step_terminates,
            reset_fn=_mock_reset,
        )
        assert result.num_episodes == 1
        assert result.episodes[0].length < 100

    def test_collect_episode_data_shapes(self) -> None:
        config = BatchCollectorConfig(max_steps=5)
        collector = BatchCollector(config=config)
        result = collector.collect(
            num_episodes=1,
            step_fn=_mock_step,
            reset_fn=_mock_reset,
        )
        ep = result.episodes[0]
        assert ep.observations.shape == (6, 18)  # T+1 observations
        assert ep.actions.shape == (5,)
        assert ep.rewards.shape == (5,)
        assert ep.dones.shape == (5,)

    def test_collect_mean_metrics(self) -> None:
        config = BatchCollectorConfig(max_steps=10)
        collector = BatchCollector(config=config)
        result = collector.collect(
            num_episodes=5,
            step_fn=_mock_step,
            reset_fn=_mock_reset,
        )
        assert result.mean_reward > 0
        assert result.mean_length > 0
        assert result.total_time_secs > 0

    def test_collect_deterministic_with_seed(self) -> None:
        config = BatchCollectorConfig(max_steps=5, seed=42)
        c1 = BatchCollector(config=config)
        c2 = BatchCollector(config=config)
        r1 = c1.collect(2, _mock_step, _mock_reset)
        r2 = c2.collect(2, _mock_step, _mock_reset)
        for e1, e2 in zip(r1.episodes, r2.episodes):
            assert e1.seed == e2.seed

    def test_action_space_size_config(self) -> None:
        config = BatchCollectorConfig(max_steps=3, action_space_size=10)
        collector = BatchCollector(config=config)
        result = collector.collect(
            num_episodes=5,
            step_fn=_mock_step,
            reset_fn=_mock_reset,
        )
        for ep in result.episodes:
            assert np.all(ep.actions < 10)
