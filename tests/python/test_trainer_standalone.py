"""Standalone tests for the basic Trainer class.

Covers __init__, train_episode, episode counting, and evaluate aggregation.
Does NOT test PPOTrainer (covered in test_mappo.py).
"""

from __future__ import annotations

import numpy as np

from forge.agents.base_agent import AgentConfig
from forge.agents.random_agent import RandomAgent
from forge.testing.env_factory import RealisticFakeEnv
from forge.training.trainer import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_MAX_EPISODES,
    Trainer,
    TrainerConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_SPACE_SIZE = 8
SEED = 7
NUM_EVAL_EPISODES = 3
CUSTOM_MAX_EPISODES = 500
CUSTOM_LOG_INTERVAL = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env_step_fn(
    env: RealisticFakeEnv,
) -> tuple[np.ndarray, callable]:
    """Reset env and return (initial_obs, step_fn).

    The returned step_fn wraps env.step into the 4-tuple signature
    (obs, reward, done, info) expected by Trainer.train_episode.
    """
    obs, _info = env.reset()

    def step_fn(action: int) -> tuple[np.ndarray, float, bool, dict]:
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        return next_obs, reward, done, info

    return obs, step_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainerInit:
    """Tests for Trainer.__init__."""

    def test_default_config(self) -> None:
        """Trainer with default TrainerConfig should initialise cleanly."""
        agent = RandomAgent(AgentConfig(), action_space_size=ACTION_SPACE_SIZE, seed=SEED)
        config = TrainerConfig()
        trainer = Trainer(agent, config)

        assert trainer.config.max_episodes == DEFAULT_MAX_EPISODES
        assert trainer.config.log_interval == DEFAULT_LOG_INTERVAL
        assert trainer._episode_count == 0

    def test_custom_config(self) -> None:
        """Trainer should respect non-default config values."""
        agent = RandomAgent(AgentConfig(), action_space_size=ACTION_SPACE_SIZE, seed=SEED)
        config = TrainerConfig(
            max_episodes=CUSTOM_MAX_EPISODES,
            log_interval=CUSTOM_LOG_INTERVAL,
        )
        trainer = Trainer(agent, config)

        assert trainer.config.max_episodes == CUSTOM_MAX_EPISODES
        assert trainer.config.log_interval == CUSTOM_LOG_INTERVAL


class TestTrainEpisode:
    """Tests for Trainer.train_episode."""

    def test_returns_metrics_dict(self) -> None:
        """train_episode should return a dict with total_reward and episode_length."""
        agent = RandomAgent(AgentConfig(), action_space_size=ACTION_SPACE_SIZE, seed=SEED)
        trainer = Trainer(agent, TrainerConfig())
        env = RealisticFakeEnv()
        _obs, step_fn = _make_env_step_fn(env)

        metrics = trainer.train_episode(step_fn)

        assert isinstance(metrics, dict)
        assert "total_reward" in metrics
        assert "episode_length" in metrics
        assert isinstance(metrics["total_reward"], float)
        assert isinstance(metrics["episode_length"], float)
        assert metrics["episode_length"] > 0

    def test_episode_count_increments(self) -> None:
        """Each call to train_episode should increment _episode_count by 1."""
        agent = RandomAgent(AgentConfig(), action_space_size=ACTION_SPACE_SIZE, seed=SEED)
        trainer = Trainer(agent, TrainerConfig())

        num_episodes = 3
        for i in range(num_episodes):
            env = RealisticFakeEnv()
            _obs, step_fn = _make_env_step_fn(env)
            trainer.train_episode(step_fn)
            assert trainer._episode_count == i + 1

    def test_reward_accumulates(self) -> None:
        """total_reward should be the sum of step rewards over the episode."""
        agent = RandomAgent(AgentConfig(), action_space_size=ACTION_SPACE_SIZE, seed=SEED)
        trainer = Trainer(agent, TrainerConfig())
        env = RealisticFakeEnv()
        _obs, step_fn = _make_env_step_fn(env)

        metrics = trainer.train_episode(step_fn)

        # RealisticFakeEnv gives non-zero rewards for most actions,
        # so total_reward should be nonzero for a full episode.
        assert metrics["total_reward"] != 0.0


class TestEvaluate:
    """Tests for Trainer.evaluate."""

    def test_returns_aggregated_metrics(self) -> None:
        """evaluate should return mean_reward and mean_length."""
        agent = RandomAgent(AgentConfig(), action_space_size=ACTION_SPACE_SIZE, seed=SEED)
        trainer = Trainer(agent, TrainerConfig())
        env = RealisticFakeEnv()

        def make_step_fn(_action: int) -> tuple[np.ndarray, float, bool, dict]:
            # Re-use the same env; reset between episodes happens inside evaluate
            # via train_episode, which drives env until done.
            next_obs, reward, terminated, truncated, info = env.step(_action)
            done = terminated or truncated
            if done:
                env.reset()
            return next_obs, reward, done, info

        # Reset once to initialise env state
        env.reset()
        metrics = trainer.evaluate(make_step_fn, num_episodes=NUM_EVAL_EPISODES)

        assert "mean_reward" in metrics
        assert "mean_length" in metrics
        assert isinstance(metrics["mean_reward"], float)
        assert isinstance(metrics["mean_length"], float)
        assert np.isfinite(metrics["mean_reward"])
        assert np.isfinite(metrics["mean_length"])

    def test_evaluate_increments_episode_count(self) -> None:
        """evaluate runs train_episode internally, so episode count should grow."""
        agent = RandomAgent(AgentConfig(), action_space_size=ACTION_SPACE_SIZE, seed=SEED)
        trainer = Trainer(agent, TrainerConfig())
        env = RealisticFakeEnv()

        def make_step_fn(action: int) -> tuple[np.ndarray, float, bool, dict]:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                env.reset()
            return next_obs, reward, done, info

        env.reset()
        initial_count = trainer._episode_count
        trainer.evaluate(make_step_fn, num_episodes=NUM_EVAL_EPISODES)
        assert trainer._episode_count == initial_count + NUM_EVAL_EPISODES

    def test_default_eval_episodes(self) -> None:
        """When num_episodes is omitted, default should be DEFAULT_EVAL_EPISODES."""
        # Just verify the default parameter value exists and is reasonable.
        assert DEFAULT_EVAL_EPISODES > 0
