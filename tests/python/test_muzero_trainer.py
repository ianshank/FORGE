"""Tests for MuZero trainer: self-play and training loop."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.agents.muzero_mcts import MuZeroMCTSConfig  # noqa: E402
from forge.models.muzero_config import MuZeroConfig  # noqa: E402
from forge.models.muzero_world_model import MuZeroWorldModel  # noqa: E402
from forge.training.muzero_buffer import GameHistory, MuZeroBufferConfig  # noqa: E402
from forge.training.muzero_trainer import MuZeroTrainer, MuZeroTrainerConfig  # noqa: E402

OBS_DIM = 11 * 11 * 7 + 73
ACTION_DIM = 5
LATENT_DIM = 16
HIDDEN_DIM = 16


def _make_model() -> MuZeroWorldModel:
    return MuZeroWorldModel(MuZeroConfig(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=1,
        num_unroll_steps=2,
        reward_support_size=11,
        value_support_size=11,
        cnn_channels=(8,),
        cnn_kernel_sizes=(3,),
        cnn_strides=(1,),
    ))


def _make_env() -> MagicMock:
    """Create a mock Gymnasium environment."""
    env = MagicMock()
    env.reset.return_value = (np.random.randn(OBS_DIM).astype(np.float32), {})

    step_count = {"n": 0}

    def step_side_effect(action: int):
        step_count["n"] += 1
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        reward = float(np.random.randn())
        terminated = step_count["n"] >= 5  # End after 5 steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    env.step.side_effect = step_side_effect
    env.close.return_value = None
    return env


def _make_trainer() -> MuZeroTrainer:
    model = _make_model()
    config = MuZeroTrainerConfig(
        training_steps_per_iter=2,
        self_play_games_per_iter=1,
        batch_size=4,
        max_episode_steps=10,
        buffer_config=MuZeroBufferConfig(capacity=100),
        mcts_config=MuZeroMCTSConfig(num_simulations=3, add_exploration_noise=False),
    )
    return MuZeroTrainer(config, model)


# ---------------------------------------------------------------------------
# MuZeroTrainerConfig
# ---------------------------------------------------------------------------


class TestMuZeroTrainerConfig:
    def test_defaults(self) -> None:
        cfg = MuZeroTrainerConfig()
        assert cfg.batch_size == 256
        assert cfg.training_steps_per_iter == 100

    def test_custom(self) -> None:
        cfg = MuZeroTrainerConfig(batch_size=64, self_play_games_per_iter=5)
        assert cfg.batch_size == 64
        assert cfg.self_play_games_per_iter == 5


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------


class TestSelfPlay:
    def test_generates_valid_history(self) -> None:
        trainer = _make_trainer()
        env = _make_env()
        history = trainer.self_play(env)

        assert isinstance(history, GameHistory)
        assert history.length > 0
        assert len(history.observations) == history.length + 1
        assert len(history.actions) == history.length
        assert len(history.rewards) == history.length

    def test_increments_game_count(self) -> None:
        trainer = _make_trainer()
        assert trainer.total_games == 0
        trainer.self_play(_make_env())
        assert trainer.total_games == 1
        trainer.self_play(_make_env())
        assert trainer.total_games == 2

    def test_self_play_env_close_called(self) -> None:
        trainer = _make_trainer()
        env = _make_env()
        trainer.self_play(env)
        # env.close is NOT called in self_play (only in train loop)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTraining:
    def test_train_step_raises_empty_buffer(self) -> None:
        trainer = _make_trainer()
        with pytest.raises(RuntimeError, match="empty"):
            trainer.train_step()

    def test_train_step_after_self_play(self) -> None:
        trainer = _make_trainer()
        env = _make_env()
        history = trainer.self_play(env)
        trainer._buffer.save_game(history)

        metrics = trainer.train_step()
        assert "loss" in metrics
        assert np.isfinite(metrics["loss"])
        assert trainer.total_train_steps == 1

    def test_temperature_schedule(self) -> None:
        trainer = _make_trainer()
        # At 0 games, temperature should be init
        temp = trainer.current_temperature()
        assert abs(temp - trainer._config.temperature_init) < 1e-6

    def test_current_temperature_zero_schedule(self) -> None:
        model = _make_model()
        config = MuZeroTrainerConfig(
            temperature_schedule_steps=0,
            temperature_final=0.1,
            training_steps_per_iter=1,
            self_play_games_per_iter=1,
            batch_size=2,
            mcts_config=MuZeroMCTSConfig(num_simulations=2, add_exploration_noise=False),
            buffer_config=MuZeroBufferConfig(capacity=10),
        )
        trainer = MuZeroTrainer(config, model)
        assert trainer.current_temperature() == 0.1  # Should be final when schedule=0


# ---------------------------------------------------------------------------
# Full train loop (smoke test)
# ---------------------------------------------------------------------------


class TestTrainLoop:
    def test_single_iteration(self) -> None:
        trainer = _make_trainer()

        def env_factory():
            return _make_env()

        history = trainer.train(env_factory, num_iterations=1)
        assert len(history["loss"]) > 0
        assert trainer.total_games >= 1
        assert trainer.total_train_steps >= 1
