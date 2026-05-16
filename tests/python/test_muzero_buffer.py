"""Tests for MuZero replay buffer."""
from __future__ import annotations

import numpy as np
import pytest

from forge.training.muzero_buffer import (
    DEFAULT_BUFFER_CAPACITY,
    GameHistory,
    MuZeroBufferConfig,
    MuZeroReplayBuffer,
)

ACTION_DIM = 5


def _make_game(length: int = 10) -> GameHistory:
    """Create a synthetic game history."""
    history = GameHistory()
    for _i in range(length + 1):
        history.observations.append(np.random.randn(20).astype(np.float32))
    for i in range(length):
        history.actions.append(np.random.randint(0, ACTION_DIM))
        history.rewards.append(float(np.random.randn()))
        history.root_values.append(float(np.random.randn()))
        history.child_visits.append(
            np.random.rand(ACTION_DIM).astype(np.float32)
        )
        history.dones.append(i == length - 1)
    return history


# ---------------------------------------------------------------------------
# GameHistory
# ---------------------------------------------------------------------------


class TestGameHistory:
    def test_length(self) -> None:
        game = _make_game(15)
        assert game.length == 15

    def test_validate_valid(self) -> None:
        game = _make_game(10)
        assert game.validate()

    def test_validate_mismatched_obs(self) -> None:
        game = _make_game(10)
        game.observations.pop()  # Remove one observation
        assert not game.validate()

    def test_validate_mismatched_rewards(self) -> None:
        game = _make_game(10)
        game.rewards.pop()
        assert not game.validate()

    def test_validate_dones_mismatch(self) -> None:
        game = _make_game(10)
        game.dones.pop()
        assert not game.validate()

    def test_empty_game(self) -> None:
        game = GameHistory()
        assert game.length == 0


# ---------------------------------------------------------------------------
# MuZeroBufferConfig
# ---------------------------------------------------------------------------


class TestMuZeroBufferConfig:
    def test_defaults(self) -> None:
        cfg = MuZeroBufferConfig()
        assert cfg.capacity == DEFAULT_BUFFER_CAPACITY

    def test_custom(self) -> None:
        cfg = MuZeroBufferConfig(capacity=500, seed=123)
        assert cfg.capacity == 500
        assert cfg.seed == 123


# ---------------------------------------------------------------------------
# MuZeroReplayBuffer
# ---------------------------------------------------------------------------


class TestMuZeroReplayBuffer:
    def test_empty_buffer(self) -> None:
        buf = MuZeroReplayBuffer()
        assert buf.num_games == 0
        assert buf.total_steps == 0

    def test_save_game(self) -> None:
        buf = MuZeroReplayBuffer()
        game = _make_game(10)
        buf.save_game(game)
        assert buf.num_games == 1
        assert buf.total_steps == 10

    def test_capacity_eviction(self) -> None:
        buf = MuZeroReplayBuffer(MuZeroBufferConfig(capacity=3))
        for _ in range(5):
            buf.save_game(_make_game(5))
        assert buf.num_games == 3

    def test_sample_from_empty_raises(self) -> None:
        buf = MuZeroReplayBuffer()
        with pytest.raises(ValueError, match="empty"):
            buf.sample_batch(batch_size=4, num_unroll_steps=3, td_steps=5, discount=0.99)

    def test_sample_batch_shapes(self) -> None:
        buf = MuZeroReplayBuffer()
        for _ in range(5):
            buf.save_game(_make_game(20))

        batch = buf.sample_batch(
            batch_size=8, num_unroll_steps=3, td_steps=5, discount=0.99,
        )
        assert batch["observations"].shape[0] == 8
        assert batch["actions"].shape == (8, 3)
        assert batch["target_values"].shape == (8, 4)  # K+1
        assert batch["target_rewards"].shape == (8, 3)
        assert batch["target_policies"].shape == (8, 4, ACTION_DIM)  # K+1
        assert batch["weights"].shape == (8,)

    def test_sample_weights_normalized(self) -> None:
        buf = MuZeroReplayBuffer()
        for _ in range(3):
            buf.save_game(_make_game(10))

        batch = buf.sample_batch(
            batch_size=4, num_unroll_steps=2, td_steps=3, discount=0.99,
        )
        assert batch["weights"].max() <= 1.0 + 1e-6

    def test_update_priorities(self) -> None:
        buf = MuZeroReplayBuffer()
        buf.save_game(_make_game(5), priority=1.0)
        buf.save_game(_make_game(5), priority=2.0)
        buf.update_priorities([0], [10.0])
        # No crash; priorities updated internally

    def test_sample_batch_single_game(self) -> None:
        buf = MuZeroReplayBuffer()
        buf.save_game(_make_game(20))
        batch = buf.sample_batch(batch_size=4, num_unroll_steps=2, td_steps=3, discount=0.99)
        assert batch["observations"].shape[0] == 4

    def test_n_step_return(self) -> None:
        buf = MuZeroReplayBuffer()
        game = GameHistory()
        game.observations = [np.zeros(5) for _ in range(6)]
        game.actions = [0] * 5
        game.rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        game.root_values = [0.5, 0.4, 0.3, 0.2, 0.1]
        game.child_visits = [np.ones(3) for _ in range(5)]
        game.dones = [False, False, False, False, True]

        value = buf._compute_n_step_return(game, position=0, td_steps=3, discount=1.0)
        # 1.0 + 1.0 + 1.0 + bootstrap(root_values[3]) = 3.0 + 0.2 = 3.2
        assert abs(value - 3.2) < 1e-6

    def test_n_step_return_at_end(self) -> None:
        """n-step return at terminal position should return 0."""
        buf = MuZeroReplayBuffer()
        game = GameHistory()
        game.observations = [np.zeros(5) for _ in range(4)]
        game.actions = [0, 0, 0]
        game.rewards = [1.0, 1.0, 1.0]
        game.root_values = [0.5, 0.4, 0.3]
        game.child_visits = [np.ones(3) for _ in range(3)]
        game.dones = [False, False, True]

        value = buf._compute_n_step_return(game, position=3, td_steps=3, discount=0.99)
        assert value == 0.0  # Beyond game length
