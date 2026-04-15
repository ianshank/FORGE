"""MuZero replay buffer with priority-based sampling.

Stores complete game histories and samples positions with associated
K-step unroll targets for MuZero training.

Usage::

    from forge.training.muzero_buffer import MuZeroReplayBuffer, MuZeroBufferConfig

    buffer = MuZeroReplayBuffer(MuZeroBufferConfig(capacity=10000))
    buffer.save_game(game_history)
    batch = buffer.sample_batch(batch_size=256, num_unroll_steps=5)
"""

from __future__ import annotations

__all__ = ["GameHistory", "MuZeroBufferConfig", "MuZeroReplayBuffer"]

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_CAPACITY: int = 10000
DEFAULT_PRIORITY_ALPHA: float = 1.0
DEFAULT_PRIORITY_BETA: float = 1.0
DEFAULT_MIN_PRIORITY: float = 1e-6


@dataclass
class GameHistory:
    """A complete game trajectory for MuZero training.

    Stores the full observation-action-reward sequence along with
    MCTS search statistics (root values and visit count distributions)
    which serve as training targets.

    Attributes:
        observations: List of flat observation arrays, shape ``(obs_dim,)`` each.
        actions: List of discrete action indices taken.
        rewards: List of scalar rewards received after each action.
        root_values: MCTS root value estimates at each step.
        child_visits: Visit count distributions at each step, shape ``(action_dim,)`` each.
        dones: Terminal flags at each step.
    """

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    root_values: list[float] = field(default_factory=list)
    child_visits: list[np.ndarray] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of transitions in this game."""
        return len(self.actions)

    def validate(self) -> bool:
        """Check that all lists have consistent lengths.

        Returns:
            True if valid, False otherwise.
        """
        n_obs = len(self.observations)
        n_act = len(self.actions)
        # observations has one more entry than actions (initial obs)
        if n_obs != n_act + 1:
            logger.warning(
                "GameHistory: observations (%d) != actions (%d) + 1",
                n_obs,
                n_act,
            )
            return False
        if len(self.rewards) != n_act:
            logger.warning("GameHistory: rewards length mismatch")
            return False
        if len(self.dones) != n_act:
            logger.warning("GameHistory: dones length mismatch")
            return False
        return True


@dataclass
class MuZeroBufferConfig:
    """Configuration for the MuZero replay buffer.

    Attributes:
        capacity: Maximum number of game histories to store.
        priority_alpha: Exponent for priority-based sampling.
        priority_beta: Importance sampling correction exponent.
        min_priority: Minimum priority to avoid zero probabilities.
        seed: Random seed for sampling.
    """

    capacity: int = DEFAULT_BUFFER_CAPACITY
    priority_alpha: float = DEFAULT_PRIORITY_ALPHA
    priority_beta: float = DEFAULT_PRIORITY_BETA
    min_priority: float = DEFAULT_MIN_PRIORITY
    seed: int = 42


class MuZeroReplayBuffer:
    """Prioritized replay buffer for MuZero training.

    Stores game histories and samples positions within games for
    multi-step unroll training. Priorities are based on prediction
    error (TD-error) for more efficient learning.

    Args:
        config: Buffer configuration.
    """

    def __init__(self, config: MuZeroBufferConfig | None = None) -> None:
        self._config = config or MuZeroBufferConfig()
        self._games: list[GameHistory] = []
        self._priorities: list[float] = []
        self._rng = np.random.default_rng(self._config.seed)
        self._total_steps: int = 0
        logger.info(
            "MuZeroReplayBuffer: capacity=%d, alpha=%.2f",
            self._config.capacity,
            self._config.priority_alpha,
        )

    @property
    def num_games(self) -> int:
        """Number of games currently stored."""
        return len(self._games)

    @property
    def total_steps(self) -> int:
        """Total number of transitions across all stored games."""
        return self._total_steps

    def save_game(self, history: GameHistory, priority: float | None = None) -> None:
        """Add a completed game to the buffer.

        If the buffer is at capacity, the oldest game is removed.

        Args:
            history: A complete game history.
            priority: Initial sampling priority. If ``None``, uses max
                existing priority (or 1.0 for the first game).
        """
        if priority is None:
            priority = max(self._priorities) if self._priorities else 1.0

        if len(self._games) >= self._config.capacity:
            removed = self._games.pop(0)
            self._priorities.pop(0)
            self._total_steps -= removed.length

        self._games.append(history)
        self._priorities.append(max(priority, self._config.min_priority))
        self._total_steps += history.length

        logger.debug(
            "Saved game: length=%d, total_games=%d, total_steps=%d",
            history.length,
            len(self._games),
            self._total_steps,
        )

    def sample_batch(
        self,
        batch_size: int,
        num_unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> dict[str, np.ndarray]:
        """Sample a training batch from the buffer.

        Each sample is a position within a game, with K-step unroll
        targets for actions, rewards, values, and policies.

        Args:
            batch_size: Number of positions to sample.
            num_unroll_steps: Number of dynamics steps to unroll (K).
            td_steps: Number of steps for n-step return computation.
            discount: Reward discount factor.

        Returns:
            Dictionary with keys:
            - ``observations``: ``(N, obs_dim)``
            - ``actions``: ``(N, K)``
            - ``target_values``: ``(N, K+1)``
            - ``target_rewards``: ``(N, K)``
            - ``target_policies``: ``(N, K+1, action_dim)``
            - ``weights``: ``(N,)`` importance sampling weights

        Raises:
            ValueError: If the buffer is empty.
        """
        if not self._games:
            msg = "Cannot sample from empty buffer"
            raise ValueError(msg)

        # Priority-based game selection
        priorities = np.array(self._priorities, dtype=np.float64)
        priorities = priorities**self._config.priority_alpha
        game_probs = priorities / priorities.sum()

        observations = []
        actions_batch = []
        target_values = []
        target_rewards = []
        target_policies = []
        weights = []

        for _ in range(batch_size):
            # Select game
            game_idx = self._rng.choice(len(self._games), p=game_probs)
            game = self._games[game_idx]

            # Select position within game
            pos = int(self._rng.integers(0, max(1, game.length)))

            obs, game_actions, game_target_values, game_target_rewards, game_target_policies = (
                self._sample_single_position(game, pos, num_unroll_steps, td_steps, discount)
            )
            observations.append(obs)
            actions_batch.append(game_actions)
            target_values.append(game_target_values)
            target_rewards.append(game_target_rewards)
            target_policies.append(game_target_policies)

            # Importance sampling weight
            weight = (1.0 / (len(self._games) * game_probs[game_idx])) ** self._config.priority_beta
            weights.append(weight)

        # Normalize weights
        weights_arr = np.array(weights, dtype=np.float32)
        max_weight = weights_arr.max()
        if max_weight > 0:
            weights_arr /= max_weight

        return {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions_batch, dtype=np.int64),
            "target_values": np.array(target_values, dtype=np.float32),
            "target_rewards": np.array(target_rewards, dtype=np.float32),
            "target_policies": np.array(target_policies, dtype=np.float32),
            "weights": weights_arr,
        }

    def update_priorities(self, game_indices: list[int], new_priorities: list[float]) -> None:
        """Update priorities for specific games.

        Args:
            game_indices: Indices of games to update.
            new_priorities: New priority values.
        """
        for idx, priority in zip(game_indices, new_priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = max(priority, self._config.min_priority)

    def _sample_single_position(
        self,
        game: GameHistory,
        pos: int,
        num_unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> tuple[np.ndarray, list[int], list[float], list[float], list[np.ndarray]]:
        """Extract observation and unroll targets for a single position.

        Args:
            game: The game history to sample from.
            pos: Position within the game.
            num_unroll_steps: Number of dynamics steps to unroll (K).
            td_steps: Number of steps for n-step return computation.
            discount: Reward discount factor.

        Returns:
            Tuple of (observation, actions, target_values, target_rewards, target_policies).
        """
        obs = game.observations[pos]

        game_actions: list[int] = []
        game_target_values: list[float] = []
        game_target_rewards: list[float] = []
        game_target_policies: list[np.ndarray] = []

        for k in range(num_unroll_steps + 1):
            step = pos + k

            if step < game.length:
                value = self._compute_n_step_return(game, step, td_steps, discount)
                game_target_values.append(value)
            else:
                game_target_values.append(0.0)

            if k < num_unroll_steps:
                if step < game.length:
                    game_actions.append(game.actions[step])
                    game_target_rewards.append(game.rewards[step])
                else:
                    game_actions.append(0)  # Padding
                    game_target_rewards.append(0.0)

            if step < len(game.child_visits) and len(game.child_visits[step]) > 0:
                visits = game.child_visits[step]
                total = visits.sum()
                policy = visits / total if total > 0 else np.ones_like(visits) / len(visits)
                game_target_policies.append(policy)
            else:
                action_dim = len(game.child_visits[0]) if game.child_visits else 1
                game_target_policies.append(np.ones(action_dim, dtype=np.float32) / action_dim)

        return obs, game_actions, game_target_values, game_target_rewards, game_target_policies

    def _compute_n_step_return(
        self,
        game: GameHistory,
        position: int,
        td_steps: int,
        discount: float,
    ) -> float:
        """Compute the n-step bootstrapped return from a position.

        Args:
            game: The game history.
            position: Starting position.
            td_steps: Number of steps to look ahead.
            discount: Reward discount factor.

        Returns:
            The n-step return value.
        """
        value = 0.0
        for i in range(td_steps):
            step = position + i
            if step >= game.length:
                break
            value += (discount**i) * game.rewards[step]

        # Bootstrap from the value at the end of the n-step window
        bootstrap_pos = position + td_steps
        if bootstrap_pos < len(game.root_values):
            value += (discount**td_steps) * game.root_values[bootstrap_pos]

        return value
