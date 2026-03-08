"""Rollout buffer for collecting and sampling training data."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RNG_SEED = 0


class RolloutBuffer:
    """Fixed-capacity buffer for storing rollout transitions."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        seed: int = DEFAULT_RNG_SEED,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self._rng = np.random.default_rng(seed)
        self._observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self._rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._dones: np.ndarray = np.zeros(capacity, dtype=np.bool_)
        self._infos: list[dict[str, Any]] = []
        self._size: int = 0
        self._pos: int = 0
        logger.info(
            "RolloutBuffer created: capacity=%d, obs_shape=%s", capacity, obs_shape
        )

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Add a transition to the buffer."""
        self._observations[self._pos] = obs
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = done
        if self._pos < len(self._infos):
            self._infos[self._pos] = info
        else:
            self._infos.append(info)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch from the buffer."""
        if self._size == 0:
            msg = "Cannot sample from empty buffer"
            raise ValueError(msg)
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observations": self._observations[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "dones": self._dones[indices],
        }

    def clear(self) -> None:
        """Reset the buffer."""
        self._size = 0
        self._pos = 0
        self._infos.clear()

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self._size

    def is_full(self) -> bool:
        """Return True if the buffer is at capacity."""
        return self._size >= self.capacity
