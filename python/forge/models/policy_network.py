"""Policy network interface and stub implementations."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class PolicyNetwork(ABC):
    """Abstract base class for policy networks."""

    @abstractmethod
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Compute action probabilities from an observation."""

    @abstractmethod
    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform a single training step. Returns metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""


class RandomPolicyNetwork(PolicyNetwork):
    """Policy network that returns uniform random action probabilities."""

    def __init__(self, action_size: int = 8) -> None:
        self.action_size = action_size
        logger.info("RandomPolicyNetwork initialized with action_size=%d", action_size)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Return uniform action probabilities."""
        return np.ones(self.action_size, dtype=np.float32) / self.action_size

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op training step."""
        return {}

    def save(self, path: str) -> None:
        """No-op save."""
        logger.debug("RandomPolicyNetwork.save called (no-op)")

    def load(self, path: str) -> None:
        """No-op load."""
        logger.debug("RandomPolicyNetwork.load called (no-op)")
