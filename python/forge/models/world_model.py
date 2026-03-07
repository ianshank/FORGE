"""World model interface and stub implementations."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class WorldModel(ABC):
    """Abstract base class for world models."""

    @abstractmethod
    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """Predict the next state given current state and action."""

    @abstractmethod
    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform a single training step. Returns metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""


class IdentityWorldModel(WorldModel):
    """World model that returns the state unchanged."""

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """Return state unchanged."""
        return state.copy()

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op training step."""
        return {}

    def save(self, path: str) -> None:
        """No-op save."""
        logger.debug("IdentityWorldModel.save called (no-op)")

    def load(self, path: str) -> None:
        """No-op load."""
        logger.debug("IdentityWorldModel.load called (no-op)")
