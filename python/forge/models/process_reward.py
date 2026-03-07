"""Process reward model interface and stub implementations."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONSTANT_SCORE = 1.0


class ProcessRewardModel(ABC):
    """Abstract base class for process reward models."""

    @abstractmethod
    def score_trace(self, trace_sequence: list[dict[str, object]]) -> float:
        """Score a sequence of decision traces."""

    @abstractmethod
    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform a single training step. Returns metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""


class ConstantRewardModel(ProcessRewardModel):
    """Process reward model that returns a constant score."""

    def __init__(self, score: float = DEFAULT_CONSTANT_SCORE) -> None:
        self.score = score
        logger.info("ConstantRewardModel initialized with score=%.2f", score)

    def score_trace(self, trace_sequence: list[dict[str, object]]) -> float:
        """Return a constant score regardless of trace content."""
        return self.score

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op training step."""
        return {}

    def save(self, path: str) -> None:
        """No-op save."""
        logger.debug("ConstantRewardModel.save called (no-op)")

    def load(self, path: str) -> None:
        """No-op load."""
        logger.debug("ConstantRewardModel.load called (no-op)")
