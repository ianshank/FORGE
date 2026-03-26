"""Base agent ABC and configuration."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge.config import DEFAULT_HIDDEN_SIZES

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str = "agent"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    hidden_sizes: list[int] = field(default_factory=lambda: list(DEFAULT_HIDDEN_SIZES))


class BaseAgent(ABC):
    """Abstract base class for FORGE agents."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._step_count: int = 0
        logger.info("Initialized %s agent: %s", type(self).__name__, config.name)

    @abstractmethod
    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Select an action. Returns (action_id, trace_info)."""

    @abstractmethod
    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Update policy. Returns metrics dict."""

    def save(self, path: str) -> None:
        """Save agent state to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w") as f:
            json.dump(
                {"config": self.config.__dict__, "step_count": self._step_count}, f
            )
        logger.info("Saved agent to %s", path)

    def load(self, path: str) -> None:
        """Load agent state from disk."""
        with Path(path).open() as f:
            data = json.load(f)
        self._step_count = data.get("step_count", 0)
        logger.info("Loaded agent from %s", path)

    @property
    def step_count(self) -> int:
        """Return the current step count."""
        return self._step_count
