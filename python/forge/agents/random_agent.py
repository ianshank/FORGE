"""Random agent implementation for baseline comparisons."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from forge.agents.base_agent import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)

DEFAULT_ACTION_SPACE_SIZE = 8
DEFAULT_SEED = 42


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random."""

    def __init__(
        self,
        config: AgentConfig,
        action_space_size: int = DEFAULT_ACTION_SPACE_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        super().__init__(config)
        self.action_space_size = action_space_size
        self._rng = np.random.default_rng(seed)
        logger.info(
            "RandomAgent initialized with action_space_size=%d, seed=%d",
            action_space_size,
            seed,
        )

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Select a random action from the action space."""
        action = int(self._rng.integers(0, self.action_space_size))
        self._step_count += 1
        return action, {"random": True}

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op learning for random agent. Returns empty metrics."""
        return {}
