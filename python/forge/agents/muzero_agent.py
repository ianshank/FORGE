"""MuZero agent: uses learned world model with MCTS for action selection.

Wraps :class:`~forge.agents.muzero_mcts.MuZeroMCTS` as a
:class:`~forge.agents.base_agent.BaseAgent` for integration with the
FORGE evaluation and training pipelines.

Usage::

    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.agents.muzero_agent import MuZeroAgent
    from forge.agents.base_agent import AgentConfig

    model = MuZeroWorldModel(MuZeroConfig(obs_dim=920, action_dim=75))
    agent = MuZeroAgent(AgentConfig(name="muzero"), model)
    action, info = agent.act(observation)
"""
from __future__ import annotations

__all__ = ["MuZeroAgent"]

import logging
from typing import TYPE_CHECKING, Any

from forge.agents.base_agent import AgentConfig, BaseAgent
from forge.agents.muzero_mcts import MuZeroMCTS, MuZeroMCTSConfig

if TYPE_CHECKING:
    import numpy as np

    from forge.models.muzero_world_model import MuZeroWorldModel

logger = logging.getLogger(__name__)


class MuZeroAgent(BaseAgent):
    """Agent that uses MuZero MCTS for action selection.

    Combines a :class:`MuZeroWorldModel` with :class:`MuZeroMCTS`
    to select actions via latent-space tree search.

    Args:
        config: Agent configuration.
        model: Trained MuZero world model.
        mcts_config: MCTS search configuration. Uses defaults if ``None``.
    """

    def __init__(
        self,
        config: AgentConfig,
        model: MuZeroWorldModel,
        mcts_config: MuZeroMCTSConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._model = model
        self._mcts = MuZeroMCTS(model, mcts_config)
        self._temperature: float = self._mcts.config.temperature
        logger.info(
            "MuZeroAgent initialized: name=%s, simulations=%d",
            config.name,
            self._mcts.config.num_simulations,
        )

    @property
    def temperature(self) -> float:
        """Current action selection temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the action selection temperature."""
        self._temperature = value

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Select an action using MuZero MCTS.

        Args:
            observation: Flat observation array of shape ``(obs_dim,)``.

        Returns:
            Tuple of (action_id, info_dict) with MCTS search statistics.
        """
        action_id, info = self._mcts.search(observation, temperature=self._temperature)
        self._step_count += 1

        logger.debug(
            "MuZeroAgent step %d: action=%d, root_value=%.3f",
            self._step_count,
            action_id,
            info.get("root_value", 0.0),
        )
        return action_id, info

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Delegate training to the world model.

        Args:
            batch: Training batch (see :meth:`MuZeroWorldModel.train_step`).

        Returns:
            Training metrics dictionary.
        """
        return self._model.train_step(batch)
