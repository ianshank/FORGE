"""PettingZoo Parallel API wrapper for multi-agent FORGE environments.

Wraps the native Rust ForgeEnv to provide PettingZoo's Parallel API for
simultaneous multi-agent interactions.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None

__all__ = ["ForgeParallelEnv"]


class ForgeParallelEnv:
    """PettingZoo Parallel API wrapper for FORGE multi-agent environments.

    All agents act simultaneously each step. The environment returns
    observations, rewards, terminations, truncations, and infos as
    dicts keyed by agent name.

    Args:
        n_agents: Number of agents in the environment.
        config: Optional configuration dict matching ForgeConfig structure.
        render_mode: Optional render mode ('ascii' or None).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["ascii"], "name": "forge_v0"}

    def __init__(
        self,
        n_agents: int = 2,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        if _NativeEnv is None:
            raise ImportError(
                "forge_env native module not found. "
                "Install with: pip install -e . (requires maturin)"
            )

        self.render_mode = render_mode
        self._n_agents = n_agents

        # Merge agent count into config
        full_config = config.copy() if config else {}
        if "agents" not in full_config:
            full_config["agents"] = {}
        full_config["agents"]["num_agents"] = n_agents

        self._env = _NativeEnv(config=full_config)

        # Agent names
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)

        # Spaces (same for all agents in homogeneous mode)
        self._obs_space = self._env.observation_space
        self._act_space = self._env.action_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment.

        Returns:
            (observations, infos) dicts keyed by agent name.
        """
        obs, info = self._env.reset(seed=seed, options=options)

        # If the native env returns per-agent data, distribute it
        observations: dict[str, Any] = {}
        infos: dict[str, Any] = {}
        for _i, agent_name in enumerate(self.agents):
            observations[agent_name] = obs  # TODO: per-agent obs from native
            infos[agent_name] = info

        self.agents = list(self.possible_agents)
        return observations, infos

    def step(
        self, actions: dict[str, int],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """Step the environment with simultaneous actions from all agents.

        Args:
            actions: Dict mapping agent names to discrete action integers.

        Returns:
            (observations, rewards, terminations, truncations, infos) dicts.
        """
        # Convert dict of actions to list ordered by agent index
        action_list = [actions.get(agent_name, 0) for agent_name in self.possible_agents]

        # Step with first agent's action (simplified -- full multi-agent
        # requires native multi-action step support)
        obs, reward, terminated, truncated, info = self._env.step(action_list[0])

        observations: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, Any] = {}

        for agent_name in self.agents:
            observations[agent_name] = obs
            rewards[agent_name] = reward
            terminations[agent_name] = terminated
            truncations[agent_name] = truncated
            infos[agent_name] = info

        # Remove terminated/truncated agents
        self.agents = [
            a
            for a in self.agents
            if not terminations.get(a, False) and not truncations.get(a, False)
        ]

        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> dict[str, Any]:
        """Return observation space for the given agent."""
        return self._obs_space  # type: ignore[no-any-return]

    def action_space(self, agent: str) -> dict[str, Any]:
        """Return action space for the given agent."""
        return self._act_space  # type: ignore[no-any-return]

    def render(self) -> str | None:
        """Render the environment."""
        if self.render_mode == "ascii":
            return self._env.render()  # type: ignore[no-any-return]
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
