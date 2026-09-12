"""PettingZoo Parallel API wrapper for multi-agent FORGE environments.

Wraps the native Rust ForgeEnv and forwards one action / observation per
agent via ``reset_all`` / ``step_multi``.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from forge_env.gymnasium_env import build_forge_spaces, coerce_observation

logger = logging.getLogger(__name__)

try:
    from pettingzoo.utils.env import ParallelEnv as _ParallelEnvBase
except ImportError:
    _ParallelEnvBase = object  # type: ignore[assignment,misc]

try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None


__all__ = ["ForgeParallelEnv"]


class ForgeParallelEnv(_ParallelEnvBase):
    """PettingZoo Parallel API wrapper for FORGE multi-agent environments.

    All agents act simultaneously each step. Observations, rewards,
    terminations, truncations, and infos are dicts keyed by agent name.

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

        full_config = config.copy() if config else {}
        if "agents" not in full_config:
            full_config["agents"] = {}
        full_config["agents"]["num_agents"] = n_agents

        self._env = _NativeEnv(config=full_config)
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)
        self._gym_obs_space, self._gym_act_space = build_forge_spaces(
            self._env.observation_space,
            self._env.action_space,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment.

        Returns:
            (observations, infos) dicts keyed by agent name.
        """
        obs_list, info = self._env.reset_all(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        observations = {
            name: coerce_observation(obs_list[i], self._gym_obs_space)
            for i, name in enumerate(self.possible_agents)
        }
        infos = dict.fromkeys(self.possible_agents, info)
        return observations, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """Step with simultaneous actions from living agents.

        Args:
            actions: Dict mapping agent names to discrete action integers.

        Returns:
            (observations, rewards, terminations, truncations, infos) dicts.
        """
        acting = list(self.agents)
        ordered = [int(actions.get(name, 0)) for name in self.possible_agents]
        obs_list, rewards_list, terminated, truncated, info = self._env.step_multi(ordered)
        alive = list(info.get("agents_alive", [True] * self._n_agents))

        observations: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, Any] = {}

        for i, name in enumerate(self.possible_agents):
            if name not in acting:
                continue
            observations[name] = coerce_observation(obs_list[i], self._gym_obs_space)
            rewards[name] = float(rewards_list[i])
            terminations[name] = bool(terminated or not alive[i])
            truncations[name] = bool(truncated)
            infos[name] = info

        terminations["__all__"] = bool(acting) and all(terminations[a] for a in acting)
        truncations["__all__"] = bool(acting) and all(truncations[a] for a in acting)

        self.agents = [
            a for a in acting if not terminations.get(a, False) and not truncations.get(a, False)
        ]
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> Any:
        """Return the Gymnasium observation space for ``agent``."""
        del agent
        return self._gym_obs_space

    def action_space(self, agent: str) -> Any:
        """Return the Gymnasium action space for ``agent``."""
        del agent
        return self._gym_act_space

    def render(self) -> str | None:
        """Render the environment."""
        if self.render_mode == "ascii":
            return self._env.render()  # type: ignore[no-any-return]
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
