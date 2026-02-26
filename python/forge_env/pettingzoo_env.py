"""PettingZoo Parallel API wrapper for multi-agent FORGE environments.

Wraps the native Rust ForgeEnv to provide PettingZoo's Parallel API for
simultaneous multi-agent interactions.
"""

from typing import Any, Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None


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

    metadata = {"render_modes": ["ascii"], "name": "forge_v0"}

    def __init__(
        self,
        n_agents: int = 2,
        config: Optional[dict] = None,
        render_mode: Optional[str] = None,
    ):
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
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Resets the environment.

        Returns:
            (observations, infos) dicts keyed by agent name.
        """
        obs, info = self._env.reset(seed=seed, options=options)

        # If the native env returns per-agent data, distribute it
        observations = {}
        infos = {}
        for i, agent_name in enumerate(self.agents):
            observations[agent_name] = obs  # TODO: per-agent obs from native
            infos[agent_name] = info

        self.agents = list(self.possible_agents)
        return observations, infos

    def step(
        self, actions: dict
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Steps the environment with simultaneous actions from all agents.

        Args:
            actions: Dict mapping agent names to discrete action integers.

        Returns:
            (observations, rewards, terminations, truncations, infos) dicts.
        """
        # Convert dict of actions to list ordered by agent index
        action_list = []
        for i, agent_name in enumerate(self.possible_agents):
            action_list.append(actions.get(agent_name, 0))

        # Step with first agent's action (simplified — full multi-agent
        # requires native multi-action step support)
        obs, reward, terminated, truncated, info = self._env.step(action_list[0])

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

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

    def observation_space(self, agent: str) -> dict:
        """Returns observation space for the given agent."""
        return self._obs_space

    def action_space(self, agent: str) -> dict:
        """Returns action space for the given agent."""
        return self._act_space

    def render(self) -> Optional[str]:
        """Renders the environment."""
        if self.render_mode == "ascii":
            return self._env.render()
        return None

    def close(self):
        """Closes the environment."""
        self._env.close()
