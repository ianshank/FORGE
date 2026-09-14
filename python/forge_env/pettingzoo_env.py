"""PettingZoo Parallel API wrapper for multi-agent FORGE environments.

Wraps the native Rust ``ForgeEnv`` as a real :class:`pettingzoo.ParallelEnv`, so
the environment passes ``pettingzoo.test.parallel_api_test`` rather than merely
resembling the API. Three things this wrapper gets right that a resemblance does
not:

* **Every agent's action is applied.** Actions go through the native
  ``step_multi``, which takes one discrete action per agent and drives the same
  multi-agent ``WorldState::step`` the Rust benchmarks exercise. The previous
  implementation forwarded only ``agent_0``'s action to the single-agent
  ``step`` and broadcast the result, so the other agents' actions were silently
  discarded.
* **Observations are per agent.** ``reset_all`` / ``step_multi`` return one
  observation per agent instead of one shared observation copied N times.
* **Spaces are real gymnasium spaces, returned by identity.** ``parallel_api_test``
  asserts ``observation_space(agent) is observation_space(agent)``, because
  space seeding breaks if a fresh copy is handed out each call.

Termination is a property of the whole simulation rather than of an individual
agent — ``WorldState::step`` returns one ``terminated`` / ``truncated`` pair —
so all live agents terminate on the same tick. That is reported honestly: every
live agent receives the same flag and the agent list empties together.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from forge_env.space_builder import build_action_space, build_observation_space, fit_observation

logger = logging.getLogger(__name__)

try:
    from pettingzoo import ParallelEnv as _ParallelEnvBase

    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False

try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Always the real base for the type checker; at runtime it degrades to
    # ``object`` so importing ``forge_env`` never requires pettingzoo.
    from collections.abc import Mapping, Sequence

    from gymnasium import spaces as spaces_t
    from pettingzoo import ParallelEnv as _EnvBase
else:  # pragma: no cover - trivial branch
    _EnvBase = _ParallelEnvBase if HAS_PETTINGZOO else object

__all__ = ["ForgeParallelEnv"]

#: Agent-name template. ``parallel_api_test`` only requires stable, hashable
#: ids; this keeps the historical ``agent_0``, ``agent_1``, ... naming.
AGENT_NAME_TEMPLATE: str = "agent_{index}"

#: Fallback agent count when neither ``n_agents`` nor the config supplies one.
#: Two, matching this wrapper's historical default — note the Rust
#: ``DEFAULT_NUM_AGENTS`` is 1, and the mismatch is exactly the bug that used to
#: produce two agent *names* over a one-agent simulation.
DEFAULT_AGENT_COUNT: int = 2

#: Action index applied on behalf of an agent that submitted none. Index 0 is
#: ``Action::Noop`` for every action-space configuration
#: (``crates/forge-python/src/env.rs::test_action_noop_is_zero``).
NOOP_ACTION_INDEX: int = 0


class ForgeParallelEnv(_EnvBase):
    """PettingZoo Parallel API environment backed by the native FORGE engine.

    Args:
        n_agents: Number of agents. When ``None``, taken from
            ``config["agents"]["num_agents"]`` if present, else
            :data:`DEFAULT_AGENT_COUNT`. When given, it is authoritative and is
            pushed into the simulation config, so the agent list and the
            simulation can no longer disagree.
        config: Optional configuration dict matching ``ForgeConfig``. Never
            mutated — a copy carries the resolved agent count.
        render_mode: Optional render mode. Must be ``None`` or a member of
            ``metadata["render_modes"]``.

    Raises:
        ImportError: If the native extension or pettingzoo is unavailable.
        ValueError: If ``n_agents`` is not positive, or ``render_mode`` is not a
            supported mode.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["ascii"], "name": "forge_v0"}

    def __init__(
        self,
        n_agents: int | None = None,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        if _NativeEnv is None:
            raise ImportError(
                "forge_env native module not found. "
                "Install with: pip install -e . (requires maturin)"
            )
        if not HAS_PETTINGZOO:
            raise ImportError("pettingzoo not installed. Install with: pip install pettingzoo")

        supported_modes = self.metadata["render_modes"]
        if render_mode is not None and render_mode not in supported_modes:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}. "
                f"Supported modes: {supported_modes} (or None)."
            )

        resolved_config = copy.deepcopy(config) if config else {}
        agent_count = self._resolve_agent_count(n_agents, resolved_config)
        if agent_count < 1:
            raise ValueError(f"n_agents must be >= 1, got {agent_count}.")

        # The simulation must agree with the agent list: `step_multi` requires
        # exactly `config.agents.num_agents` actions.
        resolved_config.setdefault("agents", {})["num_agents"] = agent_count

        self._env = _NativeEnv(config=resolved_config)
        self._config = resolved_config
        self.render_mode = render_mode

        self.possible_agents: list[str] = [
            AGENT_NAME_TEMPLATE.format(index=index) for index in range(agent_count)
        ]
        self.agents: list[str] = list(self.possible_agents)
        self._agent_index: dict[str, int] = {
            agent: index for index, agent in enumerate(self.possible_agents)
        }

        # One space instance shared by every agent, held so that
        # `observation_space(agent) is observation_space(agent)` -- an identity
        # `parallel_api_test` asserts, because per-agent space seeding relies on
        # it. The `*_spaces` dicts are the base class's declared attributes.
        observation_space = build_observation_space(self._env.observation_space)
        action_space = build_action_space(self._env.action_space)
        self.observation_spaces: dict[str, spaces_t.Dict] = dict.fromkeys(
            self.possible_agents, observation_space
        )
        self.action_spaces: dict[str, spaces_t.Discrete] = dict.fromkeys(
            self.possible_agents, action_space
        )

    @staticmethod
    def _resolve_agent_count(n_agents: int | None, config: Mapping[str, Any]) -> int:
        """Resolve the agent count from the explicit argument, then the config."""
        if n_agents is not None:
            return int(n_agents)
        configured = config.get("agents", {}).get("num_agents")
        if configured is not None:
            return int(configured)
        return DEFAULT_AGENT_COUNT

    def _actions_for_all_agents(self, actions: Mapping[str, int]) -> list[int]:
        """Expand a per-live-agent action mapping to one action per agent slot.

        ``parallel_api_test`` submits actions only for agents that are still
        live, while ``step_multi`` requires one action per configured agent, so
        the gaps are filled with a no-op.
        """
        full = [NOOP_ACTION_INDEX] * len(self.possible_agents)
        for agent, action in actions.items():
            index = self._agent_index.get(agent)
            if index is None:
                logger.debug("Ignoring action for unknown agent %r.", agent)
                continue
            full[index] = int(action)
        return full

    def _fit_all(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Fit each agent's raw observation to the declared observation space."""
        return {
            agent: fit_observation(observations[self._agent_index[agent]], self.observation_spaces[agent])
            for agent in self.agents
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Reset the environment and revive every agent.

        Args:
            seed: Optional seed for a deterministic reset.
            options: Passed through to the native env.

        Returns:
            An ``(observations, infos)`` pair, both keyed by agent name.
        """
        self.agents = list(self.possible_agents)
        observations, info = self._env.reset_all(seed=seed, options=options)
        infos = {agent: dict(info) for agent in self.agents}
        return self._fit_all(observations), infos

    def step(
        self,
        actions: Mapping[str, int],
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Step every live agent simultaneously.

        Args:
            actions: Discrete action per live agent. Agents absent from the
                mapping act with :data:`NOOP_ACTION_INDEX`.

        Returns:
            ``(observations, rewards, terminations, truncations, infos)``, each
            keyed by the agents that were live for this step. When the episode
            ends, every live agent is flagged and the agent list empties.
        """
        observations, rewards, terminated, truncated, info = self._env.step_multi(
            self._actions_for_all_agents(actions)
        )

        stepped_agents = list(self.agents)
        fitted = self._fit_all(observations)
        reward_map = {
            agent: float(rewards[self._agent_index[agent]]) for agent in stepped_agents
        }
        terminations = dict.fromkeys(stepped_agents, bool(terminated))
        truncations = dict.fromkeys(stepped_agents, bool(truncated))
        infos = {agent: dict(info) for agent in stepped_agents}

        # Termination is simulation-wide, so the whole roster retires together.
        if terminated or truncated:
            logger.debug(
                "Episode ended (terminated=%s, truncated=%s); retiring %d agent(s).",
                terminated,
                truncated,
                len(stepped_agents),
            )
            self.agents = []

        return fitted, reward_map, terminations, truncations, infos

    def observation_space(self, agent: str) -> spaces_t.Dict:
        """Return the observation space for ``agent`` (same object every call)."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces_t.Discrete:
        """Return the action space for ``agent`` (same object every call)."""
        return self.action_spaces[agent]

    def render(self) -> str | None:
        """Render the environment, or return ``None`` when no mode is set."""
        if self.render_mode == "ascii":
            return self._env.render()  # type: ignore[no-any-return]
        return None

    def close(self) -> None:
        """Close the environment and release the native handle's resources."""
        self._env.close()

    @property
    def native(self) -> Any:
        """The wrapped native ``ForgeEnv`` handle."""
        return self._env
