"""Tests for the PettingZoo Parallel API environment wrapper.

Uses the real compiled Rust native extension instead of mocks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def parallel_env() -> object:
    """Create a ForgeParallelEnv with 3 agents using real native backend."""
    _skip_if_no_native()
    from forge_env import pettingzoo_env
    from forge_env.pettingzoo_env import ForgeParallelEnv

    if pettingzoo_env._NativeEnv is None:
        pytest.skip("forge_env running in pure-Python mode (no native backend)")

    env = ForgeParallelEnv(n_agents=3, config=None)
    yield env
    env.close()


class TestAgentNamesGenerated:
    """Agent names should be generated as 'agent_0', 'agent_1', etc."""

    def test_agent_names_generated(self, parallel_env: object) -> None:
        assert parallel_env.possible_agents == ["agent_0", "agent_1", "agent_2"]
        assert parallel_env.agents == ["agent_0", "agent_1", "agent_2"]


class TestResetReturnsPerAgentObs:
    """reset() should return observations and infos keyed by agent name."""

    def test_reset_returns_per_agent_obs(self, parallel_env: object) -> None:
        obs, infos = parallel_env.reset(seed=42)

        assert isinstance(obs, dict)
        assert isinstance(infos, dict)

        for agent_name in parallel_env.possible_agents:
            assert agent_name in obs
            assert agent_name in infos
            assert "grid_view" in obs[agent_name]
            assert "health" in obs[agent_name]


class TestStepReturnsPerAgentResults:
    """step() should return per-agent dicts for obs, rewards, terms, truncs, infos."""

    def test_step_returns_per_agent_results(self, parallel_env: object) -> None:
        parallel_env.reset()
        actions = dict.fromkeys(parallel_env.agents, 0)
        obs, rewards, terminations, truncations, infos = parallel_env.step(actions)

        for agent_name in parallel_env.possible_agents:
            assert agent_name in obs
            assert agent_name in rewards
            assert agent_name in terminations
            assert agent_name in truncations
            assert agent_name in infos
            assert isinstance(rewards[agent_name], (int, float))
            assert isinstance(terminations[agent_name], bool)
            assert isinstance(truncations[agent_name], bool)


class TestObservationSpacePerAgent:
    """observation_space() should return a real gymnasium space for an agent.

    These used to assert the raw native descriptor `dict` was returned.
    PettingZoo's Parallel API requires `gymnasium.spaces.Space` instances —
    `parallel_api_test` samples from them and checks observation containment —
    so returning the descriptor made the environment unusable by any
    PettingZoo-based trainer.
    """

    def test_observation_space_per_agent(self, parallel_env: object) -> None:
        spaces = pytest.importorskip("gymnasium.spaces")
        for agent_name in parallel_env.possible_agents:
            space = parallel_env.observation_space(agent_name)
            assert isinstance(space, spaces.Dict)
            assert "grid_view" in space.spaces

    def test_observation_space_is_stable_by_identity(self, parallel_env: object) -> None:
        """`parallel_api_test` requires the same object, not an equal copy."""
        for agent_name in parallel_env.possible_agents:
            assert parallel_env.observation_space(agent_name) is parallel_env.observation_space(
                agent_name
            )


class TestActionSpacePerAgent:
    """action_space() should return a real gymnasium Discrete space."""

    def test_action_space_per_agent(self, parallel_env: object) -> None:
        spaces = pytest.importorskip("gymnasium.spaces")
        for agent_name in parallel_env.possible_agents:
            space = parallel_env.action_space(agent_name)
            assert isinstance(space, spaces.Discrete)
            assert space.n > 0


class TestRenderAndClose:
    """render() and close() should work correctly."""

    def test_render_without_mode_returns_none(self, parallel_env: object) -> None:
        assert parallel_env.render_mode is None
        assert parallel_env.render() is None

    def test_render_ascii_delegates_to_native(self) -> None:
        _skip_if_no_native()
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2, render_mode="ascii")
        try:
            assert isinstance(env.render(), str)
        finally:
            env.close()

    def test_native_property_exposes_the_handle(self, parallel_env: object) -> None:
        assert parallel_env.native is not None
        assert hasattr(parallel_env.native, "step_multi")


class TestAgentCountResolution:
    """`n_agents` and the simulation config must never disagree.

    The wrapper used to build N agent *names* while passing the caller's config
    through untouched, so a default construction produced two agents over a
    one-agent simulation (`DEFAULT_NUM_AGENTS` is 1 in Rust). `step_multi`
    requires exactly `config.agents.num_agents` actions, which turns that latent
    mismatch into a hard error — so the count is now resolved once and pushed
    into the config.
    """

    def test_explicit_n_agents_wins(self) -> None:
        _skip_if_no_native()
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=4)
        try:
            assert len(env.possible_agents) == 4
            observations, _infos = env.reset(seed=1)
            assert len(observations) == 4
        finally:
            env.close()

    def test_agent_count_is_read_from_the_config_when_omitted(self) -> None:
        _skip_if_no_native()
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(config={"agents": {"num_agents": 3}})
        try:
            assert env.possible_agents == ["agent_0", "agent_1", "agent_2"]
        finally:
            env.close()

    def test_defaults_to_the_historical_wrapper_count(self) -> None:
        _skip_if_no_native()
        from forge_env.pettingzoo_env import DEFAULT_AGENT_COUNT, ForgeParallelEnv

        env = ForgeParallelEnv()
        try:
            assert len(env.possible_agents) == DEFAULT_AGENT_COUNT
        finally:
            env.close()

    def test_the_callers_config_is_not_mutated(self) -> None:
        _skip_if_no_native()
        from forge_env.pettingzoo_env import ForgeParallelEnv

        config: dict = {"world": {"seed": 3}}
        env = ForgeParallelEnv(n_agents=2, config=config)
        try:
            assert "agents" not in config, "the caller's config dict was mutated"
        finally:
            env.close()

    def test_rejects_a_non_positive_agent_count(self) -> None:
        _skip_if_no_native()
        from forge_env.pettingzoo_env import ForgeParallelEnv

        with pytest.raises(ValueError, match="n_agents must be >= 1"):
            ForgeParallelEnv(n_agents=0)


class TestActionMappingEdgeCases:
    """Actions arrive keyed by live agent, which need not be every agent."""

    def test_missing_agents_act_with_the_noop(self) -> None:
        """`parallel_api_test` submits actions only for live agents."""
        _skip_if_no_native()
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=3)
        try:
            env.reset(seed=5)
            # Only one of three agents submits an action.
            observations, rewards, _term, _trunc, _infos = env.step({"agent_1": 4})
            assert set(observations) == {"agent_0", "agent_1", "agent_2"}
            assert set(rewards) == {"agent_0", "agent_1", "agent_2"}
        finally:
            env.close()

    def test_unknown_agent_names_are_ignored(self) -> None:
        _skip_if_no_native()
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2)
        try:
            env.reset(seed=5)
            observations, _rewards, _term, _trunc, _infos = env.step(
                {"agent_0": 0, "not_a_real_agent": 3}
            )
            assert set(observations) == {"agent_0", "agent_1"}
        finally:
            env.close()


class TestConstructorGuards:
    """Missing dependencies and bad arguments must fail with a clear message."""

    def test_raises_without_native_backend(self) -> None:
        from forge_env import pettingzoo_env

        with (
            patch.object(pettingzoo_env, "_NativeEnv", None),
            pytest.raises(ImportError, match="native module not found"),
        ):
            pettingzoo_env.ForgeParallelEnv()

    def test_raises_without_pettingzoo(self) -> None:
        from forge_env import pettingzoo_env

        with (
            patch.object(pettingzoo_env, "_NativeEnv", MagicMock()),
            patch.object(pettingzoo_env, "HAS_PETTINGZOO", False),
            pytest.raises(ImportError, match="pettingzoo not installed"),
        ):
            pettingzoo_env.ForgeParallelEnv()

    def test_rejects_an_unsupported_render_mode(self) -> None:
        _skip_if_no_native()
        from forge_env.pettingzoo_env import ForgeParallelEnv

        with pytest.raises(ValueError, match="Unsupported render_mode"):
            ForgeParallelEnv(n_agents=2, render_mode="human")


def _skip_if_no_native() -> None:
    """Skip when pettingzoo or the native extension is unavailable."""
    pytest.importorskip("pettingzoo")
    from forge_env import pettingzoo_env

    if pettingzoo_env._NativeEnv is None:
        pytest.skip("forge_env running in pure-Python mode (no native backend)")
