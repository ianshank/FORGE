"""Tests for the PettingZoo Parallel API environment wrapper.

Uses the real compiled Rust native extension instead of mocks.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def parallel_env() -> object:
    """Create a ForgeParallelEnv with 3 agents using real native backend."""
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
    """observation_space() should return the space for a given agent."""

    def test_observation_space_per_agent(self, parallel_env: object) -> None:
        for agent_name in parallel_env.possible_agents:
            space = parallel_env.observation_space(agent_name)
            assert isinstance(space, dict)


class TestActionSpacePerAgent:
    """action_space() should return the space for a given agent."""

    def test_action_space_per_agent(self, parallel_env: object) -> None:
        for agent_name in parallel_env.possible_agents:
            space = parallel_env.action_space(agent_name)
            assert isinstance(space, dict)
            assert space.get("n", 0) > 0


class TestRenderAndClose:
    """render() and close() should work correctly."""

    def test_render_without_mode_returns_none(self, parallel_env: object) -> None:
        assert parallel_env.render_mode is None
        assert parallel_env.render() is None
