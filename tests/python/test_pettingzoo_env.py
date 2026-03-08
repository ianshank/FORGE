"""Tests for the PettingZoo Parallel API environment wrapper.

The native forge_env module is mocked so tests run without Rust extension.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_MOCK_OBS = {
    "grid_view": [[[0] * 7] * 11] * 11,
    "inventory": [[0, 0]] * 10,
    "health": 0.8,
    "stamina": 0.9,
    "position": [5, 5],
    "messages": [],
    "day_phase": 0,
}

_NATIVE_OBS_SPACE = {
    "grid_view_height": 11,
    "grid_view_width": 11,
    "grid_view_channels": 7,
    "inventory_capacity": 10,
}

_NATIVE_ACT_SPACE = {"n": 8}


def _make_mock_native_env() -> MagicMock:
    """Create a fresh mock native ForgeEnv instance."""
    env = MagicMock()
    env.reset.return_value = (_MOCK_OBS, {"tick": 0})
    env.step.return_value = (_MOCK_OBS, 1.0, False, False, {"tick": 1})
    env.render.return_value = "ascii_frame"
    env.observation_space = _NATIVE_OBS_SPACE
    env.action_space = _NATIVE_ACT_SPACE
    return env


@pytest.fixture()
def parallel_env() -> object:
    """Create a ForgeParallelEnv with 3 agents using mocked native backend."""
    mock_native = _make_mock_native_env()
    mock_native_cls = MagicMock(return_value=mock_native)

    with patch("forge_env.pettingzoo_env._NativeEnv", mock_native_cls):
        from forge_env.pettingzoo_env import ForgeParallelEnv  # noqa: PLC0415

        env = ForgeParallelEnv(n_agents=3, config=None)
    return env


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
            assert isinstance(rewards[agent_name], float)
            assert isinstance(terminations[agent_name], bool)
            assert isinstance(truncations[agent_name], bool)


class TestObservationSpacePerAgent:
    """observation_space() should return the space for a given agent."""

    def test_observation_space_per_agent(self, parallel_env: object) -> None:
        for agent_name in parallel_env.possible_agents:
            space = parallel_env.observation_space(agent_name)
            assert space is not None
            assert isinstance(space, dict)


class TestActionSpacePerAgent:
    """action_space() should return the space for a given agent."""

    def test_action_space_per_agent(self, parallel_env: object) -> None:
        for agent_name in parallel_env.possible_agents:
            space = parallel_env.action_space(agent_name)
            assert space is not None
            assert isinstance(space, dict)
            assert space.get("n") == 8


class TestRenderAndClose:
    """render() and close() should work correctly."""

    def test_render_without_mode_returns_none(self, parallel_env: object) -> None:
        assert parallel_env.render_mode is None
        assert parallel_env.render() is None

    def test_close_calls_native(self, parallel_env: object) -> None:
        parallel_env.close()
        parallel_env._env.close.assert_called_once()
