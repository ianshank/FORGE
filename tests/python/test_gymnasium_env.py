"""Tests for the Gymnasium environment wrapper.

The native forge_core / forge_env module is mocked so that tests can run
without the compiled Rust extension.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from conftest import NATIVE_ACT_SPACE, make_mock_native_env


@pytest.fixture()
def env():
    """Create a ForgeGymnasiumEnv with mocked native backend."""
    gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
    mock_native = make_mock_native_env(include_messages_in_obs_space=True)
    mock_native_cls = MagicMock(return_value=mock_native)

    with patch("forge_env.gymnasium_env._NativeEnv", mock_native_cls):
        from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

        wrapper = ForgeGymnasiumEnv(config={"seed": 42})
    return wrapper


class TestResetReturnsObsAndInfo:
    """reset() should return an (observation, info) tuple."""

    def test_reset_returns_obs_and_info(self, env: object) -> None:
        obs, info = env.reset(seed=123)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert "grid_view" in obs
        assert "health" in obs
        assert "tick" in info


class TestStepReturnsFiveTuple:
    """step() should return (obs, reward, terminated, truncated, info)."""

    def test_step_returns_five_tuple(self, env: object) -> None:
        env.reset()
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestObservationSpaceStructure:
    """observation_space should be a gymnasium Dict with expected keys."""

    def test_observation_space_structure(self, env: object) -> None:
        obs_space = env.observation_space
        expected_keys = {
            "grid_view",
            "inventory",
            "health",
            "stamina",
            "position",
            "messages",
            "day_phase",
        }
        assert set(obs_space.spaces.keys()) == expected_keys


class TestActionSpaceSize:
    """action_space should be a Discrete space matching native config."""

    def test_action_space_size(self, env: object) -> None:
        assert env.action_space.n == NATIVE_ACT_SPACE["n"]


class TestRenderReturnsNoneWithoutRenderMode:
    """render() should return None when render_mode is not set."""

    def test_render_returns_none_without_render_mode(self, env: object) -> None:
        assert env.render_mode is None
        result = env.render()
        assert result is None
