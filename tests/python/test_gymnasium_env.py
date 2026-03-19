"""Tests for the Gymnasium environment wrapper.

Uses the real compiled Rust native extension instead of mocks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from forge_env.gymnasium_env import ForgeGymnasiumEnv

# Skip entire module when native extension is not built
_native_available = True
try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv  # noqa: F401
except ImportError:
    _native_available = False

pytestmark = pytest.mark.skipif(
    not _native_available,
    reason="forge_env native extension not available (run maturin develop first)",
)


@pytest.fixture()
def env() -> Generator[ForgeGymnasiumEnv]:
    """Create a ForgeGymnasiumEnv with the real native backend."""
    gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    wrapper = ForgeGymnasiumEnv(config={"seed": 42})
    yield wrapper
    wrapper.close()


class TestResetReturnsObsAndInfo:
    """reset() should return an (observation, info) tuple."""

    def test_reset_returns_obs_and_info(self, env: ForgeGymnasiumEnv) -> None:
        obs, info = env.reset(seed=123)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert "grid_view" in obs
        assert "health" in obs


class TestStepReturnsFiveTuple:
    """step() should return (obs, reward, terminated, truncated, info)."""

    def test_step_returns_five_tuple(self, env: ForgeGymnasiumEnv) -> None:
        env.reset()
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestObservationSpaceStructure:
    """observation_space should be a gymnasium Dict with expected keys."""

    def test_observation_space_structure(self, env: ForgeGymnasiumEnv) -> None:
        obs_space = env.observation_space
        expected_keys = {
            "grid_view", "inventory", "health", "stamina",
            "position", "messages", "day_phase",
        }
        assert set(obs_space.spaces.keys()) == expected_keys


class TestActionSpaceSize:
    """action_space should be a Discrete space."""

    def test_action_space_is_positive(self, env: ForgeGymnasiumEnv) -> None:
        assert env.action_space.n > 0


class TestRenderReturnsNoneWithoutRenderMode:
    """render() should return None when render_mode is not set."""

    def test_render_returns_none_without_render_mode(self, env: ForgeGymnasiumEnv) -> None:
        assert env.render_mode is None
        result = env.render()
        assert result is None
