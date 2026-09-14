"""Tests for the Gymnasium environment wrapper.

Uses the real compiled Rust native extension instead of mocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from forge_env.gymnasium_env import ForgeGymnasiumEnv


@pytest.fixture()
def env() -> Generator[ForgeGymnasiumEnv]:
    """Create a ForgeGymnasiumEnv with the real native backend."""
    gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
    from forge_env import gymnasium_env
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    if gymnasium_env._NativeEnv is None:
        pytest.skip("forge_env running in pure-Python mode (no native backend)")

    wrapper = ForgeGymnasiumEnv(config={"world": {"seed": 42}})
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
    """action_space should be a Discrete space."""

    def test_action_space_is_positive(self, env: ForgeGymnasiumEnv) -> None:
        assert env.action_space.n > 0


class TestRenderReturnsNoneWithoutRenderMode:
    """render() should return None when render_mode is not set."""

    def test_render_returns_none_without_render_mode(self, env: ForgeGymnasiumEnv) -> None:
        assert env.render_mode is None
        result = env.render()
        assert result is None


class TestRenderModeValidation:
    """An unsupported render mode must fail loudly at construction.

    Gymnasium's checker requires `render_mode` to be None or a member of
    `metadata["render_modes"]`; accepting anything else produces an env that
    silently renders nothing.
    """

    def test_rejects_an_unsupported_render_mode(self) -> None:
        pytest.importorskip("gymnasium")
        from forge_env import gymnasium_env

        if gymnasium_env._NativeEnv is None:
            pytest.skip("forge_env running in pure-Python mode (no native backend)")

        with pytest.raises(ValueError, match="Unsupported render_mode"):
            gymnasium_env.ForgeGymnasiumEnv(render_mode="human")

    def test_accepts_the_declared_modes(self) -> None:
        pytest.importorskip("gymnasium")
        from forge_env import gymnasium_env

        if gymnasium_env._NativeEnv is None:
            pytest.skip("forge_env running in pure-Python mode (no native backend)")

        for mode in gymnasium_env.ForgeGymnasiumEnv.metadata["render_modes"]:
            env = gymnasium_env.ForgeGymnasiumEnv(render_mode=mode)
            try:
                assert env.render_mode == mode
            finally:
                env.close()


def _make_native_env_mock() -> MagicMock:
    native_env = MagicMock()
    native_env.observation_space = {
        "messages": {},
    }
    native_env.action_space = {}
    native_env.reset.return_value = ({"health": 1.0}, {"tick": 0})
    native_env.step.return_value = ({"health": 0.9}, 1.0, False, False, {"tick": 1})
    native_env.render.return_value = "ascii-frame"
    return native_env


class TestPurePythonBranches:
    """Tests for constructor and helper branches that do not require the native backend."""

    def test_init_raises_without_native_backend(self) -> None:
        from forge_env import gymnasium_env

        with (
            patch.object(gymnasium_env, "_NativeEnv", None),
            pytest.raises(ImportError, match="native module not found"),
        ):
            gymnasium_env.ForgeGymnasiumEnv()

    def test_init_raises_without_gymnasium(self) -> None:
        from forge_env import gymnasium_env

        native_factory = MagicMock(return_value=_make_native_env_mock())
        with (
            patch.object(gymnasium_env, "_NativeEnv", native_factory),
            patch.object(gymnasium_env, "HAS_GYMNASIUM", False),
            pytest.raises(ImportError, match="gymnasium not installed"),
        ):
            gymnasium_env.ForgeGymnasiumEnv()

    def test_fallback_space_metadata_defaults(self) -> None:
        pytest.importorskip("gymnasium")
        from forge_env import gymnasium_env

        native_env = _make_native_env_mock()
        native_factory = MagicMock(return_value=native_env)
        with patch.object(gymnasium_env, "_NativeEnv", native_factory):
            env = gymnasium_env.ForgeGymnasiumEnv()

        grid_view = env.observation_space.spaces["grid_view"]
        inventory = env.observation_space.spaces["inventory"]
        messages = env.observation_space.spaces["messages"]
        assert grid_view.shape == (11, 11, 7)
        assert inventory.shape == (10, 2)
        assert messages.shape == (0,)
        assert env.action_space.n == 40
        env.close()

    def test_render_ascii_delegates_to_native(self) -> None:
        pytest.importorskip("gymnasium")
        from forge_env import gymnasium_env

        native_env = _make_native_env_mock()
        with patch.object(gymnasium_env, "_NativeEnv", MagicMock(return_value=native_env)):
            env = gymnasium_env.ForgeGymnasiumEnv(render_mode="ascii")

        assert env.render() == "ascii-frame"
        native_env.render.assert_called_once()
        env.close()

    def test_unwrapped_returns_self_and_native_exposes_the_handle(self) -> None:
        """`unwrapped` follows the Gymnasium contract; `native` is the handle.

        `unwrapped` used to return the native PyO3 object. That breaks the
        Gymnasium contract — `unwrapped` must yield the base `gymnasium.Env`, so
        a wrapper chain such as `TimeLimit(env).unwrapped` resolves to an env
        rather than to a foreign object — and it is asserted by the upstream
        checker. The native handle moved to the explicit `native` property.
        """
        pytest.importorskip("gymnasium")
        from forge_env import gymnasium_env

        native_env = _make_native_env_mock()
        with patch.object(gymnasium_env, "_NativeEnv", MagicMock(return_value=native_env)):
            env = gymnasium_env.ForgeGymnasiumEnv()

        assert env.unwrapped is env
        assert env.native is native_env
        env.close()
