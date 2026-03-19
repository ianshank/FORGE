"""Mock-based tests for forge_env modules that require the native Rust extension.

These tests patch _NativeEnv with a mock to exercise all Python-side logic
in gymnasium_env, pettingzoo_env, and utils without needing `maturin develop`.
"""
from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from conftest import (
    MOCK_OBS,
    NATIVE_ACT_SPACE,
    NATIVE_OBS_SPACE_WITH_MESSAGES,
    make_mock_native_env,
)

# ---------------------------------------------------------------------------
# ForgeGymnasiumEnv (mock-based)
# ---------------------------------------------------------------------------


class TestGymnasiumEnvMocked:
    """Test ForgeGymnasiumEnv with a mocked native backend."""

    @pytest.fixture()
    def _patch_native(self) -> Any:
        """Patch _NativeEnv in gymnasium_env so we can construct without Rust."""
        mock_env = make_mock_native_env(include_messages_in_obs_space=True)
        mock_cls = mock.MagicMock(return_value=mock_env)
        with mock.patch("forge_env.gymnasium_env._NativeEnv", mock_cls):
            yield mock_cls, mock_env

    def test_init_builds_spaces(self, _patch_native: Any) -> None:
        """Constructor should build gymnasium spaces from native obs/act info."""
        gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        env = ForgeGymnasiumEnv(config={"seed": 1})
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.action_space.n == NATIVE_ACT_SPACE["n"]

    def test_init_uses_native_obs_dimensions(self, _patch_native: Any) -> None:
        """Observation space dimensions should come from native obs_space dict."""
        gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        env = ForgeGymnasiumEnv()
        grid_view_space = env.observation_space["grid_view"]
        expected_h = NATIVE_OBS_SPACE_WITH_MESSAGES["grid_view_height"]
        expected_w = NATIVE_OBS_SPACE_WITH_MESSAGES["grid_view_width"]
        expected_c = NATIVE_OBS_SPACE_WITH_MESSAGES["grid_view_channels"]
        assert grid_view_space.shape == (expected_h, expected_w, expected_c)

    def test_reset_delegates_to_native(self, _patch_native: Any) -> None:
        """reset() should delegate to the native env."""
        gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        _, mock_env = _patch_native
        env = ForgeGymnasiumEnv()
        obs, _info = env.reset(seed=42)
        mock_env.reset.assert_called_once_with(seed=42, options=None)
        assert obs == MOCK_OBS

    def test_step_delegates_to_native(self, _patch_native: Any) -> None:
        """step() should delegate to the native env."""
        gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        _, mock_env = _patch_native
        env = ForgeGymnasiumEnv()
        result = env.step(3)
        mock_env.step.assert_called_once_with(3)
        assert len(result) == 5

    def test_render_ascii_delegates(self, _patch_native: Any) -> None:
        """render() with ascii mode should delegate to native."""
        gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        env = ForgeGymnasiumEnv(render_mode="ascii")
        result = env.render()
        assert result == "ascii_frame"

    def test_render_no_mode_returns_none(self, _patch_native: Any) -> None:
        """render() without render_mode should return None."""
        gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        env = ForgeGymnasiumEnv(render_mode=None)
        assert env.render() is None

    def test_close_delegates(self, _patch_native: Any) -> None:
        """close() should delegate to the native env."""
        gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        _, mock_env = _patch_native
        env = ForgeGymnasiumEnv()
        env.close()
        mock_env.close.assert_called_once()

    def test_unwrapped_returns_native(self, _patch_native: Any) -> None:
        """unwrapped property should return the native env."""
        gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        _, mock_env = _patch_native
        env = ForgeGymnasiumEnv()
        assert env.unwrapped is mock_env

    def test_raises_without_native(self) -> None:
        """Constructor should raise ImportError when native is None."""
        pytest.importorskip("gymnasium")
        with mock.patch("forge_env.gymnasium_env._NativeEnv", None):
            from forge_env.gymnasium_env import ForgeGymnasiumEnv

            with pytest.raises(ImportError, match="native module not found"):
                ForgeGymnasiumEnv()


# ---------------------------------------------------------------------------
# ForgeParallelEnv (mock-based)
# ---------------------------------------------------------------------------


class TestParallelEnvMocked:
    """Test ForgeParallelEnv with a mocked native backend."""

    @pytest.fixture()
    def _patch_native(self) -> Any:
        """Patch _NativeEnv in pettingzoo_env."""
        mock_env = make_mock_native_env()
        mock_cls = mock.MagicMock(return_value=mock_env)
        with mock.patch("forge_env.pettingzoo_env._NativeEnv", mock_cls):
            yield mock_cls, mock_env

    def test_init_sets_agents(self, _patch_native: Any) -> None:
        """Constructor should create agent names."""
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=3)
        assert env.possible_agents == ["agent_0", "agent_1", "agent_2"]
        assert env.agents == ["agent_0", "agent_1", "agent_2"]

    def test_init_merges_config(self, _patch_native: Any) -> None:
        """Constructor should merge n_agents into config dict."""
        mock_cls, _ = _patch_native
        from forge_env.pettingzoo_env import ForgeParallelEnv

        ForgeParallelEnv(n_agents=2, config={"world": {"width": 16}})
        call_kwargs = mock_cls.call_args
        passed_config = call_kwargs[1]["config"] if "config" in call_kwargs[1] else call_kwargs[0][0]
        assert passed_config["agents"]["num_agents"] == 2
        assert passed_config["world"]["width"] == 16

    def test_reset_returns_per_agent_dicts(self, _patch_native: Any) -> None:
        """reset() should return observations and infos keyed by agent name."""
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2)
        obs, infos = env.reset(seed=42)
        assert set(obs.keys()) == {"agent_0", "agent_1"}
        assert set(infos.keys()) == {"agent_0", "agent_1"}

    def test_step_returns_per_agent_dicts(self, _patch_native: Any) -> None:
        """step() should distribute results to all agents."""
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2)
        env.reset()
        actions = {"agent_0": 0, "agent_1": 1}
        _obs, rewards, _terms, _truncs, _infos = env.step(actions)
        assert set(rewards.keys()) == {"agent_0", "agent_1"}
        assert all(isinstance(v, (int, float)) for v in rewards.values())

    def test_step_removes_terminated_agents(self, _patch_native: Any) -> None:
        """step() should remove agents that are terminated."""
        _, mock_env = _patch_native
        mock_env.step.return_value = (MOCK_OBS, 0.0, True, False, {})
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2)
        env.reset()
        actions = {"agent_0": 0, "agent_1": 0}
        env.step(actions)
        assert env.agents == []

    def test_observation_space_per_agent(self, _patch_native: Any) -> None:
        """observation_space() should return space for given agent."""
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2)
        space = env.observation_space("agent_0")
        assert isinstance(space, dict)

    def test_action_space_per_agent(self, _patch_native: Any) -> None:
        """action_space() should return space for given agent."""
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2)
        space = env.action_space("agent_0")
        assert isinstance(space, dict)

    def test_render_ascii_mode(self, _patch_native: Any) -> None:
        """render() with ascii mode should return string."""
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2, render_mode="ascii")
        result = env.render()
        assert result == "ascii_frame"

    def test_render_none_mode(self, _patch_native: Any) -> None:
        """render() without mode returns None."""
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2, render_mode=None)
        assert env.render() is None

    def test_close_delegates(self, _patch_native: Any) -> None:
        """close() should delegate to native env."""
        _, mock_env = _patch_native
        from forge_env.pettingzoo_env import ForgeParallelEnv

        env = ForgeParallelEnv(n_agents=2)
        env.close()
        mock_env.close.assert_called_once()

    def test_raises_without_native(self) -> None:
        """Constructor should raise ImportError when native is None."""
        with mock.patch("forge_env.pettingzoo_env._NativeEnv", None):
            from forge_env.pettingzoo_env import ForgeParallelEnv

            with pytest.raises(ImportError, match="native module not found"):
                ForgeParallelEnv(n_agents=2)


# ---------------------------------------------------------------------------
# utils (mock-based)
# ---------------------------------------------------------------------------


class TestUtilsMocked:
    """Test forge_env.utils functions with mocked native backend."""

    @pytest.fixture()
    def _patch_native(self) -> Any:
        """Patch _NativeEnv in utils module."""
        mock_env = make_mock_native_env()
        mock_cls = mock.MagicMock(return_value=mock_env)
        with mock.patch("forge_env.utils._NativeEnv", mock_cls):
            yield mock_cls, mock_env

    def test_make_env_creates_env(self, _patch_native: Any) -> None:
        """make_env should create a native env when available."""
        from forge_env.utils import make_env

        env = make_env(config={"seed": 1})
        assert env is not None

    def test_make_env_applies_wrappers(self, _patch_native: Any) -> None:
        """make_env should apply wrappers in order."""
        from forge_env.utils import make_env

        wrapper_calls: list[str] = []

        def wrapper_a(e: Any) -> Any:
            wrapper_calls.append("a")
            return e

        def wrapper_b(e: Any) -> Any:
            wrapper_calls.append("b")
            return e

        make_env(wrappers=[wrapper_a, wrapper_b])
        assert wrapper_calls == ["a", "b"]

    def test_make_env_calls_reset_with_seed(self, _patch_native: Any) -> None:
        """make_env with seed should call reset(seed=...)."""
        _, mock_env = _patch_native
        from forge_env.utils import make_env

        make_env(seed=123)
        mock_env.reset.assert_called_once_with(seed=123)

    def test_check_env_passes_valid_env(self, _patch_native: Any) -> None:
        """check_env should return True for a valid mock env."""
        _, mock_env = _patch_native
        from forge_env.utils import check_env

        assert check_env(mock_env) is True

    def test_check_env_validates_step_length(self) -> None:
        """check_env should reject step() returning wrong tuple length."""
        from forge_env.utils import check_env

        class _BadStepEnv:
            def reset(self, **kwargs: Any) -> tuple[dict, dict]:
                return {}, {}

            def step(self, action: int) -> tuple[dict, float, bool]:
                return {}, 0.0, False  # Missing truncated and info

        with pytest.raises(AssertionError, match="5-tuple"):
            check_env(_BadStepEnv())

    def test_check_env_validates_terminated_type(self) -> None:
        """check_env should reject non-bool terminated."""
        from forge_env.utils import check_env

        class _BadTermEnv:
            def reset(self, **kwargs: Any) -> tuple[dict, dict]:
                return {}, {}

            def step(self, action: int) -> tuple[dict, float, int, bool, dict]:
                return {}, 0.0, 1, False, {}  # terminated is int, not bool

        with pytest.raises(AssertionError, match="terminated must be bool"):
            check_env(_BadTermEnv())

    def test_check_env_validates_truncated_type(self) -> None:
        """check_env should reject non-bool truncated."""
        from forge_env.utils import check_env

        class _BadTruncEnv:
            def reset(self, **kwargs: Any) -> tuple[dict, dict]:
                return {}, {}

            def step(self, action: int) -> tuple[dict, float, bool, int, dict]:
                return {}, 0.0, False, 1, {}  # truncated is int, not bool

        with pytest.raises(AssertionError, match="truncated must be bool"):
            check_env(_BadTruncEnv())

    def test_benchmark_fps_returns_positive(self, _patch_native: Any) -> None:
        """benchmark_fps should return a positive float."""
        _, mock_env = _patch_native
        from forge_env.utils import benchmark_fps

        fps = benchmark_fps(mock_env, n_steps=100)
        assert isinstance(fps, float)
        assert fps > 0

    def test_benchmark_fps_handles_episode_end(self, _patch_native: Any) -> None:
        """benchmark_fps should reset env when episode ends."""
        _, mock_env = _patch_native
        # Make step return terminated=True every 10 steps
        call_count = 0

        def step_side_effect(action: int) -> tuple:
            nonlocal call_count
            call_count += 1
            terminated = call_count % 10 == 0
            return MOCK_OBS, 1.0, terminated, False, {}

        mock_env.step.side_effect = step_side_effect
        from forge_env.utils import benchmark_fps

        fps = benchmark_fps(mock_env, n_steps=50)
        assert fps > 0
        # Should have reset multiple times (initial + 5 episode endings)
        assert mock_env.reset.call_count >= 2

    def test_seed_everything_without_torch(self) -> None:
        """seed_everything should work when torch is not installed."""
        from forge_env.utils import seed_everything

        # Should not raise even without torch
        seed_everything(99)
