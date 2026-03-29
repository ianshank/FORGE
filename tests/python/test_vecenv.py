"""Tests for forge_env.vecenv — ForgeSyncVecEnv, ForgeAsyncVecEnv, make_forge_vec_env."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — build mock ForgeGymnasiumEnv without the native Rust extension
# ---------------------------------------------------------------------------

MOCK_OBS_DICT: dict[str, Any] = {
    "grid_view": np.zeros((11, 11, 7), dtype=np.uint8),
    "inventory": np.zeros((10, 2), dtype=np.uint16),
    "health": np.float32(0.8),
    "stamina": np.float32(0.9),
    "position": np.array([5, 5], dtype=np.uint16),
    "messages": np.zeros((0,), dtype=np.uint16),
    "day_phase": 0,
}


def _make_gym_env_mock(
    action_n: int = 8,
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
) -> MagicMock:
    """Return a MagicMock that mimics ForgeGymnasiumEnv."""
    env = MagicMock()
    env.reset.return_value = (MOCK_OBS_DICT.copy(), {"tick": 0})
    env.step.return_value = (
        MOCK_OBS_DICT.copy(),
        reward,
        terminated,
        truncated,
        {"tick": 1},
    )
    env.render.return_value = "ascii"
    env.close.return_value = None

    # Minimal spaces
    act_space = MagicMock()
    act_space.n = action_n
    env.action_space = act_space

    obs_space = MagicMock()
    env.observation_space = obs_space
    return env


def _env_fn_factory(mock_env: MagicMock) -> Callable[[], MagicMock]:
    """Return a zero-argument callable that returns *mock_env*."""
    return lambda: mock_env


# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------
from forge_env.vecenv import (  # noqa: E402
    ForgeAsyncVecEnv,
    ForgeSyncVecEnv,
    _stack_obs,
    make_forge_vec_env,
)

# ---------------------------------------------------------------------------
# _stack_obs
# ---------------------------------------------------------------------------


class TestStackObs:
    """Tests for the internal _stack_obs helper."""

    def test_empty_list_returns_empty_dict(self) -> None:
        assert _stack_obs([]) == {}

    def test_single_env_preserves_shapes(self) -> None:
        obs_list = [MOCK_OBS_DICT.copy()]
        batched = _stack_obs(obs_list)
        assert batched["grid_view"].shape == (1, 11, 11, 7)
        assert batched["position"].shape == (1, 2)

    def test_multiple_envs_batch_dimension(self) -> None:
        obs_list = [MOCK_OBS_DICT.copy() for _ in range(4)]
        batched = _stack_obs(obs_list)
        assert batched["grid_view"].shape == (4, 11, 11, 7)
        assert batched["inventory"].shape == (4, 10, 2)

    def test_scalar_obs_stacked(self) -> None:
        obs_list = [{"health": np.float32(0.5)}, {"health": np.float32(0.7)}]
        batched = _stack_obs(obs_list)
        assert batched["health"].shape == (2,)
        np.testing.assert_allclose(batched["health"], [0.5, 0.7])


# ---------------------------------------------------------------------------
# ForgeSyncVecEnv
# ---------------------------------------------------------------------------


class TestForgeSyncVecEnv:
    """Tests for ForgeSyncVecEnv."""

    def _make_vec(self, n: int = 2, **kwargs: Any) -> ForgeSyncVecEnv:
        envs = [_make_gym_env_mock(**kwargs) for _ in range(n)]
        env_fns = [_env_fn_factory(e) for e in envs]
        return ForgeSyncVecEnv(env_fns)

    # -- construction --

    def test_num_envs_attribute(self) -> None:
        vec = self._make_vec(n=3)
        assert vec.num_envs == 3

    def test_observation_action_spaces_from_first_env(self) -> None:
        vec = self._make_vec(n=2)
        assert vec.action_space is not None
        assert vec.observation_space is not None

    def test_envs_property_length(self) -> None:
        vec = self._make_vec(n=2)
        assert len(vec.envs) == 2

    def test_empty_env_fns_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ForgeSyncVecEnv([])

    # -- reset --

    def test_reset_returns_batched_obs_and_infos(self) -> None:
        vec = self._make_vec(n=3)
        obs, infos = vec.reset()
        assert isinstance(obs, dict)
        assert "grid_view" in obs
        assert obs["grid_view"].shape[0] == 3
        assert len(infos) == 3

    def test_reset_with_seed_increments_per_env(self) -> None:
        envs = [_make_gym_env_mock() for _ in range(2)]
        env_fns = [_env_fn_factory(e) for e in envs]
        vec = ForgeSyncVecEnv(env_fns)
        vec.reset(seed=10)
        _, kwargs0 = envs[0].reset.call_args
        _, kwargs1 = envs[1].reset.call_args
        assert kwargs0["seed"] == 10
        assert kwargs1["seed"] == 11

    def test_reset_seed_none_passes_none(self) -> None:
        envs = [_make_gym_env_mock()]
        env_fns = [_env_fn_factory(e) for e in envs]
        vec = ForgeSyncVecEnv(env_fns)
        vec.reset(seed=None)
        _, kwargs = envs[0].reset.call_args
        assert kwargs["seed"] is None

    # -- step --

    def test_step_returns_correct_shapes(self) -> None:
        vec = self._make_vec(n=4)
        vec.reset()
        actions = np.zeros(4, dtype=np.int64)
        obs, rewards, terminated, truncated, infos = vec.step(actions)
        assert obs["grid_view"].shape[0] == 4
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)
        assert len(infos) == 4

    def test_step_auto_reset_on_done(self) -> None:
        """When an env terminates, it should auto-reset."""
        env = _make_gym_env_mock(terminated=True)
        env_fn = _env_fn_factory(env)
        vec = ForgeSyncVecEnv([env_fn])
        vec.reset()
        actions = np.array([0])
        _obs, _, terminated, _, infos = vec.step(actions)
        # Env should have been reset after termination
        assert "terminal_observation" in infos[0]
        assert terminated[0]

    def test_step_rewards_dtype_float32(self) -> None:
        vec = self._make_vec(n=2)
        vec.reset()
        _, rewards, _, _, _ = vec.step(np.zeros(2, dtype=np.int64))
        assert rewards.dtype == np.float32

    # -- close / render --

    def test_close_calls_inner_close(self) -> None:
        envs = [_make_gym_env_mock() for _ in range(2)]
        env_fns = [_env_fn_factory(e) for e in envs]
        vec = ForgeSyncVecEnv(env_fns)
        vec.close()
        for env in envs:
            env.close.assert_called_once()

    def test_render_returns_list(self) -> None:
        vec = self._make_vec(n=2)
        result = vec.render()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# make_forge_vec_env
# ---------------------------------------------------------------------------


class TestMakeForgeVecEnv:
    """Tests for the make_forge_vec_env factory."""

    def test_returns_sync_by_default(self) -> None:
        with patch("forge_env.vecenv.ForgeGymnasiumEnv", return_value=_make_gym_env_mock()):
            vec = make_forge_vec_env(n_envs=2, seed=0, asynchronous=False)
        assert isinstance(vec, ForgeSyncVecEnv)
        vec.close()

    def test_n_envs_respected(self) -> None:
        with patch("forge_env.vecenv.ForgeGymnasiumEnv", return_value=_make_gym_env_mock()):
            vec = make_forge_vec_env(n_envs=3, seed=0)
        assert vec.num_envs == 3
        vec.close()

    def test_zero_envs_raises(self) -> None:
        with pytest.raises(ValueError, match="n_envs"):
            make_forge_vec_env(n_envs=0)

    def test_wrapper_fns_applied(self) -> None:
        """Wrapper callables should be applied to each env in the factory."""
        wrapper_calls: list[int] = []

        def counting_wrapper(env: Any) -> Any:
            wrapper_calls.append(1)
            return env  # pass-through

        with patch("forge_env.vecenv.ForgeGymnasiumEnv", return_value=_make_gym_env_mock()):
            vec = make_forge_vec_env(n_envs=2, wrapper_fns=[counting_wrapper])
        # Each of the 2 envs should have been wrapped once.
        assert sum(wrapper_calls) == 2
        vec.close()

    def test_config_forwarded(self) -> None:
        """The config dict should reach ForgeGymnasiumEnv."""
        created_configs: list[Any] = []

        def mock_ctor(**kwargs: Any) -> MagicMock:
            created_configs.append(kwargs.get("config"))
            return _make_gym_env_mock()

        with patch("forge_env.vecenv.ForgeGymnasiumEnv", side_effect=mock_ctor):
            make_forge_vec_env(
                config={"world": {"width": 16}},
                n_envs=1,
            )
        assert created_configs[0] == {"world": {"width": 16}}


# ---------------------------------------------------------------------------
# ForgeAsyncVecEnv — smoke tests (spawn context is slow; use mock)
# ---------------------------------------------------------------------------


class TestForgeAsyncVecEnv:
    """Smoke tests for ForgeAsyncVecEnv using the sync fallback path."""

    def test_empty_env_fns_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ForgeAsyncVecEnv([])

    def test_render_returns_none_list(self) -> None:
        """render() in async mode returns a list of None without error."""
        # We can instantiate sync and check the method contract without
        # actually spawning processes.
        vec = self._make_sync_vec(n=2)
        # ForgeAsyncVecEnv.render is tested via a mock of num_envs
        async_mock = MagicMock(spec=ForgeAsyncVecEnv)
        async_mock.num_envs = 2
        # Call the real render implementation via unbound method
        result = ForgeAsyncVecEnv.render(async_mock)
        assert result == [None, None]
        vec.close()

    # Helper — use sync vec to test shared logic
    def _make_sync_vec(self, n: int = 2) -> ForgeSyncVecEnv:
        envs = [_make_gym_env_mock() for _ in range(n)]
        env_fns = [_env_fn_factory(e) for e in envs]
        return ForgeSyncVecEnv(env_fns)

    def _make_async_stub(self) -> ForgeAsyncVecEnv:
        vec = object.__new__(ForgeAsyncVecEnv)
        vec.num_envs = 1
        vec._parent_pipes = []
        vec._processes = []
        return vec

    def test_reset_sends_incremented_seeds(self) -> None:
        vec = self._make_async_stub()
        pipe_a = MagicMock()
        pipe_b = MagicMock()
        pipe_a.recv.return_value = (MOCK_OBS_DICT.copy(), {"seed": 10})
        pipe_b.recv.return_value = (MOCK_OBS_DICT.copy(), {"seed": 11})
        vec._parent_pipes = [pipe_a, pipe_b]
        vec.num_envs = 2

        _obs, infos = ForgeAsyncVecEnv.reset(vec, seed=10, options={"mode": "test"})

        pipe_a.send.assert_any_call(("reset", (10, {"mode": "test"})))
        pipe_b.send.assert_any_call(("reset", (11, {"mode": "test"})))
        assert infos[0]["seed"] == 10
        assert infos[1]["seed"] == 11

    def test_reset_exception_propagates(self) -> None:
        vec = self._make_async_stub()
        pipe = MagicMock()
        pipe.recv.return_value = RuntimeError("boom")
        vec._parent_pipes = [pipe]

        with pytest.raises(RuntimeError, match="boom"):
            ForgeAsyncVecEnv.reset(vec)

    def test_step_auto_resets_and_merges_info(self) -> None:
        vec = self._make_async_stub()
        pipe = MagicMock()
        terminal_obs = MOCK_OBS_DICT.copy()
        reset_obs = MOCK_OBS_DICT.copy()
        reset_obs["health"] = np.float32(0.1)
        pipe.recv.side_effect = [
            (terminal_obs, 1.0, True, False, {"tick": 1}),
            (reset_obs, {"reset": True}),
        ]
        vec._parent_pipes = [pipe]

        obs, rewards, terminated, truncated, infos = ForgeAsyncVecEnv.step(
            vec, np.array([0], dtype=np.int64)
        )

        assert rewards.shape == (1,)
        assert terminated[0]
        assert not truncated[0]
        assert infos[0]["reset"] is True
        assert "terminal_observation" in infos[0]
        assert obs["health"][0] == pytest.approx(0.1)

    def test_close_terminates_hung_process(self) -> None:
        vec = self._make_async_stub()
        pipe = MagicMock()
        process = MagicMock()
        process.is_alive.return_value = True
        vec._parent_pipes = [pipe]
        vec._processes = [process]

        ForgeAsyncVecEnv.close(vec)

        process.join.assert_called_once_with(timeout=5)
        process.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# NumPy unavailability
# ---------------------------------------------------------------------------


def test_numpy_unavailable_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ForgeSyncVecEnv raises ImportError when numpy is missing."""
    import forge_env.vecenv as vecenv_mod  # noqa: PLC0415

    monkeypatch.setattr(vecenv_mod, "HAS_NUMPY", False)
    with pytest.raises(ImportError, match="numpy"):
        ForgeSyncVecEnv([_make_gym_env_mock])
