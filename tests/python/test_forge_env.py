"""Integration tests for the FORGE Python environment wrappers.

These tests verify the Python-side API contracts:
  - gymnasium_env.ForgeGymnasiumEnv
  - pettingzoo_env.ForgeParallelEnv
  - wrappers (Flatten, NormalizeReward, TimeLimit, RecordEpisodeStatistics)
  - utils (make_env, check_env, benchmark_fps)

Tests that require the native Rust extension (maturin build) are marked
with ``@pytest.mark.skipif`` so the suite still runs in pure-Python CI.

Run with::

    pytest tests/python/ -v
"""

from __future__ import annotations

from typing import Any

import pytest

from forge_env.wrappers import DEFAULT_REWARD_CLIP

# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_import_forge_env() -> None:
    """The forge_env package itself should always be importable."""
    import forge_env

    assert hasattr(forge_env, "__version__")


def test_import_gymnasium_env() -> None:
    """The gymnasium wrapper module should be importable."""
    from forge_env import gymnasium_env

    assert hasattr(gymnasium_env, "ForgeGymnasiumEnv")


def test_import_pettingzoo_env() -> None:
    """The PettingZoo wrapper module should be importable."""
    from forge_env import pettingzoo_env

    assert hasattr(pettingzoo_env, "ForgeParallelEnv")


def test_import_wrappers() -> None:
    """All wrapper classes should be importable."""
    from forge_env.wrappers import (
        FlattenObservationWrapper,
        NormalizeRewardWrapper,
        RecordEpisodeStatistics,
        TimeLimit,
    )

    assert FlattenObservationWrapper is not None
    assert NormalizeRewardWrapper is not None
    assert TimeLimit is not None
    assert RecordEpisodeStatistics is not None


def test_import_utils() -> None:
    """Utility functions should be importable."""
    from forge_env.utils import benchmark_fps, check_env, make_env, seed_everything

    assert callable(make_env)
    assert callable(check_env)
    assert callable(benchmark_fps)
    assert callable(seed_everything)


# ---------------------------------------------------------------------------
# Native-dependent tests — skip gracefully when Rust extension is unavailable
# ---------------------------------------------------------------------------

def _skip_if_no_native() -> None:
    """Skip the calling test when the forge_env native extension is not built."""
    try:
        from forge_env import gymnasium_env
        if gymnasium_env._NativeEnv is None:
            pytest.skip("forge_env running in pure-Python mode (no native backend)")
    except ImportError as exc:
        pytest.skip(f"forge_env native extension not available: {exc}")


def test_env_creation() -> None:
    """Creating a ForgeGymnasiumEnv with default config should not raise."""
    _skip_if_no_native()
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    env = ForgeGymnasiumEnv()
    assert env is not None
    env.close()


def test_reset_returns_tuple() -> None:
    """reset() must return a (obs, info) 2-tuple."""
    _skip_if_no_native()
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    env = ForgeGymnasiumEnv()
    result = env.reset(seed=42)
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    env.close()


def test_step_returns_tuple() -> None:
    """step() must return a (obs, reward, term, trunc, info) 5-tuple."""
    _skip_if_no_native()
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    env = ForgeGymnasiumEnv()
    env.reset(seed=42)
    result = env.step(0)
    assert isinstance(result, tuple)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_observation_keys() -> None:
    """Observation dict should contain the expected keys."""
    _skip_if_no_native()
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    env = ForgeGymnasiumEnv()
    obs, _info = env.reset(seed=42)
    expected_keys = {
        "grid_view",
        "inventory",
        "health",
        "stamina",
        "position",
        "messages",
        "day_phase",
    }
    assert expected_keys.issubset(set(obs.keys())), (
        f"Missing keys: {expected_keys - set(obs.keys())}"
    )
    env.close()


def test_deterministic_seed() -> None:
    """Same seed should produce the same initial observation."""
    _skip_if_no_native()
    import numpy as np

    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    env1 = ForgeGymnasiumEnv()
    obs1, _ = env1.reset(seed=123)
    env1.close()

    env2 = ForgeGymnasiumEnv()
    obs2, _ = env2.reset(seed=123)
    env2.close()

    np.testing.assert_array_equal(obs1["grid_view"], obs2["grid_view"])
    assert obs1["health"] == obs2["health"]
    assert obs1["position"] == obs2["position"]


def test_wrapper_flatten() -> None:
    """FlattenObservationWrapper should produce a 1-D array."""
    _skip_if_no_native()
    import numpy as np

    from forge_env.gymnasium_env import ForgeGymnasiumEnv
    from forge_env.wrappers import FlattenObservationWrapper

    env = FlattenObservationWrapper(ForgeGymnasiumEnv())
    obs, _info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert obs.dtype == np.float32
    env.close()


def test_wrapper_time_limit() -> None:
    """TimeLimit wrapper should truncate at max_steps."""
    _skip_if_no_native()
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
    from forge_env.wrappers import TimeLimit

    env = TimeLimit(ForgeGymnasiumEnv(), max_steps=10)
    env.reset(seed=42)

    truncated = False
    for _ in range(20):
        _obs, _reward, terminated, truncated, _info = env.step(0)
        if terminated or truncated:
            break

    assert truncated, "Episode should have been truncated at max_steps=10"
    env.close()


def test_multi_agent_env() -> None:
    """ForgeParallelEnv with 2 agents should return per-agent dicts."""
    _skip_if_no_native()
    from forge_env.pettingzoo_env import ForgeParallelEnv

    env = ForgeParallelEnv(n_agents=2)
    observations, _infos = env.reset(seed=42)

    assert len(observations) == 2
    assert "agent_0" in observations
    assert "agent_1" in observations

    actions = {"agent_0": 0, "agent_1": 1}
    _obs, rewards, _terms, _truncs, _infos = env.step(actions)
    assert "agent_0" in rewards
    assert "agent_1" in rewards
    env.close()


def test_benchmark_fps() -> None:
    """benchmark_fps should return a positive float."""
    _skip_if_no_native()
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
    from forge_env.utils import benchmark_fps

    env = ForgeGymnasiumEnv()
    fps = benchmark_fps(env, n_steps=100)
    assert isinstance(fps, float)
    assert fps > 0, "FPS should be positive"
    env.close()


def test_check_env() -> None:
    """check_env should pass for a valid environment."""
    _skip_if_no_native()
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
    from forge_env.utils import check_env

    env = ForgeGymnasiumEnv()
    result = check_env(env)
    assert result is True
    env.close()


# ---------------------------------------------------------------------------
# Pure-Python wrapper tests (no native module needed)
# ---------------------------------------------------------------------------


class _DummyEnv:
    """Minimal env mock for testing wrappers without native dependencies."""

    def __init__(self) -> None:
        self._step_count = 0

    def reset(self, **kwargs: Any) -> tuple[dict[str, list[float]], dict[str, int]]:
        """Reset the dummy environment."""
        self._step_count = 0
        obs: dict[str, list[float]] = {"x": [1.0, 2.0], "y": [3.0]}
        info: dict[str, int] = {"tick": 0}
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, list[float]], float, bool, bool, dict[str, int]]:
        """Step the dummy environment."""
        self._step_count += 1
        obs: dict[str, list[float]] = {"x": [1.0, 2.0], "y": [3.0]}
        reward = 1.0
        terminated = False
        truncated = False
        info: dict[str, int] = {"tick": self._step_count}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the dummy environment (no-op)."""


def test_time_limit_wrapper_pure() -> None:
    """TimeLimit truncates after max_steps (no native module needed)."""
    from forge_env.wrappers import TimeLimit

    env = TimeLimit(_DummyEnv(), max_steps=5)
    env.reset()

    truncated = False
    for _ in range(10):
        _obs, _r, _term, truncated, _info = env.step(0)
        if truncated:
            break

    assert truncated, "Should truncate after 5 steps"


def test_normalize_reward_wrapper_pure() -> None:
    """NormalizeRewardWrapper should produce values in [-10, 10]."""
    from forge_env.wrappers import NormalizeRewardWrapper

    env = NormalizeRewardWrapper(_DummyEnv())
    env.reset()

    for _ in range(100):
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert -DEFAULT_REWARD_CLIP <= reward <= DEFAULT_REWARD_CLIP, f"Normalised reward out of range: {reward}"


def test_record_episode_statistics_pure() -> None:
    """RecordEpisodeStatistics should inject episode info on termination."""
    from forge_env.wrappers import RecordEpisodeStatistics, TimeLimit

    env = RecordEpisodeStatistics(TimeLimit(_DummyEnv(), max_steps=5))
    env.reset()

    info: dict[str, Any] = {}
    for _ in range(10):
        _obs, _r, _term, truncated, info = env.step(0)
        if truncated:
            break

    assert "episode" in info, "info should contain 'episode' key on termination"
    assert "r" in info["episode"], "episode info should have 'r' (return)"
    assert "l" in info["episode"], "episode info should have 'l' (length)"
    assert "t" in info["episode"], "episode info should have 't' (time)"
    assert info["episode"]["l"] == 5, "episode length should be 5"


def test_flatten_observation_wrapper_pure() -> None:
    """FlattenObservationWrapper should produce 1-D float32 arrays."""
    np = pytest.importorskip("numpy")
    from forge_env.wrappers import FlattenObservationWrapper

    env = FlattenObservationWrapper(_DummyEnv())
    obs, _info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert obs.dtype == np.float32
    # {"x": [1, 2], "y": [3]} -> 3 elements
    assert obs.shape[0] == 3


def test_seed_everything() -> None:
    """seed_everything should be callable and not raise."""
    from forge_env.utils import seed_everything

    seed_everything(42)


def test_time_limit_resets_counter() -> None:
    """TimeLimit should reset its _current_step counter on reset()."""
    from forge_env.wrappers import TimeLimit

    env = TimeLimit(_DummyEnv(), max_steps=10)
    env.reset()

    # Take a few steps to advance the counter.
    for _ in range(5):
        env.step(0)
    assert env._current_step == 5

    # After reset the counter must be back to zero.
    env.reset()
    assert env._current_step == 0


def test_normalize_reward_updates_stats() -> None:
    """NormalizeRewardWrapper.count, reward_mean, and reward_var should update after steps."""
    from forge_env.wrappers import NormalizeRewardWrapper

    env = NormalizeRewardWrapper(_DummyEnv())
    env.reset()

    assert env.count == 0.0
    assert env.reward_mean == 0.0

    # _DummyEnv always returns reward=1.0
    env.step(0)
    assert env.count == 1.0
    # After a single observation of 1.0 the mean must be 1.0.
    assert env.reward_mean == 1.0

    env.step(0)
    assert env.count == 2.0
    # Two observations of 1.0 -> mean still 1.0.
    assert env.reward_mean == 1.0


def test_normalize_reward_clips_extreme() -> None:
    """Normalised reward must stay in [-10, 10] even with extreme inputs."""
    from forge_env.wrappers import NormalizeRewardWrapper

    class _ExtremeRewardEnv(_DummyEnv):
        """Dummy env that returns extreme reward values."""

        def step(
            self,
            action: int,
        ) -> tuple[dict[str, list[float]], float, bool, bool, dict[str, int]]:
            self._step_count += 1
            obs: dict[str, list[float]] = {"x": [1.0, 2.0], "y": [3.0]}
            reward = 1e12 if self._step_count % 2 == 0 else -1e12
            info: dict[str, int] = {"tick": self._step_count}
            return obs, reward, False, False, info

    env = NormalizeRewardWrapper(_ExtremeRewardEnv())
    env.reset()

    for _ in range(50):
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert -DEFAULT_REWARD_CLIP <= reward <= DEFAULT_REWARD_CLIP, f"Normalised reward out of range: {reward}"


def test_record_episode_statistics_resets_on_new_episode() -> None:
    """RecordEpisodeStatistics should reset counters when reset() is called."""
    from forge_env.wrappers import RecordEpisodeStatistics, TimeLimit

    env = RecordEpisodeStatistics(TimeLimit(_DummyEnv(), max_steps=3))

    # First episode.
    env.reset()
    for _ in range(3):
        env.step(0)

    # Internal counters should reflect the finished episode.
    assert env._episode_length == 3
    assert env._episode_return == 3.0

    # After reset the counters must be zeroed out.
    env.reset()
    assert env._episode_length == 0
    assert env._episode_return == 0.0


def test_flatten_observation_step() -> None:
    """FlattenObservationWrapper should flatten observations from step() too."""
    np = pytest.importorskip("numpy")
    from forge_env.wrappers import FlattenObservationWrapper

    env = FlattenObservationWrapper(_DummyEnv())
    env.reset()

    obs, _reward, _terminated, _truncated, _info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert obs.dtype == np.float32
    # {"x": [1, 2], "y": [3]} -> 3 elements
    assert obs.shape[0] == 3


def test_base_wrapper_delegates_attributes() -> None:
    """_BaseWrapper.__getattr__ should forward attribute access to the inner env."""
    from forge_env.wrappers import TimeLimit

    inner = _DummyEnv()
    inner.custom_attr = "hello"  # type: ignore[attr-defined]

    wrapped = TimeLimit(inner, max_steps=10)
    # The wrapper itself has no 'custom_attr'; it should be forwarded.
    assert wrapped.custom_attr == "hello"
    # The inner env's _step_count should also be reachable via delegation.
    assert wrapped._step_count == inner._step_count


def test_time_limit_does_not_affect_terminated() -> None:
    """TimeLimit must not override terminated=True coming from the inner env."""
    from forge_env.wrappers import TimeLimit

    class _TerminatingEnv(_DummyEnv):
        """Dummy env that terminates on the very first step."""

        def step(
            self,
            action: int,
        ) -> tuple[dict[str, list[float]], float, bool, bool, dict[str, int]]:
            self._step_count += 1
            obs: dict[str, list[float]] = {"x": [1.0, 2.0], "y": [3.0]}
            info: dict[str, int] = {"tick": self._step_count}
            return obs, 0.0, True, False, info

    env = TimeLimit(_TerminatingEnv(), max_steps=100)
    env.reset()

    _obs, _r, terminated, truncated, _info = env.step(0)
    assert terminated is True, "terminated flag from inner env must be preserved"
    assert truncated is False, "TimeLimit should not set truncated on step 1 of 100"


def test_seed_everything_with_numpy() -> None:
    """seed_everything should set the numpy random seed correctly."""
    np = pytest.importorskip("numpy")
    from forge_env.utils import seed_everything

    seed_everything(12345)
    a = np.random.rand(5)

    seed_everything(12345)
    b = np.random.rand(5)

    np.testing.assert_array_equal(a, b)


def test_make_env_raises_without_native() -> None:
    """make_env should raise ImportError when the native module is not available."""
    from unittest import mock

    import forge_env.utils as utils_mod

    with (
        mock.patch.object(utils_mod, "_NativeEnv", None),
        pytest.raises(ImportError, match="native module not found"),
    ):
        utils_mod.make_env()


def test_check_env_validates_reset_type() -> None:
    """check_env should raise AssertionError when reset() returns the wrong type."""
    from forge_env.utils import check_env

    class _BadEnv:
        """Env whose reset() returns a plain dict instead of a tuple."""

        def reset(self, **kwargs: Any) -> dict[str, int]:
            return {"bad": 1}

        def step(self, action: int) -> None:
            pass  # pragma: no cover

        def close(self) -> None:
            pass  # pragma: no cover

    with pytest.raises(AssertionError, match=r"reset.*must return a tuple"):
        check_env(_BadEnv())


# ---------------------------------------------------------------------------
# forge_env.__init__ — ImportError path (ForgeEnv = None)
# ---------------------------------------------------------------------------


def test_forge_env_init_native_import_failure() -> None:
    """forge_env.__init__ sets ForgeEnv=None when native module is unavailable."""
    import importlib
    import sys

    import forge_env as fe_mod

    # Remember current state so we can restore it.
    original_forge_env_native = sys.modules.get("forge_env.forge_env", "ABSENT")
    original_ForgeEnv = fe_mod.ForgeEnv  # noqa: F841

    try:
        # Setting sys.modules entry to None causes `import forge_env.forge_env`
        # to raise ImportError when the module is re-imported.
        sys.modules["forge_env.forge_env"] = None  # type: ignore[assignment]
        importlib.reload(fe_mod)
        assert fe_mod.ForgeEnv is None
    finally:
        # Restore sys.modules to its prior state.
        if original_forge_env_native == "ABSENT":
            sys.modules.pop("forge_env.forge_env", None)
        else:
            sys.modules["forge_env.forge_env"] = original_forge_env_native  # type: ignore[assignment]
        # Reload again to restore the module's ForgeEnv attribute.
        importlib.reload(fe_mod)
        # After restoration the attribute should be back to whatever it was.


def test_forge_env_init_feature_extractors_import_failure() -> None:
    """forge_env.__init__ remains importable when feature extractors are unavailable."""
    import importlib
    import sys

    import forge_env as fe_mod

    original_feature_extractors = sys.modules.get("forge_env.feature_extractors", "ABSENT")

    try:
        sys.modules["forge_env.feature_extractors"] = None  # type: ignore[assignment]
        importlib.reload(fe_mod)
        assert fe_mod._HAS_EXTRACTORS is False
        assert fe_mod.ForgeGridCnnExtractor is None
        assert fe_mod.ForgeObsExtractor is None
        assert fe_mod.ForgeGymnasiumEnv is not None
    finally:
        if original_feature_extractors == "ABSENT":
            sys.modules.pop("forge_env.feature_extractors", None)
        else:
            sys.modules["forge_env.feature_extractors"] = original_feature_extractors  # type: ignore[assignment]
        importlib.reload(fe_mod)


def test_forge_env_init_feature_extractors_dependency_unavailable() -> None:
    """forge_env.__init__ hides extractors when torch or SB3 is unavailable."""
    import importlib

    import forge_env as fe_mod
    import forge_env.feature_extractors as feature_extractors_mod

    original_has_torch = feature_extractors_mod.HAS_TORCH
    original_has_sb3 = feature_extractors_mod.HAS_SB3

    try:
        feature_extractors_mod.HAS_TORCH = False
        feature_extractors_mod.HAS_SB3 = True
        importlib.reload(fe_mod)
        assert fe_mod._HAS_EXTRACTORS is False
        assert fe_mod.ForgeGridCnnExtractor is None
        assert fe_mod.ForgeObsExtractor is None
    finally:
        feature_extractors_mod.HAS_TORCH = original_has_torch
        feature_extractors_mod.HAS_SB3 = original_has_sb3
        importlib.reload(fe_mod)


def test_forge_env_init_sb3_callbacks_import_failure() -> None:
    """forge_env.__init__ remains importable when SB3 callbacks are unavailable."""
    import importlib
    import sys

    import forge_env as fe_mod

    original_callbacks = sys.modules.get("forge_env.sb3_callbacks", "ABSENT")

    try:
        sys.modules["forge_env.sb3_callbacks"] = None  # type: ignore[assignment]
        importlib.reload(fe_mod)
        assert fe_mod._HAS_CALLBACKS is False
        assert fe_mod.ForgeCurriculumCallback is None
        assert fe_mod.ForgeMetricsCallback is None
        assert fe_mod.ForgeSyncVecEnv is not None
    finally:
        if original_callbacks == "ABSENT":
            sys.modules.pop("forge_env.sb3_callbacks", None)
        else:
            sys.modules["forge_env.sb3_callbacks"] = original_callbacks  # type: ignore[assignment]
        importlib.reload(fe_mod)


def test_forge_env_init_sb3_callbacks_dependency_unavailable() -> None:
    """forge_env.__init__ hides callbacks when Stable Baselines 3 is unavailable."""
    import importlib

    import forge_env as fe_mod
    import forge_env.sb3_callbacks as sb3_callbacks_mod

    original_has_sb3 = sb3_callbacks_mod.HAS_SB3

    try:
        sb3_callbacks_mod.HAS_SB3 = False
        importlib.reload(fe_mod)
        assert fe_mod._HAS_CALLBACKS is False
        assert fe_mod.ForgeCurriculumCallback is None
        assert fe_mod.ForgeMetricsCallback is None
    finally:
        sb3_callbacks_mod.HAS_SB3 = original_has_sb3
        importlib.reload(fe_mod)


# ---------------------------------------------------------------------------
# forge_env.utils — make_env wrappers/seed paths, check_env assertion branches
# ---------------------------------------------------------------------------


def test_make_env_with_wrappers() -> None:
    """make_env should apply wrappers in order."""
    from unittest.mock import MagicMock, patch

    import forge_env.utils as utils_mod

    # Create a mock native env and a mock wrapper.
    mock_inner = MagicMock()
    mock_wrapped = MagicMock()
    MockNative = MagicMock(return_value=mock_inner)
    wrapper_fn = MagicMock(return_value=mock_wrapped)

    with patch.object(utils_mod, "_NativeEnv", MockNative):
        env = utils_mod.make_env(wrappers=[wrapper_fn])

    wrapper_fn.assert_called_once_with(mock_inner)
    assert env is mock_wrapped


def test_make_env_with_seed() -> None:
    """make_env calls env.reset(seed=seed) when seed is provided."""
    from unittest.mock import MagicMock, patch

    import forge_env.utils as utils_mod

    mock_env = MagicMock()
    MockNative = MagicMock(return_value=mock_env)

    with patch.object(utils_mod, "_NativeEnv", MockNative):
        utils_mod.make_env(seed=42)

    mock_env.reset.assert_called_once_with(seed=42)


def test_check_env_bad_reset_length() -> None:
    """check_env raises AssertionError when reset() returns wrong-length tuple."""
    from forge_env.utils import check_env

    class _BadResetLen:
        def reset(self, **kwargs: Any) -> tuple[dict, dict, dict]:
            return ({}, {}, {})  # length 3 instead of 2

        def step(self, action: int) -> None:
            pass  # pragma: no cover

    with pytest.raises(AssertionError, match=r"2-tuple"):
        check_env(_BadResetLen())


def test_check_env_bad_step_type() -> None:
    """check_env raises AssertionError when step() does not return a tuple."""
    from forge_env.utils import check_env

    class _BadStepType:
        def reset(self, **kwargs: Any) -> tuple[dict, dict]:
            return ({}, {})

        def step(self, action: int) -> dict:  # type: ignore[override]
            return {"not": "a tuple"}

    with pytest.raises(AssertionError, match=r"must return a tuple"):
        check_env(_BadStepType())


def test_check_env_bad_step_length() -> None:
    """check_env raises AssertionError when step() returns wrong-length tuple."""
    from forge_env.utils import check_env

    class _BadStepLen:
        def reset(self, **kwargs: Any) -> tuple[dict, dict]:
            return ({}, {})

        def step(self, action: int) -> tuple[dict, float, bool]:
            return ({}, 0.0, False)  # length 3 instead of 5

    with pytest.raises(AssertionError, match=r"5-tuple"):
        check_env(_BadStepLen())


def test_check_env_non_bool_terminated() -> None:
    """check_env raises AssertionError when terminated is not bool."""
    from forge_env.utils import check_env

    class _NonBoolTerminated:
        def reset(self, **kwargs: Any) -> tuple[dict, dict]:
            return ({}, {})

        def step(self, action: int) -> tuple[dict, float, int, bool, dict]:
            return ({}, 0.0, 1, False, {})  # terminated is int, not bool

    with pytest.raises(AssertionError, match=r"terminated must be bool"):
        check_env(_NonBoolTerminated())


def test_check_env_non_bool_truncated() -> None:
    """check_env raises AssertionError when truncated is not bool."""
    from forge_env.utils import check_env

    class _NonBoolTruncated:
        def reset(self, **kwargs: Any) -> tuple[dict, dict]:
            return ({}, {})

        def step(self, action: int) -> tuple[dict, float, bool, int, dict]:
            return ({}, 0.0, False, 0, {})  # truncated is int, not bool

    with pytest.raises(AssertionError, match=r"truncated must be bool"):
        check_env(_NonBoolTruncated())


# ---------------------------------------------------------------------------
# forge_env.utils — additional branch coverage
# ---------------------------------------------------------------------------


def test_benchmark_fps_resets_on_episode_end() -> None:
    """benchmark_fps should call env.reset() when an episode terminates."""
    from forge_env.utils import benchmark_fps

    class _TermEvery3Steps(_DummyEnv):
        """Dummy env that terminates every 3 steps."""

        def step(
            self, action: int,
        ) -> tuple[dict[str, list[float]], float, bool, bool, dict[str, int]]:
            self._step_count += 1
            obs: dict[str, list[float]] = {"x": [1.0, 2.0], "y": [3.0]}
            terminated = self._step_count % 3 == 0
            info: dict[str, int] = {"tick": self._step_count}
            return obs, 1.0, terminated, False, info

    env = _TermEvery3Steps()
    fps = benchmark_fps(env, n_steps=12)
    assert fps > 0
    # The env should have been reset after each termination (steps 3, 6, 9, 12).
    # After the final step with n_steps=12, _step_count has been reset multiple times.
    # Just verify benchmark_fps returns without error and FPS is positive.


def test_seed_everything_without_forge_seed_module() -> None:
    """seed_everything uses inline fallback when forge.utils.seed is not importable."""
    import sys
    from unittest.mock import patch

    from forge_env.utils import seed_everything

    # Patch forge.utils.seed to be unavailable, forcing the except ImportError branch.
    with patch.dict(sys.modules, {"forge.utils.seed": None}):
        # Should not raise; falls back to random.seed / np.random.seed
        seed_everything(99)


def test_seed_everything_without_torch() -> None:
    """seed_everything silently skips torch seeding when torch is not installed."""
    import sys
    from unittest.mock import patch

    from forge_env.utils import seed_everything

    # Setting sys.modules["torch"] = None makes `import torch` raise ImportError.
    with patch.dict(sys.modules, {"torch": None}):
        seed_everything(7)  # Should complete without error
