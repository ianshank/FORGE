"""test_wrappers_extended.py \u2014 Comprehensive pure-Python wrapper tests.

Tests all wrapper classes without requiring the native Rust extension.
Covers boundary conditions, statistical properties, wrapper chaining,
attribute delegation, and the new __repr__ / unwrapped API.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import pytest

from forge_env import wrappers as _wrappers_mod

# ---------------------------------------------------------------------------
# Shared _DummyEnv fixture (no native deps)
# ---------------------------------------------------------------------------


class _DummyEnv:
    """Minimal environment mock for wrapper testing."""

    _FIXED_OBS: ClassVar[dict[str, list[float]]] = {"x": [1.0, 2.0], "y": [3.0]}

    def __init__(self, reward: float = 1.0) -> None:
        self._step_count: int = 0
        self._reward = reward
        # Expose fake spaces so delegates work
        self.observation_space: dict[str, Any] = {"shape": (3,)}
        self.action_space: dict[str, Any] = {"n": 4}

    def reset(self, **kwargs: Any) -> tuple[dict[str, list[float]], dict[str, int]]:
        """Reset the dummy environment."""
        self._step_count = 0
        return self._FIXED_OBS.copy(), {"tick": 0}

    def step(
        self, action: int
    ) -> tuple[dict[str, list[float]], float, bool, bool, dict[str, int]]:
        """Step the dummy environment."""
        self._step_count += 1
        return self._FIXED_OBS.copy(), self._reward, False, False, {"tick": self._step_count}

    def close(self) -> None:
        """No-op close."""

    def __repr__(self) -> str:
        return "_DummyEnv()"


# ---------------------------------------------------------------------------
# _BaseWrapper / BaseWrapper tests
# (These features require the locally-modified source; skip gracefully if
# the installed/built package pre-dates them.)
# ---------------------------------------------------------------------------

_HAS_REPR = hasattr(_wrappers_mod._BaseWrapper, "__repr__") and "__repr__" in vars(
    _wrappers_mod._BaseWrapper
)
_HAS_UNWRAPPED = "unwrapped" in vars(_wrappers_mod._BaseWrapper)
_HAS_BASE_WRAPPER_ALIAS = hasattr(_wrappers_mod, "BaseWrapper")

_skip_new_api = pytest.mark.skipif(
    not (_HAS_REPR and _HAS_UNWRAPPED and _HAS_BASE_WRAPPER_ALIAS),
    reason="New wrapper API (repr/unwrapped/BaseWrapper) not in installed package — build & install with maturin develop",
)


@_skip_new_api
def test_base_wrapper_repr() -> None:
    """__repr__ shows the class name and inner env repr."""
    from forge_env.wrappers import TimeLimit  # noqa: PLC0415

    env = _DummyEnv()
    w = TimeLimit(env, max_steps=10)
    r = repr(w)
    assert "TimeLimit" in r
    assert "_DummyEnv" in r


@_skip_new_api
def test_base_wrapper_unwrapped_single() -> None:
    """unwrapped on a single wrapper returns the base env."""
    from forge_env.wrappers import TimeLimit  # noqa: PLC0415

    env = _DummyEnv()
    w = TimeLimit(env, max_steps=5)
    assert w.unwrapped is env


@_skip_new_api
def test_base_wrapper_unwrapped_nested() -> None:
    """unwrapped traverses a multi-layer wrapper chain to the base env."""
    from forge_env.wrappers import (  # noqa: PLC0415
        NormalizeRewardWrapper,
        RecordEpisodeStatistics,
        TimeLimit,
    )

    base = _DummyEnv()
    w = RecordEpisodeStatistics(NormalizeRewardWrapper(TimeLimit(base, max_steps=10)))
    assert w.unwrapped is base


def test_base_wrapper_getattr_delegation() -> None:
    """Attributes not on the wrapper itself are forwarded to the inner env."""
    from forge_env.wrappers import TimeLimit  # noqa: PLC0415

    env = _DummyEnv()
    w = TimeLimit(env, max_steps=5)
    # observation_space is on _DummyEnv but not on TimeLimit
    assert w.observation_space == {"shape": (3,)}


@_skip_new_api
def test_base_wrapper_alias() -> None:
    """BaseWrapper public alias is available and is _BaseWrapper."""
    from forge_env.wrappers import (  # noqa: PLC0415
        BaseWrapper,
        TimeLimit,
    )

    assert issubclass(TimeLimit, BaseWrapper)


# ---------------------------------------------------------------------------
# TimeLimit edge conditions
# ---------------------------------------------------------------------------


def test_time_limit_exact_boundary() -> None:
    """Truncation triggers on the exact max_steps-th step, not before."""
    from forge_env.wrappers import TimeLimit  # noqa: PLC0415

    env = TimeLimit(_DummyEnv(), max_steps=3)
    env.reset()

    # Steps 1, 2 should NOT truncate
    for _ in range(2):
        _obs, _r, _term, truncated, _info = env.step(0)
        assert not truncated, "Should not truncate before max_steps"

    # Step 3 should truncate
    _obs, _r, _term, truncated, _info = env.step(0)
    assert truncated, "Should truncate exactly at max_steps=3"


def test_time_limit_reset_resets_counter() -> None:
    """Resetting the env resets the step counter so truncation starts fresh."""
    from forge_env.wrappers import TimeLimit  # noqa: PLC0415

    env = TimeLimit(_DummyEnv(), max_steps=2)
    env.reset()
    env.step(0)
    env.step(0)  # truncated here

    env.reset()  # counter back to 0
    _obs, _r, _term, truncated, _info = env.step(0)
    assert not truncated, "Counter should have reset after env.reset()"


# ---------------------------------------------------------------------------
# NormalizeRewardWrapper statistical properties
# ---------------------------------------------------------------------------


def test_normalize_reward_clip() -> None:
    """Extreme rewards are clipped to the default clip=10 range."""
    from forge_env.wrappers import NormalizeRewardWrapper  # noqa: PLC0415

    # Use a reward that is guaranteed to be extreme after normalisation
    env = NormalizeRewardWrapper(_DummyEnv(reward=1e9))
    env.reset()
    # Step many times so the statistics have time to settle
    for _ in range(200):
        _obs, reward, _t, _tr, _info = env.step(0)
        assert -10.0 <= reward <= 10.0, f"Clipped reward out of range: {reward}"


def test_normalize_reward_stats_persist_across_resets() -> None:
    """Running statistics are NOT reset when the env is reset (by design)."""
    from forge_env.wrappers import NormalizeRewardWrapper  # noqa: PLC0415

    env = NormalizeRewardWrapper(_DummyEnv())
    env.reset()
    for _ in range(10):
        env.step(0)

    count_before = env.count
    env.reset()
    assert env.count == count_before, "count should not reset on env.reset()"


def test_normalize_reward_mean_converges() -> None:
    """After many constant-reward steps mean should converge to that value."""
    from forge_env.wrappers import NormalizeRewardWrapper  # noqa: PLC0415

    constant_reward = 5.0
    env = NormalizeRewardWrapper(_DummyEnv(reward=constant_reward))
    env.reset()
    for _ in range(500):
        env.step(0)

    assert math.isclose(env.reward_mean, constant_reward, rel_tol=0.01), (
        f"Expected mean≈{constant_reward}, got {env.reward_mean}"
    )


# ---------------------------------------------------------------------------
# RecordEpisodeStatistics
# ---------------------------------------------------------------------------


def test_record_episode_statistics_multiple_episodes() -> None:
    """Statistics accumulate correctly across multiple episodes."""
    from forge_env.wrappers import RecordEpisodeStatistics, TimeLimit  # noqa: PLC0415

    env = RecordEpisodeStatistics(TimeLimit(_DummyEnv(), max_steps=3))

    episodes: list[dict[str, Any]] = []
    for _ep in range(3):
        env.reset()
        info: dict[str, Any] = {}
        for _ in range(5):
            _obs, _r, _term, truncated, info = env.step(0)
            if truncated:
                break
        assert "episode" in info, "episode key expected after truncation"
        episodes.append(info["episode"])

    for ep in episodes:
        assert ep["l"] == 3, "Each episode should last exactly 3 steps"
        assert ep["r"] == pytest.approx(3.0), "Return = 1.0/step * 3 steps"


def test_record_episode_statistics_timing() -> None:
    """Episode timing key 't' is a non-negative float."""
    from forge_env.wrappers import RecordEpisodeStatistics, TimeLimit  # noqa: PLC0415

    env = RecordEpisodeStatistics(TimeLimit(_DummyEnv(), max_steps=2))
    env.reset()
    info: dict[str, Any] = {}
    for _ in range(5):
        _obs, _r, _term, truncated, info = env.step(0)
        if truncated:
            break

    assert isinstance(info["episode"]["t"], float)
    assert info["episode"]["t"] >= 0.0


# ---------------------------------------------------------------------------
# FlattenObservationWrapper
# ---------------------------------------------------------------------------


def test_flatten_obs_sorted_keys() -> None:
    """Keys are concatenated in sorted order giving a deterministic shape."""
    np = pytest.importorskip("numpy")
    from forge_env.wrappers import FlattenObservationWrapper  # noqa: PLC0415

    # obs: {"x": [1, 2], "y": [3]} sorted → x then y → 3 elements
    env = FlattenObservationWrapper(_DummyEnv())
    obs, _ = env.reset()
    assert obs.shape == (3,), f"Expected shape (3,), got {obs.shape}"
    # Verify actual values: sorted(["x","y"]) = ["x","y"]
    np.testing.assert_array_almost_equal(obs, [1.0, 2.0, 3.0])


def test_flatten_obs_step_consistent() -> None:
    """Flattened obs from step() has same shape as from reset()."""
    np = pytest.importorskip("numpy")
    from forge_env.wrappers import FlattenObservationWrapper  # noqa: PLC0415

    env = FlattenObservationWrapper(_DummyEnv())
    reset_obs, _ = env.reset()
    step_obs, _, _, _, _ = env.step(0)

    assert reset_obs.shape == step_obs.shape
    np.testing.assert_array_almost_equal(reset_obs, step_obs)


# ---------------------------------------------------------------------------
# Nested wrappers — end-to-end chain
# ---------------------------------------------------------------------------


def test_nested_wrappers_chain() -> None:
    """TimeLimit(NormalizeReward(FlattenObs(env))) works end-to-end."""
    np = pytest.importorskip("numpy")
    from forge_env.wrappers import (  # noqa: PLC0415
        FlattenObservationWrapper,
        NormalizeRewardWrapper,
        TimeLimit,
    )

    env: Any = _DummyEnv()
    env = FlattenObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    env = TimeLimit(env, max_steps=5)

    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)

    truncated = False
    for _ in range(10):
        obs, reward, _term, truncated, _ = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert -10.0 <= reward <= 10.0
        if truncated:
            break

    assert truncated, "TimeLimit should have truncated after 5 steps"
