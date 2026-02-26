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

import importlib
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Check native module availability
# ---------------------------------------------------------------------------

_native_available = importlib.util.find_spec("forge_env.forge_env") is not None

skip_native = pytest.mark.skipif(
    not _native_available,
    reason="Native forge_env module not built (run: maturin develop)",
)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_import_forge_env() -> None:
    """The forge_env package itself should always be importable."""
    import forge_env  # noqa: PLC0415

    assert hasattr(forge_env, "__version__")


def test_import_gymnasium_env() -> None:
    """The gymnasium wrapper module should be importable."""
    from forge_env import gymnasium_env  # noqa: PLC0415

    assert hasattr(gymnasium_env, "ForgeGymnasiumEnv")


def test_import_pettingzoo_env() -> None:
    """The PettingZoo wrapper module should be importable."""
    from forge_env import pettingzoo_env  # noqa: PLC0415

    assert hasattr(pettingzoo_env, "ForgeParallelEnv")


def test_import_wrappers() -> None:
    """All wrapper classes should be importable."""
    from forge_env.wrappers import (  # noqa: PLC0415
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
    from forge_env.utils import benchmark_fps, check_env, make_env, seed_everything  # noqa: PLC0415

    assert callable(make_env)
    assert callable(check_env)
    assert callable(benchmark_fps)
    assert callable(seed_everything)


# ---------------------------------------------------------------------------
# Native-dependent tests
# ---------------------------------------------------------------------------


@skip_native
def test_env_creation() -> None:
    """Creating a ForgeGymnasiumEnv with default config should not raise."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

    env = ForgeGymnasiumEnv()
    assert env is not None
    env.close()


@skip_native
def test_reset_returns_tuple() -> None:
    """reset() must return a (obs, info) 2-tuple."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

    env = ForgeGymnasiumEnv()
    result = env.reset(seed=42)
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    env.close()


@skip_native
def test_step_returns_tuple() -> None:
    """step() must return a (obs, reward, term, trunc, info) 5-tuple."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

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


@skip_native
def test_observation_keys() -> None:
    """Observation dict should contain the expected keys."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

    env = ForgeGymnasiumEnv()
    obs, _info = env.reset(seed=42)
    expected_keys = {
        "grid_view", "inventory", "health", "stamina", "position", "messages", "day_phase",
    }
    assert expected_keys.issubset(
        set(obs.keys())
    ), f"Missing keys: {expected_keys - set(obs.keys())}"
    env.close()


@skip_native
def test_deterministic_seed() -> None:
    """Same seed should produce the same initial observation."""
    import numpy as np  # noqa: PLC0415

    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

    env1 = ForgeGymnasiumEnv()
    obs1, _ = env1.reset(seed=123)
    env1.close()

    env2 = ForgeGymnasiumEnv()
    obs2, _ = env2.reset(seed=123)
    env2.close()

    np.testing.assert_array_equal(obs1["grid_view"], obs2["grid_view"])
    assert obs1["health"] == obs2["health"]
    assert obs1["position"] == obs2["position"]


@skip_native
def test_wrapper_flatten() -> None:
    """FlattenObservationWrapper should produce a 1-D array."""
    import numpy as np  # noqa: PLC0415

    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415
    from forge_env.wrappers import FlattenObservationWrapper  # noqa: PLC0415

    env = FlattenObservationWrapper(ForgeGymnasiumEnv())
    obs, _info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert obs.dtype == np.float32
    env.close()


@skip_native
def test_wrapper_time_limit() -> None:
    """TimeLimit wrapper should truncate at max_steps."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415
    from forge_env.wrappers import TimeLimit  # noqa: PLC0415

    env = TimeLimit(ForgeGymnasiumEnv(), max_steps=10)
    env.reset(seed=42)

    truncated = False
    for _ in range(20):
        _obs, _reward, terminated, truncated, _info = env.step(0)
        if terminated or truncated:
            break

    assert truncated, "Episode should have been truncated at max_steps=10"
    env.close()


@skip_native
def test_multi_agent_env() -> None:
    """ForgeParallelEnv with 2 agents should return per-agent dicts."""
    from forge_env.pettingzoo_env import ForgeParallelEnv  # noqa: PLC0415

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


@skip_native
def test_benchmark_fps() -> None:
    """benchmark_fps should return a positive float."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415
    from forge_env.utils import benchmark_fps  # noqa: PLC0415

    env = ForgeGymnasiumEnv()
    fps = benchmark_fps(env, n_steps=100)
    assert isinstance(fps, float)
    assert fps > 0, "FPS should be positive"
    env.close()


@skip_native
def test_check_env() -> None:
    """check_env should pass for a valid environment."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415
    from forge_env.utils import check_env  # noqa: PLC0415

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
        self, action: int,
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
    from forge_env.wrappers import TimeLimit  # noqa: PLC0415

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
    from forge_env.wrappers import NormalizeRewardWrapper  # noqa: PLC0415

    env = NormalizeRewardWrapper(_DummyEnv())
    env.reset()

    for _ in range(100):
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert -10.0 <= reward <= 10.0, f"Normalised reward out of range: {reward}"


def test_record_episode_statistics_pure() -> None:
    """RecordEpisodeStatistics should inject episode info on termination."""
    from forge_env.wrappers import RecordEpisodeStatistics, TimeLimit  # noqa: PLC0415

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
    from forge_env.wrappers import FlattenObservationWrapper  # noqa: PLC0415

    env = FlattenObservationWrapper(_DummyEnv())
    obs, _info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert obs.dtype == np.float32
    # {"x": [1, 2], "y": [3]} -> 3 elements
    assert obs.shape[0] == 3


def test_seed_everything() -> None:
    """seed_everything should be callable and not raise."""
    from forge_env.utils import seed_everything  # noqa: PLC0415

    seed_everything(42)
