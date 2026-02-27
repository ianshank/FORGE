"""test_utils_extended.py \u2014 Extended pure-Python tests for forge_env.utils.

Tests check_env failure modes, make_env error paths, benchmark_fps behaviour,
and seed_everything correctness \u2014 all without requiring the native Rust extension.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal env mocks
# ---------------------------------------------------------------------------


class _GoodEnv:
    """A conforming environment for baseline pass verification."""

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"obs": 0}, {}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return {"obs": 0}, 0.0, False, False, {}

    def close(self) -> None:
        pass


class _BadResetLengthEnv:
    """reset() returns a 3-tuple (invalid)."""

    def reset(self, **kwargs: Any) -> tuple[Any, ...]:  # type: ignore[override]
        return {"obs": 0}, {}, "extra"

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return {"obs": 0}, 0.0, False, False, {}


class _BadStepLengthEnv:
    """step() returns a 6-tuple (invalid)."""

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"obs": 0}, {}

    def step(self, action: int) -> tuple[Any, ...]:  # type: ignore[override]
        return {"obs": 0}, 0.0, False, False, {}, "extra"


class _NonBoolTerminatedEnv:
    """step() returns a non-bool for terminated."""

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"obs": 0}, {}

    def step(self, action: int) -> tuple[Any, float, Any, bool, dict[str, Any]]:  # type: ignore[override]
        return {"obs": 0}, 0.0, 0, False, {}  # terminated = int, not bool


class _NonBoolTruncatedEnv:
    """step() returns a non-bool for truncated."""

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"obs": 0}, {}

    def step(self, action: int) -> tuple[Any, float, bool, Any, dict[str, Any]]:  # type: ignore[override]
        return {"obs": 0}, 0.0, False, 0, {}  # truncated = int, not bool


class _InstantEnv:
    """Environment that returns instantly (for FPS test)."""

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"obs": 0}, {}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return {"obs": 0}, 0.0, False, False, {}


# ---------------------------------------------------------------------------
# check_env tests
# ---------------------------------------------------------------------------


def test_check_env_passes_good_env() -> None:
    """check_env returns True for a conforming environment."""
    from forge_env.utils import check_env  # noqa: PLC0415

    assert check_env(_GoodEnv()) is True


def test_check_env_bad_reset_tuple_length() -> None:
    """check_env raises AssertionError when reset() returns wrong-length tuple."""
    from forge_env.utils import check_env  # noqa: PLC0415

    with pytest.raises(AssertionError, match="2-tuple"):
        check_env(_BadResetLengthEnv())  # type: ignore[arg-type]


def test_check_env_bad_step_tuple_length() -> None:
    """check_env raises AssertionError when step() returns wrong-length tuple."""
    from forge_env.utils import check_env  # noqa: PLC0415

    with pytest.raises(AssertionError, match="5-tuple"):
        check_env(_BadStepLengthEnv())  # type: ignore[arg-type]


def test_check_env_non_bool_terminated() -> None:
    """check_env raises AssertionError when terminated is not bool."""
    from forge_env.utils import check_env  # noqa: PLC0415

    with pytest.raises(AssertionError, match="terminated must be bool"):
        check_env(_NonBoolTerminatedEnv())  # type: ignore[arg-type]


def test_check_env_non_bool_truncated() -> None:
    """check_env raises AssertionError when truncated is not bool."""
    from forge_env.utils import check_env  # noqa: PLC0415

    with pytest.raises(AssertionError, match="truncated must be bool"):
        check_env(_NonBoolTruncatedEnv())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# make_env tests
# ---------------------------------------------------------------------------


def test_make_env_raises_without_native() -> None:
    """make_env raises ImportError when the native extension is absent."""
    import forge_env.utils as _utils_mod  # noqa: PLC0415

    original = _utils_mod._NativeEnv
    try:
        _utils_mod._NativeEnv = None  # type: ignore[assignment]
        with pytest.raises(ImportError, match="forge_env native module"):
            _utils_mod.make_env()
    finally:
        _utils_mod._NativeEnv = original


def test_make_env_applies_wrappers_in_order() -> None:
    """make_env wraps the env in the declared order (first wrapper outermost last)."""
    import forge_env.utils as _utils_mod  # noqa: PLC0415
    from forge_env.wrappers import NormalizeRewardWrapper, TimeLimit  # noqa: PLC0415

    mock_native = MagicMock()
    mock_native.return_value = _GoodEnv()

    original = _utils_mod._NativeEnv
    try:
        _utils_mod._NativeEnv = mock_native  # type: ignore[assignment]
        env = _utils_mod.make_env(
            wrappers=[
                lambda e: TimeLimit(e, max_steps=10),
                NormalizeRewardWrapper,
            ]
        )
    finally:
        _utils_mod._NativeEnv = original

    # The outermost wrapper should be NormalizeRewardWrapper
    assert isinstance(env, NormalizeRewardWrapper)
    # Traverse the chain manually in case `unwrapped` isn't available
    inner = env.env  # NormalizeRewardWrapper.env → TimeLimit
    inner = inner.env  # TimeLimit.env → _GoodEnv
    assert isinstance(inner, _GoodEnv)


def test_make_env_calls_reset_with_seed() -> None:
    """make_env calls reset(seed=<seed>) when seed is provided."""
    import forge_env.utils as _utils_mod  # noqa: PLC0415

    inner = _GoodEnv()
    inner.reset = MagicMock(return_value=({"obs": 0}, {}))  # type: ignore[method-assign]

    mock_native = MagicMock(return_value=inner)
    original = _utils_mod._NativeEnv
    try:
        _utils_mod._NativeEnv = mock_native  # type: ignore[assignment]
        _utils_mod.make_env(seed=99)
    finally:
        _utils_mod._NativeEnv = original

    inner.reset.assert_called_once_with(seed=99)


# ---------------------------------------------------------------------------
# benchmark_fps tests
# ---------------------------------------------------------------------------


def test_benchmark_fps_returns_positive_float() -> None:
    """benchmark_fps returns a positive float for a fast environment."""
    from forge_env.utils import benchmark_fps  # noqa: PLC0415

    fps = benchmark_fps(_InstantEnv(), n_steps=100)
    assert isinstance(fps, float)
    assert fps > 0.0


def test_benchmark_fps_resets_on_episode_end() -> None:
    """benchmark_fps calls reset when the episode ends mid-benchmark."""
    from forge_env.utils import benchmark_fps  # noqa: PLC0415

    # Environment that terminates after 3 steps
    class _ShortEpisodeEnv:
        def __init__(self) -> None:
            self._steps = 0
            self.reset_count = 0

        def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            self.reset_count += 1
            self._steps = 0
            return {"obs": 0}, {}

        def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self._steps += 1
            terminated = self._steps >= 3
            return {"obs": 0}, 0.0, terminated, False, {}

    env = _ShortEpisodeEnv()
    benchmark_fps(env, n_steps=10)
    # Should have reset at least once (initial reset) + once after termination
    assert env.reset_count >= 2


# ---------------------------------------------------------------------------
# seed_everything tests
# ---------------------------------------------------------------------------


def test_seed_everything_callable_no_error() -> None:
    """seed_everything runs without raising for a standard seed."""
    from forge_env.utils import seed_everything  # noqa: PLC0415

    seed_everything(0)
    seed_everything(42)
    seed_everything(2**31 - 1)


def test_seed_everything_idempotent() -> None:
    """Calling seed_everything twice with the same seed does not raise."""
    from forge_env.utils import seed_everything  # noqa: PLC0415

    seed_everything(123)
    seed_everything(123)


def test_seed_everything_seeds_random_module() -> None:
    """seed_everything produces deterministic random.random() output."""
    import random  # noqa: PLC0415

    from forge_env.utils import seed_everything  # noqa: PLC0415

    seed_everything(7)
    val1 = random.random()
    seed_everything(7)
    val2 = random.random()
    assert val1 == val2, "random.random() should be reproducible after seed_everything"


def test_seed_everything_seeds_numpy_if_available() -> None:
    """seed_everything produces deterministic numpy output (if installed)."""
    np = pytest.importorskip("numpy")
    from forge_env.utils import seed_everything  # noqa: PLC0415

    seed_everything(13)
    arr1 = np.random.rand(5)
    seed_everything(13)
    arr2 = np.random.rand(5)
    np.testing.assert_array_equal(arr1, arr2)


def test_seed_everything_mocks_torch_if_installed() -> None:
    """seed_everything calls torch.manual_seed when torch is available."""
    from forge_env.utils import seed_everything  # noqa: PLC0415

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        seed_everything(42)

    fake_torch.manual_seed.assert_called_once_with(42)
