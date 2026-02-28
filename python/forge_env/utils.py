"""Utility functions for the FORGE environment.

Provides convenience helpers for creating, validating, and benchmarking
ForgeEnv instances, as well as a global seeding helper.

Functions
---------
make_env
    Factory that creates a ForgeEnv and optionally wraps it.
check_env
    Basic environment validation (reset / step contract).
benchmark_fps
    Measure raw step throughput in frames-per-second.
seed_everything
    Set random seeds across Python, numpy, and (optionally) torch for
    reproducibility.
"""

from __future__ import annotations

import logging
import random
import time as _time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None

__all__ = [
    "benchmark_fps",
    "check_env",
    "make_env",
    "seed_everything",
]


# ---------------------------------------------------------------------------
# make_env
# ---------------------------------------------------------------------------


def make_env(
    config: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[..., Any]] | None = None,
    seed: int | None = None,
) -> Any:
    """Create a ForgeEnv instance and optionally apply a chain of wrappers.

    Args:
        config: Optional configuration dict forwarded to ``ForgeEnv``.
        wrappers: An optional sequence of wrapper constructors (callables).
            Each callable must accept a single positional argument (the
            environment) and return a wrapped environment.  Wrappers are
            applied in order -- the first element wraps the base env, the
            second wraps the result, and so on.
        seed: If provided, ``reset(seed=seed)`` is called on the final
            (possibly wrapped) environment before it is returned, ensuring
            a deterministic initial state.

    Returns:
        The (optionally wrapped) environment instance.

    Raises:
        ImportError: If the native ``ForgeEnv`` module is not available.

    Example::

        from forge_env.utils import make_env
        from forge_env.wrappers import TimeLimit, RecordEpisodeStatistics

        env = make_env(
            config={"world": {"width": 64, "height": 64}},
            wrappers=[
                lambda e: TimeLimit(e, max_steps=1000),
                RecordEpisodeStatistics,
            ],
            seed=42,
        )
    """
    if _NativeEnv is None:
        raise ImportError(
            "forge_env native module not found. "
            "Install with: pip install -e . (requires maturin)"
        )

    env: Any = _NativeEnv(config=config)

    if wrappers is not None:
        for wrapper_fn in wrappers:
            env = wrapper_fn(env)
            logger.debug("Applied wrapper %s", type(env).__name__)

    if seed is not None:
        env.reset(seed=seed)

    return env


# ---------------------------------------------------------------------------
# check_env
# ---------------------------------------------------------------------------


def check_env(env: Any) -> bool:
    """Run basic sanity checks on an environment instance.

    The function calls ``reset()`` and ``step(0)`` and verifies that the
    return values conform to the expected tuple structure:

    * ``reset()`` must return a ``(obs, info)`` 2-tuple.
    * ``step(action)`` must return a ``(obs, reward, terminated, truncated, info)``
      5-tuple where ``terminated`` and ``truncated`` are booleans.

    Args:
        env: An environment instance exposing ``reset`` and ``step`` methods.

    Returns:
        ``True`` if all checks pass.

    Raises:
        AssertionError: If any check fails, with a descriptive message.
    """
    # -- reset --------------------------------------------------------------
    reset_result = env.reset()
    assert isinstance(reset_result, tuple), (
        f"reset() must return a tuple, got {type(reset_result).__name__}"
    )
    assert len(reset_result) == 2, (
        f"reset() must return a 2-tuple (obs, info), got length {len(reset_result)}"
    )
    _obs, _info = reset_result

    # -- step ---------------------------------------------------------------
    step_result = env.step(0)
    assert isinstance(step_result, tuple), (
        f"step() must return a tuple, got {type(step_result).__name__}"
    )
    assert len(step_result) == 5, (
        f"step() must return a 5-tuple (obs, reward, terminated, truncated, info), "
        f"got length {len(step_result)}"
    )
    _step_obs, _reward, terminated, truncated, _step_info = step_result

    assert isinstance(terminated, bool), (
        f"terminated must be bool, got {type(terminated).__name__}"
    )
    assert isinstance(truncated, bool), (
        f"truncated must be bool, got {type(truncated).__name__}"
    )

    return True


# ---------------------------------------------------------------------------
# benchmark_fps
# ---------------------------------------------------------------------------


def benchmark_fps(env: Any, n_steps: int = 10000) -> float:
    """Measure environment step throughput in frames per second.

    The environment is reset once, then stepped ``n_steps`` times using
    action ``0``.  If the episode ends before ``n_steps`` is reached the
    environment is reset and stepping continues.

    Args:
        env: An environment instance exposing ``reset`` and ``step`` methods.
        n_steps: Number of steps to benchmark over.

    Returns:
        Steps per second as a float.
    """
    env.reset()

    start = _time.perf_counter()
    for _ in range(n_steps):
        _obs, _reward, terminated, truncated, _info = env.step(0)
        if terminated or truncated:
            env.reset()
    elapsed = _time.perf_counter() - start

    fps = n_steps / elapsed if elapsed > 0 else float("inf")
    logger.info("Benchmark: %.1f FPS over %d steps", fps, n_steps)
    return fps


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Set random seeds across multiple libraries for reproducibility.

    Seeds the following (when available):

    * ``random`` (Python stdlib)
    * ``numpy.random``
    * ``torch.manual_seed`` and ``torch.cuda.manual_seed_all``

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)

    if HAS_NUMPY:
        # Seed the legacy global RNG (used by Gymnasium/SB3 etc.)
        np.random.seed(seed)

    # Optional: seed PyTorch if installed
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
