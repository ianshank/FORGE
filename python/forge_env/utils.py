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
import os
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
            "forge_env native module not found. Install with: pip install -e . (requires maturin)"
        )

    env: Any = _NativeEnv(config=config)

    if wrappers is not None:
        for wrapper_fn in wrappers:
            env = wrapper_fn(env)

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

    assert isinstance(terminated, bool), f"terminated must be bool, got {type(terminated).__name__}"
    assert isinstance(truncated, bool), f"truncated must be bool, got {type(truncated).__name__}"

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


# Fallback-path mirrors of the canonical constants in `forge.utils.seed`.
# They are duplicated (not imported) on purpose: the branch that uses them
# is precisely the branch where `forge.utils.seed` is NOT importable. Keep
# the two in lock-step — `tests/python/test_seed.py` asserts they match.

#: Default for :func:`seed_everything`'s ``deterministic`` switch — off, so
#: existing callers keep their current behaviour and throughput.
DEFAULT_DETERMINISTIC = False
#: Only affects child processes; the parent's hash seed is fixed at startup.
PYTHONHASHSEED_ENV_VAR = "PYTHONHASHSEED"
#: Required before ``use_deterministic_algorithms(True)`` will allow CUDA matmuls.
CUBLAS_WORKSPACE_CONFIG_ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"
DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def seed_everything(seed: int, *, deterministic: bool = DEFAULT_DETERMINISTIC) -> None:
    """Set random seeds across multiple libraries for reproducibility.

    Delegates to :func:`forge.utils.seed.set_all_seeds` when available,
    falling back to an inline implementation otherwise. Both paths seed the
    same generators.

    Seeds the following (when available):

    * ``random`` (Python stdlib)
    * ``numpy.random``
    * ``torch.manual_seed`` — always, whenever torch is importable. This is
      the generator a CPU-only host uses, so seeding it is what makes torch
      reproducible off-GPU.
    * ``torch.cuda.manual_seed_all`` — additionally, when CUDA is available.

    Args:
        seed: The integer seed value.
        deterministic: Opt into strict determinism — additionally sets
            ``torch.use_deterministic_algorithms(True)``, the cuDNN
            determinism/autotune flags, and ``PYTHONHASHSEED``. Defaults to
            :data:`DEFAULT_DETERMINISTIC` (off): it costs throughput and
            makes some kernels raise rather than fall back, so it is opt-in.
    """
    try:
        from forge.utils.seed import set_all_seeds
    except ImportError:
        # Inline fallback: `forge` is not importable (e.g. a stripped
        # `forge_env`-only install). Mirror set_all_seeds' behaviour.
        random.seed(seed)
        if HAS_NUMPY:
            np.random.seed(seed)
        _seed_torch_fallback(seed, deterministic=deterministic)
        return

    set_all_seeds(seed, deterministic=deterministic)


def _seed_torch_fallback(seed: int, *, deterministic: bool) -> None:
    """Seed torch without depending on the ``forge`` package.

    Used only by :func:`seed_everything`'s ``forge.utils.seed``-unavailable
    branch; the canonical implementation is
    :func:`forge.utils.seed.seed_torch`.
    """
    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return

    os.environ[PYTHONHASHSEED_ENV_VAR] = str(seed)
    os.environ.setdefault(CUBLAS_WORKSPACE_CONFIG_ENV_VAR, DEFAULT_CUBLAS_WORKSPACE_CONFIG)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except (AttributeError, RuntimeError) as exc:
        logger.warning("strict torch determinism unavailable: %s", exc)
