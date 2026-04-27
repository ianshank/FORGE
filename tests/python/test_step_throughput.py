"""PyO3 end-to-end step latency test for Phase 0 audit.

Measures ``env.step(action)`` round-trip latency through the real Python
wrapper used by ``examples/train_ppo.py`` and ``examples/train_sac_cleanrl.py``
rather than the cheapest ``Action::Noop`` path. Emits a JSON report with
median, p99, and derived steps/sec so the Phase 0 claims-verification table
has a reference row for the ``<8 μs/step`` claim.

The test is opt-in behind ``FORGE_RUN_STEP_THROUGHPUT=1`` so CI does not pay
the 100k-iteration cost by default. Running locally:

    FORGE_RUN_STEP_THROUGHPUT=1 \
        pytest tests/python/test_step_throughput.py -s

Writing the JSON report next to the audit tree (only when running):

    FORGE_RUN_STEP_THROUGHPUT=1 \
    FORGE_STEP_THROUGHPUT_OUT=/tmp/pyo3_step.json \
        pytest tests/python/test_step_throughput.py -s
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)

#: Environment variable that opts the heavy measurement into the test run.
ENV_RUN: str = "FORGE_RUN_STEP_THROUGHPUT"
#: Environment variable that, when set, writes the JSON report there.
ENV_OUT: str = "FORGE_STEP_THROUGHPUT_OUT"
#: Environment variable overriding the number of measured step() calls.
ENV_ITERS: str = "FORGE_STEP_THROUGHPUT_ITERS"
#: Environment variable overriding warm-up iterations before measurement.
ENV_WARMUP: str = "FORGE_STEP_THROUGHPUT_WARMUP"
#: Environment variable overriding the deterministic env seed.
ENV_SEED: str = "FORGE_STEP_THROUGHPUT_SEED"

#: Default measurement iteration count (matches the plan's 100k target).
DEFAULT_ITERS: int = 100_000
#: Default warm-up iterations before measurement starts.
DEFAULT_WARMUP: int = 1_000
#: Default deterministic seed for reproducible measurements.
DEFAULT_SEED: int = 42
#: Minimum pass rate for "step must complete" as a smoke check in the
#: fast-default mode (not the opt-in heavy run).
SMOKE_ITERS: int = 32

#: Action id for ``Action::Move(Direction::Right)`` in the base action space.
#: This matches the path that ``train_ppo.py`` exercises rather than the
#: cheaper ``Noop`` path.
ACTION_ID_MOVE_RIGHT: int = 4


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning("ignoring non-integer %s=%r; using default %d", name, raw, default)
        return default
    if value < minimum:
        LOGGER.warning("ignoring %s=%d below minimum %d; using default %d", name, value, minimum, default)
        return default
    return value


def _percentile(samples: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile (0-100) using linear interpolation.

    Implemented locally to avoid pulling ``numpy`` into the test surface when
    the only consumer is a tiny summary statistic. ``samples`` must be
    non-empty; callers guard this.
    """
    if not samples:
        raise ValueError("_percentile requires at least one sample")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("wrote step-throughput report to %s", path)


def _native_env_factory() -> Any:
    """Return the native ``ForgeEnv`` class if the extension is built.

    Skips the test when running in the pure-Python CI matrix where the
    maturin build has not produced ``forge_env.forge_env``.
    """
    try:
        from forge_env.forge_env import ForgeEnv
    except ImportError:
        pytest.skip("forge_env native extension not built (run `maturin develop`)")
    return ForgeEnv


def _drive(env: Any, action_id: int, iters: int) -> list[float]:
    """Step ``env`` ``iters`` times recording nanosecond latency per call.

    Resets on terminal transitions so the measurement loop never pays the
    cost of a dead environment. Returns per-step durations in nanoseconds.
    """
    samples: list[float] = [0.0] * iters
    perf_ns = time.perf_counter_ns
    for i in range(iters):
        start = perf_ns()
        result = env.step(action_id)
        end = perf_ns()
        samples[i] = float(end - start)
        # Result shape is (obs, reward, terminated, truncated, info) per
        # the Gymnasium protocol; we only care about termination flags.
        if isinstance(result, tuple) and len(result) >= 4:
            terminated = bool(result[2])
            truncated = bool(result[3])
            if terminated or truncated:
                env.reset()
    return samples


def _summarise(samples: list[float], *, iters: int, warmup: int, seed: int) -> dict[str, Any]:
    median_ns = statistics.median(samples)
    p99_ns = _percentile(samples, 99.0)
    mean_ns = statistics.fmean(samples)
    stdev_ns = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    steps_per_sec = 1e9 / mean_ns if mean_ns > 0 else float("inf")
    return {
        "iters": iters,
        "warmup": warmup,
        "seed": seed,
        "action_id": ACTION_ID_MOVE_RIGHT,
        "median_ns": median_ns,
        "p99_ns": p99_ns,
        "mean_ns": mean_ns,
        "stdev_ns": stdev_ns,
        "steps_per_sec": steps_per_sec,
    }


def _iter_environments(env_cls: Any, seed: int) -> Iterable[Any]:
    """Yield a single configured env, ensuring it is closed after use."""
    env = env_cls(config={"seed": seed})
    try:
        env.reset()
        yield env
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def test_pyo3_step_smoke() -> None:
    """Fast smoke check: the native path completes ``SMOKE_ITERS`` steps.

    Runs unconditionally (when the extension is available) so the PyO3
    integration stays covered even if nobody opts into the heavy run. Uses
    ``Action::Move(Direction::Right)`` — the same path the heavy test
    measures — so any regression in that dispatch is caught here too.
    """
    env_cls = _native_env_factory()
    seed = _env_int(ENV_SEED, DEFAULT_SEED)
    for env in _iter_environments(env_cls, seed):
        samples = _drive(env, ACTION_ID_MOVE_RIGHT, SMOKE_ITERS)
        assert len(samples) == SMOKE_ITERS
        assert all(s >= 0.0 for s in samples)


@pytest.mark.skipif(
    os.environ.get(ENV_RUN) != "1",
    reason=f"set {ENV_RUN}=1 to run the heavy throughput measurement",
)
def test_pyo3_step_throughput() -> None:
    """Measure PyO3 round-trip step latency under the realistic action path."""
    env_cls = _native_env_factory()
    warmup = _env_int(ENV_WARMUP, DEFAULT_WARMUP)
    iters = _env_int(ENV_ITERS, DEFAULT_ITERS)
    seed = _env_int(ENV_SEED, DEFAULT_SEED)

    LOGGER.info("running step-throughput: warmup=%d iters=%d seed=%d", warmup, iters, seed)
    for env in _iter_environments(env_cls, seed):
        _drive(env, ACTION_ID_MOVE_RIGHT, warmup)
        samples = _drive(env, ACTION_ID_MOVE_RIGHT, iters)

    summary = _summarise(samples, iters=iters, warmup=warmup, seed=seed)
    LOGGER.info("step-throughput summary: %s", summary)

    out = os.environ.get(ENV_OUT)
    if out:
        _write_report(Path(out), summary)

    # Soft correctness assertions — the test's primary purpose is the report,
    # not a strict performance gate. Hard numeric thresholds belong in the
    # audit disposition table, not in the test suite.
    assert summary["iters"] == iters
    assert summary["median_ns"] > 0.0
    assert summary["steps_per_sec"] > 0.0
