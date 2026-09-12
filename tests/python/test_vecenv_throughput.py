"""Karten 2026 random-action SPS for ForgeAsyncVecEnv at N parallel envs.

Measures ``n_envs ∈ {1, 8, 16, 32, 64}``. This is **not** a JAX ``vmap`` of
the physics — ``ForgeJaxEnv`` is ``io_callback`` around native envs and must
not be quoted as GPU JAX SPS. ``RealisticFakeEnv`` is a mock and is not a
Python physics baseline.

The heavy sweep is opt-in (``FORGE_RUN_VECENV_THROUGHPUT=1``). CI still runs
a tiny native smoke so the import path stays covered. Commit the report to
``benchmarks/baselines/cloud_agent/vecenv_step.json`` and keep README
``SPS @ N`` claims at or below those numbers
(``tests/python/test_throughput_claim.py``).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

LOGGER = logging.getLogger(__name__)

ENV_RUN: str = "FORGE_RUN_VECENV_THROUGHPUT"
ENV_OUT: str = "FORGE_VECENV_THROUGHPUT_OUT"
ENV_ITERS: str = "FORGE_VECENV_THROUGHPUT_ITERS"
ENV_WARMUP: str = "FORGE_VECENV_THROUGHPUT_WARMUP"
ENV_SEED: str = "FORGE_VECENV_THROUGHPUT_SEED"
ENV_MIN_WALL: str = "FORGE_VECENV_THROUGHPUT_MIN_WALL_S"

DEFAULT_N_ENVS: tuple[int, ...] = (1, 8, 16, 32, 64)
DEFAULT_ITERS: int = 50_000
DEFAULT_WARMUP: int = 16
DEFAULT_SEED: int = 42
DEFAULT_MIN_WALL_S: float = 2.0
SMOKE_ITERS: int = 4
ACTION_ID_MOVE_RIGHT: int = 4

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "baselines" / "cloud_agent" / "vecenv_step.json"

_MEASURE_CONFIG: dict[str, Any] = {
    "world": {"width": 16, "height": 16, "seed": DEFAULT_SEED},
    "agents": {"num_agents": 1, "comm_vocab_size": 0},
    "task": {"max_episode_length": 10_000, "enabled": False},
}


class _PicklableGymFactory:
    """Spawn-safe zero-arg factory for :class:`ForgeGymnasiumEnv`."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def __call__(self) -> Any:
        from forge_env.gymnasium_env import ForgeGymnasiumEnv

        return ForgeGymnasiumEnv(config=self.config)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning("ignoring non-integer %s=%r", name, raw)
        return default
    return default if value < minimum else value


def _native_available() -> bool:
    try:
        from forge_env.forge_env import ForgeEnv
    except ImportError:
        return False
    return ForgeEnv is not None


def _require_native() -> None:
    if not _native_available():
        pytest.skip("forge_env native extension not built (run `maturin develop`)")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except OSError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        LOGGER.warning("ignoring non-float %s=%r", name, raw)
        return default
    return default if value < minimum else value


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _detect_hardware() -> dict[str, str]:
    return {
        "arch": platform.machine() or "unknown",
        "cpu": _cpu_model(),
        "os": platform.system() or "unknown",
        "os_release": platform.release() or "unknown",
        "python": platform.python_version(),
    }


def _random_actions(rng: np.random.Generator, n_envs: int, n_actions: int) -> np.ndarray:
    return rng.integers(0, n_actions, size=n_envs, dtype=np.int64)


def _measure_async(
    n_envs: int,
    *,
    max_iters: int,
    warmup: int,
    seed: int,
    min_wall_s: float,
) -> dict[str, Any]:
    from forge_env.vecenv import ForgeAsyncVecEnv

    factory = _PicklableGymFactory(_MEASURE_CONFIG)
    vec = ForgeAsyncVecEnv([factory] * n_envs, context="spawn")
    try:
        n_actions = int(vec.action_space.n)
        rng = np.random.default_rng(seed)
        vec.reset(seed=seed)
        for _ in range(warmup):
            vec.step(_random_actions(rng, n_envs, n_actions))
        started = time.perf_counter()
        iters = 0
        while iters < max_iters:
            vec.step(_random_actions(rng, n_envs, n_actions))
            iters += 1
            if min_wall_s > 0.0 and time.perf_counter() - started >= min_wall_s:
                break
        wall_s = time.perf_counter() - started
    finally:
        vec.close()

    total_env_steps = n_envs * iters
    steps_per_sec = total_env_steps / wall_s if wall_s > 0 else 0.0
    return {
        "n_envs": n_envs,
        "iters": iters,
        "warmup": warmup,
        "wall_s": wall_s,
        "total_env_steps": total_env_steps,
        "steps_per_sec": steps_per_sec,
        "backend": "ForgeAsyncVecEnv",
        "action_sampling": "uniform_random",
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("wrote vecenv throughput report to %s", path)


def test_async_vecenv_smoke() -> None:
    """Native smoke: one async env completes a few random-action steps."""
    _require_native()
    row = _measure_async(
        1, max_iters=SMOKE_ITERS, warmup=1, seed=DEFAULT_SEED, min_wall_s=0.0
    )
    assert row["total_env_steps"] == SMOKE_ITERS
    assert row["steps_per_sec"] > 0.0


@pytest.mark.skipif(
    os.environ.get(ENV_RUN) != "1",
    reason=f"set {ENV_RUN}=1 to run the Karten SPS@N sweep",
)
def test_async_vecenv_sps_at_n() -> None:
    """Random-action SPS for ``n_envs ∈ {1,8,16,32,64}`` via ForgeAsyncVecEnv."""
    _require_native()
    base_iters = _env_int(ENV_ITERS, DEFAULT_ITERS)
    warmup = _env_int(ENV_WARMUP, DEFAULT_WARMUP)
    seed = _env_int(ENV_SEED, DEFAULT_SEED)
    min_wall_s = _env_float(ENV_MIN_WALL, DEFAULT_MIN_WALL_S)
    by_n: dict[str, Any] = {}
    for n_envs in DEFAULT_N_ENVS:
        LOGGER.info(
            "measuring ForgeAsyncVecEnv n=%d max_iters=%d min_wall_s=%.1f",
            n_envs,
            base_iters,
            min_wall_s,
        )
        by_n[str(n_envs)] = _measure_async(
            n_envs,
            max_iters=base_iters,
            warmup=warmup,
            seed=seed,
            min_wall_s=min_wall_s,
        )

    payload: dict[str, Any] = {
        "producer": "test_vecenv_throughput",
        "protocol": "Karten 2026 random-action SPS",
        "profile": "cloud_agent",
        "git_sha": _git_sha(),
        "seed": seed,
        "n_envs": list(DEFAULT_N_ENVS),
        "min_wall_s": min_wall_s,
        "by_n": by_n,
        "hardware": _detect_hardware(),
        "notes": (
            "ForgeAsyncVecEnv is process-parallel PyO3, not a JAX vmap of "
            "physics. Do not cite ForgeJaxEnv or RealisticFakeEnv as this number."
        ),
    }
    out = os.environ.get(ENV_OUT, str(DEFAULT_OUT))
    _write_report(Path(out), payload)
    for n_envs, row in by_n.items():
        assert row["steps_per_sec"] > 0.0, f"SPS @ {n_envs} was non-positive"
