"""FORGE Environment -- Fast Open-source Runtime for Generalist Environments.

Provides Gymnasium-compatible single-agent and PettingZoo-compatible
multi-agent environments backed by a high-performance Rust simulation engine.

Submodules:
    gymnasium_env  -- Gymnasium single-agent wrapper
    pettingzoo_env -- PettingZoo Parallel API multi-agent wrapper
    jax_env        -- JAX-vectorized batched environment
    wrappers       -- Observation/reward/time-limit wrappers
    utils          -- Factory, validation, and benchmarking utilities
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# Native Rust extension (built via maturin)
try:
    from forge_env.forge_env import ForgeEnv
except ImportError:
    ForgeEnv = None  # native ext not built yet
    logger.debug("Native forge_env module not available; running in pure-Python mode.")

# Convenience re-exports
from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: E402
from forge_env.pettingzoo_env import ForgeParallelEnv  # noqa: E402
from forge_env.utils import benchmark_fps, check_env, make_env  # noqa: E402

__all__ = [
    "ForgeEnv",
    "ForgeGymnasiumEnv",
    "ForgeParallelEnv",
    "__version__",
    "benchmark_fps",
    "check_env",
    "make_env",
]
