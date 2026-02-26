"""FORGE Environment — Fast Open-source Runtime for Generalist Environments.

Provides Gymnasium-compatible single-agent and PettingZoo-compatible
multi-agent environments backed by a high-performance Rust simulation engine.

Submodules:
    gymnasium_env  — Gymnasium single-agent wrapper
    pettingzoo_env — PettingZoo Parallel API multi-agent wrapper
    jax_env        — JAX-vectorized batched environment
    wrappers       — Observation/reward/time-limit wrappers
    utils          — Factory, validation, and benchmarking utilities
"""

__version__ = "0.1.0"

# Native Rust extension (built via maturin)
try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None

# Convenience re-exports
from forge_env.gymnasium_env import ForgeGymnasiumEnv
from forge_env.pettingzoo_env import ForgeParallelEnv
from forge_env.utils import make_env, check_env, benchmark_fps

__all__ = [
    "ForgeGymnasiumEnv",
    "ForgeParallelEnv",
    "make_env",
    "check_env",
    "benchmark_fps",
    "__version__",
]
