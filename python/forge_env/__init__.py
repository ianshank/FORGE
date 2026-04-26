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
    ForgeEnv = None
    logger.debug("Native forge_env module not available; running in pure-Python mode.")

# Convenience re-exports. These are intentionally placed after the native
# extension probe above so that callers see a consistent error path
# ("Native module unavailable") before any pure-Python wrapper raises a
# secondary ImportError. The `# noqa: E402` is therefore deliberate, not
# accidental.
from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: E402
from forge_env.pettingzoo_env import ForgeParallelEnv  # noqa: E402
from forge_env.utils import benchmark_fps, check_env, make_env, seed_everything  # noqa: E402
from forge_env.vecenv import ForgeAsyncVecEnv, ForgeSyncVecEnv, make_forge_vec_env  # noqa: E402

# Optional imports — guarded so forge_env remains importable without SB3/torch.
ForgeGridCnnExtractor = None
ForgeObsExtractor = None
try:
    import forge_env.feature_extractors as _feature_extractors
except ImportError:
    _HAS_EXTRACTORS = False
else:
    _HAS_EXTRACTORS = _feature_extractors.HAS_TORCH and _feature_extractors.HAS_SB3
    if _HAS_EXTRACTORS:
        ForgeGridCnnExtractor = _feature_extractors.ForgeGridCnnExtractor
        ForgeObsExtractor = _feature_extractors.ForgeObsExtractor

ForgeCurriculumCallback = None
ForgeMetricsCallback = None
try:
    import forge_env.sb3_callbacks as _sb3_callbacks
except ImportError:
    _HAS_CALLBACKS = False
else:
    _HAS_CALLBACKS = _sb3_callbacks.HAS_SB3
    if _HAS_CALLBACKS:
        ForgeCurriculumCallback = _sb3_callbacks.ForgeCurriculumCallback
        ForgeMetricsCallback = _sb3_callbacks.ForgeMetricsCallback

__all__ = [
    "ForgeAsyncVecEnv",
    "ForgeCurriculumCallback",
    "ForgeEnv",
    "ForgeGridCnnExtractor",
    "ForgeGymnasiumEnv",
    "ForgeMetricsCallback",
    "ForgeObsExtractor",
    "ForgeParallelEnv",
    "ForgeSyncVecEnv",
    "__version__",
    "benchmark_fps",
    "check_env",
    "make_env",
    "make_forge_vec_env",
    "seed_everything",
]
