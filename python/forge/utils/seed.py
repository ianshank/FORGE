"""Seed utilities for reproducible experiments.

Two layers, deliberately separated:

* :func:`set_all_seeds` — the *cheap* default. Seeds Python's ``random``,
  NumPy and (when importable) PyTorch on both CPU and every visible CUDA
  device. Costs nothing at runtime and is safe to call from library code.
* The opt-in ``deterministic=True`` switch — additionally forces PyTorch's
  strict-determinism mode (deterministic kernel selection, cuDNN
  determinism, no cuDNN autotuning) and pins ``PYTHONHASHSEED`` for child
  processes. This trades throughput for bit-reproducibility, so it is
  **off by default**; callers that need reproducible numerics (regression
  baselines, published experiment runs) opt in explicitly.

No hard-coded values: the default for the switch and the environment
variable names/values it writes are module-level constants.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CUBLAS_WORKSPACE_CONFIG_ENV_VAR",
    "DEFAULT_CUBLAS_WORKSPACE_CONFIG",
    "DEFAULT_DETERMINISTIC",
    "MAX_SEED",
    "PYTHONHASHSEED_ENV_VAR",
    "derive_seed",
    "enable_torch_determinism",
    "seed_torch",
    "set_all_seeds",
]

MAX_SEED = 2**32 - 1

#: Default for the ``deterministic`` switch on :func:`set_all_seeds` and
#: :func:`seed_torch`. ``False`` keeps the historical behaviour (and the
#: historical throughput) for every existing caller; strict determinism is
#: something a caller asks for, never something it inherits silently.
DEFAULT_DETERMINISTIC: Final[bool] = False

#: Hash randomisation for `str`/`bytes` is fixed at interpreter start, so
#: writing this only affects *child* processes (DataLoader workers,
#: subprocess-spawned trainers). Documented rather than silently assumed.
PYTHONHASHSEED_ENV_VAR: Final[str] = "PYTHONHASHSEED"

#: cuBLAS needs a fixed workspace size before it will honour
#: ``torch.use_deterministic_algorithms(True)`` for CUDA matmuls; without
#: it PyTorch raises at the first offending op. Only set when the caller
#: opts into strict determinism, and never overrides an operator's own value.
CUBLAS_WORKSPACE_CONFIG_ENV_VAR: Final[str] = "CUBLAS_WORKSPACE_CONFIG"
DEFAULT_CUBLAS_WORKSPACE_CONFIG: Final[str] = ":4096:8"


def _import_torch() -> Any | None:
    """Return the ``torch`` module, or ``None`` when it is not installed.

    Torch is an optional dependency (the ``sb3`` / ``cleanrl`` /
    ``minecraft`` extras), so every seeding path has to tolerate its
    absence.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch


def enable_torch_determinism(torch: Any) -> None:
    """Force PyTorch into strict-determinism mode.

    Sets ``torch.use_deterministic_algorithms(True)`` plus the cuDNN
    determinism/autotune flags, and pins ``CUBLAS_WORKSPACE_CONFIG`` if the
    operator has not already chosen a value.

    This is best-effort: an old torch build without one of these knobs logs
    a warning rather than raising, so opting into determinism can never turn
    a working run into a crashed one.

    Args:
        torch: The imported ``torch`` module.
    """
    os.environ.setdefault(CUBLAS_WORKSPACE_CONFIG_ENV_VAR, DEFAULT_CUBLAS_WORKSPACE_CONFIG)

    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError) as exc:
        logger.warning("torch.use_deterministic_algorithms(True) unavailable: %s", exc)

    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is None:
        logger.debug("torch.backends.cudnn unavailable; skipping cuDNN determinism flags")
        return
    try:
        cudnn.deterministic = True
        cudnn.benchmark = False
    except (AttributeError, RuntimeError) as exc:  # pragma: no cover - exotic torch builds
        logger.warning("could not set torch.backends.cudnn determinism flags: %s", exc)


def seed_torch(seed: int, *, deterministic: bool = DEFAULT_DETERMINISTIC) -> bool:
    """Seed PyTorch's CPU and CUDA generators.

    ``torch.manual_seed`` is called **unconditionally** whenever torch is
    importable — it seeds the CPU generator, which is the only generator a
    CPU-only host (every CI runner) ever uses.
    ``torch.cuda.manual_seed_all`` is additionally called when CUDA is
    actually available.

    Args:
        seed: The seed value to use.
        deterministic: When ``True``, also apply
            :func:`enable_torch_determinism`. Defaults to
            :data:`DEFAULT_DETERMINISTIC` (off) so existing callers keep
            their current behaviour and performance.

    Returns:
        ``True`` if torch was importable and seeded, ``False`` otherwise.
    """
    torch = _import_torch()
    if torch is None:
        logger.debug("torch not installed; skipping torch seeding")
        return False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        enable_torch_determinism(torch)
    return True


def set_all_seeds(seed: int, *, deterministic: bool = DEFAULT_DETERMINISTIC) -> None:
    """Set seeds for all random number generators.

    Seeds Python's ``random`` module, NumPy, and — when torch is installed —
    ``torch.manual_seed`` plus ``torch.cuda.manual_seed_all`` on CUDA hosts.

    Args:
        seed: The seed value to use.
        deterministic: Opt into strict determinism. In addition to seeding,
            this sets ``torch.use_deterministic_algorithms(True)``, the cuDNN
            determinism flags, and ``PYTHONHASHSEED`` (which only affects
            child processes — the parent's hash seed is fixed at interpreter
            start). Off by default; it costs throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch_seeded = seed_torch(seed, deterministic=deterministic)

    if deterministic:
        os.environ[PYTHONHASHSEED_ENV_VAR] = str(seed)

    logger.info(
        "All seeds set to %d (torch=%s, deterministic=%s)",
        seed,
        torch_seeded,
        deterministic,
    )


def derive_seed(base: int, component: str) -> int:
    """Derive a deterministic seed from a base seed and component name.

    Args:
        base: The base seed value.
        component: A string identifier for the component.

    Returns:
        A derived integer seed.
    """
    hash_input = f"{base}:{component}".encode()
    hash_digest = hashlib.sha256(hash_input).hexdigest()
    return int(hash_digest[:8], 16) % (MAX_SEED + 1)
