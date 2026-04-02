"""Seed utilities for reproducible experiments."""

from __future__ import annotations

import hashlib
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

MAX_SEED = 2**32 - 1


def set_all_seeds(seed: int) -> None:
    """Set seeds for all random number generators.

    Sets seeds for Python's random module and NumPy.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info("All seeds set to %d", seed)


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
