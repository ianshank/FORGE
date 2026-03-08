"""Observation flattening utilities for FORGE environments.

Provides a single, reusable implementation for converting dict observations
(as returned by ForgeGymnasiumEnv) into flat numpy arrays suitable for
neural network input.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def flatten_obs(obs: dict[str, Any]) -> np.ndarray:
    """Flatten a dict observation into a 1-D float32 numpy array.

    Keys are sorted alphabetically for deterministic ordering, ensuring
    the same observation structure always produces the same flat layout.

    Args:
        obs: Dictionary mapping observation keys to array-like values.
            Each value is converted to a float32 array and ravelled.

    Returns:
        A 1-D float32 numpy array containing all observation values
        concatenated in sorted-key order.

    Raises:
        ValueError: If *obs* is empty.

    Examples:
        >>> flatten_obs({"position": [1, 2], "health": 0.5})
        array([0.5, 1. , 2. ], dtype=float32)
    """
    if not obs:
        msg = "Cannot flatten empty observation dict"
        raise ValueError(msg)

    parts = [
        np.asarray(obs[key], dtype=np.float32).ravel()
        for key in sorted(obs.keys())
    ]
    result = np.concatenate(parts)
    logger.debug("Flattened obs: %d keys -> %d dims", len(obs), result.shape[0])
    return result


def compute_obs_dim(env: Any) -> int:
    """Compute the flat observation dimensionality from an environment.

    Performs a probe reset to measure the flattened observation size.

    Args:
        env: A Gymnasium-compatible environment with ``reset()`` returning
            ``(obs_dict, info)``.

    Returns:
        The number of dimensions in the flattened observation.

    Raises:
        RuntimeError: If the environment reset fails.
    """
    try:
        obs, _info = env.reset()
    except Exception as exc:
        msg = f"Failed to probe observation dimensions: {exc}"
        raise RuntimeError(msg) from exc

    dim: int = int(flatten_obs(obs).shape[0])
    logger.info("Probed obs_dim=%d from environment", dim)
    return dim
