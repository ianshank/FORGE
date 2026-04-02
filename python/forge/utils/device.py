"""Device detection utilities for compute backend selection."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Detect and return the best available compute device.

    Returns:
        One of "cuda", "mps", or "cpu".
    """
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            logger.info("CUDA device detected")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS device detected")
            return "mps"
    except ImportError:
        logger.debug("PyTorch not installed, defaulting to CPU")

    logger.info("Using CPU device")
    return "cpu"


def is_gpu_available() -> bool:
    """Check whether any GPU device is available.

    Returns:
        True if CUDA or MPS is available, False otherwise.
    """
    device = get_device()
    return device in ("cuda", "mps")
