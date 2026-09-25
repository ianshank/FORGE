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
        import torch

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


def ensure_device_available(device: str) -> str:
    """Fail fast when an explicitly requested accelerator is unusable.

    ``"auto"`` and ``"cpu"`` always pass. ``"cuda"`` / ``"cuda:N"`` require a
    CUDA-enabled PyTorch build that can see a GPU (and, for ``cuda:N``, at
    least ``N + 1`` devices); ``"mps"`` requires the Apple MPS backend.
    Catching this before the environment and network are built turns an
    opaque ``torch`` error deep in the first update into an actionable one.

    Args:
        device: Requested device string.

    Returns:
        ``device`` unchanged, for call-site chaining.

    Raises:
        ValueError: If ``device`` is not a recognised device string, or the
            requested accelerator is not available in this environment.
    """
    kind, sep, index = device.partition(":")
    if kind in ("auto", "cpu") and not sep:
        return device
    if kind not in ("cuda", "mps") or (sep and (kind != "cuda" or not index.isdigit())):
        msg = f"Unknown device {device!r}; expected auto, cpu, cuda, cuda:N or mps"
        raise ValueError(msg)
    try:
        import torch
    except ImportError as exc:
        msg = f"device={device!r} requires PyTorch (pip install torch)"
        raise ValueError(msg) from exc

    if kind == "cuda":
        if not torch.cuda.is_available():
            if torch.version.cuda is None:
                hint = "this is a CPU-only PyTorch build; install a CUDA wheel"
            else:
                hint = (
                    f"PyTorch is built for CUDA {torch.version.cuda} but sees no GPU; "
                    "check `nvidia-smi`, the driver version, and CUDA_VISIBLE_DEVICES"
                )
            msg = f"device={device!r} requested but torch.cuda.is_available() is False ({hint})"
            raise ValueError(msg)
        count = torch.cuda.device_count()
        if index and int(index) >= count:
            msg = f"device={device!r} requested but only {count} CUDA device(s) are visible"
            raise ValueError(msg)
    elif not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        msg = f"device={device!r} requested but the MPS backend is not available"
        raise ValueError(msg)
    logger.info("Requested device %s is available", device)
    return device
