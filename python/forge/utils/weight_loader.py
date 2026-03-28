"""Reusable utility for downloading and caching model weights from HuggingFace Hub.

Provides a :class:`WeightLoader` that resolves files from any HuggingFace
repository, with lazy imports so that ``huggingface_hub`` and ``torch``
remain optional dependencies.

Usage::

    from forge.utils.weight_loader import WeightLoader, WeightLoaderConfig

    loader = WeightLoader(WeightLoaderConfig(repo_id="ianshank/mousedroid-weights"))
    checkpoint = loader.load_pt("rssm/final.pt", device="cpu")
    arrays = loader.load_npz("mcts/policy_init.npz")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_HF_MISSING_MSG = (
    "huggingface_hub is required for weight downloading. "
    "Install it with: pip install huggingface-hub>=0.20"
)

_TORCH_MISSING_MSG = (
    "torch is required to load PyTorch checkpoints. "
    "Install it with: pip install torch>=2.0"
)

DEFAULT_REPO_ID = "ianshank/mousedroid-weights"
DEFAULT_REVISION = "main"


@dataclass
class WeightLoaderConfig:
    """Configuration for :class:`WeightLoader`.

    Attributes:
        repo_id: HuggingFace repository identifier (e.g. ``"user/model"``).
        revision: Git revision (branch, tag, or commit hash).
        cache_dir: Local cache directory.  ``None`` uses the HuggingFace default.
        force_download: Re-download even if the file is already cached.
    """

    repo_id: str = DEFAULT_REPO_ID
    revision: str = DEFAULT_REVISION
    cache_dir: str | None = None
    force_download: bool = False


def _require_huggingface_hub() -> Any:
    """Import and return ``huggingface_hub``, raising a clear error if missing."""
    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_HF_MISSING_MSG) from exc
    return huggingface_hub


class WeightLoader:
    """Download and cache model weights from a HuggingFace repository.

    All network access is deferred to method calls — construction is free.

    Args:
        config: Loader configuration.
    """

    def __init__(self, config: WeightLoaderConfig | None = None) -> None:
        self._config = config or WeightLoaderConfig()
        logger.info(
            "WeightLoader initialised: repo=%s revision=%s",
            self._config.repo_id,
            self._config.revision,
        )

    @property
    def config(self) -> WeightLoaderConfig:
        """Return the loader configuration."""
        return self._config

    def resolve_path(self, filename: str) -> Path:
        """Download *filename* (if needed) and return its local cache path.

        Args:
            filename: Path within the repository (e.g. ``"rssm/final.pt"``).

        Returns:
            Absolute path to the cached file.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
        """
        hf = _require_huggingface_hub()
        local: str = hf.hf_hub_download(
            repo_id=self._config.repo_id,
            filename=filename,
            revision=self._config.revision,
            cache_dir=self._config.cache_dir,
            force_download=self._config.force_download,
        )
        path = Path(local)
        logger.debug("Resolved %s -> %s", filename, path)
        return path

    def load_npz(self, filename: str) -> dict[str, np.ndarray]:
        """Download and load a NumPy ``.npz`` archive.

        Args:
            filename: Path within the repository (e.g. ``"mcts/policy_init.npz"``).

        Returns:
            Dictionary mapping array names to :class:`numpy.ndarray` values.
        """
        path = self.resolve_path(filename)
        data = dict(np.load(str(path), allow_pickle=False))
        logger.info(
            "Loaded .npz %s: %d arrays (%s)",
            filename,
            len(data),
            ", ".join(f"{k}={v.shape}" for k, v in data.items()),
        )
        return data

    def load_pt(self, filename: str, device: str = "cpu") -> dict[str, Any]:
        """Download and load a PyTorch ``.pt`` checkpoint.

        Args:
            filename: Path within the repository (e.g. ``"rssm/final.pt"``).
            device: Target device for tensor mapping.

        Returns:
            The checkpoint dictionary.

        Raises:
            ImportError: If ``torch`` is not installed.
        """
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(_TORCH_MISSING_MSG) from exc

        path = self.resolve_path(filename)
        checkpoint: dict[str, Any] = torch.load(
            str(path), map_location=device, weights_only=True
        )
        logger.info("Loaded .pt %s on device=%s", filename, device)
        return checkpoint

    def list_files(self) -> list[str]:
        """List all files in the remote repository.

        Returns:
            Sorted list of filenames in the repository.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
        """
        hf = _require_huggingface_hub()
        info = hf.model_info(
            self._config.repo_id,
            revision=self._config.revision,
        )
        files = sorted(s.rfilename for s in (info.siblings or []))
        logger.info("Listed %d files in %s", len(files), self._config.repo_id)
        return files
