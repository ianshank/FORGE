"""Weight exporter for MangoMAS integration.

Exports trained weights (.npz) and configuration (JSON) bundles
for transfer to MangoMAS agent initialization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExportManifest:
    """Manifest describing an exported weight bundle."""

    version: str = "1.0"
    platform: str = "drone"
    components: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class WeightExporter:
    """Exports trained weight bundles for MangoMAS transfer.

    Creates a directory with:
    - bdi_weights.npz (BDI GRU+MLP weights)
    - constitutional_weights.npz (policy/value with constraint awareness)
    - rssm_weights.npz (GRU transition + heads)
    - mcts_config.json (optimal MCTS hyperparameters)
    - curiosity_weights.json (optimized curiosity channel blend)
    - curriculum_state.json (current curriculum progress)
    - manifest.json (bundle metadata)
    """

    def __init__(self, output_dir: str | Path, platform: str = "drone") -> None:
        self.output_dir = Path(output_dir)
        self.platform = platform
        self._manifest = ExportManifest(platform=platform)
        logger.info("WeightExporter: output_dir=%s, platform=%s", output_dir, platform)

    def export_bdi_weights(self, weights: dict[str, np.ndarray]) -> Path:
        """Export BDI GRU+MLP weights."""
        path = self.output_dir / "bdi_weights.npz"
        self._ensure_dir()
        np.savez(str(path), **weights)  # type: ignore[arg-type]
        self._manifest.components.append("bdi")
        self._manifest.metadata["bdi_arrays"] = list(weights.keys())
        logger.info("Exported BDI weights: %s", path)
        return path

    def export_constitutional_weights(self, weights: dict[str, np.ndarray]) -> Path:
        """Export constitutional RL policy/value weights."""
        path = self.output_dir / "constitutional_weights.npz"
        self._ensure_dir()
        np.savez(str(path), **weights)  # type: ignore[arg-type]
        self._manifest.components.append("constitutional")
        self._manifest.metadata["constitutional_arrays"] = list(weights.keys())
        logger.info("Exported constitutional weights: %s", path)
        return path

    def export_rssm_weights(self, weights: dict[str, np.ndarray]) -> Path:
        """Export RSSM world model weights."""
        path = self.output_dir / "rssm_weights.npz"
        self._ensure_dir()
        np.savez(str(path), **weights)  # type: ignore[arg-type]
        self._manifest.components.append("rssm")
        self._manifest.metadata["rssm_arrays"] = list(weights.keys())
        logger.info("Exported RSSM weights: %s", path)
        return path

    def export_mcts_config(self, config: dict[str, Any]) -> Path:
        """Export optimal MCTS configuration."""
        path = self.output_dir / "mcts_config.json"
        self._ensure_dir()
        with path.open("w") as f:
            json.dump(config, f, indent=2)
        self._manifest.components.append("mcts")
        logger.info("Exported MCTS config: %s", path)
        return path

    def export_curiosity_weights(self, weights: dict[str, float]) -> Path:
        """Export optimized curiosity channel weights."""
        path = self.output_dir / "curiosity_weights.json"
        self._ensure_dir()
        with path.open("w") as f:
            json.dump(weights, f, indent=2)
        self._manifest.components.append("curiosity")
        logger.info("Exported curiosity weights: %s", path)
        return path

    def export_muzero_weights(self, weights: dict[str, np.ndarray]) -> Path:
        """Export MuZero world model weights."""
        path = self.output_dir / "muzero_weights.npz"
        self._ensure_dir()
        np.savez(str(path), **weights)  # type: ignore[arg-type]
        self._manifest.components.append("muzero")
        self._manifest.metadata["muzero_arrays"] = list(weights.keys())
        logger.info("Exported MuZero weights: %s", path)
        return path

    def export_curriculum_state(self, state: dict[str, Any]) -> Path:
        """Export curriculum state."""
        path = self.output_dir / "curriculum_state.json"
        self._ensure_dir()
        with path.open("w") as f:
            json.dump(state, f, indent=2)
        self._manifest.components.append("curriculum")
        logger.info("Exported curriculum state: %s", path)
        return path

    def finalize(self) -> Path:
        """Write the manifest and return the output directory."""
        manifest_path = self.output_dir / "manifest.json"
        self._ensure_dir()
        with manifest_path.open("w") as f:
            json.dump(
                {
                    "version": self._manifest.version,
                    "platform": self._manifest.platform,
                    "components": self._manifest.components,
                    "metadata": self._manifest.metadata,
                },
                f,
                indent=2,
            )
        logger.info(
            "Export finalized: %d components in %s",
            len(self._manifest.components),
            self.output_dir,
        )
        return self.output_dir

    def _ensure_dir(self) -> None:
        """Ensure the output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
