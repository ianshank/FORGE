"""Action and observation space adapters for FORGE↔MangoMAS bridge.

ActionSpaceAdapter: continuous (MangoMAS) ↔ discrete (FORGE) action mapping.
ObservationAdapter: FORGE grid observations → flat state vectors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from forge.mangomas.config import (
    ActionAdapterConfig,
    ObservationAdapterConfig,
)

logger = logging.getLogger(__name__)


class ActionSpaceAdapter:
    """Bidirectional continuous↔discrete action space mapping.

    Car mode: 2D continuous (steering, throttle) → FORGE discrete
    Drone mode: 4D continuous (dx, dy, dz, yaw) → FORGE discrete
    """

    def __init__(
        self,
        config: ActionAdapterConfig | None = None,
        platform: str = "drone",
    ) -> None:
        self.config = config or ActionAdapterConfig()
        self.platform = platform
        self._bins = self.config.bins_per_axis
        self._lo = self.config.continuous_range_min
        self._hi = self.config.continuous_range_max
        logger.debug("ActionSpaceAdapter: platform=%s, bins=%d", platform, self._bins)

    @property
    def continuous_dims(self) -> int:
        """Number of continuous action dimensions."""
        return 2 if self.platform == "car" else 4

    def continuous_to_discrete(self, continuous: np.ndarray) -> int:
        """Map a continuous action vector to a FORGE discrete action ID."""
        dims = self.continuous_dims
        c = np.clip(continuous[:dims], self._lo, self._hi)
        bins = self._quantize(c)
        # Flatten multi-dim bin indices to single integer
        action_id = 0
        for i, b in enumerate(bins):
            action_id += int(b) * (self._bins ** (dims - 1 - i))
        return action_id

    def discrete_to_continuous(self, action_id: int) -> np.ndarray:
        """Map a FORGE discrete action ID to a continuous action vector."""
        dims = self.continuous_dims
        bins = []
        remaining = action_id
        for i in range(dims):
            divisor = self._bins ** (dims - 1 - i)
            bins.append(remaining // divisor)
            remaining %= divisor
        return np.array([self._bin_center(b) for b in bins], dtype=np.float32)  # type: ignore[no-any-return]

    def _quantize(self, values: np.ndarray) -> list[int]:
        """Quantize continuous values to bin indices."""
        normalized = (values - self._lo) / (self._hi - self._lo)
        indices = np.clip((normalized * self._bins).astype(int), 0, self._bins - 1)
        return indices.tolist()  # type: ignore[no-any-return]

    def _bin_center(self, bin_idx: int) -> float:
        """Get the center value for a bin index."""
        bin_idx = max(0, min(bin_idx, self._bins - 1))
        step = (self._hi - self._lo) / self._bins
        return self._lo + step * (bin_idx + 0.5)

    @property
    def total_action_space(self) -> int:
        """Total number of discrete actions in the mapped space."""
        return self._bins**self.continuous_dims  # type: ignore[no-any-return]


class ObservationAdapter:
    """Converts FORGE observations to flat state vectors for MangoMAS.

    Extracts grid summary statistics, scalar fields, inventory,
    and optional drone fields.
    """

    def __init__(
        self,
        config: ObservationAdapterConfig | None = None,
        platform: str = "drone",
    ) -> None:
        self.config = config or ObservationAdapterConfig()
        self.platform = platform
        self._include_drone = platform == "drone" or self.config.include_drone_fields
        self._grid_channels = self.config.grid_channels
        self._scalar_fields = self.config.scalar_fields
        self._inventory_fields = self.config.inventory_fields
        self._drone_fields = self.config.drone_fields
        logger.debug(
            "ObservationAdapter: platform=%s, drone_fields=%s",
            platform,
            self._include_drone,
        )

    @property
    def output_dim(self) -> int:
        """Dimensionality of the output state vector."""
        base = self._grid_channels + self._scalar_fields + self._inventory_fields
        return base + self._drone_fields if self._include_drone else base

    def adapt(self, obs: dict[str, Any]) -> np.ndarray:
        """Convert a FORGE observation dict to a flat state vector."""
        parts: list[np.ndarray] = []

        # Grid summary: count agents, resources, obstacles + 8 terrain types
        grid = np.array(obs.get("grid_view", []), dtype=np.float32)
        if grid.size > 0:
            parts.append(self._grid_summary(grid))
        else:
            parts.append(np.zeros(self._grid_channels, dtype=np.float32))

        # Scalar fields
        parts.append(
            np.array(
                [
                    float(obs.get("health", 1.0)),
                    float(obs.get("stamina", 1.0)),
                    float(obs.get("position", [0, 0])[0]) / self.config.position_scale,  # normalize
                    float(obs.get("position", [0, 0])[1]) / self.config.position_scale,
                    float(obs.get("day_phase", 0.0)),
                ],
                dtype=np.float32,
            )
        )

        # Inventory summary
        inv = obs.get("inventory", {})
        if isinstance(inv, dict):
            parts.append(
                np.array(
                    [float(len(inv)), float(sum(inv.values()))],
                    dtype=np.float32,
                )
            )
        else:
            parts.append(np.zeros(2, dtype=np.float32))

        # Drone fields
        if self._include_drone:
            parts.append(
                np.array(
                    [
                        float(obs.get("altitude", 0.0)),
                        float(obs.get("battery", 1.0)),
                        float(obs.get("morphology", 0.0)),
                        float(obs.get("heading", 0.0)),
                    ],
                    dtype=np.float32,
                )
            )

        return np.concatenate(parts)  # type: ignore[no-any-return]

    def _grid_summary(self, grid: np.ndarray) -> np.ndarray:
        """Extract summary statistics from the observation grid."""
        if grid.ndim < 3:
            return np.zeros(self._grid_channels, dtype=np.float32)  # type: ignore[no-any-return]
        # Channels: agents(0), resources(1), obstacles(2), terrain(3-10)
        n_cells = float(grid.shape[0] * grid.shape[1])
        if n_cells == 0:
            return np.zeros(self._grid_channels, dtype=np.float32)  # type: ignore[no-any-return]
        summary = np.zeros(self._grid_channels, dtype=np.float32)
        n_channels = min(grid.shape[2], self._grid_channels)
        for c in range(min(3, n_channels)):
            summary[c] = float(np.sum(grid[:, :, c] > 0)) / n_cells
        for c in range(3, n_channels):
            summary[c] = float(np.mean(grid[:, :, c]))
        return summary  # type: ignore[no-any-return]
