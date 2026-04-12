"""MuZero world model configuration.

All dimensions and hyperparameters are configurable via dataclass fields.
No hard-coded values — constants are module-level defaults that can be
overridden through constructor arguments or TOML configuration.

Usage::

    from forge.models.muzero_config import MuZeroConfig

    config = MuZeroConfig(obs_dim=920, action_dim=75, latent_dim=256)
"""
from __future__ import annotations

__all__ = ["MuZeroConfig"]

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- Module-level defaults (override via dataclass fields) ---

DEFAULT_LATENT_DIM: int = 256
DEFAULT_HIDDEN_DIM: int = 256
DEFAULT_NUM_BLOCKS: int = 4
DEFAULT_REWARD_SUPPORT_SIZE: int = 31
DEFAULT_VALUE_SUPPORT_SIZE: int = 31
DEFAULT_DISCOUNT: float = 0.997
DEFAULT_NUM_UNROLL_STEPS: int = 5
DEFAULT_TD_STEPS: int = 10
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_LEARNING_RATE: float = 3e-4

# Grid observation defaults (matching FORGE's 2*vision_radius+1)
DEFAULT_GRID_HEIGHT: int = 11
DEFAULT_GRID_WIDTH: int = 11
DEFAULT_GRID_CHANNELS: int = 7  # OBS_FEATURES_PER_TILE

# Vector observation defaults (inventory + scalars + comm + day + task_progress + drone + agri)
DEFAULT_VECTOR_DIM: int = 73

# CNN feature extractor defaults
DEFAULT_CNN_CHANNELS: tuple[int, ...] = (32, 64, 64)
DEFAULT_CNN_KERNEL_SIZES: tuple[int, ...] = (3, 3, 3)
DEFAULT_CNN_STRIDES: tuple[int, ...] = (1, 1, 1)


@dataclass
class MuZeroConfig:
    """Configuration for the MuZero world model and training pipeline.

    Observation dimensions should be derived from the FORGE environment
    at runtime (e.g., via ``env.observation_space`` and ``env.action_space``).
    Grid and vector dimensions can be computed from ``ForgeConfig`` parameters.

    Attributes:
        obs_dim: Total flat observation dimensionality.
        action_dim: Number of discrete actions.
        latent_dim: Dimensionality of the latent state vector.
        hidden_dim: Width of hidden layers in all networks.
        num_blocks: Number of residual blocks in representation and dynamics.
        reward_support_size: Number of bins for categorical reward prediction.
        value_support_size: Number of bins for categorical value prediction.
        discount: Reward discount factor for n-step returns.
        num_unroll_steps: Number of dynamics steps unrolled during training (K).
        td_steps: Number of steps for n-step return computation.
        weight_decay: L2 regularization coefficient.
        learning_rate: Adam optimizer learning rate.
        grid_height: Height of the ego-centric grid observation.
        grid_width: Width of the ego-centric grid observation.
        grid_channels: Number of feature channels per grid tile.
        vector_dim: Dimensionality of the non-spatial observation vector.
        cnn_channels: Output channels for each CNN layer.
        cnn_kernel_sizes: Kernel sizes for each CNN layer.
        cnn_strides: Strides for each CNN layer.
        device: Torch device string.
    """

    obs_dim: int = 0  # Must be set from environment
    action_dim: int = 0  # Must be set from environment
    latent_dim: int = DEFAULT_LATENT_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_blocks: int = DEFAULT_NUM_BLOCKS
    reward_support_size: int = DEFAULT_REWARD_SUPPORT_SIZE
    value_support_size: int = DEFAULT_VALUE_SUPPORT_SIZE
    discount: float = DEFAULT_DISCOUNT
    num_unroll_steps: int = DEFAULT_NUM_UNROLL_STEPS
    td_steps: int = DEFAULT_TD_STEPS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    learning_rate: float = DEFAULT_LEARNING_RATE
    grid_height: int = DEFAULT_GRID_HEIGHT
    grid_width: int = DEFAULT_GRID_WIDTH
    grid_channels: int = DEFAULT_GRID_CHANNELS
    vector_dim: int = DEFAULT_VECTOR_DIM
    cnn_channels: tuple[int, ...] = DEFAULT_CNN_CHANNELS
    cnn_kernel_sizes: tuple[int, ...] = DEFAULT_CNN_KERNEL_SIZES
    cnn_strides: tuple[int, ...] = DEFAULT_CNN_STRIDES
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate configuration and compute derived fields."""
        grid_dim = self.grid_height * self.grid_width * self.grid_channels
        expected_obs = grid_dim + self.vector_dim

        if self.obs_dim == 0:
            self.obs_dim = expected_obs
            logger.debug(
                "MuZeroConfig: obs_dim auto-computed as %d (grid=%d + vector=%d)",
                self.obs_dim,
                grid_dim,
                self.vector_dim,
            )
        elif self.obs_dim != expected_obs:
            logger.warning(
                "MuZeroConfig: obs_dim=%d differs from grid+vector=%d; "
                "spatial encoding may be misaligned.",
                self.obs_dim,
                expected_obs,
            )

        if self.action_dim <= 0:
            logger.warning(
                "MuZeroConfig: action_dim=%d is non-positive; "
                "must be set from environment before use.",
                self.action_dim,
            )

        if len(self.cnn_channels) != len(self.cnn_kernel_sizes):
            msg = (
                f"cnn_channels length ({len(self.cnn_channels)}) must match "
                f"cnn_kernel_sizes length ({len(self.cnn_kernel_sizes)})"
            )
            raise ValueError(msg)

        if len(self.cnn_channels) != len(self.cnn_strides):
            msg = (
                f"cnn_channels length ({len(self.cnn_channels)}) must match "
                f"cnn_strides length ({len(self.cnn_strides)})"
            )
            raise ValueError(msg)

    @property
    def grid_flat_dim(self) -> int:
        """Total number of elements in the flattened grid observation."""
        return self.grid_height * self.grid_width * self.grid_channels

    @property
    def reward_support_range(self) -> tuple[int, int]:
        """Min and max values of the categorical reward support."""
        half = self.reward_support_size // 2
        return (-half, half)

    @property
    def value_support_range(self) -> tuple[int, int]:
        """Min and max values of the categorical value support."""
        half = self.value_support_size // 2
        return (-half, half)
