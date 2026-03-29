"""Custom SB3 feature extractors for FORGE's Dict observation space.

FORGE observations contain heterogeneous data: a 3-D grid view (suitable
for a CNN) and several scalar / vector fields (health, stamina, position,
inventory, day_phase, task_progress).  Standard SB3 policies handle Dict
spaces with a ``CombinedExtractor``, but this module provides FORGE-specific
extractors that expose cleaner CNN architectures and dynamic sizing.

Classes
-------
ForgeGridCnnExtractor
    CNN-only extractor for the ``grid_view`` key.  Useful when scalar
    fields are handled separately or when wrapping with
    :class:`~forge_env.wrappers.FlattenObservationWrapper` first.
ForgeObsExtractor
    Combined extractor: CNN branch for ``grid_view`` + MLP branch for all
    remaining (scalar/vector) keys.  The output ``features_dim`` is the
    sum of both branches and is computed dynamically from the observation
    space — no hard-coded sizes.

Both classes require PyTorch and Stable Baselines 3 to be installed.  They
degrade gracefully with an :class:`ImportError` if either is missing.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    BaseFeaturesExtractor = object

if TYPE_CHECKING:
    import gymnasium as gym

__all__ = [
    "ForgeGridCnnExtractor",
    "ForgeObsExtractor",
]

# Keys that contain spatial (grid) data processed by the CNN branch.
_GRID_KEYS: frozenset[str] = frozenset({"grid_view"})

# Keys that are skipped entirely (variable-length or non-numeric).
_SKIP_KEYS: frozenset[str] = frozenset({"messages"})


def _require_torch_sb3() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for FORGE feature extractors. "
            "Install with: pip install torch"
        )
    if not HAS_SB3:
        raise ImportError(
            "Stable Baselines 3 is required for FORGE feature extractors. "
            "Install with: pip install stable-baselines3"
        )


def _build_cnn(
    in_channels: int,
    cnn_channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    strides: tuple[int, ...],
) -> nn.Sequential:
    """Build a configurable CNN with ReLU activations.

    Args:
        in_channels: Number of input channels (e.g. 7 for grid_view).
        cnn_channels: Output channels per convolutional layer.
        kernel_sizes: Kernel size per layer; must have same length as
            *cnn_channels*.
        strides: Stride per layer; must have same length as *cnn_channels*.

    Returns:
        A :class:`torch.nn.Sequential` CNN ending with a :class:`Flatten`.
    """
    layers: list[nn.Module] = []
    current_channels = in_channels
    for out_ch, k, s in zip(cnn_channels, kernel_sizes, strides):
        layers.append(nn.Conv2d(current_channels, out_ch, kernel_size=k, stride=s))
        layers.append(nn.ReLU())
        current_channels = out_ch
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


def _cnn_output_dim(
    cnn: nn.Sequential,
    grid_h: int,
    grid_w: int,
    in_channels: int,
    device: torch.device,
) -> int:
    """Compute the flat output dimension of a CNN by running a dummy forward pass.

    Args:
        cnn: The CNN module.
        grid_h: Grid view height.
        grid_w: Grid view width.
        in_channels: Number of input channels.
        device: Device to run the dummy pass on.

    Returns:
        Integer flat output dimension.
    """
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, grid_h, grid_w, device=device)
        out = cnn(dummy)
    return int(out.shape[1])


def _scalar_obs_dim(observation_space: gym.spaces.Dict) -> int:
    """Compute the total flattened dimension of non-grid observation keys.

    Args:
        observation_space: The Dict observation space.

    Returns:
        Total number of scalar features (int).
    """
    total = 0
    for key, space in observation_space.spaces.items():
        if key in _GRID_KEYS or key in _SKIP_KEYS:
            continue
        total += int(math.prod(space.shape)) if space.shape else 1
    return total


# ---------------------------------------------------------------------------
# ForgeGridCnnExtractor
# ---------------------------------------------------------------------------


class ForgeGridCnnExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for the FORGE ``grid_view`` observation.

    Processes only the ``grid_view`` key of the Dict observation space.
    All other keys are ignored.  Suitable when scalar observations have
    already been merged (e.g. via
    :class:`~forge_env.wrappers.FlattenObservationWrapper`).

    Args:
        observation_space: A ``gymnasium.spaces.Dict`` that must contain a
            ``"grid_view"`` key with shape ``(H, W, C)``.
        features_dim: Dimensionality of the output feature vector.
            The CNN's raw output is projected to this size via a linear
            layer.
        cnn_channels: Number of output channels per convolutional layer.
        cnn_kernel_sizes: Kernel size per layer.
        cnn_strides: Stride per layer.

    Raises:
        ImportError: If PyTorch or Stable Baselines 3 is not installed.
        KeyError: If ``"grid_view"`` is not in the observation space.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_channels: tuple[int, ...] = (32, 64),
        cnn_kernel_sizes: tuple[int, ...] = (3, 3),
        cnn_strides: tuple[int, ...] = (1, 1),
    ) -> None:
        _require_torch_sb3()
        super().__init__(observation_space, features_dim=features_dim)

        if "grid_view" not in observation_space.spaces:
            raise KeyError(
                "'grid_view' key not found in observation_space. "
                "ForgeGridCnnExtractor requires a Dict space with a 'grid_view' entry."
            )

        grid_space = observation_space.spaces["grid_view"]
        grid_h, grid_w, in_channels = grid_space.shape

        self._cnn = _build_cnn(in_channels, cnn_channels, cnn_kernel_sizes, cnn_strides)

        # Determine CNN flat output size.
        device = torch.device("cpu")
        cnn_out_dim = _cnn_output_dim(
            self._cnn, grid_h, grid_w, in_channels, device
        )

        self._linear = nn.Linear(cnn_out_dim, features_dim)
        logger.debug(
            "ForgeGridCnnExtractor: grid=(%d,%d,%d) cnn_out=%d features_dim=%d",
            grid_h, grid_w, in_channels, cnn_out_dim, features_dim,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from the ``grid_view`` observation.

        Args:
            observations: Batch of observations (Dict[str, Tensor]).

        Returns:
            Feature tensor of shape ``(batch, features_dim)``.
        """
        # grid_view arrives as (batch, H, W, C) — permute to (batch, C, H, W)
        grid = observations["grid_view"].float() / 255.0
        grid = grid.permute(0, 3, 1, 2)
        return self._linear(self._cnn(grid))


# ---------------------------------------------------------------------------
# ForgeObsExtractor
# ---------------------------------------------------------------------------


class ForgeObsExtractor(BaseFeaturesExtractor):
    """Combined CNN + MLP feature extractor for FORGE Dict observations.

    Two branches:

    * **CNN branch** — processes ``grid_view`` (HxWxC) with configurable
      convolutional layers.
    * **MLP branch** — flattens and concatenates all remaining keys
      (health, stamina, position, inventory, day_phase, task_progress, …)
      and passes them through a configurable MLP.

    The outputs of both branches are concatenated to produce the final
    feature vector.  The total ``features_dim`` is the sum of both branch
    output sizes and is computed dynamically — no hard-coded values.

    Args:
        observation_space: A ``gymnasium.spaces.Dict``.
        cnn_out_dim: Output dimension of the CNN branch.
        cnn_channels: Output channels per convolutional layer.
        cnn_kernel_sizes: Kernel size per layer.
        cnn_strides: Stride per layer.
        mlp_hidden_sizes: Hidden layer sizes for the scalar MLP branch.

    Raises:
        ImportError: If PyTorch or Stable Baselines 3 is not installed.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_out_dim: int = 256,
        cnn_channels: tuple[int, ...] = (32, 64),
        cnn_kernel_sizes: tuple[int, ...] = (3, 3),
        cnn_strides: tuple[int, ...] = (1, 1),
        mlp_hidden_sizes: tuple[int, ...] = (128,),
    ) -> None:
        _require_torch_sb3()

        # Compute total features_dim dynamically before calling super().__init__
        scalar_in = _scalar_obs_dim(observation_space)
        mlp_out = mlp_hidden_sizes[-1] if mlp_hidden_sizes else scalar_in
        total_features_dim = cnn_out_dim + mlp_out

        super().__init__(observation_space, features_dim=total_features_dim)

        self._scalar_keys: list[str] = sorted(
            k
            for k in observation_space.spaces
            if k not in _GRID_KEYS and k not in _SKIP_KEYS
        )

        # CNN branch
        if "grid_view" in observation_space.spaces:
            grid_h, grid_w, in_channels = observation_space.spaces["grid_view"].shape
            self._cnn: nn.Module = _build_cnn(
                in_channels, cnn_channels, cnn_kernel_sizes, cnn_strides
            )
            device = torch.device("cpu")
            raw_cnn_dim = _cnn_output_dim(
                self._cnn, grid_h, grid_w, in_channels, device
            )
            self._cnn_linear: nn.Module = nn.Linear(raw_cnn_dim, cnn_out_dim)
            self._has_grid = True
        else:
            self._has_grid = False
            # Adjust features_dim when there is no grid key.
            self._features_dim = mlp_out

        # MLP branch
        mlp_layers: list[nn.Module] = []
        current_dim = scalar_in
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(current_dim, hidden_size))
            mlp_layers.append(nn.ReLU())
            current_dim = hidden_size
        self._mlp: nn.Module = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()

        logger.debug(
            "ForgeObsExtractor: scalar_in=%d mlp_out=%d cnn_out=%d features_dim=%d",
            scalar_in, mlp_out, cnn_out_dim if self._has_grid else 0, self._features_dim,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and concatenate CNN + MLP features.

        Args:
            observations: Batch of observations (Dict[str, Tensor]).

        Returns:
            Feature tensor of shape ``(batch, features_dim)``.
        """
        parts: list[torch.Tensor] = []

        # CNN branch
        if self._has_grid:
            grid = observations["grid_view"].float() / 255.0
            grid = grid.permute(0, 3, 1, 2)
            cnn_feat = self._cnn_linear(self._cnn(grid))
            parts.append(cnn_feat)

        # MLP branch — concatenate all scalar keys
        scalar_parts = [
            observations[k].float().reshape(observations[k].shape[0], -1)
            for k in self._scalar_keys
            if k in observations
        ]
        if scalar_parts:
            scalar_cat = torch.cat(scalar_parts, dim=-1)
            mlp_feat = self._mlp(scalar_cat)
            parts.append(mlp_feat)

        if not parts:
            # Observation space has neither a grid nor any scalar fields.
            # Return a zero tensor of the declared features_dim so callers
            # always receive a consistently-shaped output.
            some_tensor = next(iter(observations.values()))
            return torch.zeros(
                some_tensor.shape[0], self._features_dim, device=some_tensor.device
            )
        return torch.cat(parts, dim=-1)
