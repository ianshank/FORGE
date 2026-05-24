"""MuZero neural network components: representation, dynamics, and prediction.

Implements the three core MuZero functions as PyTorch modules:

- **RepresentationNetwork**: Encodes raw observations into latent states.
  Uses a dual-stream architecture (CNN for spatial grid, MLP for vector features)
  matching FORGE's observation structure.

- **DynamicsNetwork**: Predicts the next latent state and reward given a
  latent state and action. Uses residual MLP blocks.

- **PredictionNetwork**: Predicts policy logits and value from a latent state.
  Value and reward are predicted as categorical distributions for stability.

Usage::

    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_networks import (
        RepresentationNetwork, DynamicsNetwork, PredictionNetwork,
    )

    config = MuZeroConfig(obs_dim=920, action_dim=75)
    rep_net = RepresentationNetwork(config)
    dyn_net = DynamicsNetwork(config)
    pred_net = PredictionNetwork(config)
"""
from __future__ import annotations

__all__ = [
    "DynamicsNetwork",
    "PredictionNetwork",
    "RepresentationNetwork",
    "ResidualBlock",
    "scalar_to_support",
    "support_to_scalar",
]

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from forge.models.muzero_config import MuZeroConfig

logger = logging.getLogger(__name__)


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert scalar values to a categorical distribution over support bins.

    Uses the MuZero transform: distributes each scalar across the two
    nearest support bins proportionally.

    Args:
        x: Scalar tensor of shape ``(...)``.
        support_size: Number of bins in the categorical support.

    Returns:
        Categorical distribution of shape ``(..., support_size)``.
    """
    import torch

    half = support_size // 2
    x = torch.clamp(x, -half, half)
    floor_val = x.floor()
    frac = x - floor_val
    floor_idx = (floor_val + half).long()
    floor_idx = torch.clamp(floor_idx, 0, support_size - 1)
    ceil_idx = torch.clamp(floor_idx + 1, 0, support_size - 1)

    result = torch.zeros(*x.shape, support_size, device=x.device, dtype=x.dtype)
    result.scatter_add_(-1, floor_idx.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    result.scatter_add_(-1, ceil_idx.unsqueeze(-1), frac.unsqueeze(-1))
    return result


def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert categorical logits to scalar values via softmax expectation.

    Args:
        logits: Logit tensor of shape ``(..., support_size)``.
        support_size: Number of bins in the categorical support.

    Returns:
        Scalar tensor of shape ``(...)``.
    """
    import torch

    half = support_size // 2
    probs = torch.softmax(logits, dim=-1)
    support = torch.arange(-half, half + 1, device=logits.device, dtype=logits.dtype)
    # Trim if support_size is even
    support = support[:support_size]
    return (probs * support).sum(dim=-1)


class ResidualBlock:
    """A simple residual MLP block: x + MLP(LayerNorm(x)).

    Not a ``nn.Module`` subclass to avoid registering in ``__init__``.
    Use :meth:`build` to create the actual ``nn.Module``.
    """

    @staticmethod
    def build(dim: int) -> torch.nn.Module:
        """Build a residual block as a ``nn.Module``.

        Args:
            dim: Input and output dimensionality.

        Returns:
            A module that computes ``x + relu(linear(layer_norm(x)))``.
        """
        from torch import nn

        class _ResBlock(nn.Module):
            def __init__(self, d: int) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.fc1 = nn.Linear(d, d)
                self.fc2 = nn.Linear(d, d)
                self.act = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                out = self.norm(x)
                out = self.act(self.fc1(out))
                out = self.fc2(out)
                return residual + out

        return _ResBlock(dim)


class RepresentationNetwork:
    """Encodes raw FORGE observations into latent state vectors.

    Architecture:
        - **Spatial stream**: Reshapes grid observations to (B, C, H, W),
          processes through Conv2d layers, and flattens.
        - **Vector stream**: Processes non-spatial features through an MLP.
        - **Fusion**: Concatenates spatial and vector features, maps to
          latent dimension with LayerNorm.

    Args:
        config: MuZero configuration.
    """

    def __init__(self, config: MuZeroConfig) -> None:
        import torch
        from torch import nn

        self._config = config
        self._device = torch.device(config.device)

        # Spatial stream: CNN over grid observations
        cnn_layers: list[nn.Module] = []
        if config.use_raw_block_id:
            self.block_embeddings: nn.Embedding | None = nn.Embedding(
                num_embeddings=config.num_block_embeddings,
                embedding_dim=config.block_embedding_dim,
            ).to(self._device)
            in_channels = config.block_embedding_dim + config.grid_channels - 1
        else:
            self.block_embeddings = None
            in_channels = config.grid_channels

        use_conv3d = config.grid_depth > 1
        for out_ch, kernel, stride in zip(
            config.cnn_channels, config.cnn_kernel_sizes, config.cnn_strides
        ):
            if use_conv3d:
                cnn_layers.append(nn.Conv3d(in_channels, out_ch, kernel, stride, padding=kernel // 2))
            else:
                cnn_layers.append(nn.Conv2d(in_channels, out_ch, kernel, stride, padding=kernel // 2))
            cnn_layers.append(nn.ReLU())
            in_channels = out_ch
        self.cnn = nn.Sequential(*cnn_layers).to(self._device)

        # Compute CNN output dimension
        with torch.no_grad():
            if use_conv3d:
                dummy = torch.zeros(
                    1,
                    config.block_embedding_dim + config.grid_channels - 1 if config.use_raw_block_id else config.grid_channels,
                    config.grid_depth,
                    config.grid_height,
                    config.grid_width,
                )
            else:
                dummy = torch.zeros(
                    1,
                    config.block_embedding_dim + config.grid_channels - 1 if config.use_raw_block_id else config.grid_channels,
                    config.grid_height,
                    config.grid_width,
                )
            cnn_out = self.cnn(dummy.to(self._device))
            self._cnn_flat_dim = int(cnn_out.numel())

        # Vector stream: MLP
        self.vector_mlp = nn.Sequential(
            nn.Linear(config.vector_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        ).to(self._device)

        # Fusion: spatial_flat + vector_hidden -> latent
        fusion_input_dim = self._cnn_flat_dim + config.hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        ).to(self._device)

        # Residual blocks for refinement
        self.res_blocks = nn.Sequential(
            *[ResidualBlock.build(config.latent_dim) for _ in range(config.num_blocks)]
        ).to(self._device)

        # Collect all modules for parameter access
        modules = []
        if self.block_embeddings is not None:
            modules.append(self.block_embeddings)
        modules.extend([self.cnn, self.vector_mlp, self.fusion, self.res_blocks])
        self.modules_list = nn.ModuleList(modules).to(self._device)

        total_params = sum(p.numel() for p in self.modules_list.parameters())
        logger.info(
            "RepresentationNetwork: grid=(%d,%d,%d), vector=%d, latent=%d, params=%d",
            config.grid_channels, config.grid_height, config.grid_width,
            config.vector_dim, config.latent_dim, total_params,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode a raw observation into a latent state.

        Args:
            observation: Flat observation of shape ``(B, obs_dim)``
                or ``(obs_dim,)``.

        Returns:
            Latent state of shape ``(B, latent_dim)`` or ``(latent_dim,)``.
        """
        import torch

        squeezed = observation.dim() == 1
        if squeezed:
            observation = observation.unsqueeze(0)

        c = self._config
        grid_dim = c.grid_flat_dim

        # Split observation into spatial and vector components
        grid_flat = observation[:, :grid_dim]
        vector = observation[:, grid_dim:]

        # Spatial stream
        if c.grid_depth > 1:
            grid = grid_flat.view(-1, c.grid_channels, c.grid_depth, c.grid_height, c.grid_width)
            if self.block_embeddings is not None:
                # Extract first channel (block type hash/index)
                # block_types shape: (B, D, H, W)
                block_types = grid[:, 0, :, :, :].long()
                block_types = torch.clamp(block_types, 0, c.num_block_embeddings - 1)
                # block_emb shape: (B, D, H, W, block_embedding_dim)
                block_emb = self.block_embeddings(block_types)
                # permute to shape: (B, block_embedding_dim, D, H, W)
                block_emb = block_emb.permute(0, 4, 1, 2, 3)
                # Concatenate with remaining channels along channel dimension (dim=1)
                # remaining channels shape: (B, grid_channels-1, D, H, W)
                grid = torch.cat([block_emb, grid[:, 1:, :, :, :]], dim=1)
        else:
            grid = grid_flat.view(-1, c.grid_channels, c.grid_height, c.grid_width)
            if self.block_embeddings is not None:
                # Extract first channel (block type hash/index)
                # block_types shape: (B, H, W)
                block_types = grid[:, 0, :, :].long()
                block_types = torch.clamp(block_types, 0, c.num_block_embeddings - 1)
                # block_emb shape: (B, H, W, block_embedding_dim)
                block_emb = self.block_embeddings(block_types)
                # permute to shape: (B, block_embedding_dim, H, W)
                block_emb = block_emb.permute(0, 3, 1, 2)
                # Concatenate with remaining channels along channel dimension (dim=1)
                # remaining channels shape: (B, grid_channels-1, H, W)
                grid = torch.cat([block_emb, grid[:, 1:, :, :]], dim=1)

        spatial_features = self.cnn(grid).flatten(start_dim=1)

        # Vector stream
        vector_features = self.vector_mlp(vector)

        # Fusion
        combined = self.fusion(
            self._cat(spatial_features, vector_features)
        )
        latent = self.res_blocks(combined)

        if squeezed:
            latent = latent.squeeze(0)
        return latent

    @staticmethod
    def _cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        import torch

        return torch.cat([a, b], dim=-1)

    def parameters(self) -> list[torch.nn.Parameter]:
        """Return all trainable parameters."""
        return list(self.modules_list.parameters())


class DynamicsNetwork:
    """Predicts the next latent state and reward from a latent state and action.

    Architecture:
        - Concatenates latent state with one-hot action encoding.
        - Passes through an MLP to produce the next latent state.
        - Applies residual blocks and LayerNorm for stability.
        - Reward predicted via a categorical head.

    Args:
        config: MuZero configuration.
    """

    def __init__(self, config: MuZeroConfig) -> None:
        import torch
        from torch import nn

        self._config = config
        self._device = torch.device(config.device)

        input_dim = config.latent_dim + config.action_dim

        # State transition: (latent + action) -> next_latent
        self.transition = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        ).to(self._device)

        # Residual refinement
        self.res_blocks = nn.Sequential(
            *[ResidualBlock.build(config.latent_dim) for _ in range(config.num_blocks)]
        ).to(self._device)

        # Reward head: latent -> reward categorical logits
        self.reward_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.reward_support_size),
        ).to(self._device)

        self.modules_list = nn.ModuleList([
            self.transition, self.res_blocks, self.reward_head,
        ]).to(self._device)

        total_params = sum(p.numel() for p in self.modules_list.parameters())
        logger.info(
            "DynamicsNetwork: latent=%d, action=%d, reward_bins=%d, params=%d",
            config.latent_dim, config.action_dim,
            config.reward_support_size, total_params,
        )

    def forward(
        self, latent_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next latent state and reward logits.

        Args:
            latent_state: Current latent state of shape ``(B, latent_dim)``.
            action: One-hot action tensor of shape ``(B, action_dim)``.

        Returns:
            Tuple of (next_latent_state, reward_logits) with shapes
            ``(B, latent_dim)`` and ``(B, reward_support_size)``.
        """
        import torch

        x = torch.cat([latent_state, action], dim=-1)
        next_latent = self.transition(x)
        next_latent = self.res_blocks(next_latent)
        reward_logits = self.reward_head(next_latent)
        return next_latent, reward_logits

    def parameters(self) -> list[torch.nn.Parameter]:
        """Return all trainable parameters."""
        return list(self.modules_list.parameters())


class PredictionNetwork:
    """Predicts policy logits and value from a latent state.

    Architecture:
        - Shared trunk of residual blocks.
        - Policy head: Linear -> action logits (unnormalized).
        - Value head: Linear -> categorical value logits.

    Args:
        config: MuZero configuration.
    """

    def __init__(self, config: MuZeroConfig) -> None:
        import torch
        from torch import nn

        self._config = config
        self._device = torch.device(config.device)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
        ).to(self._device)

        # Policy head
        self.policy_head = nn.Linear(config.hidden_dim, config.action_dim).to(self._device)

        # Value head (categorical)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.value_support_size),
        ).to(self._device)

        self.modules_list = nn.ModuleList([
            self.trunk, self.policy_head, self.value_head,
        ]).to(self._device)

        total_params = sum(p.numel() for p in self.modules_list.parameters())
        logger.info(
            "PredictionNetwork: latent=%d, action=%d, value_bins=%d, params=%d",
            config.latent_dim, config.action_dim,
            config.value_support_size, total_params,
        )

    def forward(
        self, latent_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy logits and value logits from a latent state.

        Args:
            latent_state: Latent state of shape ``(B, latent_dim)``
                or ``(latent_dim,)``.

        Returns:
            Tuple of (policy_logits, value_logits) with shapes
            ``(B, action_dim)`` and ``(B, value_support_size)``.
        """
        squeezed = latent_state.dim() == 1
        if squeezed:
            latent_state = latent_state.unsqueeze(0)

        features = self.trunk(latent_state)
        policy_logits = self.policy_head(features)
        value_logits = self.value_head(features)

        if squeezed:
            policy_logits = policy_logits.squeeze(0)
            value_logits = value_logits.squeeze(0)
        return policy_logits, value_logits

    def parameters(self) -> list[torch.nn.Parameter]:
        """Return all trainable parameters."""
        return list(self.modules_list.parameters())
