"""Unit tests for the MuZero neural network models."""

from __future__ import annotations

import torch

from forge.models.muzero_config import MuZeroConfig
from forge.models.muzero_networks import DynamicsNetwork, PredictionNetwork, RepresentationNetwork


def test_representation_network_without_embeddings() -> None:
    """Verify RepresentationNetwork works correctly without learned embeddings (backward compatibility)."""
    # 11 * 11 * 7 + 73 = 920
    config = MuZeroConfig(
        obs_dim=920,
        action_dim=5,
        use_raw_block_id=False,
        latent_dim=16,
        hidden_dim=16,
        num_blocks=1,
        cnn_channels=(8,),
        cnn_kernel_sizes=(3,),
        cnn_strides=(1,),
    )
    rep_net = RepresentationNetwork(config)
    assert rep_net.block_embeddings is None

    # Check forward pass
    obs = torch.randn(2, config.obs_dim)
    latent = rep_net.forward(obs)
    assert latent.shape == (2, config.latent_dim)

    # Single observation
    obs_single = torch.randn(config.obs_dim)
    latent_single = rep_net.forward(obs_single)
    assert latent_single.shape == (config.latent_dim,)


def test_representation_network_with_embeddings() -> None:
    """Verify RepresentationNetwork works correctly with learned block embeddings."""
    config = MuZeroConfig(
        obs_dim=920,
        action_dim=5,
        use_raw_block_id=True,
        num_block_embeddings=36,
        block_embedding_dim=8,
        latent_dim=16,
        hidden_dim=16,
        num_blocks=1,
        cnn_channels=(8,),
        cnn_kernel_sizes=(3,),
        cnn_strides=(1,),
    )
    rep_net = RepresentationNetwork(config)
    assert rep_net.block_embeddings is not None
    assert isinstance(rep_net.block_embeddings, torch.nn.Embedding)

    # Check parameter registration (should include block_embeddings.weight)
    params = list(rep_net.parameters())
    assert any(p.shape == (36, 8) for p in params)

    # Check forward pass
    obs = torch.randn(2, config.obs_dim)
    # Set the first channel to valid integer values (e.g., indices 0 to 35)
    flat_grid = obs[:, : config.grid_flat_dim].view(
        -1, config.grid_channels, config.grid_height, config.grid_width
    )
    flat_grid[:, 0, :, :] = torch.randint(0, 36, (2, config.grid_height, config.grid_width)).float()

    latent = rep_net.forward(obs)
    assert latent.shape == (2, config.latent_dim)

    # Check gradients propagate through block embeddings
    loss = latent.sum()
    loss.backward()
    assert rep_net.block_embeddings.weight.grad is not None


def test_representation_network_3d_grid() -> None:
    """Verify RepresentationNetwork works correctly with a 3D block-grid variant (grid_depth > 1)."""
    config = MuZeroConfig(
        obs_dim=2614,
        action_dim=5,
        grid_depth=3,
        use_raw_block_id=True,
        num_block_embeddings=36,
        block_embedding_dim=8,
        latent_dim=16,
        hidden_dim=16,
        num_blocks=1,
        cnn_channels=(8,),
        cnn_kernel_sizes=(3,),
        cnn_strides=(1,),
    )
    rep_net = RepresentationNetwork(config)
    assert rep_net.block_embeddings is not None
    assert isinstance(rep_net.block_embeddings, torch.nn.Embedding)

    obs = torch.randn(2, config.obs_dim)
    flat_grid = obs[:, : config.grid_flat_dim].view(
        -1, config.grid_channels, config.grid_depth, config.grid_height, config.grid_width
    )
    flat_grid[:, 0, :, :, :] = torch.randint(
        0, 36, (2, config.grid_depth, config.grid_height, config.grid_width)
    ).float()

    latent = rep_net.forward(obs)
    assert latent.shape == (2, config.latent_dim)


def test_dynamics_and_prediction_networks() -> None:
    """Verify DynamicsNetwork and PredictionNetwork forward passes and output shapes."""
    config = MuZeroConfig(obs_dim=920, action_dim=5, latent_dim=16, hidden_dim=16)
    dyn_net = DynamicsNetwork(config)
    pred_net = PredictionNetwork(config)

    latent = torch.randn(2, config.latent_dim)
    action = torch.zeros(2, config.action_dim)
    action[:, 2] = 1.0  # action index 2

    next_latent, reward_logits = dyn_net.forward(latent, action)
    assert next_latent.shape == (2, config.latent_dim)
    assert reward_logits.shape == (2, config.reward_support_size)

    policy_logits, value_logits = pred_net.forward(next_latent)
    assert policy_logits.shape == (2, config.action_dim)
    assert value_logits.shape == (2, config.value_support_size)
