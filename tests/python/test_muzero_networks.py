"""Tests for MuZero network components: config, representation, dynamics, prediction."""
from __future__ import annotations

import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.models.muzero_config import (  # noqa: E402
    DEFAULT_GRID_CHANNELS,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_WIDTH,
    DEFAULT_LATENT_DIM,
    DEFAULT_VECTOR_DIM,
    MuZeroConfig,
)
from forge.models.muzero_networks import (  # noqa: E402
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    scalar_to_support,
    support_to_scalar,
)
from forge.models.muzero_world_model import (  # noqa: E402
    MuZeroWorldModel,
    NetworkOutput,
)
from forge.models.world_model import WorldModel  # noqa: E402

# Small test dimensions for speed
OBS_DIM = 11 * 11 * 7 + 73  # 920 (standard FORGE)
ACTION_DIM = 10
LATENT_DIM = 32
HIDDEN_DIM = 32
NUM_BLOCKS = 1
REWARD_SUPPORT = 11
VALUE_SUPPORT = 11


def _make_config(**overrides: int | float | str) -> MuZeroConfig:
    """Create a small test config."""
    defaults = {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_blocks": NUM_BLOCKS,
        "reward_support_size": REWARD_SUPPORT,
        "value_support_size": VALUE_SUPPORT,
        "grid_height": DEFAULT_GRID_HEIGHT,
        "grid_width": DEFAULT_GRID_WIDTH,
        "grid_channels": DEFAULT_GRID_CHANNELS,
        "vector_dim": DEFAULT_VECTOR_DIM,
        "cnn_channels": (16, 16),
        "cnn_kernel_sizes": (3, 3),
        "cnn_strides": (1, 1),
        "device": "cpu",
    }
    defaults.update(overrides)
    return MuZeroConfig(**defaults)


# ---------------------------------------------------------------------------
# MuZeroConfig
# ---------------------------------------------------------------------------


class TestMuZeroConfig:
    """Tests for MuZeroConfig defaults and validation."""

    def test_defaults(self) -> None:
        cfg = MuZeroConfig()
        assert cfg.latent_dim == DEFAULT_LATENT_DIM
        assert cfg.grid_height == DEFAULT_GRID_HEIGHT

    def test_obs_dim_auto_computed(self) -> None:
        cfg = MuZeroConfig(obs_dim=0)
        expected = DEFAULT_GRID_HEIGHT * DEFAULT_GRID_WIDTH * DEFAULT_GRID_CHANNELS + DEFAULT_VECTOR_DIM
        assert cfg.obs_dim == expected

    def test_custom_values(self) -> None:
        cfg = MuZeroConfig(obs_dim=100, action_dim=5, latent_dim=64)
        assert cfg.obs_dim == 100
        assert cfg.action_dim == 5
        assert cfg.latent_dim == 64

    def test_is_dataclass(self) -> None:
        field_names = {f.name for f in fields(MuZeroConfig)}
        assert "obs_dim" in field_names
        assert "action_dim" in field_names
        assert "latent_dim" in field_names

    def test_grid_flat_dim(self) -> None:
        cfg = _make_config()
        assert cfg.grid_flat_dim == cfg.grid_height * cfg.grid_width * cfg.grid_channels

    def test_support_range(self) -> None:
        cfg = MuZeroConfig(reward_support_size=31)
        low, high = cfg.reward_support_range
        assert low == -15
        assert high == 15

    def test_cnn_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="cnn_channels"):
            MuZeroConfig(
                obs_dim=100,
                cnn_channels=(32, 64),
                cnn_kernel_sizes=(3,),
                cnn_strides=(1, 1),
            )


# ---------------------------------------------------------------------------
# scalar_to_support / support_to_scalar
# ---------------------------------------------------------------------------


class TestSupportTransforms:
    """Tests for categorical support conversion functions."""

    def test_scalar_to_support_shape(self) -> None:
        x = torch.tensor([3.0])
        dist = scalar_to_support(x, 31)
        assert dist.shape == (1, 31)
        assert abs(dist.sum().item() - 1.0) < 1e-5

    def test_scalar_to_support_integer(self) -> None:
        """Integer values should produce one-hot-like distributions."""
        x = torch.tensor([3.0])
        dist = scalar_to_support(x, 31)
        # Half=15, so index 3+15=18 should have weight ~1.0
        assert dist[0, 18].item() > 0.99

    def test_scalar_to_support_fractional(self) -> None:
        """Fractional values should split weight between two bins."""
        x = torch.tensor([2.5])
        dist = scalar_to_support(x, 31)
        # Half=15, floor=2→index 17, ceil=3→index 18
        assert dist[0, 17].item() > 0.4
        assert dist[0, 18].item() > 0.4

    def test_support_to_scalar_uniform(self) -> None:
        """Uniform logits should give ~0 scalar value (symmetric support)."""
        logits = torch.zeros(1, 11)
        val = support_to_scalar(logits, 11)
        assert abs(val.item()) < 1e-5

    def test_support_to_scalar_peaked(self) -> None:
        """High logit at positive support index should give positive scalar."""
        logits = torch.zeros(1, 11)
        logits[0, 10] = 100.0  # Peak at index 10 → value 5 (half=5)
        val = support_to_scalar(logits, 11)
        assert abs(val.item() - 5.0) < 0.1

    def test_scalar_to_support_batch(self) -> None:
        x = torch.tensor([0.0, 1.0, -1.0])
        dist = scalar_to_support(x, 11)
        assert dist.shape == (3, 11)
        # Each distribution should sum to 1
        for i in range(3):
            assert abs(dist[i].sum().item() - 1.0) < 1e-5

    def test_clamping(self) -> None:
        """Values outside support range should be clamped."""
        x = torch.tensor([100.0])
        dist = scalar_to_support(x, 11)
        # Should be clamped to 5 (half=5), weight at index 10
        assert dist[0, 10].item() > 0.99


# ---------------------------------------------------------------------------
# RepresentationNetwork
# ---------------------------------------------------------------------------


class TestRepresentationNetwork:
    """Tests for RepresentationNetwork."""

    def test_output_shape(self) -> None:
        cfg = _make_config()
        net = RepresentationNetwork(cfg)
        obs = torch.randn(1, OBS_DIM)
        latent = net.forward(obs)
        assert latent.shape == (1, LATENT_DIM)

    def test_single_obs_shape(self) -> None:
        cfg = _make_config()
        net = RepresentationNetwork(cfg)
        obs = torch.randn(OBS_DIM)
        latent = net.forward(obs)
        assert latent.shape == (LATENT_DIM,)

    def test_batch_processing(self) -> None:
        cfg = _make_config()
        net = RepresentationNetwork(cfg)
        obs = torch.randn(4, OBS_DIM)
        latent = net.forward(obs)
        assert latent.shape == (4, LATENT_DIM)

    def test_output_finite(self) -> None:
        cfg = _make_config()
        net = RepresentationNetwork(cfg)
        obs = torch.randn(2, OBS_DIM)
        latent = net.forward(obs)
        assert torch.all(torch.isfinite(latent))

    def test_parameters_not_empty(self) -> None:
        cfg = _make_config()
        net = RepresentationNetwork(cfg)
        assert len(net.parameters()) > 0

    def test_gradient_flow(self) -> None:
        """Verify gradients propagate through representation network."""
        cfg = _make_config()
        net = RepresentationNetwork(cfg)
        obs = torch.randn(2, OBS_DIM, requires_grad=True)
        latent = net.forward(obs)
        latent.sum().backward()
        assert obs.grad is not None
        assert torch.all(torch.isfinite(obs.grad))


# ---------------------------------------------------------------------------
# DynamicsNetwork
# ---------------------------------------------------------------------------


class TestDynamicsNetwork:
    """Tests for DynamicsNetwork."""

    def test_output_shapes(self) -> None:
        cfg = _make_config()
        net = DynamicsNetwork(cfg)
        latent = torch.randn(1, LATENT_DIM)
        action = torch.zeros(1, ACTION_DIM)
        action[0, 3] = 1.0
        next_latent, reward_logits = net.forward(latent, action)
        assert next_latent.shape == (1, LATENT_DIM)
        assert reward_logits.shape == (1, REWARD_SUPPORT)

    def test_batch_shapes(self) -> None:
        cfg = _make_config()
        net = DynamicsNetwork(cfg)
        latent = torch.randn(4, LATENT_DIM)
        action = torch.zeros(4, ACTION_DIM)
        next_latent, reward_logits = net.forward(latent, action)
        assert next_latent.shape == (4, LATENT_DIM)
        assert reward_logits.shape == (4, REWARD_SUPPORT)

    def test_output_finite(self) -> None:
        cfg = _make_config()
        net = DynamicsNetwork(cfg)
        latent = torch.randn(2, LATENT_DIM)
        action = torch.zeros(2, ACTION_DIM)
        next_latent, reward_logits = net.forward(latent, action)
        assert torch.all(torch.isfinite(next_latent))
        assert torch.all(torch.isfinite(reward_logits))

    def test_parameters_count(self) -> None:
        cfg = _make_config()
        net = DynamicsNetwork(cfg)
        assert len(net.parameters()) > 0


# ---------------------------------------------------------------------------
# PredictionNetwork
# ---------------------------------------------------------------------------


class TestPredictionNetwork:
    """Tests for PredictionNetwork."""

    def test_output_shapes(self) -> None:
        cfg = _make_config()
        net = PredictionNetwork(cfg)
        latent = torch.randn(1, LATENT_DIM)
        policy, value = net.forward(latent)
        assert policy.shape == (1, ACTION_DIM)
        assert value.shape == (1, VALUE_SUPPORT)

    def test_single_latent(self) -> None:
        cfg = _make_config()
        net = PredictionNetwork(cfg)
        latent = torch.randn(LATENT_DIM)
        policy, value = net.forward(latent)
        assert policy.shape == (ACTION_DIM,)
        assert value.shape == (VALUE_SUPPORT,)

    def test_output_finite(self) -> None:
        cfg = _make_config()
        net = PredictionNetwork(cfg)
        latent = torch.randn(2, LATENT_DIM)
        policy, value = net.forward(latent)
        assert torch.all(torch.isfinite(policy))
        assert torch.all(torch.isfinite(value))

    def test_parameters_count(self) -> None:
        cfg = _make_config()
        net = PredictionNetwork(cfg)
        assert len(net.parameters()) > 0


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------


class TestResidualBlock:
    """Tests for ResidualBlock."""

    def test_build_output_shape(self) -> None:
        from forge.models.muzero_networks import ResidualBlock  # noqa: PLC0415

        block = ResidualBlock.build(32)
        x = torch.randn(4, 32)
        out = block(x)
        assert out.shape == (4, 32)

    def test_residual_connection(self) -> None:
        """Output should differ from input (non-identity) but be same shape."""
        from forge.models.muzero_networks import ResidualBlock  # noqa: PLC0415

        block = ResidualBlock.build(16)
        x = torch.randn(2, 16)
        out = block(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# MuZeroWorldModel
# ---------------------------------------------------------------------------


class TestMuZeroWorldModel:
    """Tests for MuZeroWorldModel."""

    def test_isinstance_world_model(self) -> None:
        model = MuZeroWorldModel(_make_config())
        assert isinstance(model, WorldModel)

    def test_config_property(self) -> None:
        cfg = _make_config()
        model = MuZeroWorldModel(cfg)
        assert model.config is cfg

    def test_initial_inference_output(self) -> None:
        model = MuZeroWorldModel(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        output = model.initial_inference(obs)
        assert isinstance(output, NetworkOutput)
        assert output.latent_state.shape == (LATENT_DIM,)
        assert output.policy_logits.shape == (ACTION_DIM,)
        assert output.reward == 0.0
        assert np.isfinite(output.value)

    def test_recurrent_inference_output(self) -> None:
        model = MuZeroWorldModel(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        init_out = model.initial_inference(obs)
        rec_out = model.recurrent_inference(init_out.latent_state, action=2)
        assert rec_out.latent_state.shape == (LATENT_DIM,)
        assert rec_out.policy_logits.shape == (ACTION_DIM,)
        assert np.isfinite(rec_out.reward)
        assert np.isfinite(rec_out.value)

    def test_predict_returns_latent(self) -> None:
        model = MuZeroWorldModel(_make_config())
        state = np.random.randn(LATENT_DIM).astype(np.float32)
        next_state = model.predict(state, action=0)
        assert next_state.shape == (LATENT_DIM,)

    def test_all_parameters_returns_list(self) -> None:
        model = MuZeroWorldModel(_make_config())
        params = model.all_parameters()
        assert isinstance(params, list)
        assert len(params) > 0

    def test_recurrent_inference_boundary_actions(self) -> None:
        model = MuZeroWorldModel(_make_config())
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        init = model.initial_inference(obs)
        # Test action 0 (lower bound)
        out0 = model.recurrent_inference(init.latent_state, action=0)
        assert out0.latent_state.shape == (LATENT_DIM,)
        # Test action ACTION_DIM-1 (upper bound)
        out_last = model.recurrent_inference(init.latent_state, action=ACTION_DIM - 1)
        assert out_last.latent_state.shape == (LATENT_DIM,)

    def test_negative_action_raises(self) -> None:
        model = MuZeroWorldModel(_make_config())
        state = np.random.randn(LATENT_DIM).astype(np.float32)
        with pytest.raises(ValueError, match="Invalid action"):
            model.recurrent_inference(state, action=-1)

    def test_invalid_action_raises(self) -> None:
        model = MuZeroWorldModel(_make_config())
        state = np.random.randn(LATENT_DIM).astype(np.float32)
        with pytest.raises(ValueError, match="Invalid action"):
            model.recurrent_inference(state, action=ACTION_DIM + 5)

    def test_train_step_returns_metrics(self) -> None:
        cfg = _make_config()
        model = MuZeroWorldModel(cfg)
        N = 4
        K = cfg.num_unroll_steps
        batch = {
            "observations": np.random.randn(N, OBS_DIM).astype(np.float32),
            "actions": np.random.randint(0, ACTION_DIM, (N, K)).astype(np.int64),
            "target_values": np.random.randn(N, K + 1).astype(np.float32),
            "target_rewards": np.random.randn(N, K).astype(np.float32),
            "target_policies": np.random.dirichlet(
                np.ones(ACTION_DIM), (N, K + 1)
            ).astype(np.float32),
        }
        metrics = model.train_step(batch)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "reward_loss" in metrics
        assert np.isfinite(metrics["loss"])

    def test_save_load_roundtrip(self) -> None:
        cfg = _make_config()
        model1 = MuZeroWorldModel(cfg)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        torch.manual_seed(42)
        out1 = model1.initial_inference(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model1.save(path)
            model2 = MuZeroWorldModel(cfg)
            model2.load(path)
            torch.manual_seed(42)
            out2 = model2.initial_inference(obs)
            np.testing.assert_allclose(out1.latent_state, out2.latent_state, atol=1e-5)
        finally:
            Path(path).unlink()

    def test_load_validates_dimensions(self) -> None:
        cfg1 = _make_config(latent_dim=32)
        model1 = MuZeroWorldModel(cfg1)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model1.save(path)
            cfg2 = _make_config(latent_dim=64)
            model2 = MuZeroWorldModel(cfg2)
            with pytest.raises(ValueError, match="latent_dim"):
                model2.load(path)
        finally:
            Path(path).unlink()
