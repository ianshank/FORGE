"""Standalone tests for policy network classes.

Covers RandomPolicyNetwork (forward, train_step, save/load roundtrip)
and ActorCriticNetwork save/load with dimension mismatch validation.

ActorCriticNetwork forward/get_action_and_value shapes are already
thoroughly tested in test_mappo.py; this file focuses on edge cases
and the RandomPolicyNetwork which is not covered there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from forge.config import DEFAULT_ACTION_SIZE
from forge.models.policy_network import RandomPolicyNetwork

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUSTOM_ACTION_SIZE = 6
OBS_DIM = 16
ACTION_DIM = 5
HIDDEN_SIZES = [32, 32]
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# RandomPolicyNetwork
# ---------------------------------------------------------------------------


class TestRandomPolicyNetworkStandalone:
    """Additional tests for RandomPolicyNetwork beyond test_mappo.py."""

    def test_forward_correct_shape(self) -> None:
        """forward() should return array of shape (action_size,)."""
        net = RandomPolicyNetwork(action_size=CUSTOM_ACTION_SIZE)
        obs = np.zeros(10, dtype=np.float32)
        probs = net.forward(obs)
        assert probs.shape == (CUSTOM_ACTION_SIZE,)

    def test_forward_sums_to_one(self) -> None:
        """Uniform probabilities should sum to 1.0."""
        net = RandomPolicyNetwork(action_size=CUSTOM_ACTION_SIZE)
        obs = np.random.default_rng(0).standard_normal(10).astype(np.float32)
        probs = net.forward(obs)
        assert pytest.approx(float(np.sum(probs)), abs=1e-6) == 1.0

    def test_forward_ignores_observation(self) -> None:
        """Output should be identical regardless of observation content."""
        net = RandomPolicyNetwork(action_size=CUSTOM_ACTION_SIZE)
        obs_a = np.zeros(10, dtype=np.float32)
        obs_b = np.ones(10, dtype=np.float32)
        np.testing.assert_array_equal(net.forward(obs_a), net.forward(obs_b))

    def test_train_step_returns_empty_dict(self) -> None:
        """train_step is a no-op and returns empty dict."""
        net = RandomPolicyNetwork(action_size=CUSTOM_ACTION_SIZE)
        result = net.train_step({"obs": np.zeros((4, 8), dtype=np.float32)})
        assert result == {}

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """save/load should not raise (both are no-ops)."""
        net = RandomPolicyNetwork(action_size=CUSTOM_ACTION_SIZE)
        path = str(tmp_path / "random_model.bin")
        net.save(path)
        net.load(path)
        # After load, forward should still work identically
        probs = net.forward(np.zeros(5, dtype=np.float32))
        assert probs.shape == (CUSTOM_ACTION_SIZE,)

    def test_default_action_size(self) -> None:
        """Default action_size should match DEFAULT_ACTION_SIZE from config."""
        net = RandomPolicyNetwork()
        assert net.action_size == DEFAULT_ACTION_SIZE


# ---------------------------------------------------------------------------
# ActorCriticNetwork (torch-dependent)
# ---------------------------------------------------------------------------


class TestActorCriticNetworkStandalone:
    """Edge-case tests for ActorCriticNetwork not covered in test_mappo.py."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip("torch")

    def test_save_load_roundtrip_via_tmp_path(self, tmp_path: Path) -> None:
        """Save and load through a tmp_path directory."""
        import torch  # noqa: PLC0415
        from forge.models.policy_network import ActorCriticNetwork  # noqa: PLC0415

        net1 = ActorCriticNetwork(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES
        )
        obs = torch.randn(1, OBS_DIM)
        logits1, val1 = net1.forward(obs)

        path = str(tmp_path / "model.pt")
        net1.save(path)

        net2 = ActorCriticNetwork(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES
        )
        net2.load(path)
        logits2, val2 = net2.forward(obs)

        assert torch.allclose(logits1, logits2, atol=1e-6)
        assert torch.allclose(val1, val2, atol=1e-6)

    def test_load_dimension_mismatch_obs(self, tmp_path: Path) -> None:
        """Loading a checkpoint with wrong obs_dim should raise ValueError."""
        from forge.models.policy_network import ActorCriticNetwork  # noqa: PLC0415

        net1 = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        path = str(tmp_path / "model.pt")
        net1.save(path)

        mismatched_obs_dim = OBS_DIM + 4
        net2 = ActorCriticNetwork(
            obs_dim=mismatched_obs_dim, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES
        )
        with pytest.raises(ValueError, match="obs_dim"):
            net2.load(path)

    def test_load_dimension_mismatch_action(self, tmp_path: Path) -> None:
        """Loading a checkpoint with wrong action_dim should raise ValueError."""
        from forge.models.policy_network import ActorCriticNetwork  # noqa: PLC0415

        net1 = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        path = str(tmp_path / "model.pt")
        net1.save(path)

        mismatched_action_dim = ACTION_DIM + 2
        net2 = ActorCriticNetwork(
            obs_dim=OBS_DIM, action_dim=mismatched_action_dim, hidden_sizes=HIDDEN_SIZES
        )
        with pytest.raises(ValueError, match="action_dim"):
            net2.load(path)

    def test_parameters_returns_list(self) -> None:
        """parameters() should return a non-empty list of torch Parameters."""
        import torch  # noqa: PLC0415
        from forge.models.policy_network import ActorCriticNetwork  # noqa: PLC0415

        net = ActorCriticNetwork(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES
        )
        params = net.parameters()
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)
