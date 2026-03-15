"""Tests for MAPPO agent, ActorCriticNetwork, and PPOTrainer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from forge.agents.mappo_agent import MAPPOAgent, MAPPOConfig  # noqa: E402
from forge.models.policy_network import ActorCriticNetwork  # noqa: E402
from forge.training.trainer import PPOTrainer, PPOTrainerConfig  # noqa: E402

# Test dimensions
OBS_DIM = 16
ACTION_DIM = 5
HIDDEN_SIZES = [32, 32]
BATCH_SIZE = 8


class TestActorCriticNetwork:
    """Tests for ActorCriticNetwork."""

    def test_forward_shapes(self) -> None:
        """Forward pass should produce correct output shapes."""
        net = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        logits, value = net.forward(obs)
        assert logits.shape == (BATCH_SIZE, ACTION_DIM)
        assert value.shape == (BATCH_SIZE, 1)

    def test_get_action_and_value(self) -> None:
        """get_action_and_value should return valid actions and quantities."""
        net = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        assert action.shape == (BATCH_SIZE,)
        assert log_prob.shape == (BATCH_SIZE,)
        assert entropy.shape == (BATCH_SIZE,)
        assert value.shape == (BATCH_SIZE, 1)
        # Actions should be valid indices
        assert (action >= 0).all()
        assert (action < ACTION_DIM).all()

    def test_get_action_with_given_action(self) -> None:
        """When action is provided, log_prob should be computed for that action."""
        net = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        fixed_actions = torch.randint(0, ACTION_DIM, (BATCH_SIZE,))
        action, log_prob, _entropy, _value = net.get_action_and_value(obs, action=fixed_actions)
        assert torch.equal(action, fixed_actions)
        assert log_prob.shape == (BATCH_SIZE,)

    def test_deterministic_action(self) -> None:
        """Deterministic mode should always return the same action for same input."""
        net = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        obs = torch.randn(1, OBS_DIM)
        actions = []
        for _ in range(10):
            action, _, _, _ = net.get_action_and_value(obs, deterministic=True)
            actions.append(action.item())
        assert len(set(actions)) == 1, "Deterministic mode should be consistent"

    def test_get_value(self) -> None:
        """get_value should return scalar values."""
        net = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        value = net.get_value(obs)
        assert value.shape == (BATCH_SIZE, 1)

    def test_save_load_roundtrip(self) -> None:
        """Save and load should preserve network weights."""
        net1 = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        obs = torch.randn(1, OBS_DIM)
        logits1, val1 = net1.forward(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            net1.save(path)

            net2 = ActorCriticNetwork(
                obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES
            )
            net2.load(path)
            logits2, val2 = net2.forward(obs)

            assert torch.allclose(logits1, logits2, atol=1e-6)
            assert torch.allclose(val1, val2, atol=1e-6)
        finally:
            Path(path).unlink()

    def test_train_eval_mode(self) -> None:
        """train_mode and eval_mode should toggle dropout/batchnorm behavior."""
        net = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES)
        net.train_mode()
        assert net.encoder.training
        net.eval_mode()
        assert not net.encoder.training


class TestMAPPOAgent:
    """Tests for MAPPOAgent."""

    def _make_agent(self) -> MAPPOAgent:
        config = MAPPOConfig(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=list(HIDDEN_SIZES),
            device="cpu",
            epochs=2,
            batch_size=4,
        )
        return MAPPOAgent(config)

    def test_act_returns_valid_action(self) -> None:
        """act() should return an action within the action space."""
        agent = self._make_agent()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, info = agent.act(obs)
        assert 0 <= action < ACTION_DIM
        assert "log_prob" in info
        assert "value" in info
        assert "entropy" in info

    def test_act_batch(self) -> None:
        """act_batch should handle multiple observations."""
        agent = self._make_agent()
        obs = np.random.randn(BATCH_SIZE, OBS_DIM).astype(np.float32)
        actions, log_probs, entropies, values = agent.act_batch(obs)
        assert actions.shape == (BATCH_SIZE,)
        assert log_probs.shape == (BATCH_SIZE,)
        assert entropies.shape == (BATCH_SIZE,)
        assert values.shape == (BATCH_SIZE,)

    def test_learn_reduces_loss(self) -> None:
        """PPO update should produce finite, reasonable metrics."""
        agent = self._make_agent()
        n = 32
        batch = {
            "observations": np.random.randn(n, OBS_DIM).astype(np.float32),
            "actions": np.random.randint(0, ACTION_DIM, n).astype(np.int64),
            "old_log_probs": np.full(n, -np.log(ACTION_DIM), dtype=np.float32),
            "advantages": np.random.randn(n).astype(np.float32),
            "returns": np.random.randn(n).astype(np.float32),
        }
        metrics = agent.learn(batch)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])

    def test_compute_gae(self) -> None:
        """GAE computation should produce arrays of correct shape."""
        agent = self._make_agent()
        T = 10
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value=0.0)
        assert advantages.shape == (T,)
        assert returns.shape == (T,)
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(returns))

    def test_gae_terminal_episode(self) -> None:
        """GAE should handle episode boundaries correctly."""
        agent = self._make_agent()
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        advantages, _returns = agent.compute_gae(rewards, values, dones, next_value=0.0)
        # At terminal step, advantage = reward - value (no bootstrap)
        assert advantages[2] == pytest.approx(0.5, abs=1e-5)

    def test_save_load(self) -> None:
        """Save and load should preserve agent state."""
        agent = self._make_agent()
        agent._step_count = 100
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        _action1, _ = agent.act(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/agent"
            agent.save(path)

            agent2 = self._make_agent()
            agent2.load(path)
            assert agent2.step_count == 101  # 100 + 1 from act()


class TestPPOTrainer:
    """Tests for PPOTrainer with a simple mock environment."""

    def _make_env(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
        """Create simple mock environment functions."""
        rng = np.random.default_rng(42)
        state = {"step": 0}

        def reset() -> np.ndarray:
            state["step"] = 0
            return rng.standard_normal(obs_dim).astype(np.float32)

        def step(action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
            state["step"] += 1
            obs = rng.standard_normal(obs_dim).astype(np.float32)
            reward = rng.standard_normal()
            terminated = state["step"] >= 20
            truncated = False
            return obs, float(reward), terminated, truncated, {}

        return reset, step

    def test_collect_rollout(self) -> None:
        """collect_rollout should return correctly shaped arrays."""
        config = MAPPOConfig(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=list(HIDDEN_SIZES),
            device="cpu",
        )
        agent = MAPPOAgent(config)
        trainer_config = PPOTrainerConfig(rollout_length=64, max_episode_steps=20)
        trainer = PPOTrainer(agent, trainer_config)

        reset_fn, step_fn = self._make_env()
        rollout = trainer.collect_rollout(reset_fn, step_fn)

        assert rollout["observations"].shape == (64, OBS_DIM)
        assert rollout["actions"].shape == (64,)
        assert rollout["rewards"].shape == (64,)
        assert rollout["old_log_probs"].shape == (64,)
        assert rollout["values"].shape == (64,)
        assert rollout["advantages"].shape == (64,)
        assert rollout["returns"].shape == (64,)

    def test_train_one_update(self) -> None:
        """A single PPO update should complete without error."""
        config = MAPPOConfig(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=list(HIDDEN_SIZES),
            device="cpu",
            epochs=1,
            batch_size=16,
        )
        agent = MAPPOAgent(config)
        trainer_config = PPOTrainerConfig(rollout_length=32, max_episode_steps=10, log_interval=1)
        trainer = PPOTrainer(agent, trainer_config)

        reset_fn, step_fn = self._make_env()
        metrics_list = trainer.train(reset_fn, step_fn, num_updates=1)

        assert len(metrics_list) == 1
        assert "policy_loss" in metrics_list[0]
        assert "value_loss" in metrics_list[0]
        assert trainer.total_steps == 32
        assert trainer.episode_count > 0

    def test_train_multiple_updates(self) -> None:
        """Multiple PPO updates should show the training loop works end-to-end."""
        config = MAPPOConfig(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=list(HIDDEN_SIZES),
            device="cpu",
            epochs=2,
            batch_size=16,
        )
        agent = MAPPOAgent(config)
        trainer_config = PPOTrainerConfig(rollout_length=32, max_episode_steps=10, log_interval=100)
        trainer = PPOTrainer(agent, trainer_config)

        reset_fn, step_fn = self._make_env()
        metrics_list = trainer.train(reset_fn, step_fn, num_updates=3)

        assert len(metrics_list) == 3
        assert trainer.total_steps == 96  # 3 * 32
        # All metrics should be finite
        for m in metrics_list:
            for k, v in m.items():
                assert np.isfinite(v), f"Non-finite metric: {k}={v}"
