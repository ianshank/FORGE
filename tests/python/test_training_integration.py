"""Integration tests for the training pipeline.

These tests verify that the training script components wire together
correctly: environment -> agent -> trainer -> checkpoints.

Also covers previously untested code paths:
- Trainer.train_episode() and evaluate()
- MCTS tree internals (_select, _expand, _simulate, _backpropagate)
- train.py parse_args()
- Shared observation flattening utility
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure python/ and scripts/ are importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from forge.agents.base_agent import AgentConfig
from forge.agents.mcts_agent import MCTSAgent, MCTSConfig, MCTSNode
from forge.agents.random_agent import RandomAgent
from forge.training.trainer import Trainer, TrainerConfig
from forge.utils.observation import compute_obs_dim, flatten_obs
from train import _DEFAULT_EPISODES, _DEFAULT_SEED, parse_args

# ============================================================
# Shared observation utility tests
# ============================================================


class TestFlattenObs:
    """Tests for the shared observation flattening utility."""

    def test_sorted_key_order(self) -> None:
        """Keys are sorted alphabetically for deterministic ordering."""
        obs: dict[str, Any] = {"z_val": [3.0], "a_val": [1.0], "m_val": [2.0]}
        result = flatten_obs(obs)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_multidimensional_values(self) -> None:
        """Multi-dimensional values are ravelled before concatenation."""
        obs: dict[str, Any] = {"grid": np.ones((2, 3)), "scalar": np.array([5.0])}
        result = flatten_obs(obs)
        assert result.shape == (7,)  # 2*3 + 1
        assert result.dtype == np.float32

    def test_empty_obs_raises(self) -> None:
        """Empty observation dict raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            flatten_obs({})

    def test_deterministic(self) -> None:
        """Same input produces identical output."""
        obs: dict[str, Any] = {"a": [1, 2], "b": [3]}
        r1 = flatten_obs(obs)
        r2 = flatten_obs(obs)
        np.testing.assert_array_equal(r1, r2)


# ============================================================
# Trainer.train_episode() and evaluate() tests
# ============================================================


class TestTrainer:
    """Tests for the base Trainer class."""

    @staticmethod
    def _make_env_step(
        max_steps: int = 10,
    ) -> tuple[Any, Any]:
        """Create a mock env_step_fn that terminates after max_steps."""
        state = {"steps": 0, "max": max_steps}

        def step_fn(action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
            state["steps"] += 1
            done = state["steps"] >= state["max"]
            if done:
                state["steps"] = 0  # auto-reset for multi-episode evaluation
            return np.zeros(4, dtype=np.float32), 1.0, done, {}

        def reset_fn() -> None:
            state["steps"] = 0

        return step_fn, reset_fn

    def test_train_episode_returns_metrics(self) -> None:
        """train_episode returns dict with total_reward and episode_length."""
        agent = RandomAgent(AgentConfig(), action_space_size=4, seed=0)
        trainer = Trainer(agent=agent, config=TrainerConfig(log_interval=1))
        step_fn, _ = self._make_env_step(max_steps=5)
        metrics = trainer.train_episode(step_fn)
        assert "total_reward" in metrics
        assert "episode_length" in metrics
        assert metrics["episode_length"] == 5.0
        assert metrics["total_reward"] == 5.0

    def test_evaluate_returns_aggregated(self) -> None:
        """evaluate returns mean_reward and mean_length."""
        agent = RandomAgent(AgentConfig(), action_space_size=4, seed=0)
        trainer = Trainer(agent=agent, config=TrainerConfig(log_interval=1))
        step_fn, _ = self._make_env_step(max_steps=3)
        eval_metrics = trainer.evaluate(step_fn, num_episodes=4)
        assert "mean_reward" in eval_metrics
        assert "mean_length" in eval_metrics
        assert eval_metrics["mean_length"] == pytest.approx(3.0)

    def test_episode_count_increments(self) -> None:
        """Each train_episode call increments the internal counter."""
        agent = RandomAgent(AgentConfig(), action_space_size=2, seed=0)
        trainer = Trainer(agent=agent, config=TrainerConfig())
        step_fn, _ = self._make_env_step(max_steps=2)
        trainer.train_episode(step_fn)
        trainer.train_episode(step_fn)
        assert trainer._episode_count == 2


# ============================================================
# MCTS tree internals tests
# ============================================================


class TestMCTSInternals:
    """Tests for MCTS tree operations (_select, _expand, _simulate, _backpropagate)."""

    @staticmethod
    def _make_agent(sims: int = 10, depth: int = 5) -> MCTSAgent:
        config = MCTSConfig(num_simulations=sims, max_depth=depth)
        return MCTSAgent(config, action_space_size=4, seed=42)

    def test_select_returns_leaf(self) -> None:
        """_select traverses to a leaf node."""
        agent = self._make_agent()
        root = MCTSNode()
        # Root is a leaf, so select should return it
        result = agent._select(root)
        assert result is root

    def test_select_follows_best_child(self) -> None:
        """_select follows UCB1 best children down the tree."""
        agent = self._make_agent()
        root = MCTSNode()
        root.visit_count = 10
        child = MCTSNode(parent=root, action=0)
        child.visit_count = 3
        child.total_value = 2.0
        root.children[0] = child
        child2 = MCTSNode(parent=root, action=1)
        child2.visit_count = 1
        child2.total_value = 0.5
        root.children[1] = child2

        result = agent._select(root)
        assert result.is_leaf()

    def test_expand_adds_children(self) -> None:
        """_expand adds children to a visited leaf node."""
        agent = self._make_agent()
        root = MCTSNode()
        root.visit_count = 1  # Must be visited to expand
        expanded = agent._expand(root)
        assert len(root.children) == 4  # action_space_size
        assert expanded is not None

    def test_expand_root_unvisited(self) -> None:
        """_expand on unvisited root still expands (parent is None)."""
        agent = self._make_agent()
        root = MCTSNode()
        agent._expand(root)
        assert len(root.children) == 4

    def test_simulate_returns_float(self) -> None:
        """_simulate returns a value in [0, 1)."""
        agent = self._make_agent()
        node = MCTSNode()
        for _ in range(20):
            value = agent._simulate(node)
            assert isinstance(value, float)
            assert 0.0 <= value < 1.0

    def test_backpropagate_updates_ancestors(self) -> None:
        """_backpropagate updates visit_count and total_value up the tree."""
        agent = self._make_agent()
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        grandchild = MCTSNode(parent=child, action=1)

        agent._backpropagate(grandchild, 0.5)

        assert grandchild.visit_count == 1
        assert grandchild.total_value == 0.5
        assert child.visit_count == 1
        assert child.total_value == 0.5
        assert root.visit_count == 1
        assert root.total_value == 0.5

    def test_backpropagate_accumulates(self) -> None:
        """Multiple backpropagations accumulate values."""
        agent = self._make_agent()
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child

        agent._backpropagate(child, 0.3)
        agent._backpropagate(child, 0.7)

        assert child.visit_count == 2
        assert child.total_value == pytest.approx(1.0)
        assert root.visit_count == 2

    def test_compute_tree_depth_nested(self) -> None:
        """Tree depth is computed correctly for multi-level trees."""
        agent = self._make_agent()
        root = MCTSNode()
        c1 = MCTSNode(parent=root, action=0)
        c2 = MCTSNode(parent=c1, action=0)
        c3 = MCTSNode(parent=c2, action=0)
        root.children[0] = c1
        c1.children[0] = c2
        c2.children[0] = c3
        assert agent._compute_tree_depth(root) == 3


# ============================================================
# train.py parse_args tests
# ============================================================


class TestParseArgs:
    """Tests for the training script argument parser."""

    def test_defaults(self) -> None:
        """Default arguments should match constants."""
        args = parse_args([])
        assert args.agent == "random"
        assert args.episodes == _DEFAULT_EPISODES
        assert args.seed == _DEFAULT_SEED
        assert args.log_level == "INFO"

    def test_custom_args(self) -> None:
        """Custom arguments are parsed correctly."""
        args = parse_args([
            "--agent", "mappo",
            "--num-updates", "5",
            "--seed", "99",
            "--log-level", "DEBUG",
        ])
        assert args.agent == "mappo"
        assert args.num_updates == 5
        assert args.seed == 99
        assert args.log_level == "DEBUG"

    def test_invalid_agent_raises(self) -> None:
        """Invalid agent type raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--agent", "nonexistent"])


# ============================================================
# Integration tests requiring ForgeGymnasiumEnv
# ============================================================


@pytest.fixture()
def env():
    """Create a ForgeGymnasiumEnv for testing."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

    e = ForgeGymnasiumEnv()
    yield e
    e.close()


class TestComputeObsDim:
    """Tests for compute_obs_dim utility."""

    def test_returns_positive_int(self, env: Any) -> None:
        """compute_obs_dim returns a positive integer."""
        dim = compute_obs_dim(env)
        assert isinstance(dim, int)
        assert dim > 0


class TestRandomAgentTraining:
    """Test random agent episode loop."""

    def test_random_agent_completes_episodes(self, env: Any) -> None:
        """Random agent can run multiple episodes without error."""
        action_dim = env.action_space.n
        agent = RandomAgent(
            config=AgentConfig(name="test_random"),
            action_space_size=action_dim,
            seed=42,
        )

        episodes_completed = 0
        for _ in range(5):
            obs, _info = env.reset()
            flat_obs = flatten_obs(obs)
            done = False
            steps = 0
            while not done and steps < 200:
                action, _trace = agent.act(flat_obs)
                obs, _reward, terminated, truncated, _info = env.step(action)
                flat_obs = flatten_obs(obs)
                done = terminated or truncated
                steps += 1
            episodes_completed += 1

        assert episodes_completed == 5


_torch_available = True
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _torch_available = False


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
class TestMAPPOTraining:
    """Test MAPPO agent with PPOTrainer."""

    def test_mappo_training_completes(self, env: Any) -> None:
        """MAPPO training runs for a small number of updates without error."""
        from forge.agents.mappo_agent import MAPPOAgent, MAPPOConfig  # noqa: PLC0415
        from forge.training.trainer import PPOTrainer, PPOTrainerConfig  # noqa: PLC0415

        obs_dim = compute_obs_dim(env)
        action_dim = env.action_space.n

        config = MAPPOConfig(
            name="test_mappo",
            obs_dim=obs_dim,
            action_dim=action_dim,
            device="cpu",
        )
        agent = MAPPOAgent(config, obs_dim=obs_dim, action_dim=action_dim)

        trainer_config = PPOTrainerConfig(
            rollout_length=64,
            max_episode_steps=50,
            log_interval=1,
        )
        trainer = PPOTrainer(agent=agent, config=trainer_config)

        def reset_fn() -> np.ndarray:
            o, _ = env.reset()
            return flatten_obs(o)

        def step_fn(action: int) -> tuple:
            o, r, term, trunc, info = env.step(action)
            return flatten_obs(o), r, term, trunc, info

        all_metrics = trainer.train(reset_fn, step_fn, num_updates=2)

        assert len(all_metrics) == 2
        assert trainer.total_steps > 0

    def test_training_metrics_keys(self, env: Any) -> None:
        """Training metrics contain expected keys."""
        from forge.agents.mappo_agent import MAPPOAgent, MAPPOConfig  # noqa: PLC0415
        from forge.training.trainer import PPOTrainer, PPOTrainerConfig  # noqa: PLC0415

        obs_dim = compute_obs_dim(env)
        action_dim = env.action_space.n

        config = MAPPOConfig(
            name="test_keys",
            obs_dim=obs_dim,
            action_dim=action_dim,
            device="cpu",
        )
        agent = MAPPOAgent(config, obs_dim=obs_dim, action_dim=action_dim)
        trainer = PPOTrainer(
            agent=agent,
            config=PPOTrainerConfig(rollout_length=32, max_episode_steps=30),
        )

        def reset_fn() -> np.ndarray:
            o, _ = env.reset()
            return flatten_obs(o)

        def step_fn(action: int) -> tuple:
            o, r, term, trunc, info = env.step(action)
            return flatten_obs(o), r, term, trunc, info

        metrics_list = trainer.train(reset_fn, step_fn, num_updates=1)
        assert len(metrics_list) == 1

        m = metrics_list[0]
        assert "policy_loss" in m
        assert "value_loss" in m
        assert "entropy" in m
        assert "total_steps" in m
