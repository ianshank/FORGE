"""Tests for FORGE agent framework."""

from __future__ import annotations

import random
import tempfile

import numpy as np
import pytest
from forge.agents.base_agent import AgentConfig, BaseAgent
from forge.agents.mcts_agent import MCTSAgent, MCTSConfig, MCTSNode
from forge.agents.random_agent import RandomAgent
from forge.traces.decision_trace import DecisionTrace
from forge.training.buffer import RolloutBuffer
from forge.training.checkpointing import CheckpointManager
from forge.utils.device import get_device
from forge.utils.seed import set_all_seeds


class TestBaseAgent:
    """Tests for BaseAgent ABC."""

    def test_cannot_instantiate(self) -> None:
        """BaseAgent is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent(AgentConfig())


class TestRandomAgent:
    """Tests for RandomAgent."""

    @pytest.mark.parametrize("action_space_size", [1, 4, 8, 16, 40])
    def test_act_returns_valid_action(self, action_space_size: int) -> None:
        """act() should return an action within the action space."""
        agent = RandomAgent(AgentConfig(), action_space_size=action_space_size, seed=0)
        obs = np.zeros(10, dtype=np.float32)
        for _ in range(20):
            action, info = agent.act(obs)
            assert 0 <= action < action_space_size
            assert isinstance(info, dict)

    def test_learn_returns_dict(self) -> None:
        """learn() should return an empty dict for random agent."""
        agent = RandomAgent(AgentConfig())
        result = agent.learn({})
        assert isinstance(result, dict)
        assert len(result) == 0

    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_deterministic_with_same_seed(self, seed: int) -> None:
        """Two agents with the same seed should produce identical actions."""
        agent_a = RandomAgent(AgentConfig(), action_space_size=10, seed=seed)
        agent_b = RandomAgent(AgentConfig(), action_space_size=10, seed=seed)
        obs = np.zeros(10, dtype=np.float32)
        for _ in range(20):
            a, _ = agent_a.act(obs)
            b, _ = agent_b.act(obs)
            assert a == b


class TestMCTSNode:
    """Tests for MCTSNode UCB1 scoring."""

    def test_ucb1_unvisited_is_inf(self) -> None:
        """Unvisited nodes should have infinite UCB1 score."""
        node = MCTSNode()
        score = node.ucb1_score(parent_visits=10, exploration_constant=1.414)
        assert score == float("inf")

    @pytest.mark.parametrize(
        "visit_count,total_value,parent_visits",
        [(5, 3.0, 20), (1, 0.5, 2), (100, 50.0, 1000)],
    )
    def test_ucb1_visited(
        self, visit_count: int, total_value: float, parent_visits: int
    ) -> None:
        """Visited nodes should return a finite UCB1 score."""
        node = MCTSNode()
        node.visit_count = visit_count
        node.total_value = total_value
        score = node.ucb1_score(parent_visits=parent_visits, exploration_constant=1.414)
        assert isinstance(score, float)
        assert score > 0
        assert score != float("inf")

    def test_is_leaf(self) -> None:
        """A node with no children is a leaf."""
        node = MCTSNode()
        assert node.is_leaf()
        node.children[0] = MCTSNode(parent=node, action=0)
        assert not node.is_leaf()


class TestMCTSAgent:
    """Tests for MCTSAgent."""

    def test_act_returns_action_and_trace(self) -> None:
        """act() should return a valid action and trace dict."""
        config = MCTSConfig(num_simulations=10, max_depth=5)
        agent = MCTSAgent(config, action_space_size=4, seed=42)
        obs = np.zeros(10, dtype=np.float32)
        action, trace = agent.act(obs)
        assert 0 <= action < 4
        assert "search_depth" in trace
        assert "ucb1_scores" in trace
        assert "visit_counts" in trace


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_add_and_len(self) -> None:
        """Adding items should increase buffer length."""
        buf = RolloutBuffer(capacity=10, obs_shape=(4,))
        assert len(buf) == 0
        buf.add(np.zeros(4), 0, 1.0, False, {})
        assert len(buf) == 1

    def test_sample(self) -> None:
        """Sampling should return correctly shaped arrays."""
        buf = RolloutBuffer(capacity=10, obs_shape=(4,))
        for i in range(5):
            buf.add(np.ones(4) * i, i % 3, float(i), False, {})
        batch = buf.sample(batch_size=3)
        assert batch["observations"].shape == (3, 4)
        assert batch["actions"].shape == (3,)
        assert batch["rewards"].shape == (3,)

    def test_clear(self) -> None:
        """clear() should reset the buffer to empty."""
        buf = RolloutBuffer(capacity=10, obs_shape=(4,))
        buf.add(np.zeros(4), 0, 1.0, False, {})
        buf.clear()
        assert len(buf) == 0

    @pytest.mark.parametrize("capacity", [1, 3, 10, 100])
    def test_is_full(self, capacity: int) -> None:
        """is_full() should return True when buffer is at capacity."""
        buf = RolloutBuffer(capacity=capacity, obs_shape=(2,))
        for _i in range(capacity):
            buf.add(np.zeros(2), 0, 0.0, False, {})
        assert buf.is_full()


class TestDecisionTrace:
    """Tests for DecisionTrace."""

    def test_roundtrip(self) -> None:
        """to_dict/from_dict should preserve all fields."""
        trace = DecisionTrace(
            tick=42,
            agent_id="test",
            action=3,
            confidence=0.95,
            search_depth=5,
            ucb1_score=1.5,
            intent_label="explore",
            preconditions=["has_key"],
            expected_outcome="open_door",
        )
        d = trace.to_dict()
        restored = DecisionTrace.from_dict(d)
        assert restored == trace


class TestSeed:
    """Tests for seed utilities."""

    def test_deterministic(self) -> None:
        """set_all_seeds should produce deterministic numpy and stdlib results."""
        set_all_seeds(123)
        a_np = np.random.random()
        a_py = random.random()
        set_all_seeds(123)
        b_np = np.random.random()
        b_py = random.random()
        assert a_np == b_np
        assert a_py == b_py


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_and_load(self) -> None:
        """save/load_latest should roundtrip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_checkpoints=3)
            agent = RandomAgent(AgentConfig(name="test_agent"))
            agent._step_count = 42

            manager.save(agent, episode=10, metrics={"reward": 1.5})

            new_agent = RandomAgent(AgentConfig(name="test_agent"))
            metadata = manager.load_latest(new_agent)
            assert metadata is not None
            assert metadata["episode"] == 10
            assert new_agent.step_count == 42

    def test_list_checkpoints(self) -> None:
        """list_checkpoints should return saved checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            agent = RandomAgent(AgentConfig())
            manager.save(agent, episode=1, metrics={})
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 1


class TestRandomAgentDeterminism:
    """Tests for RandomAgent seed determinism."""

    def test_same_seed_same_actions(self) -> None:
        """Two RandomAgents with the same seed should produce identical action sequences."""
        num_steps = 20
        obs = np.zeros(10, dtype=np.float32)

        agent_a = RandomAgent(AgentConfig(), action_space_size=4, seed=99)
        agent_b = RandomAgent(AgentConfig(), action_space_size=4, seed=99)

        actions_a = [agent_a.act(obs)[0] for _ in range(num_steps)]
        actions_b = [agent_b.act(obs)[0] for _ in range(num_steps)]
        assert actions_a == actions_b

    def test_different_seed_different_actions(self) -> None:
        """Two RandomAgents with different seeds should (very likely) differ."""
        num_steps = 50
        obs = np.zeros(10, dtype=np.float32)

        agent_a = RandomAgent(AgentConfig(), action_space_size=8, seed=1)
        agent_b = RandomAgent(AgentConfig(), action_space_size=8, seed=2)

        actions_a = [agent_a.act(obs)[0] for _ in range(num_steps)]
        actions_b = [agent_b.act(obs)[0] for _ in range(num_steps)]
        # With 8 actions over 50 steps, identical sequences are astronomically unlikely.
        assert actions_a != actions_b


class TestMCTSAgentEdgeCases:
    """Edge-case tests for MCTSAgent."""

    def test_zero_simulations(self) -> None:
        """With 0 simulations the agent should still return a valid action (random fallback)."""
        config = MCTSConfig(num_simulations=0, max_depth=5)
        agent = MCTSAgent(config, action_space_size=4, seed=42)
        obs = np.zeros(10, dtype=np.float32)
        action, trace = agent.act(obs)
        assert 0 <= action < 4
        # With 0 simulations, visit_counts should be empty (random fallback).
        assert trace["visit_counts"] == {}

    def test_single_action_space(self) -> None:
        """With action_space_size=1 the agent has no real choice."""
        config = MCTSConfig(num_simulations=10, max_depth=5)
        agent = MCTSAgent(config, action_space_size=1, seed=42)
        obs = np.zeros(10, dtype=np.float32)
        action, _trace = agent.act(obs)
        assert action == 0

    def test_single_simulation(self) -> None:
        """With num_simulations=1 the agent should still function."""
        config = MCTSConfig(num_simulations=1, max_depth=5)
        agent = MCTSAgent(config, action_space_size=4, seed=42)
        obs = np.zeros(10, dtype=np.float32)
        action, trace = agent.act(obs)
        assert 0 <= action < 4
        assert trace["search_depth"] >= 0


class TestDevice:
    """Tests for device detection."""

    def test_get_device_returns_string(self) -> None:
        """get_device() should return a recognized device string."""
        device = get_device()
        assert device in ("cuda", "mps", "cpu")
