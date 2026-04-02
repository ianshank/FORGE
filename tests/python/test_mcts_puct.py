"""Tests for PUCT search, Dirichlet noise, virtual loss, and MCTSEvaluator."""

from __future__ import annotations

import numpy as np
from forge.agents.mcts_agent import MCTSAgent, MCTSConfig, MCTSEvaluator, MCTSNode

# ---------------------------------------------------------------------------
# Mock evaluator for testing
# ---------------------------------------------------------------------------

class UniformMockEvaluator:
    """Mock evaluator that returns uniform priors and a fixed value."""

    def __init__(self, action_space_size: int, value: float = 0.5) -> None:
        self._action_space_size = action_space_size
        self._value = value

    def evaluate(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Return uniform policy and fixed value."""
        priors = np.ones(self._action_space_size, dtype=np.float64) / self._action_space_size
        return priors, self._value


class BiasedMockEvaluator:
    """Mock evaluator that puts most prior mass on action 0."""

    def __init__(self, action_space_size: int) -> None:
        self._action_space_size = action_space_size

    def evaluate(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Return biased policy favouring action 0."""
        priors = np.full(self._action_space_size, 0.01, dtype=np.float64)
        priors[0] = 0.9
        priors /= priors.sum()
        return priors, 0.5


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPUCTScore:
    """Tests for the MCTSNode.puct_score method."""

    def test_puct_score_prefers_high_prior(self) -> None:
        """A node with prior=0.8 should score higher than prior=0.1 at equal visits."""
        high = MCTSNode(prior_prob=0.8)
        low = MCTSNode(prior_prob=0.1)
        # Give them the same visit count
        high.visit_count = 3
        high.total_value = 1.5
        low.visit_count = 3
        low.total_value = 1.5
        parent_visits = 20
        c_puct = 1.5
        assert high.puct_score(parent_visits, c_puct) > low.puct_score(parent_visits, c_puct)

    def test_puct_score_non_negative(self) -> None:
        """PUCT score should be >= 0 for valid (non-negative value) inputs."""
        node = MCTSNode(prior_prob=0.5)
        node.visit_count = 10
        node.total_value = 5.0  # Q = 0.5
        score = node.puct_score(parent_visits=100, c_puct=1.5)
        assert score >= 0.0

    def test_puct_score_unvisited_node(self) -> None:
        """An unvisited node should still return a finite non-negative score."""
        node = MCTSNode(prior_prob=0.25)
        score = node.puct_score(parent_visits=10, c_puct=1.5)
        assert score >= 0.0
        assert score != float("inf")


class TestDirichletNoise:
    """Tests for Dirichlet noise application at the root."""

    def test_dirichlet_noise_applied_at_root(self) -> None:
        """Root priors should be modified after Dirichlet noise injection."""
        config = MCTSConfig(use_puct=True, num_simulations=0)
        evaluator = UniformMockEvaluator(action_space_size=4)
        agent = MCTSAgent(config, action_space_size=4, seed=42, evaluator=evaluator)

        # Build a root and set uniform priors
        root = MCTSNode()
        uniform_prior = 0.25
        for a in range(4):
            root.children[a] = MCTSNode(parent=root, action=a, prior_prob=uniform_prior)

        original_priors = [child.prior_prob for child in root.children.values()]

        # Apply noise
        agent._apply_dirichlet_noise(root)

        noisy_priors = [child.prior_prob for child in root.children.values()]

        # At least one prior should have changed
        assert original_priors != noisy_priors
        # Priors should still sum to approximately 1
        assert abs(sum(noisy_priors) - 1.0) < 1e-6


class TestBackwardsCompat:
    """Tests ensuring UCB1 default behaviour is unchanged."""

    def test_backwards_compat_ucb1_default(self) -> None:
        """MCTSConfig() should use UCB1 and produce valid actions."""
        config = MCTSConfig(num_simulations=10, max_depth=5)
        assert config.use_puct is False
        agent = MCTSAgent(config, action_space_size=4, seed=42)
        obs = np.zeros(10, dtype=np.float32)
        action, trace = agent.act(obs)
        assert 0 <= action < 4
        assert "search_depth" in trace
        assert "ucb1_scores" in trace
        assert "visit_counts" in trace


class TestMCTSWithEvaluator:
    """Tests for PUCT mode with a mock evaluator."""

    def test_mcts_with_mock_evaluator(self) -> None:
        """With use_puct=True and a mock evaluator, agent selects valid actions."""
        config = MCTSConfig(use_puct=True, num_simulations=20, max_depth=5)
        evaluator = UniformMockEvaluator(action_space_size=4)
        agent = MCTSAgent(config, action_space_size=4, seed=42, evaluator=evaluator)
        obs = np.zeros(10, dtype=np.float32)
        action, trace = agent.act(obs)
        assert 0 <= action < 4
        assert trace["visit_counts"]  # should have visits

    def test_evaluator_protocol_conformance(self) -> None:
        """Mock evaluators should satisfy the MCTSEvaluator protocol."""
        assert isinstance(UniformMockEvaluator(4), MCTSEvaluator)
        assert isinstance(BiasedMockEvaluator(4), MCTSEvaluator)


class TestVirtualLoss:
    """Tests for virtual loss bookkeeping."""

    def test_virtual_loss_incremented(self) -> None:
        """Virtual loss should increase during selection traversal."""
        config = MCTSConfig(use_puct=True, num_simulations=1, max_depth=10)
        evaluator = UniformMockEvaluator(action_space_size=4)
        agent = MCTSAgent(config, action_space_size=4, seed=42, evaluator=evaluator)

        # Build a small tree manually
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0, prior_prob=0.5)
        grandchild = MCTSNode(parent=child, action=1, prior_prob=0.5)
        root.children[0] = child
        child.children[1] = grandchild
        # Give visits so selection traverses through
        root.visit_count = 5
        child.visit_count = 3
        child.total_value = 1.0

        assert child.virtual_loss == 0
        # Selection should traverse through child, incrementing its virtual_loss
        agent._select(root)
        # virtual_loss on child should have been incremented
        assert child.virtual_loss >= 1

    def test_virtual_loss_decremented_after_backup(self) -> None:
        """Virtual loss should be decremented during backpropagation."""
        node = MCTSNode(prior_prob=0.5)
        node.virtual_loss = 2
        parent = MCTSNode()
        node.parent = parent

        config = MCTSConfig()
        agent = MCTSAgent(config, action_space_size=4, seed=0)
        agent._backpropagate(node, 0.5)
        # virtual_loss should have been decremented by 1
        assert node.virtual_loss == 1
