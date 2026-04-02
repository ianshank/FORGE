"""MCTS agent with UCB1 and PUCT tree search."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from forge.agents.base_agent import AgentConfig, BaseAgent
from forge.agents.random_agent import DEFAULT_ACTION_SPACE_SIZE, DEFAULT_SEED

logger = logging.getLogger(__name__)

DEFAULT_NUM_SIMULATIONS = 100
DEFAULT_MAX_DEPTH = 20
DEFAULT_EXPLORATION_CONSTANT = 1.414
DEFAULT_TEMPERATURE = 1.0
DEFAULT_C_PUCT = 1.5
DEFAULT_DIRICHLET_ALPHA = 0.03
DEFAULT_DIRICHLET_EPSILON = 0.25


@runtime_checkable
class MCTSEvaluator(Protocol):
    """Protocol for neural network evaluation in PUCT MCTS.

    Implementations provide policy priors and value estimates for
    a given observation, replacing random rollouts with learned guidance.
    """

    def evaluate(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (policy_prior, value_estimate) for the given observation.

        Args:
            observation: The current state observation.

        Returns:
            A tuple of:
                policy_prior: ndarray of shape (action_space_size,) with prior probabilities.
                value_estimate: float state value estimate in [0, 1].

        """
        ...


@dataclass
class MCTSConfig(AgentConfig):
    """Configuration for MCTS agent.

    Supports both UCB1 (default) and PUCT search strategies.
    When ``use_puct`` is False (default), behaviour is identical to
    plain UCB1 MCTS.  Set ``use_puct=True`` to enable PUCT selection
    with optional neural-network guidance via an ``MCTSEvaluator``.
    """

    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    max_depth: int = DEFAULT_MAX_DEPTH
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT
    temperature: float = DEFAULT_TEMPERATURE
    c_puct: float = DEFAULT_C_PUCT
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON
    use_puct: bool = False


class MCTSNode:
    """A node in the MCTS search tree.

    Supports both UCB1 and PUCT scoring strategies, as well as
    virtual-loss accounting for future parallel search.
    """

    def __init__(
        self,
        parent: MCTSNode | None = None,
        action: int | None = None,
        prior_prob: float = 0.0,
    ) -> None:
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.children: dict[int, MCTSNode] = {}
        self.parent: MCTSNode | None = parent
        self.action: int | None = action
        self.prior_prob: float = prior_prob
        self.virtual_loss: int = 0

    def ucb1_score(self, parent_visits: int, exploration_constant: float) -> float:
        """Compute the UCB1 score for this node.

        Args:
            parent_visits: Total visit count of the parent node.
            exploration_constant: Controls exploration vs exploitation.

        Returns:
            The UCB1 score (infinity for unvisited nodes).

        """
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.total_value / self.visit_count
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration

    def puct_score(self, parent_visits: int, c_puct: float) -> float:
        """Compute the PUCT score for this node.

        Implements the formula:
            Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))

        Args:
            parent_visits: Total visit count of the parent node.
            c_puct: PUCT exploration constant.

        Returns:
            The PUCT score (always non-negative for valid inputs).

        """
        effective_visits = self.visit_count + self.virtual_loss
        q_value = 0.0 if effective_visits == 0 else self.total_value / effective_visits
        exploration = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + effective_visits)
        return q_value + exploration

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def best_child(self, exploration_constant: float) -> tuple[int, MCTSNode]:
        """Return the child with the highest UCB1 score.

        Args:
            exploration_constant: UCB1 exploration constant.

        Returns:
            A tuple of (action, child_node) for the best child.

        Raises:
            ValueError: If the node has no children.

        """
        best_action = -1
        best_node: MCTSNode | None = None
        best_score = float("-inf")
        for action, child in self.children.items():
            score = child.ucb1_score(self.visit_count, exploration_constant)
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child
        if best_node is None:
            msg = "No children to select from"
            raise ValueError(msg)
        return best_action, best_node

    def best_child_puct(self, c_puct: float) -> tuple[int, MCTSNode]:
        """Return the child with the highest PUCT score.

        Args:
            c_puct: PUCT exploration constant.

        Returns:
            A tuple of (action, child_node) for the best child.

        Raises:
            ValueError: If the node has no children.

        """
        best_action = -1
        best_node: MCTSNode | None = None
        best_score = float("-inf")
        for action, child in self.children.items():
            score = child.puct_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child
        if best_node is None:
            msg = "No children to select from"
            raise ValueError(msg)
        return best_action, best_node


class MCTSAgent(BaseAgent):
    """Agent that uses Monte Carlo Tree Search for action selection.

    Supports two search strategies:
    - UCB1 (default): classic MCTS with random rollouts.
    - PUCT: neural-guided search with prior probabilities and optional
      Dirichlet noise for exploration.

    The strategy is selected via ``MCTSConfig.use_puct``.
    """

    def __init__(
        self,
        config: MCTSConfig,
        action_space_size: int = DEFAULT_ACTION_SPACE_SIZE,
        seed: int = DEFAULT_SEED,
        evaluator: MCTSEvaluator | None = None,
    ) -> None:
        super().__init__(config)
        self.mcts_config = config
        self.action_space_size = action_space_size
        self._rng = np.random.default_rng(seed)
        self.evaluator = evaluator

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Perform MCTS and return the best action with trace info.

        Args:
            observation: Current environment observation.

        Returns:
            A tuple of (action, trace_dict) where trace_dict contains
            search metadata such as visit counts and scores.

        """
        root = MCTSNode()

        # For PUCT, initialise root priors from evaluator or uniform fallback
        if self.mcts_config.use_puct:
            if self.evaluator is not None:
                priors, _ = self.evaluator.evaluate(observation)
            else:
                # Uniform priors when no evaluator — equal exploration across actions
                priors = np.ones(self.action_space_size) / self.action_space_size
            for a in range(self.action_space_size):
                root.children[a] = MCTSNode(parent=root, action=a, prior_prob=float(priors[a]))
            self._apply_dirichlet_noise(root)

        for _ in range(self.mcts_config.num_simulations):
            node = self._select(root)
            node = self._expand(node, observation)
            value = self._simulate(node, observation)
            self._backpropagate(node, value)

        # Select action based on visit counts
        visit_counts: dict[int, int] = {
            action: child.visit_count for action, child in root.children.items()
        }

        if self.mcts_config.use_puct:
            scores: dict[int, float] = {
                action: child.puct_score(root.visit_count, self.mcts_config.c_puct)
                for action, child in root.children.items()
            }
        else:
            scores = {
                action: child.ucb1_score(root.visit_count, self.mcts_config.exploration_constant)
                for action, child in root.children.items()
            }

        if not visit_counts:
            action = int(self._rng.integers(0, self.action_space_size))
        else:
            action = max(visit_counts, key=lambda a: visit_counts[a])

        max_depth = self._compute_tree_depth(root)
        self._step_count += 1

        score_key = "puct_scores" if self.mcts_config.use_puct else "ucb1_scores"
        trace = {
            "search_depth": max_depth,
            score_key: scores,
            "score_type": "puct" if self.mcts_config.use_puct else "ucb1",
            "visit_counts": visit_counts,
        }
        return action, trace

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op learning for MCTS agent. Returns empty metrics."""
        return {}

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node by following best children.

        Uses PUCT or UCB1 scoring depending on config.
        Applies virtual loss during traversal.
        """
        current = node
        depth = 0
        while not current.is_leaf() and depth < self.mcts_config.max_depth:
            if self.mcts_config.use_puct:
                _, current = current.best_child_puct(self.mcts_config.c_puct)
            else:
                _, current = current.best_child(self.mcts_config.exploration_constant)
            current.virtual_loss += 1
            depth += 1
        return current

    def _expand(self, node: MCTSNode, observation: np.ndarray) -> MCTSNode:
        """Expand a leaf node by adding children for all actions.

        When using PUCT with an evaluator, child priors are set from
        the evaluator's policy output.

        Args:
            node: The leaf node to expand.
            observation: Current environment observation.

        Returns:
            A child node to simulate from.

        """
        if node.visit_count > 0 or node.parent is None:
            if self.mcts_config.use_puct and self.evaluator is not None:
                priors, _ = self.evaluator.evaluate(observation)
                for action in range(self.action_space_size):
                    if action not in node.children:
                        node.children[action] = MCTSNode(
                            parent=node, action=action, prior_prob=float(priors[action])
                        )
            else:
                for action in range(self.action_space_size):
                    if action not in node.children:
                        node.children[action] = MCTSNode(parent=node, action=action)
            if node.children:
                action = int(self._rng.integers(0, self.action_space_size))
                return node.children.get(action, next(iter(node.children.values())))
        return node

    def _simulate(self, node: MCTSNode, observation: np.ndarray) -> float:
        """Run a simulation and return a value estimate.

        When using PUCT with an evaluator, returns the evaluator's value
        estimate instead of a random rollout.

        Args:
            node: The node to simulate from.
            observation: Current environment observation.

        Returns:
            A float value estimate for backpropagation.

        """
        if self.mcts_config.use_puct and self.evaluator is not None:
            _, value = self.evaluator.evaluate(observation)
            return value
        return float(self._rng.random())

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate the simulation value back up the tree.

        Also decrements virtual loss that was applied during selection.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            if current.virtual_loss > 0:
                current.virtual_loss -= 1
            current = current.parent

    def _apply_dirichlet_noise(self, root: MCTSNode) -> None:
        """Apply Dirichlet noise to root node priors for exploration.

        Implements: P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)

        Args:
            root: The root node whose children's priors will be modified.

        """
        n_children = len(root.children)
        if n_children == 0:
            return
        noise = self._rng.dirichlet([self.mcts_config.dirichlet_alpha] * n_children)
        eps = self.mcts_config.dirichlet_epsilon
        for i, child in enumerate(root.children.values()):
            child.prior_prob = (1.0 - eps) * child.prior_prob + eps * float(noise[i])

    def _compute_tree_depth(self, node: MCTSNode) -> int:
        """Compute the maximum depth of the search tree."""
        if node.is_leaf():
            return 0
        return 1 + max(self._compute_tree_depth(child) for child in node.children.values())
