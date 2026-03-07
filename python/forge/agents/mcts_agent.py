"""MCTS agent with UCB1 tree search."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.agents.base_agent import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)

DEFAULT_NUM_SIMULATIONS = 100
DEFAULT_MAX_DEPTH = 20
DEFAULT_EXPLORATION_CONSTANT = 1.414
DEFAULT_TEMPERATURE = 1.0


@dataclass
class MCTSConfig(AgentConfig):
    """Configuration for MCTS agent."""

    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    max_depth: int = DEFAULT_MAX_DEPTH
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT
    temperature: float = DEFAULT_TEMPERATURE


class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(
        self,
        parent: MCTSNode | None = None,
        action: int | None = None,
    ) -> None:
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.children: dict[int, MCTSNode] = {}
        self.parent: MCTSNode | None = parent
        self.action: int | None = action

    def ucb1_score(
        self, parent_visits: int, exploration_constant: float
    ) -> float:
        """Compute the UCB1 score for this node."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.total_value / self.visit_count
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        return exploitation + exploration

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def best_child(
        self, exploration_constant: float
    ) -> tuple[int, MCTSNode]:
        """Return the child with the highest UCB1 score."""
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


class MCTSAgent(BaseAgent):
    """Agent that uses Monte Carlo Tree Search with UCB1 for action selection."""

    def __init__(
        self,
        config: MCTSConfig,
        action_space_size: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__(config)
        self.mcts_config = config
        self.action_space_size = action_space_size
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        """Perform MCTS and return the best action with trace info."""
        root = MCTSNode()

        for _ in range(self.mcts_config.num_simulations):
            node = self._select(root)
            node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)

        # Select action based on visit counts
        visit_counts: dict[int, int] = {
            action: child.visit_count for action, child in root.children.items()
        }
        ucb1_scores: dict[int, float] = {
            action: child.ucb1_score(
                root.visit_count, self.mcts_config.exploration_constant
            )
            for action, child in root.children.items()
        }

        if not visit_counts:
            action = int(self._rng.integers(0, self.action_space_size))
        else:
            action = max(visit_counts, key=lambda a: visit_counts[a])

        max_depth = self._compute_tree_depth(root)
        self._step_count += 1

        trace = {
            "search_depth": max_depth,
            "ucb1_scores": ucb1_scores,
            "visit_counts": visit_counts,
        }
        return action, trace

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """No-op learning for MCTS agent. Returns empty metrics."""
        return {}

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node by following UCB1 best children."""
        current = node
        depth = 0
        while not current.is_leaf() and depth < self.mcts_config.max_depth:
            _, current = current.best_child(self.mcts_config.exploration_constant)
            depth += 1
        return current

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a leaf node by adding children for all actions."""
        if node.visit_count > 0 or node.parent is None:
            for action in range(self.action_space_size):
                if action not in node.children:
                    node.children[action] = MCTSNode(parent=node, action=action)
            if node.children:
                action = int(self._rng.integers(0, self.action_space_size))
                return node.children.get(action, next(iter(node.children.values())))
        return node

    def _simulate(self, node: MCTSNode) -> float:
        """Run a random rollout and return a simulated value."""
        return float(self._rng.random())

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate the simulation value back up the tree."""
        current: MCTSNode | None = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    def _compute_tree_depth(self, node: MCTSNode) -> int:
        """Compute the maximum depth of the search tree."""
        if node.is_leaf():
            return 0
        return 1 + max(
            self._compute_tree_depth(child) for child in node.children.values()
        )
