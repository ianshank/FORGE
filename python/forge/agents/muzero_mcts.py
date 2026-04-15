"""MuZero MCTS: Monte Carlo Tree Search operating in latent space.

Unlike the standard MCTS in :mod:`forge.agents.mcts_agent` which requires a
full environment forward model, this planner uses a learned world model
(:class:`~forge.models.muzero_world_model.MuZeroWorldModel`) to search
entirely in latent space.

Usage::

    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.agents.muzero_mcts import MuZeroMCTS, MuZeroMCTSConfig

    model = MuZeroWorldModel(MuZeroConfig(obs_dim=920, action_dim=75))
    mcts = MuZeroMCTS(model, MuZeroMCTSConfig())

    action, info = mcts.search(observation)
"""

from __future__ import annotations

__all__ = ["MuZeroMCTS", "MuZeroMCTSConfig"]

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from forge.models.muzero_world_model import MuZeroWorldModel

logger = logging.getLogger(__name__)

# --- Configuration ---

DEFAULT_NUM_SIMULATIONS: int = 50
DEFAULT_C_PUCT: float = 1.25
DEFAULT_DIRICHLET_ALPHA: float = 0.3
DEFAULT_DIRICHLET_EPSILON: float = 0.25
DEFAULT_DISCOUNT: float = 0.997
DEFAULT_MAX_DEPTH: int = 50
DEFAULT_TEMPERATURE: float = 1.0


@dataclass
class MuZeroMCTSConfig:
    """Configuration for MuZero MCTS search.

    All parameters have sensible defaults and can be overridden.

    Attributes:
        num_simulations: Number of MCTS simulations per search.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Concentration parameter for Dirichlet noise at root.
        dirichlet_epsilon: Mixing weight for Dirichlet noise.
        discount: Reward discount factor.
        max_depth: Maximum search tree depth.
        temperature: Temperature for action selection from visit counts.
        add_exploration_noise: Whether to add Dirichlet noise at root.
    """

    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    c_puct: float = DEFAULT_C_PUCT
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON
    discount: float = DEFAULT_DISCOUNT
    max_depth: int = DEFAULT_MAX_DEPTH
    temperature: float = DEFAULT_TEMPERATURE
    add_exploration_noise: bool = True


# --- Tree Node ---


class _MinMaxStats:
    """Tracks min and max Q-values for normalization within the search tree.

    Normalizes Q-values to [0, 1] to ensure the PUCT exploration bonus
    is scale-invariant across different environments and training stages.
    """

    def __init__(self) -> None:
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def update(self, value: float) -> None:
        """Update tracked min/max with a new value."""
        self._min = min(self._min, value)
        self._max = max(self._max, value)

    def normalize(self, value: float) -> float:
        """Normalize a value to [0, 1] based on tracked range.

        Returns 0.0 if min == max (no spread observed).
        """
        if self._max > self._min:
            return (value - self._min) / (self._max - self._min)
        return 0.0


class _MCTSNode:
    """A node in the MuZero MCTS tree.

    Each node stores a latent state (from the world model) and statistics
    for PUCT selection.
    """

    __slots__ = (
        "children",
        "is_expanded",
        "latent_state",
        "prior",
        "reward",
        "value_sum",
        "visit_count",
    )

    def __init__(self, prior: float) -> None:
        self.latent_state: np.ndarray | None = None
        self.reward: float = 0.0
        self.prior: float = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, _MCTSNode] = {}
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        """Mean Q-value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# --- MCTS Search ---


class MuZeroMCTS:
    """MuZero Monte Carlo Tree Search planner.

    Performs MCTS in latent space using a learned world model. Each search
    call builds a fresh tree from the given observation.

    Args:
        model: A trained :class:`MuZeroWorldModel`.
        config: MCTS search configuration.
    """

    def __init__(
        self,
        model: MuZeroWorldModel,
        config: MuZeroMCTSConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config or MuZeroMCTSConfig()
        self._action_dim = model.config.action_dim
        logger.info(
            "MuZeroMCTS: simulations=%d, c_puct=%.2f, actions=%d",
            self._config.num_simulations,
            self._config.c_puct,
            self._action_dim,
        )

    @property
    def config(self) -> MuZeroMCTSConfig:
        """Return the MCTS configuration."""
        return self._config

    def search(
        self,
        observation: np.ndarray,
        *,
        temperature: float | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Run MCTS search from an observation and return the selected action.

        Args:
            observation: Flat observation array of shape ``(obs_dim,)``.
            temperature: Override the config temperature for this search.
                Use 0.0 for greedy (argmax visits), >0 for stochastic.

        Returns:
            Tuple of (action_id, info_dict) where info_dict contains:
            - ``visit_counts``: Visit count array for all actions.
            - ``action_probs``: Normalized visit-count distribution.
            - ``root_value``: Estimated value at the root.
        """
        temp = temperature if temperature is not None else self._config.temperature

        # Build root node
        root = _MCTSNode(prior=0.0)
        output = self._model.initial_inference(observation)
        root.latent_state = output.latent_state
        root.is_expanded = True

        # Expand root children with initial policy
        policy_probs = _softmax(output.policy_logits)
        for action_id in range(self._action_dim):
            root.children[action_id] = _MCTSNode(prior=float(policy_probs[action_id]))

        # Add exploration noise at root
        if self._config.add_exploration_noise:
            self._add_dirichlet_noise(root)

        # Run simulations
        min_max = _MinMaxStats()
        for _ in range(self._config.num_simulations):
            self._run_simulation(root, min_max)

        # Collect visit counts
        visit_counts = np.zeros(self._action_dim, dtype=np.float32)
        for action_id, child in root.children.items():
            visit_counts[action_id] = child.visit_count

        # Select action based on temperature
        action_probs = self._visit_counts_to_probs(visit_counts, temp)
        if temp == 0.0:
            action_id = int(np.argmax(visit_counts))
        else:
            action_id = int(np.random.choice(self._action_dim, p=action_probs))

        return action_id, {
            "visit_counts": visit_counts,
            "action_probs": action_probs,
            "root_value": output.value,
        }

    def _run_simulation(self, root: _MCTSNode, min_max: _MinMaxStats) -> None:
        """Run one MCTS simulation: select -> expand -> backpropagate."""
        node = root
        search_path: list[_MCTSNode] = [root]
        action_history: list[int] = []

        # Selection: traverse tree using PUCT until reaching an unexpanded node
        depth = 0
        while node.is_expanded and depth < self._config.max_depth:
            action_id, child = self._select_child(node, min_max)
            action_history.append(action_id)
            search_path.append(child)
            node = child
            depth += 1

        # Expansion: use dynamics model to expand the leaf node
        parent = search_path[-2] if len(search_path) >= 2 else root
        if not node.is_expanded and len(action_history) > 0:
            last_action = action_history[-1]
            parent_latent = parent.latent_state
            if parent_latent is not None:
                output = self._model.recurrent_inference(parent_latent, last_action)
                node.latent_state = output.latent_state
                node.reward = output.reward
                node.is_expanded = True

                # Create children with policy priors
                policy_probs = _softmax(output.policy_logits)
                for a in range(self._action_dim):
                    node.children[a] = _MCTSNode(prior=float(policy_probs[a]))

                value = output.value
            else:
                value = 0.0
        # Already expanded or at max depth; use existing value estimate
        elif node.latent_state is not None:
            output = self._model.recurrent_inference(
                node.latent_state, action_history[-1] if action_history else 0
            )
            value = output.value
        else:
            value = 0.0

        # Backpropagation
        self._backpropagate(search_path, value, min_max)

    def _select_child(self, node: _MCTSNode, min_max: _MinMaxStats) -> tuple[int, _MCTSNode]:
        """Select the child with highest PUCT score.

        PUCT formula:
            score = normalized_Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)

        Args:
            node: Parent node.
            min_max: Running min/max tracker for Q normalization.

        Returns:
            Tuple of (action_id, child_node).
        """
        best_score = float("-inf")
        best_action = 0
        best_child = next(iter(node.children.values()))

        parent_visits_sqrt = math.sqrt(node.visit_count)

        for action_id, child in node.children.items():
            q_value = min_max.normalize(child.value) if child.visit_count > 0 else 0.0
            exploration = (
                self._config.c_puct * child.prior * parent_visits_sqrt / (1 + child.visit_count)
            )
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_action = action_id
                best_child = child

        return best_action, best_child

    def _backpropagate(
        self,
        search_path: list[_MCTSNode],
        value: float,
        min_max: _MinMaxStats,
    ) -> None:
        """Backpropagate value through the search path with discounting."""
        discount = self._config.discount
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max.update(node.value)
            value = node.reward + discount * value

    def _add_dirichlet_noise(self, root: _MCTSNode) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        alpha = self._config.dirichlet_alpha
        epsilon = self._config.dirichlet_epsilon
        noise = np.random.dirichlet([alpha] * self._action_dim)

        for action_id, child in root.children.items():
            child.prior = (1 - epsilon) * child.prior + epsilon * float(noise[action_id])

    @staticmethod
    def _visit_counts_to_probs(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
        """Convert visit counts to a probability distribution.

        Args:
            visit_counts: Raw visit counts per action.
            temperature: Softmax temperature. 0.0 = greedy.

        Returns:
            Probability distribution over actions.
        """
        if temperature == 0.0:
            probs = np.zeros_like(visit_counts)
            best = np.argmax(visit_counts)
            probs[best] = 1.0
            return probs  # type: ignore[no-any-return]

        # Temperature-scaled softmax over log visit counts
        total = visit_counts.sum()
        if total == 0:
            return np.ones_like(visit_counts) / len(visit_counts)  # type: ignore[no-any-return]

        counts_temp = visit_counts ** (1.0 / temperature)
        total_temp = counts_temp.sum()
        if total_temp == 0:
            return np.ones_like(visit_counts) / len(visit_counts)  # type: ignore[no-any-return]
        return counts_temp / total_temp  # type: ignore[no-any-return]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1D array."""
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()  # type: ignore[no-any-return]
