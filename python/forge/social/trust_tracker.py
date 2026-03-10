"""Trust and reputation tracking for multi-agent training."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SocialConfig:
    """Configuration for the social tracking system."""

    trust_initial: float = 0.5
    trust_update_rate: float = 0.1
    cooperation_reward_weight: float = 0.3
    betrayal_penalty: float = -0.5
    alliance_threshold: float = 0.7


class TrustTracker:
    """Tracks pairwise trust between agents."""

    def __init__(self, num_agents: int, config: SocialConfig | None = None) -> None:
        self.config = config or SocialConfig()
        self.num_agents = num_agents
        self.trust_matrix: np.ndarray = np.full(
            (num_agents, num_agents), self.config.trust_initial, dtype=np.float32
        )
        self.interaction_count: np.ndarray = np.zeros(
            (num_agents, num_agents), dtype=np.int32
        )
        self.reputation: np.ndarray = np.zeros(num_agents, dtype=np.float32)
        self._coop_counts: np.ndarray = np.zeros(num_agents, dtype=np.int32)
        self._hostile_counts: np.ndarray = np.zeros(num_agents, dtype=np.int32)
        logger.info("TrustTracker created for %d agents", num_agents)

    def record_cooperation(self, agent_a: int, agent_b: int) -> None:
        """Record a cooperative interaction."""
        lr = self.config.trust_update_rate
        self.trust_matrix[agent_a, agent_b] = min(
            1.0, self.trust_matrix[agent_a, agent_b] + lr
        )
        self.trust_matrix[agent_b, agent_a] = min(
            1.0, self.trust_matrix[agent_b, agent_a] + lr
        )
        self.interaction_count[agent_a, agent_b] += 1
        self.interaction_count[agent_b, agent_a] += 1
        self._coop_counts[agent_a] += 1
        self._coop_counts[agent_b] += 1
        self._update_reputation(agent_a)
        self._update_reputation(agent_b)

    def record_hostility(self, attacker: int, defender: int) -> None:
        """Record a hostile interaction."""
        lr = self.config.trust_update_rate
        self.trust_matrix[attacker, defender] = max(
            0.0, self.trust_matrix[attacker, defender] - lr
        )
        self.trust_matrix[defender, attacker] = max(
            0.0, self.trust_matrix[defender, attacker] - lr
        )
        self.interaction_count[attacker, defender] += 1
        self.interaction_count[defender, attacker] += 1
        self._hostile_counts[attacker] += 1
        self._update_reputation(attacker)

    def compute_social_rewards(self) -> np.ndarray:
        """Compute per-agent social rewards based on trust and reputation."""
        n = self.num_agents
        rewards = np.zeros(n, dtype=np.float32)
        w = self.config.cooperation_reward_weight
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mean_trust = float(self.trust_matrix[mask, i].mean()) if n > 1 else 0.0
            trust_delta = mean_trust - self.config.trust_initial
            rewards[i] = trust_delta * w + self.reputation[i] * (1 - w)
        return rewards

    def _update_reputation(self, agent: int) -> None:
        coop = float(self._coop_counts[agent])
        hostile = float(self._hostile_counts[agent])
        total = coop + hostile
        if total > 0:
            self.reputation[agent] = (coop - hostile) / total
