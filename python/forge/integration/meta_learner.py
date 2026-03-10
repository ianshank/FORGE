"""Meta-learner: adapts learning rules based on cross-domain performance.

Implements meta-RL by adjusting hyperparameters (learning rate, entropy
coefficient, etc.) based on how quickly the agent adapts to new domains.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetaLearnerConfig:
    """Configuration for the meta-learner."""

    meta_lr: float = 0.001
    adaptation_window: int = 50
    min_lr: float = 1e-5
    max_lr: float = 1e-2


class MetaLearner:
    """Adapts learning hyperparameters based on cross-domain adaptation speed.

    Tracks how quickly rewards improve when switching domains, and adjusts
    the base learning rate accordingly: faster adaptation → maintain LR,
    slower adaptation → increase LR.
    """

    def __init__(self, config: MetaLearnerConfig | None = None) -> None:
        self.config = config or MetaLearnerConfig()
        self.current_lr: float = 3e-4
        self._domain_histories: dict[str, list[float]] = {}
        self._adaptation_scores: list[float] = []
        logger.info("MetaLearner initialized with meta_lr=%.4f", self.config.meta_lr)

    def record_domain_reward(self, domain: str, reward: float) -> None:
        """Record a reward observation for a domain."""
        if domain not in self._domain_histories:
            self._domain_histories[domain] = []
        self._domain_histories[domain].append(reward)

    def adapt(self) -> float:
        """Compute adaptation score and update learning rate.

        Returns:
            Updated learning rate.
        """
        window = self.config.adaptation_window
        scores = []

        for domain, rewards in self._domain_histories.items():
            if len(rewards) < window * 2:
                continue
            early = np.mean(rewards[-window * 2 : -window])
            late = np.mean(rewards[-window:])
            improvement = late - early
            scores.append(improvement)

        if not scores:
            return self.current_lr

        mean_improvement = float(np.mean(scores))
        self._adaptation_scores.append(mean_improvement)

        # If adaptation is slow (low improvement), increase LR
        if mean_improvement < 0.01:
            self.current_lr = min(
                self.config.max_lr,
                self.current_lr * (1 + self.config.meta_lr),
            )
        elif mean_improvement > 0.1:
            self.current_lr = max(
                self.config.min_lr,
                self.current_lr * (1 - self.config.meta_lr),
            )

        return self.current_lr

    @property
    def adaptation_history(self) -> list[float]:
        """Return the history of adaptation scores."""
        return list(self._adaptation_scores)
