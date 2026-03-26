"""Integrated trainer: orchestrates memory, social, and cognitive layers during training.

Extends the PPO training loop with hooks for memory writes, social reward
augmentation, and cross-layer evaluation.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from forge.memory.memory_store import Episode, MemoryStore, MemoryStoreConfig
from forge.social.trust_tracker import SocialConfig, TrustTracker

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_WRITE_INTERVAL = 10
DEFAULT_SOCIAL_REWARD_WEIGHT = 0.3
DEFAULT_DOMAIN_REWARD_WINDOW = 200


@dataclass
class IntegrationTrainerConfig:
    """Configuration for integrated training."""

    memory_write_interval: int = DEFAULT_MEMORY_WRITE_INTERVAL
    social_reward_weight: float = DEFAULT_SOCIAL_REWARD_WEIGHT
    meta_learning_enabled: bool = False
    meta_lr: float = 0.001
    max_episodes: int = 1000
    log_interval: int = 10
    domain_reward_window: int = DEFAULT_DOMAIN_REWARD_WINDOW
    curriculum_domains: list[str] = field(
        default_factory=lambda: ["navigation", "crafting", "social", "combat"]
    )


class IntegratedTrainer:
    """Training loop that integrates memory, social, and cognitive layers.

    This trainer wraps a standard env-agent loop, adding:
    - Memory writes after each episode
    - Social reward augmentation
    - Cross-domain curriculum tracking
    """

    def __init__(
        self,
        num_agents: int,
        config: IntegrationTrainerConfig | None = None,
    ) -> None:
        self.config = config or IntegrationTrainerConfig()
        self.num_agents = num_agents

        # Initialize subsystems
        mem_config = MemoryStoreConfig()
        self.memories = [MemoryStore(i, mem_config) for i in range(num_agents)]
        self.trust = TrustTracker(num_agents, SocialConfig())

        self._episode_count = 0
        self._total_steps = 0
        self._domain_rewards: dict[str, list[float]] = {
            d: [] for d in self.config.curriculum_domains
        }
        logger.info("IntegratedTrainer initialized for %d agents", num_agents)

    def train_episode(
        self,
        reset_fn: Callable[[], np.ndarray],
        step_fn: Callable[[int], tuple[np.ndarray, float, bool, bool, dict[str, Any]]],
        agent_act_fn: Callable[[np.ndarray], int],
        domain: str = "navigation",
    ) -> dict[str, float | str]:
        """Run a single integrated training episode.

        Args:
            reset_fn: Resets the environment, returns initial observation.
            step_fn: Steps the environment with an action.
            agent_act_fn: Agent action selection function.
            domain: Curriculum domain label for this episode.

        Returns:
            Episode metrics (numeric values and string labels).
        """
        obs = reset_fn()
        total_reward = 0.0
        steps = 0
        done = False
        events: list[str] = []

        while not done:
            action = agent_act_fn(obs)
            obs, reward, terminated, truncated, info = step_fn(action)
            done = terminated or truncated

            # Augment reward with social signal (skip computation when weight is zero)
            w = self.config.social_reward_weight
            if w > 0.0:
                social_rewards = self.trust.compute_social_rewards()
                blended_reward = reward * (1 - w) + float(social_rewards.mean()) * w
            else:
                blended_reward = reward

            total_reward += blended_reward
            steps += 1
            self._total_steps += 1

            # Record events from info
            if "event" in info:
                events.append(str(info["event"]))

        # Write episode to memory
        self._episode_count += 1
        if self._episode_count % self.config.memory_write_interval == 0:
            episode = Episode(
                tick_start=self._total_steps - steps,
                tick_end=self._total_steps,
                agent_ids=list(range(self.num_agents)),
                outcome="success" if total_reward > 0 else "failure",
                reward=total_reward,
                tags=[domain],
            )
            for mem in self.memories:
                mem.store_episode(copy.copy(episode))

        # Track per-domain rewards (capped sliding window to bound memory)
        if domain in self._domain_rewards:
            window = self._domain_rewards[domain]
            window.append(total_reward)
            if len(window) > self.config.domain_reward_window:
                self._domain_rewards[domain] = window[-self.config.domain_reward_window :]

        metrics = {
            "total_reward": total_reward,
            "episode_length": float(steps),
            "domain": domain,
            "memory_entries": float(sum(m.total_entries() for m in self.memories)),
        }

        if self._episode_count % self.config.log_interval == 0:
            logger.info("Episode %d | %s", self._episode_count, metrics)

        return metrics

    @property
    def episode_count(self) -> int:
        """Total episodes completed."""
        return self._episode_count

    @property
    def total_steps(self) -> int:
        """Total environment steps taken."""
        return self._total_steps

    def domain_mean_reward(self, domain: str) -> float:
        """Return the mean reward for a curriculum domain."""
        rewards = self._domain_rewards.get(domain, [])
        return float(np.mean(rewards)) if rewards else 0.0
