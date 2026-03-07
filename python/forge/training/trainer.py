"""Training loop for FORGE agents."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from forge.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPISODES = 1000
DEFAULT_EVAL_INTERVAL = 100
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_LOG_INTERVAL = 10


@dataclass
class TrainerConfig:
    """Configuration for the training loop."""

    max_episodes: int = DEFAULT_MAX_EPISODES
    eval_interval: int = DEFAULT_EVAL_INTERVAL
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    log_interval: int = DEFAULT_LOG_INTERVAL


class Trainer:
    """Manages the training loop for a FORGE agent."""

    def __init__(self, agent: BaseAgent, config: TrainerConfig) -> None:
        self.agent = agent
        self.config = config
        self._episode_count: int = 0
        logger.info("Trainer initialized with config: %s", config)

    def train_episode(
        self, env_step_fn: Callable[[int], tuple[np.ndarray, float, bool, dict[str, Any]]]
    ) -> dict[str, float]:
        """Run a single training episode.

        Args:
            env_step_fn: A callable that takes an action and returns
                (observation, reward, done, info).

        Returns:
            Episode metrics including total_reward and episode_length.
        """
        obs = np.zeros(1, dtype=np.float32)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = self.agent.act(obs)
            obs, reward, done, _ = env_step_fn(action)
            total_reward += reward
            steps += 1

        self._episode_count += 1
        metrics = {"total_reward": total_reward, "episode_length": float(steps)}
        if self._episode_count % self.config.log_interval == 0:
            logger.info("Episode %d: %s", self._episode_count, metrics)
        return metrics

    def evaluate(
        self,
        env_step_fn: Callable[[int], tuple[np.ndarray, float, bool, dict[str, Any]]],
        num_episodes: int = 10,
    ) -> dict[str, float]:
        """Evaluate the agent over multiple episodes.

        Args:
            env_step_fn: Environment step function.
            num_episodes: Number of evaluation episodes.

        Returns:
            Aggregated metrics with mean_reward and mean_length.
        """
        rewards: list[float] = []
        lengths: list[float] = []

        for _ in range(num_episodes):
            metrics = self.train_episode(env_step_fn)
            rewards.append(metrics["total_reward"])
            lengths.append(metrics["episode_length"])

        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
        }
