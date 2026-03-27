"""Training loop for FORGE agents.

Provides a basic Trainer for simple env-agent loops and a PPOTrainer
that implements proper rollout collection with GAE advantage estimation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from forge.agents.base_agent import BaseAgent
    from forge.agents.mappo_agent import MAPPOAgent

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPISODES = 1000
DEFAULT_EVAL_INTERVAL = 100
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_LOG_INTERVAL = 10
DEFAULT_EVAL_EPISODES = 10
DEFAULT_ROLLOUT_LENGTH = 2048
DEFAULT_MAX_EPISODE_STEPS = 512
DEFAULT_CHECKPOINT_DIR = "checkpoints"


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
        num_episodes: int = DEFAULT_EVAL_EPISODES,
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


@dataclass
class PPOTrainerConfig:
    """Configuration for PPO training loop.

    Values can be loaded from ForgeConfig via from_forge_config();
    any fields not provided there fall back to the defaults defined here.
    """

    rollout_length: int = DEFAULT_ROLLOUT_LENGTH
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    max_episodes: int = DEFAULT_MAX_EPISODES
    log_interval: int = DEFAULT_LOG_INTERVAL
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    eval_interval: int = DEFAULT_EVAL_INTERVAL
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR

    @classmethod
    def from_forge_config(cls, forge_config: Any) -> PPOTrainerConfig:
        """Build from a ForgeConfig instance."""
        tc = forge_config.training
        sc = forge_config.simulation
        return cls(
            rollout_length=tc.rollout_length,
            max_episode_steps=sc.max_episode_length,
            checkpoint_interval=tc.checkpoint_interval,
        )


class PPOTrainer:
    """Training loop that collects rollouts and performs PPO updates.

    Works with any environment exposing reset()/step() and a MAPPOAgent.

    Args:
        agent: MAPPOAgent instance.
        config: PPOTrainerConfig with loop parameters.
    """

    def __init__(self, agent: MAPPOAgent, config: PPOTrainerConfig) -> None:
        self.agent = agent
        self.config = config
        self._total_steps: int = 0
        self._episode_count: int = 0
        self._episode_rewards: list[float] = []
        logger.info("PPOTrainer initialized: %s", config)

    def collect_rollout(
        self,
        reset_fn: Callable[[], np.ndarray],
        step_fn: Callable[[int], tuple[np.ndarray, float, bool, bool, dict[str, Any]]],
    ) -> dict[str, np.ndarray]:
        """Collect a rollout of experience for PPO training.

        Args:
            reset_fn: Callable returning initial observation.
            step_fn: Callable(action) -> (obs, reward, terminated, truncated, info).

        Returns:
            Rollout batch dict with keys: observations, actions, rewards,
            dones, old_log_probs, values, advantages, returns.
        """
        cfg = self.config
        obs_dim = self.agent.obs_dim
        T = cfg.rollout_length

        # Pre-allocate arrays
        observations = np.zeros((T, obs_dim), dtype=np.float32)
        actions: np.ndarray = np.zeros(T, dtype=np.int64)
        rewards = np.zeros(T, dtype=np.float32)
        dones = np.zeros(T, dtype=np.float32)
        log_probs = np.zeros(T, dtype=np.float32)
        values = np.zeros(T, dtype=np.float32)

        obs = reset_fn()
        episode_reward = 0.0
        episode_steps = 0

        for t in range(T):
            observations[t] = obs
            action, info = self.agent.act(obs)
            actions[t] = action
            log_probs[t] = info["log_prob"]
            values[t] = info["value"]

            obs, reward, terminated, truncated, _info = step_fn(action)
            done = terminated or truncated
            rewards[t] = reward
            dones[t] = float(done)

            episode_reward += reward
            episode_steps += 1
            self._total_steps += 1

            if done or episode_steps >= cfg.max_episode_steps:
                self._episode_count += 1
                self._episode_rewards.append(episode_reward)
                if self._episode_count % cfg.log_interval == 0:
                    recent = self._episode_rewards[-cfg.log_interval :]
                    logger.info(
                        "Episode %d | steps=%d | reward=%.2f | mean_reward=%.2f",
                        self._episode_count,
                        self._total_steps,
                        episode_reward,
                        float(np.mean(recent)),
                    )
                obs = reset_fn()
                episode_reward = 0.0
                episode_steps = 0

        # Bootstrap value for GAE computation
        import torch  # noqa: PLC0415

        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=torch.device(self.agent.device)
            ).unsqueeze(0)
            next_value = float(self.agent.network.get_value(obs_tensor).item())

        # Compute GAE advantages and returns
        advantages, returns = self.agent.compute_gae(rewards, values, dones, next_value)

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "old_log_probs": log_probs,
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }

    def train(
        self,
        reset_fn: Callable[[], np.ndarray],
        step_fn: Callable[[int], tuple[np.ndarray, float, bool, bool, dict[str, Any]]],
        num_updates: int,
    ) -> list[dict[str, float]]:
        """Run the full PPO training loop.

        Args:
            reset_fn: Environment reset function returning initial observation.
            step_fn: Environment step function.
            num_updates: Number of PPO update iterations to perform.

        Returns:
            List of per-update metric dicts.
        """
        all_metrics: list[dict[str, float]] = []

        for update in range(1, num_updates + 1):
            # Collect rollout
            rollout = self.collect_rollout(reset_fn, step_fn)

            # PPO update
            metrics = self.agent.learn(rollout)
            metrics["total_steps"] = float(self._total_steps)
            metrics["episodes"] = float(self._episode_count)
            if self._episode_rewards:
                metrics["mean_reward"] = float(
                    np.mean(self._episode_rewards[-self.config.log_interval :])
                )
            all_metrics.append(metrics)

            if update % self.config.log_interval == 0:
                logger.info(
                    "Update %d/%d | %s",
                    update,
                    num_updates,
                    {k: f"{v:.4f}" for k, v in metrics.items()},
                )

        return all_metrics

    @property
    def total_steps(self) -> int:
        """Total environment steps taken."""
        return self._total_steps

    @property
    def episode_count(self) -> int:
        """Total episodes completed."""
        return self._episode_count

    @property
    def episode_rewards(self) -> list[float]:
        """History of episode rewards."""
        return list(self._episode_rewards)
