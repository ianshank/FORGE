"""Batch episode collector for MangoMAS integration.

High-level Python interface for collecting episodes from FORGE
in batch mode for pre-training data generation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from forge.mangomas.config import BatchCollectorConfig

logger = logging.getLogger(__name__)


@dataclass
class EpisodeData:
    """Data from a single collected episode."""

    observations: np.ndarray  # (T, state_dim)
    actions: np.ndarray       # (T,)
    rewards: np.ndarray       # (T,)
    dones: np.ndarray         # (T,)
    total_reward: float = 0.0
    length: int = 0
    seed: int = 0


@dataclass
class BatchResult:
    """Result from a batch episode collection."""

    episodes: list[EpisodeData]
    total_steps: int = 0
    total_time_secs: float = 0.0
    mean_reward: float = 0.0
    mean_length: float = 0.0

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def steps_per_second(self) -> float:
        if self.total_time_secs <= 0:
            return 0.0
        return self.total_steps / self.total_time_secs


class BatchCollector:
    """High-throughput episode collection from FORGE.

    Wraps the Rust batch runner (when available) or provides a pure-Python
    fallback for collecting episodes with configurable policies.
    """

    def __init__(self, config: BatchCollectorConfig | None = None) -> None:
        self.config = config or BatchCollectorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        logger.info(
            "BatchCollector: max_steps=%d, num_envs=%d",
            self.config.max_steps,
            self.config.num_envs,
        )

    def collect(
        self,
        num_episodes: int,
        step_fn: Callable[[np.ndarray, int], tuple[np.ndarray, float, bool]],
        reset_fn: Callable[[int], np.ndarray],
        policy_fn: Callable[[np.ndarray], int] | None = None,
    ) -> BatchResult:
        """Collect episodes using provided environment functions.

        Args:
            num_episodes: Number of episodes to collect.
            step_fn: Callable(obs, action) -> (next_obs, reward, done).
            reset_fn: Callable(seed) -> initial_obs.
            policy_fn: Callable(obs) -> action_id. Random if None.

        Returns:
            BatchResult with all collected episodes.
        """
        start = time.monotonic()
        episodes: list[EpisodeData] = []
        total_steps = 0

        for ep in range(num_episodes):
            seed = int(self._rng.integers(0, 2**31))
            obs = reset_fn(seed)
            state_dim = obs.shape[0] if obs.ndim > 0 else 1

            obs_list = [obs.copy()]
            action_list = []
            reward_list = []
            done_list = []

            for step in range(self.config.max_steps):
                if policy_fn is not None:
                    action = policy_fn(obs)
                else:
                    action = int(self._rng.integers(0, 75))

                next_obs, reward, done = step_fn(obs, action)
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(done)
                obs_list.append(next_obs.copy())
                obs = next_obs
                total_steps += 1

                if done:
                    break

            ep_data = EpisodeData(
                observations=np.array(obs_list, dtype=np.float32),
                actions=np.array(action_list, dtype=np.int64),
                rewards=np.array(reward_list, dtype=np.float32),
                dones=np.array(done_list, dtype=np.float32),
                total_reward=float(sum(reward_list)),
                length=len(action_list),
                seed=seed,
            )
            episodes.append(ep_data)

            if (ep + 1) % 100 == 0:
                logger.debug("Collected %d/%d episodes", ep + 1, num_episodes)

        elapsed = time.monotonic() - start
        rewards = [e.total_reward for e in episodes]
        lengths = [e.length for e in episodes]

        result = BatchResult(
            episodes=episodes,
            total_steps=total_steps,
            total_time_secs=elapsed,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_length=float(np.mean(lengths)) if lengths else 0.0,
        )

        logger.info(
            "Batch collection: %d episodes, %d steps, %.1f steps/s, mean_reward=%.2f",
            result.num_episodes,
            result.total_steps,
            result.steps_per_second,
            result.mean_reward,
        )
        return result
