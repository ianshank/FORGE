"""Real evaluation pipeline for FORGE agents.

Runs a configurable number of episodes, collects standard RL metrics,
and returns an :class:`EvalResult` summary.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    num_episodes: int = 10
    seed: int = 0
    determinism_check: bool = True
    log_per_episode: bool = False


@dataclass
class EvalResult:
    """Aggregated results from an evaluation run."""

    num_episodes: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    episode_length_mean: float
    episode_length_std: float
    tier_success_rates: dict[int, float]
    steps_per_second: float
    determinism_passed: bool
    total_steps: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "num_episodes": self.num_episodes,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "episode_length_mean": self.episode_length_mean,
            "episode_length_std": self.episode_length_std,
            "tier_success_rates": self.tier_success_rates,
            "steps_per_second": self.steps_per_second,
            "determinism_passed": self.determinism_passed,
            "total_steps": self.total_steps,
        }


@runtime_checkable
class _HasAct(Protocol):
    """Minimal agent protocol — only ``act`` is required for evaluation."""

    def act(self, observation: Any) -> tuple[int, dict[str, Any]]: ...


class Evaluator:
    """Runs evaluation episodes and aggregates RL metrics.

    Parameters
    ----------
    config:
        Evaluation settings.  Defaults to :class:`EvalConfig`.
    """

    def __init__(self, config: EvalConfig | None = None) -> None:
        self.config: EvalConfig = config or EvalConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, env: Any, agent: Any) -> EvalResult:
        """Run evaluation episodes and return aggregated metrics.

        Parameters
        ----------
        env:
            A Gymnasium-compatible environment.  Must expose ``reset()``
            and ``step(action)`` methods.
        agent:
            An agent with an ``act(observation)`` method that returns
            ``(action_id, trace_info)``.

        Returns
        -------
        EvalResult
            Aggregated statistics over all episodes.
        """
        cfg = self.config
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        # tier -> list[bool] success per episode
        tier_successes: dict[int, list[bool]] = defaultdict(list)

        total_steps = 0
        t_start = time.perf_counter()

        for ep in range(cfg.num_episodes):
            ep_seed = cfg.seed + ep
            ep_reward, ep_length, ep_tier_successes = self._run_episode(
                env, agent, seed=ep_seed
            )
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            total_steps += ep_length

            for tier, success in ep_tier_successes.items():
                tier_successes[tier].append(success)

            if cfg.log_per_episode:
                logger.info(
                    "Episode %d/%d: reward=%.3f length=%d",
                    ep + 1,
                    cfg.num_episodes,
                    ep_reward,
                    ep_length,
                )

        elapsed = time.perf_counter() - t_start
        steps_per_second = total_steps / elapsed if elapsed > 0 else 0.0

        # Determinism check: re-run episode 0 with the same seed and compare.
        determinism_passed = True
        if cfg.determinism_check and cfg.num_episodes > 0:
            determinism_passed = self._check_determinism(env, agent, episode_rewards[0])

        rewards_arr = np.array(episode_rewards, dtype=float)
        lengths_arr = np.array(episode_lengths, dtype=float)

        tier_success_rates: dict[int, float] = {
            tier: float(np.mean(successes)) for tier, successes in tier_successes.items()
        }

        result = EvalResult(
            num_episodes=cfg.num_episodes,
            reward_mean=float(np.mean(rewards_arr)),
            reward_std=float(np.std(rewards_arr)),
            reward_min=float(np.min(rewards_arr)),
            reward_max=float(np.max(rewards_arr)),
            episode_length_mean=float(np.mean(lengths_arr)),
            episode_length_std=float(np.std(lengths_arr)),
            tier_success_rates=tier_success_rates,
            steps_per_second=steps_per_second,
            determinism_passed=determinism_passed,
            total_steps=total_steps,
        )

        logger.info(
            "Evaluation complete: episodes=%d reward_mean=%.3f±%.3f "
            "steps_per_second=%.1f determinism=%s",
            result.num_episodes,
            result.reward_mean,
            result.reward_std,
            result.steps_per_second,
            result.determinism_passed,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_episode(
        self, env: Any, agent: Any, *, seed: int | None = None
    ) -> tuple[float, int, dict[int, bool]]:
        """Run a single episode.

        Parameters
        ----------
        seed:
            If given, passed to ``env.reset(seed=...)`` for reproducibility.

        Returns
        -------
        tuple[float, int, dict[int, bool]]
            ``(total_reward, step_count, tier_successes)``
        """
        try:
            obs, info = env.reset(seed=seed)
        except TypeError:
            # Fallback for envs that don't accept seed kwarg.
            obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        tier_successes: dict[int, bool] = {}

        while True:
            action, _trace = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            # Collect tier success if reported in info dict.
            # Accept both "tier"/"success" and "task_tier"/"task_success" keys
            # for compatibility across FORGE env variants.
            if isinstance(info, dict):
                tier = info.get("tier", info.get("task_tier"))
                success = info.get("success", info.get("task_success"))
                if tier is not None and success is not None:
                    tier_successes[int(tier)] = bool(success)

            if terminated or truncated:
                break

        return total_reward, steps, tier_successes

    def _check_determinism(self, env: Any, agent: Any, expected_reward: float) -> bool:
        """Re-run episode 0 with the same seed and verify reward matches.

        Returns ``True`` if the rewards are identical (within floating-point
        tolerance), ``False`` otherwise.
        """
        try:
            reward, _length, _tiers = self._run_episode(
                env, agent, seed=self.config.seed
            )
            passed = abs(reward - expected_reward) < 1e-6
            if not passed:
                logger.warning(
                    "Determinism check FAILED: expected %.6f got %.6f",
                    expected_reward,
                    reward,
                )
            return passed
        except Exception:
            logger.exception("Determinism check raised an exception")
            return False
