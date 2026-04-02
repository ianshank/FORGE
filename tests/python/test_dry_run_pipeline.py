"""Integration tests that run the full training pipeline in dry-run mode.
These tests use REAL environments (or RealisticFakeEnv) — no MagicMock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from forge.agents.base_agent import AgentConfig
from forge.agents.random_agent import RandomAgent
from forge.config import DryRunConfig, ForgeConfig
from forge.evaluation.evaluator import EvalConfig, Evaluator
from forge.testing.env_factory import FakeEnvConfig, create_env
from forge.utils.metrics import MetricsTracker


class TestDryRunPipeline:
    """Full pipeline tests using dry-run configuration with real data."""

    @staticmethod
    def _make_env(max_steps: int = 10, seed: int = 42) -> object:
        """Create a real or realistic-fake env — never MagicMock.

        ``force_fake=True`` ensures we always get a ``RealisticFakeEnv`` in
        environments where the Rust extension has not been compiled (the Python
        stub package is importable but the native ``forge_env.forge_env``
        module may not be present).
        """
        return create_env(
            config=FakeEnvConfig(max_episode_length=max_steps, seed=seed),
            force_fake=True,
        )

    def test_random_agent_dry_run_real_env(self) -> None:
        config = ForgeConfig(dry_run=DryRunConfig(enabled=True))
        effective = config.effective_simulation()
        assert effective.grid_size == 8
        env = self._make_env(max_steps=effective.max_episode_length)
        agent = RandomAgent(
            AgentConfig(name="dry_run_random"),
            action_space_size=env.action_space.n,
        )
        evaluator = Evaluator(
            EvalConfig(
                num_episodes=config.dry_run.max_episodes,
                seed=config.dry_run.seed,
            )
        )
        result = evaluator.evaluate(env, agent)
        assert result.num_episodes == config.dry_run.max_episodes
        assert result.total_steps > 0
        assert result.steps_per_second > 0

    def test_metrics_tracker_with_real_episodes(self) -> None:
        tracker = MetricsTracker()
        config = ForgeConfig(dry_run=DryRunConfig(enabled=True))
        env = self._make_env(max_steps=10)
        agent = RandomAgent(
            AgentConfig(name="metrics_test"),
            action_space_size=env.action_space.n,
        )
        for _ep in range(config.dry_run.max_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = agent.act(obs)
                obs, reward, terminated, truncated, _info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            tracker.record("episode_reward", ep_reward)
        summary = tracker.summary("episode_reward")
        assert summary["count"] == config.dry_run.max_episodes
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary

    def test_determinism_with_real_env(self) -> None:
        seed = 123
        results = []
        for _ in range(2):
            env = self._make_env(max_steps=10, seed=seed)
            agent = RandomAgent(
                AgentConfig(name="det_test"),
                action_space_size=env.action_space.n,
                seed=seed,
            )
            evaluator = Evaluator(EvalConfig(num_episodes=2, seed=seed))
            results.append(evaluator.evaluate(env, agent))
        assert results[0].reward_mean == results[1].reward_mean
        assert results[0].episode_length_mean == results[1].episode_length_mean

    def test_eval_result_serialization(self) -> None:
        env = self._make_env()
        evaluator = Evaluator(EvalConfig(num_episodes=2))
        result = evaluator.evaluate(
            env,
            RandomAgent(AgentConfig(name="ser"), action_space_size=env.action_space.n),
        )
        d = result.to_dict()
        serialized = json.dumps(d)
        assert "reward_mean" in serialized
        assert "tier_success_rates" in serialized

    def test_observation_quality(self) -> None:
        env = self._make_env(seed=42)
        obs, info = env.reset()
        assert obs.dtype == np.float32
        assert not np.allclose(obs, 0.0)
        assert np.all(np.isfinite(obs))
        assert "task_tier" in info

    def test_reward_distribution(self) -> None:
        env = self._make_env(max_steps=20, seed=7)
        agent = RandomAgent(
            AgentConfig(name="dist"),
            action_space_size=env.action_space.n,
            seed=7,
        )
        rewards = []
        for _ in range(10):
            obs, _ = env.reset()
            ep_r = 0.0
            done = False
            while not done:
                a, _ = agent.act(obs)
                obs, r, t, tr, _ = env.step(a)
                ep_r += r
                done = t or tr
            rewards.append(ep_r)
        assert len({round(r, 2) for r in rewards}) > 1
