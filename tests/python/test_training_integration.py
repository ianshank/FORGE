"""Integration tests for the training pipeline.

These tests verify that the training script components wire together
correctly: environment → agent → trainer → checkpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure python/ is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))


def _flatten_obs(obs: dict) -> np.ndarray:
    """Flatten a dict observation to 1-D float32 array (sorted keys)."""
    parts = [
        np.asarray(obs[key], dtype=np.float32).ravel() for key in sorted(obs.keys())
    ]
    return np.concatenate(parts)


@pytest.fixture()
def env():
    """Create a ForgeGymnasiumEnv for testing."""
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    e = ForgeGymnasiumEnv()
    yield e
    e.close()


class TestRandomAgentTraining:
    """Test random agent episode loop."""

    def test_random_agent_completes_episodes(self, env) -> None:
        """Random agent can run multiple episodes without error."""
        from forge.agents.base_agent import AgentConfig
        from forge.agents.random_agent import RandomAgent

        action_dim = env.action_space.n
        agent = RandomAgent(
            config=AgentConfig(name="test_random"),
            action_space_size=action_dim,
            seed=42,
        )

        episodes_completed = 0
        for _ in range(5):
            obs, _info = env.reset()
            flat_obs = _flatten_obs(obs)
            done = False
            steps = 0
            while not done and steps < 200:
                action, _trace = agent.act(flat_obs)
                obs, _reward, terminated, truncated, _info = env.step(action)
                flat_obs = _flatten_obs(obs)
                done = terminated or truncated
                steps += 1
            episodes_completed += 1

        assert episodes_completed == 5


class TestMAPPOTraining:
    """Test MAPPO agent with PPOTrainer."""

    def test_mappo_training_completes(self, env) -> None:
        """MAPPO training runs for a small number of updates without error."""
        from forge.agents.mappo_agent import MAPPOAgent, MAPPOConfig
        from forge.training.trainer import PPOTrainer, PPOTrainerConfig

        obs, _info = env.reset()
        obs_dim = _flatten_obs(obs).shape[0]
        action_dim = env.action_space.n

        config = MAPPOConfig(
            name="test_mappo",
            obs_dim=obs_dim,
            action_dim=action_dim,
            device="cpu",
        )
        agent = MAPPOAgent(config, obs_dim=obs_dim, action_dim=action_dim)

        trainer_config = PPOTrainerConfig(
            rollout_length=64,
            max_episode_steps=50,
            log_interval=1,
        )
        trainer = PPOTrainer(agent=agent, config=trainer_config)

        def reset_fn() -> np.ndarray:
            o, _ = env.reset()
            return _flatten_obs(o)

        def step_fn(action: int) -> tuple:
            o, r, term, trunc, info = env.step(action)
            return _flatten_obs(o), r, term, trunc, info

        all_metrics = trainer.train(reset_fn, step_fn, num_updates=2)

        assert len(all_metrics) == 2
        assert trainer.total_steps > 0

    def test_training_metrics_keys(self, env) -> None:
        """Training metrics contain expected keys."""
        from forge.agents.mappo_agent import MAPPOAgent, MAPPOConfig
        from forge.training.trainer import PPOTrainer, PPOTrainerConfig

        obs, _info = env.reset()
        obs_dim = _flatten_obs(obs).shape[0]
        action_dim = env.action_space.n

        config = MAPPOConfig(
            name="test_keys",
            obs_dim=obs_dim,
            action_dim=action_dim,
            device="cpu",
        )
        agent = MAPPOAgent(config, obs_dim=obs_dim, action_dim=action_dim)
        trainer = PPOTrainer(
            agent=agent,
            config=PPOTrainerConfig(rollout_length=32, max_episode_steps=30),
        )

        def reset_fn() -> np.ndarray:
            o, _ = env.reset()
            return _flatten_obs(o)

        def step_fn(action: int) -> tuple:
            o, r, term, trunc, info = env.step(action)
            return _flatten_obs(o), r, term, trunc, info

        metrics_list = trainer.train(reset_fn, step_fn, num_updates=1)
        assert len(metrics_list) == 1

        m = metrics_list[0]
        assert "policy_loss" in m
        assert "value_loss" in m
        assert "entropy" in m
        assert "total_steps" in m
