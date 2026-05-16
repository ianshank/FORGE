"""Tests for the integrated trainer module."""

from __future__ import annotations

from typing import Any

import numpy as np

from forge.integration.integrated_trainer import (
    IntegratedTrainer,
    IntegrationTrainerConfig,
)


class TestIntegrationTrainerConfig:
    """Tests for IntegrationTrainerConfig."""

    def test_defaults(self) -> None:
        config = IntegrationTrainerConfig()
        assert config.memory_write_interval == 10
        assert config.social_reward_weight == 0.3
        assert not config.meta_learning_enabled
        assert config.max_episodes == 1000
        assert len(config.curriculum_domains) == 4

    def test_custom_config(self) -> None:
        config = IntegrationTrainerConfig(
            memory_write_interval=5,
            social_reward_weight=0.5,
            max_episodes=500,
        )
        assert config.memory_write_interval == 5
        assert config.social_reward_weight == 0.5


class TestIntegratedTrainer:
    """Tests for IntegratedTrainer."""

    def test_creation(self) -> None:
        trainer = IntegratedTrainer(num_agents=4)
        assert trainer.num_agents == 4
        assert trainer.episode_count == 0
        assert trainer.total_steps == 0

    def test_creation_with_config(self) -> None:
        config = IntegrationTrainerConfig(memory_write_interval=5)
        trainer = IntegratedTrainer(num_agents=2, config=config)
        assert trainer.config.memory_write_interval == 5

    def test_memories_per_agent(self) -> None:
        trainer = IntegratedTrainer(num_agents=3)
        assert len(trainer.memories) == 3

    def test_train_episode(self) -> None:
        trainer = IntegratedTrainer(num_agents=2)
        step_count = 0

        def reset_fn() -> np.ndarray:
            return np.zeros(10, dtype=np.float32)

        def step_fn(action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            nonlocal step_count
            step_count += 1
            done = step_count >= 5
            return np.zeros(10, dtype=np.float32), 1.0, done, False, {}

        def act_fn(obs: np.ndarray) -> int:
            return 0

        metrics = trainer.train_episode(reset_fn, step_fn, act_fn)
        assert "total_reward" in metrics
        assert "episode_length" in metrics
        assert metrics["episode_length"] == 5.0
        assert trainer.episode_count == 1
        assert trainer.total_steps == 5

    def test_domain_tracking(self) -> None:
        trainer = IntegratedTrainer(num_agents=1)
        step_idx = 0

        def reset_fn() -> np.ndarray:
            nonlocal step_idx
            step_idx = 0
            return np.zeros(5, dtype=np.float32)

        def step_fn(action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            nonlocal step_idx
            step_idx += 1
            return np.zeros(5, dtype=np.float32), 0.5, step_idx >= 3, False, {}

        trainer.train_episode(reset_fn, step_fn, lambda o: 0, domain="navigation")
        trainer.train_episode(reset_fn, step_fn, lambda o: 0, domain="crafting")
        trainer.train_episode(reset_fn, step_fn, lambda o: 0, domain="navigation")

        assert trainer.domain_mean_reward("navigation") > 0
        assert trainer.domain_mean_reward("crafting") > 0
        assert trainer.domain_mean_reward("unknown") == 0.0

    def test_memory_writes_at_interval(self) -> None:
        config = IntegrationTrainerConfig(memory_write_interval=2)
        trainer = IntegratedTrainer(num_agents=1, config=config)
        step_idx = 0

        def reset_fn() -> np.ndarray:
            nonlocal step_idx
            step_idx = 0
            return np.zeros(5, dtype=np.float32)

        def step_fn(action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            nonlocal step_idx
            step_idx += 1
            return np.zeros(5, dtype=np.float32), 1.0, step_idx >= 2, False, {}

        # Run 2 episodes; memory_write_interval=2, so write at episode 2
        trainer.train_episode(reset_fn, step_fn, lambda o: 0)
        assert trainer.memories[0].total_entries() == 0  # no write at episode 1

        trainer.train_episode(reset_fn, step_fn, lambda o: 0)
        assert trainer.memories[0].total_entries() > 0  # write at episode 2

    def test_social_reward_blending(self) -> None:
        config = IntegrationTrainerConfig(social_reward_weight=0.0)
        trainer = IntegratedTrainer(num_agents=1, config=config)
        step_idx = 0

        def reset_fn() -> np.ndarray:
            nonlocal step_idx
            step_idx = 0
            return np.zeros(5, dtype=np.float32)

        def step_fn(action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            nonlocal step_idx
            step_idx += 1
            return np.zeros(5, dtype=np.float32), 1.0, step_idx >= 1, False, {}

        metrics = trainer.train_episode(reset_fn, step_fn, lambda o: 0)
        # With social_reward_weight=0, total_reward should equal task reward
        assert abs(metrics["total_reward"] - 1.0) < 0.01
