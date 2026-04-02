"""Tests for forge.testing.env_factory — RealisticFakeEnv and create_env.

All tests use the fake env directly (force_fake=True) so that no native
Rust extension is required.
"""
from __future__ import annotations

import numpy as np
import pytest

from forge.testing.env_factory import (
    FakeEnvConfig,
    RealisticFakeEnv,
    _observation_size,
    create_env,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CFG = FakeEnvConfig()


def _make_env(cfg: FakeEnvConfig | None = None) -> RealisticFakeEnv:
    return RealisticFakeEnv(cfg or DEFAULT_CFG)


# ---------------------------------------------------------------------------
# reset() contract
# ---------------------------------------------------------------------------


class TestReset:
    def test_returns_tuple_of_two(self):
        env = _make_env()
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_is_ndarray_float32(self):
        env = _make_env()
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_obs_shape_matches_config(self):
        cfg = FakeEnvConfig(grid_view_size=7, grid_channels=3, inventory_slots=5)
        env = _make_env(cfg)
        obs, _ = env.reset()
        expected = _observation_size(cfg)
        assert obs.shape == (expected,)

    def test_obs_shape_default_config(self):
        env = _make_env()
        obs, _ = env.reset()
        assert obs.shape == (_observation_size(DEFAULT_CFG),)

    def test_info_is_dict(self):
        env = _make_env()
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_info_contains_task_tier(self):
        env = _make_env()
        _, info = env.reset()
        assert "task_tier" in info

    def test_info_contains_required_keys(self):
        env = _make_env()
        _, info = env.reset()
        for key in ("task_tier", "task_success", "tick", "health", "stamina"):
            assert key in info, f"Missing key: {key}"

    def test_obs_not_all_zeros(self):
        env = _make_env()
        obs, _ = env.reset()
        assert not np.all(obs == 0.0), "Observation should not be all zeros"


# ---------------------------------------------------------------------------
# step() contract
# ---------------------------------------------------------------------------


class TestStep:
    def test_returns_five_tuple(self):
        env = _make_env()
        env.reset()
        result = env.step(0)
        assert isinstance(result, tuple) and len(result) == 5

    def test_obs_is_float32_ndarray(self):
        env = _make_env()
        env.reset()
        obs, *_ = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_obs_shape_consistent_with_reset(self):
        env = _make_env()
        obs_reset, _ = env.reset()
        obs_step, *_ = env.step(0)
        assert obs_reset.shape == obs_step.shape

    def test_reward_is_float(self):
        env = _make_env()
        env.reset()
        _, reward, *_ = env.step(0)
        assert isinstance(reward, float)

    def test_terminated_and_truncated_are_bool(self):
        env = _make_env()
        env.reset()
        _, _, terminated, truncated, _ = env.step(0)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_info_contains_task_tier(self):
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "task_tier" in info

    def test_obs_has_noise(self):
        env = _make_env()
        env.reset()
        obs, *_ = env.step(1)
        # Should not be uniform constant
        assert obs.std() > 0.0, "Observation should have non-zero variance"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_truncates_at_max_episode_length(self):
        cfg = FakeEnvConfig(max_episode_length=5)
        env = _make_env(cfg)
        env.reset()
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated:
                break
        assert truncated, "Episode should be truncated after max_episode_length steps"

    def test_not_truncated_before_max(self):
        cfg = FakeEnvConfig(max_episode_length=10)
        env = _make_env(cfg)
        env.reset()
        _, _, _, truncated, _ = env.step(0)
        assert not truncated

    def test_terminated_when_health_depleted(self):
        """Combat actions drain health; repeated combat should eventually terminate."""
        cfg = FakeEnvConfig(max_episode_length=200)
        env = _make_env(FakeEnvConfig(max_episode_length=200, seed=0))
        env.reset(seed=0)
        # Force health to near-zero directly for a deterministic test.
        env._health = 0.001
        _, _, terminated, _, _ = env.step(6)  # combat action
        assert terminated, "Should terminate when health <= 0"


# ---------------------------------------------------------------------------
# Reward variation
# ---------------------------------------------------------------------------


class TestRewards:
    def test_rewards_vary_by_action(self):
        """Different actions must produce different reward values."""
        cfg = DEFAULT_CFG
        rewards = set()
        for action in range(cfg.action_space_n):
            env = _make_env(cfg)
            env.reset(seed=cfg.seed)
            _, reward, *_ = env.step(action)
            rewards.add(reward)
        assert len(rewards) > 1, (
            "Rewards should differ across action types, got: %s" % rewards
        )

    def test_gather_reward_matches_config(self):
        cfg = FakeEnvConfig(resource_gather_reward=0.75, seed=1)
        env = _make_env(cfg)
        env.reset(seed=1)
        _, reward, *_ = env.step(5)  # gather action
        assert reward == cfg.resource_gather_reward

    def test_combat_reward_matches_config(self):
        cfg = FakeEnvConfig(combat_reward=2.0, seed=1)
        env = _make_env(cfg)
        env.reset(seed=1)
        _, reward, *_ = env.step(6)  # combat action
        # Combat reward is only granted if not terminated (health > 0)
        assert reward == cfg.combat_reward

    def test_exploration_reward_matches_config(self):
        cfg = FakeEnvConfig(exploration_reward=0.25, seed=1)
        env = _make_env(cfg)
        env.reset(seed=1)
        _, reward, *_ = env.step(1)  # move up
        assert reward == cfg.exploration_reward

    def test_noop_reward_is_zero(self):
        env = _make_env()
        env.reset()
        _, reward, *_ = env.step(0)
        assert reward == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_produces_identical_obs(self):
        cfg = FakeEnvConfig(seed=7)
        env_a = _make_env(cfg)
        env_b = _make_env(cfg)
        obs_a, _ = env_a.reset(seed=7)
        obs_b, _ = env_b.reset(seed=7)
        np.testing.assert_array_equal(obs_a, obs_b)

    def test_same_seed_same_trajectory(self):
        cfg = FakeEnvConfig(seed=99, max_episode_length=10)
        actions = [1, 2, 5, 6, 0, 3, 4, 1, 2, 0]

        def rollout(seed):
            env = _make_env(cfg)
            env.reset(seed=seed)
            rewards = []
            for a in actions:
                _, r, terminated, truncated, _ = env.step(a)
                rewards.append(r)
                if terminated or truncated:
                    break
            return rewards

        assert rollout(99) == rollout(99)

    def test_different_seeds_produce_different_obs(self):
        env = _make_env()
        obs_a, _ = env.reset(seed=1)
        obs_b, _ = env.reset(seed=2)
        assert not np.allclose(obs_a, obs_b), (
            "Different seeds should produce different observations"
        )


# ---------------------------------------------------------------------------
# action_space
# ---------------------------------------------------------------------------


class TestActionSpace:
    def test_action_space_n_attribute(self):
        env = _make_env()
        assert hasattr(env.action_space, "n")
        assert env.action_space.n == DEFAULT_CFG.action_space_n

    def test_action_space_n_from_config(self):
        cfg = FakeEnvConfig(action_space_n=16)
        env = _make_env(cfg)
        assert env.action_space.n == 16


# ---------------------------------------------------------------------------
# create_env factory
# ---------------------------------------------------------------------------


class TestCreateEnv:
    def test_force_fake_returns_realistic_fake_env(self):
        env = create_env(force_fake=True)
        assert isinstance(env, RealisticFakeEnv)

    def test_force_fake_with_config(self):
        cfg = FakeEnvConfig(grid_view_size=5, grid_channels=2)
        env = create_env(force_fake=True, config=cfg)
        assert isinstance(env, RealisticFakeEnv)
        obs, _ = env.reset()
        assert obs.shape == (_observation_size(cfg),)

    def test_create_env_respects_config_action_space(self):
        cfg = FakeEnvConfig(action_space_n=4)
        env = create_env(force_fake=True, config=cfg)
        assert env.action_space.n == 4

    def test_create_env_default_config_works(self):
        env = create_env(force_fake=True)
        obs, info = env.reset()
        assert obs.shape == (_observation_size(DEFAULT_CFG),)
        assert "task_tier" in info

    def test_create_env_full_step_cycle(self):
        env = create_env(force_fake=True)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
