"""Unit tests for the FORGE evaluation pipeline.

Uses :class:`unittest.mock.MagicMock` environments so the tests are
isolated from the native Rust extension and exercise only the
:class:`~forge.evaluation.Evaluator` logic.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from forge.evaluation import EvalConfig, EvalResult, Evaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(
    *,
    reward_per_step: float = 1.0,
    episode_steps: int = 5,
    tier: int | None = None,
    success: bool = False,
) -> MagicMock:
    """Return a mock Gymnasium-compatible environment.

    The environment always runs for *episode_steps* steps.  On the last
    step ``terminated=True`` is returned.  If *tier* is given, each
    ``step`` info dict includes ``{"tier": tier, "success": success}``.
    """
    env = MagicMock()
    obs = MagicMock(name="obs")
    info_base: dict = {}
    if tier is not None:
        info_base = {"tier": tier, "success": success}

    env.reset.return_value = (obs, {})

    def _step(_action: int):
        _step.count += 1  # type: ignore[attr-defined]
        terminated = _step.count >= episode_steps
        _step.count = _step.count if not terminated else 0
        step_info = {**info_base}
        return obs, reward_per_step, terminated, False, step_info

    _step.count = 0  # type: ignore[attr-defined]
    env.step.side_effect = _step
    return env


def _make_agent(action: int = 0) -> MagicMock:
    """Return a mock agent that always returns *action*."""
    agent = MagicMock()
    agent.act.return_value = (action, {})
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvalConfig:
    """EvalConfig defaults and construction."""

    def test_defaults(self) -> None:
        cfg = EvalConfig()
        assert cfg.num_episodes == 10
        assert cfg.seed == 0
        assert cfg.determinism_check is True
        assert cfg.log_per_episode is False

    def test_custom_values(self) -> None:
        cfg = EvalConfig(num_episodes=5, seed=42, determinism_check=False, log_per_episode=True)
        assert cfg.num_episodes == 5
        assert cfg.seed == 42
        assert cfg.determinism_check is False
        assert cfg.log_per_episode is True


class TestEvalResult:
    """EvalResult construction and serialisation."""

    def _make_result(self, **kwargs) -> EvalResult:
        defaults = {
            "num_episodes": 5,
            "reward_mean": 3.0,
            "reward_std": 0.5,
            "reward_min": 2.0,
            "reward_max": 4.0,
            "episode_length_mean": 10.0,
            "episode_length_std": 1.0,
            "tier_success_rates": {1: 0.8},
            "steps_per_second": 100.0,
            "determinism_passed": True,
            "total_steps": 50,
        }
        defaults.update(kwargs)
        return EvalResult(**defaults)

    def test_to_dict_keys(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        expected_keys = {
            "num_episodes",
            "reward_mean",
            "reward_std",
            "reward_min",
            "reward_max",
            "episode_length_mean",
            "episode_length_std",
            "tier_success_rates",
            "steps_per_second",
            "determinism_passed",
            "total_steps",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert d["num_episodes"] == 5
        assert d["reward_mean"] == pytest.approx(3.0)
        assert d["tier_success_rates"] == {1: 0.8}
        assert d["determinism_passed"] is True


class TestEvaluatorReturnsResult:
    """Evaluator.evaluate returns a properly typed EvalResult."""

    def test_returns_eval_result(self) -> None:
        env = _make_env(episode_steps=3)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=2, determinism_check=False)
        evaluator = Evaluator(cfg)
        result = evaluator.evaluate(env, agent)
        assert isinstance(result, EvalResult)

    def test_num_episodes_matches_config(self) -> None:
        env = _make_env(episode_steps=4)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=3, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.num_episodes == 3


class TestRewardStats:
    """Reward statistics are computed correctly."""

    def test_reward_mean(self) -> None:
        # Each episode: 5 steps x 1.0 reward = 5.0
        env = _make_env(reward_per_step=1.0, episode_steps=5)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=4, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.reward_mean == pytest.approx(5.0)

    def test_reward_std_zero_when_identical(self) -> None:
        env = _make_env(reward_per_step=2.0, episode_steps=3)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=5, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.reward_std == pytest.approx(0.0)

    def test_reward_min_max(self) -> None:
        env = _make_env(reward_per_step=1.0, episode_steps=5)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=3, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        # All episodes identical → min == max == mean
        assert result.reward_min == pytest.approx(result.reward_mean)
        assert result.reward_max == pytest.approx(result.reward_mean)

    def test_total_steps(self) -> None:
        env = _make_env(episode_steps=5)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=4, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.total_steps == 4 * 5


class TestEpisodeLength:
    """Episode length statistics."""

    def test_episode_length_mean(self) -> None:
        env = _make_env(episode_steps=7)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=3, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.episode_length_mean == pytest.approx(7.0)

    def test_episode_length_std_zero_when_identical(self) -> None:
        env = _make_env(episode_steps=6)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=4, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.episode_length_std == pytest.approx(0.0)


class TestTierSuccessRates:
    """Per-tier success rates are aggregated correctly."""

    def test_tier_success_rate_all_success(self) -> None:
        env = _make_env(episode_steps=3, tier=1, success=True)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=5, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert 1 in result.tier_success_rates
        assert result.tier_success_rates[1] == pytest.approx(1.0)

    def test_tier_success_rate_all_fail(self) -> None:
        env = _make_env(episode_steps=3, tier=2, success=False)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=4, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert 2 in result.tier_success_rates
        assert result.tier_success_rates[2] == pytest.approx(0.0)

    def test_no_tier_info_empty_dict(self) -> None:
        env = _make_env(episode_steps=3, tier=None)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=3, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.tier_success_rates == {}


class TestThroughput:
    """steps_per_second is a positive finite number."""

    def test_steps_per_second_positive(self) -> None:
        env = _make_env(episode_steps=10)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=3, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.steps_per_second > 0.0
        assert result.steps_per_second < float("inf")


class TestDeterminism:
    """Determinism check logic."""

    def test_determinism_passes_when_env_is_deterministic(self) -> None:
        env = _make_env(episode_steps=4, reward_per_step=1.0)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=2, determinism_check=True)
        result = Evaluator(cfg).evaluate(env, agent)
        assert result.determinism_passed is True

    def test_determinism_disabled(self) -> None:
        env = _make_env(episode_steps=3)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=2, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        # When disabled, should default to True (not checked = not failed)
        assert result.determinism_passed is True

    def test_determinism_fails_when_rewards_differ(self) -> None:
        """Evaluator marks determinism_passed=False when re-run reward differs."""
        env = MagicMock()
        obs = MagicMock()
        env.reset.return_value = (obs, {})

        # First episode: reward = 5.0 (5 steps x 1.0)
        # Determinism re-run (episode index 0 again): reward = 10.0 (10 steps x 1.0)
        call_count = [0]

        def _step(_action):
            call_count[0] += 1
            # Episodes 1-5 (first real episode): terminate at step 5
            if call_count[0] <= 5:
                terminated = call_count[0] == 5
                return obs, 1.0, terminated, False, {}
            # Episodes 6-15 (determinism re-run, 10 steps)
            else:
                idx = call_count[0] - 5
                terminated = idx == 10
                return obs, 1.0, terminated, False, {}

        env.step.side_effect = _step
        cfg = EvalConfig(num_episodes=1, determinism_check=True)
        result = Evaluator(cfg).evaluate(env, agent=_make_agent())
        assert result.determinism_passed is False


class TestLogPerEpisode:
    """log_per_episode flag triggers per-episode logging."""

    def test_log_per_episode_calls_logger(self) -> None:
        env = _make_env(episode_steps=3)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=2, determinism_check=False, log_per_episode=True)
        with patch("forge.evaluation.evaluator.logger") as mock_logger:
            Evaluator(cfg).evaluate(env, agent)
        # Expect 2 per-episode info calls + 1 summary info call
        assert mock_logger.info.call_count >= 2


class TestToDict:
    """EvalResult.to_dict round-trip and JSON-serialisability."""

    def test_to_dict_is_json_serialisable(self) -> None:
        env = _make_env(episode_steps=3, tier=1, success=True)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=2, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        d = result.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_to_dict_tier_keys_are_int(self) -> None:
        env = _make_env(episode_steps=3, tier=3, success=True)
        agent = _make_agent()
        cfg = EvalConfig(num_episodes=2, determinism_check=False)
        result = Evaluator(cfg).evaluate(env, agent)
        for key in result.to_dict()["tier_success_rates"]:
            assert isinstance(key, int)
