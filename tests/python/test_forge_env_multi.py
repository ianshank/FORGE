"""Tests for ForgeEnv.reset_all and ForgeEnv.step_multi.

These methods expose the existing Rust multi-agent WorldState::step surface
through PyO3. Single-agent reset/step must keep their current contracts.
"""

from __future__ import annotations

from typing import Any

import pytest


def _skip_if_no_native() -> None:
    """Skip when the forge_env native extension is not built."""
    try:
        from forge_env import ForgeEnv

        if ForgeEnv is None:
            pytest.skip("forge_env running in pure-Python mode (no native backend)")
    except ImportError as exc:
        pytest.skip(f"forge_env native extension not available: {exc}")


def _multi_config(n_agents: int) -> dict[str, Any]:
    return {"agents": {"num_agents": n_agents}, "world": {"seed": 42}}


@pytest.fixture()
def env_n3() -> Any:
    _skip_if_no_native()
    from forge_env import ForgeEnv

    env = ForgeEnv(config=_multi_config(3))
    yield env
    env.close()


class TestResetAllReturnsPerAgentObservations:
    """reset_all must return one observation dict per configured agent."""

    def test_reset_all_length_and_keys(self, env_n3: Any) -> None:
        observations, info = env_n3.reset_all(seed=42)

        assert isinstance(observations, list)
        assert len(observations) == 3
        assert isinstance(info, dict)
        for obs in observations:
            assert isinstance(obs, dict)
            assert "grid_view" in obs
            assert "health" in obs
            assert "position" in obs

    def test_reset_all_positions_are_per_agent(self, env_n3: Any) -> None:
        observations, _info = env_n3.reset_all(seed=7)
        positions = [tuple(obs["position"]) for obs in observations]
        assert len(positions) == 3
        assert len(set(positions)) > 1


class TestStepMultiAppliesEveryAction:
    """step_multi must apply one discrete action per agent."""

    def test_step_multi_shapes(self, env_n3: Any) -> None:
        env_n3.reset_all(seed=42)
        observations, rewards, terminated, truncated, info = env_n3.step_multi([0, 4, 1])

        assert isinstance(observations, list)
        assert len(observations) == 3
        assert isinstance(rewards, list)
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert len(info["agents_alive"]) == 3

    def test_step_multi_rejects_wrong_length(self, env_n3: Any) -> None:
        env_n3.reset_all(seed=42)
        with pytest.raises(ValueError, match="expected 3 actions"):
            env_n3.step_multi([0, 4])
        with pytest.raises(ValueError, match="expected 3 actions"):
            env_n3.step_multi([0, 4, 1, 2])


class TestSingleAgentContractUnchanged:
    """Existing reset/step remain agent-0 2-tuple / 5-tuple APIs."""

    def test_reset_and_step_still_single_agent(self) -> None:
        _skip_if_no_native()
        from forge_env import ForgeEnv

        env = ForgeEnv(config={"world": {"seed": 1}})
        try:
            obs, info = env.reset(seed=1)
            assert isinstance(obs, dict)
            assert isinstance(info, dict)
            assert "grid_view" in obs

            result = env.step(0)
            assert len(result) == 5
            step_obs, reward, terminated, truncated, step_info = result
            assert isinstance(step_obs, dict)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(step_info, dict)
        finally:
            env.close()
