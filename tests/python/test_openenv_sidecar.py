"""OpenEnv sidecar contract tests (SDK optional)."""

from __future__ import annotations

from typing import Any

import pytest

from forge_env.openenv_env import (
    ForgeAction,
    ForgeObservation,
    ForgeOpenEnv,
    create_forge_openenv_app,
)


class _FakeInner:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"tick": 0, "seed": seed}, {"ok": True}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.steps += 1
        done = self.steps >= 2
        return {"tick": self.steps, "action": action}, 1.5, done, False, {"n": self.steps}

    def close(self) -> None:
        return None


def test_openenv_reset_step_carry_reward_and_done() -> None:
    env = ForgeOpenEnv(inner=_FakeInner(), config={"world": {"width": 16}})
    obs = env.reset(seed=7)
    assert isinstance(obs, ForgeObservation)
    assert obs.done is False
    assert obs.reward == 0.0
    assert env.state.seed == 7
    stepped = env.step(ForgeAction(action_id=4))
    assert stepped.reward == 1.5
    assert stepped.done is False
    done_obs = env.step(ForgeAction(action_id=0))
    assert done_obs.done is True
    assert done_obs.terminated is True
    assert env.state.step_count == 2
    env.close()


def test_create_app_requires_openenv_sdk() -> None:
    try:
        import openenv  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="openenv is not installed"):
            create_forge_openenv_app()
        return
    app = create_forge_openenv_app(config={"agents": {"num_agents": 1}})
    assert app is not None
