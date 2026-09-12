"""Python observation/reward determinism lockstep.

Two ForgeGymnasiumEnv instances constructed with the same seed and fed the
same action sequence must produce byte-identical observations and identical
rewards. The 1,000,000-step run is opt-in via FORGE_RUN_LONG_DETERMINISM=1.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest


def _skip_if_no_native() -> None:
    """Skip when the forge_env native extension is not built."""
    from forge_env import gymnasium_env

    if gymnasium_env._NativeEnv is None:
        pytest.skip("forge_env running in pure-Python mode (no native backend)")


def _assert_obs_equal(obs_a: dict[str, Any], obs_b: dict[str, Any], step: int) -> None:
    """Fail on the first mismatched observation key with a byte offset."""
    keys = sorted(set(obs_a) | set(obs_b))
    for key in keys:
        if key not in obs_a or key not in obs_b:
            pytest.fail(f"mismatch at step {step} key {key}: missing from one env")
        left = np.asarray(obs_a[key])
        right = np.asarray(obs_b[key])
        bytes_a = left.tobytes()
        bytes_b = right.tobytes()
        if bytes_a == bytes_b:
            continue
        limit = min(len(bytes_a), len(bytes_b))
        offset = next((i for i in range(limit) if bytes_a[i] != bytes_b[i]), limit)
        pytest.fail(
            f"mismatch at step {step} key {key} byte offset {offset} "
            f"(len {len(bytes_a)} vs {len(bytes_b)})"
        )


def test_python_observation_reward_determinism(pytestconfig: pytest.Config) -> None:
    """Lockstep two envs for N steps; 0 mismatching bytes and equal rewards."""
    pytest.importorskip("gymnasium")
    _skip_if_no_native()
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    steps = int(pytestconfig.getoption("--determinism-steps"))
    if os.environ.get("FORGE_RUN_LONG_DETERMINISM") == "1":
        steps = 1_000_000

    seed = 123
    env_a = ForgeGymnasiumEnv()
    env_b = ForgeGymnasiumEnv()
    try:
        obs_a, _ = env_a.reset(seed=seed)
        obs_b, _ = env_b.reset(seed=seed)
        _assert_obs_equal(obs_a, obs_b, step=-1)

        rng = np.random.default_rng(seed)
        episode = 0
        action_n = int(env_a.action_space.n)
        for step in range(steps):
            action = int(rng.integers(0, action_n))
            obs_a, reward_a, term_a, trunc_a, _ = env_a.step(action)
            obs_b, reward_b, term_b, trunc_b, _ = env_b.step(action)
            if reward_a != reward_b:
                pytest.fail(f"mismatch at step {step} reward {reward_a} vs {reward_b}")
            if term_a != term_b or trunc_a != trunc_b:
                pytest.fail(f"mismatch at step {step} done flags")
            _assert_obs_equal(obs_a, obs_b, step=step)
            if term_a or trunc_a:
                episode += 1
                obs_a, _ = env_a.reset(seed=seed + episode)
                obs_b, _ = env_b.reset(seed=seed + episode)
                _assert_obs_equal(obs_a, obs_b, step=step)
    finally:
        env_a.close()
        env_b.close()
