"""Tests for the JAX vectorized environment wrapper.

Requires JAX to be installed (auto-skips otherwise).
The native forge_env module is mocked.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Auto-skip if JAX is not installed.
# ---------------------------------------------------------------------------
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

# ---------------------------------------------------------------------------
# Mock the native forge_env module before importing the wrapper.
# ---------------------------------------------------------------------------
_MOCK_OBS = {
    "grid_view": [[[0] * 7] * 11] * 11,
    "inventory": [[0, 0]] * 10,
    "health": 0.8,
    "stamina": 0.9,
    "position": [5, 5],
    "messages": [],
    "day_phase": 0,
}


def _make_mock_native_env():
    """Create a fresh mock native env instance."""
    inst = MagicMock()
    inst.reset.return_value = (_MOCK_OBS, {"tick": 0})
    inst.step.return_value = (_MOCK_OBS, 1.0, False, False, {"tick": 1})
    inst.observation_space = {
        "grid_view_height": 11,
        "grid_view_width": 11,
        "grid_view_channels": 7,
        "inventory_capacity": 10,
    }
    inst.action_space = {"n": 8}
    return inst


_mock_forge_env_module = MagicMock()
_mock_forge_env_module.ForgeEnv.side_effect = lambda **kwargs: _make_mock_native_env()

sys.modules.setdefault("forge_env", MagicMock())
sys.modules["forge_env.forge_env"] = _mock_forge_env_module

from forge_env.jax_env import ForgeJaxEnv  # noqa: E402


@pytest.fixture()
def jax_env():
    """Create a ForgeJaxEnv with 4 parallel envs."""
    _mock_forge_env_module.ForgeEnv.side_effect = lambda **kwargs: _make_mock_native_env()
    return ForgeJaxEnv(n_envs=4, config=None, seed=0)


class TestResetReturnsBatchedArrays:
    """reset() should return observations as batched JAX arrays."""

    def test_reset_returns_batched_arrays(self, jax_env):
        obs, _info = jax_env.reset()

        assert isinstance(obs, dict)
        # Each observation field should have leading dimension == n_envs
        assert obs["grid_view"].shape[0] == 4
        assert obs["health"].shape == (4,)
        assert obs["stamina"].shape == (4,)
        assert obs["position"].shape == (4, 2)
        assert obs["day_phase"].shape == (4,)


class TestStepReturnsBatchedResults:
    """step() should return batched obs, rewards, terminated, truncated, info."""

    def test_step_returns_batched_results(self, jax_env):
        jax_env.reset()
        actions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        obs, rewards, terminated, truncated, info = jax_env.step(actions)

        assert isinstance(obs, dict)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)
        assert isinstance(info, dict)


class TestMultipleEnvsIndependent:
    """Each env instance should be stepped independently."""

    def test_multiple_envs_independent(self, jax_env):
        jax_env.reset()

        # Each internal env should have received its own reset call
        for i, env in enumerate(jax_env._envs):
            env.reset.assert_called_once_with(seed=i)

        actions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        jax_env.step(actions)

        # Each env should have been stepped with its respective action
        for i, env in enumerate(jax_env._envs):
            env.step.assert_called_once_with(i)
