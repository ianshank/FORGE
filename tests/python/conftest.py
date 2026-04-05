"""Shared test fixtures and helpers for FORGE Python tests.

Centralises mock observation data and native-env factory so that
``test_gymnasium_env``, ``test_pettingzoo_env`` and ``test_jax_env``
share a single source of truth.
"""

from __future__ import annotations

from unittest.mock import MagicMock

try:
    from forge_env.gymnasium_env import (
        _DEFAULT_CARRY_CAPACITY,
        _DEFAULT_GRID_CHANNELS,
        _DEFAULT_VIEW_SIDE,
    )
except ImportError:
    # Fallback defaults when the native extension is not built (pure-Python CI).
    # Must match python/forge_env/gymnasium_env.py to keep mock data consistent.
    _DEFAULT_VIEW_SIDE = 11  # 2 * 5 + 1
    _DEFAULT_GRID_CHANNELS = 7  # OBS_FEATURES_PER_TILE
    _DEFAULT_CARRY_CAPACITY = 10  # DEFAULT_CARRY_CAPACITY

# ---------------------------------------------------------------------------
# Mock observation data matching the native wrapper's expected structure.
# ---------------------------------------------------------------------------

MOCK_OBS: dict = {
    "grid_view": [[[0] * _DEFAULT_GRID_CHANNELS] * _DEFAULT_VIEW_SIDE] * _DEFAULT_VIEW_SIDE,
    "inventory": [[0, 0]] * _DEFAULT_CARRY_CAPACITY,
    "health": 0.8,
    "stamina": 0.9,
    "position": [5, 5],
    "messages": [],
    "day_phase": 0,
}

NATIVE_OBS_SPACE: dict = {
    "grid_view_height": _DEFAULT_VIEW_SIDE,
    "grid_view_width": _DEFAULT_VIEW_SIDE,
    "grid_view_channels": _DEFAULT_GRID_CHANNELS,
    "inventory_capacity": _DEFAULT_CARRY_CAPACITY,
}

NATIVE_OBS_SPACE_WITH_MESSAGES: dict = {
    **NATIVE_OBS_SPACE,
    "messages": {"shape": (0,)},
}

NATIVE_ACT_SPACE: dict = {"n": 8}


def make_mock_native_env(
    *,
    include_messages_in_obs_space: bool = False,
    include_render: bool = True,
) -> MagicMock:
    """Create a fresh mock native ``ForgeEnv`` instance.

    Parameters
    ----------
    include_messages_in_obs_space:
        When *True*, the ``observation_space`` dict includes a ``"messages"``
        entry.  Only the Gymnasium wrapper requires this.
    include_render:
        When *True* (default), ``env.render()`` returns a placeholder string.
    """
    env = MagicMock()
    env.reset.return_value = (MOCK_OBS, {"tick": 0})
    env.step.return_value = (MOCK_OBS, 1.0, False, False, {"tick": 1})
    if include_render:
        env.render.return_value = "ascii_frame"
    env.observation_space = (
        NATIVE_OBS_SPACE_WITH_MESSAGES if include_messages_in_obs_space else NATIVE_OBS_SPACE
    )
    env.action_space = NATIVE_ACT_SPACE
    return env
