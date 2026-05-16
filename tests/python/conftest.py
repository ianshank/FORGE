"""Shared test fixtures and helpers for FORGE Python tests.

Centralises mock observation data and native-env factory so that
``test_gymnasium_env``, ``test_pettingzoo_env`` and ``test_jax_env``
share a single source of truth.

Also exposes shared filesystem helpers (``REPO_ROOT``, ``read_toml``)
used by config / preset / asset tests so each test file does not roll
its own ``Path(__file__).resolve().parents[N]`` and TOML loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from forge_env.gymnasium_env import (
    _DEFAULT_CARRY_CAPACITY,
    _DEFAULT_GRID_CHANNELS,
    _DEFAULT_VIEW_SIDE,
)

# ---------------------------------------------------------------------------
# Shared filesystem helpers.
# ---------------------------------------------------------------------------

#: Absolute path to the repository root, regardless of which ``tests/python/``
#: file imports it. Computed once at module import so tests stay fast.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def read_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file as a dict, falling back to ``tomli`` on py3.9/3.10.

    Used by every preset / pyproject / config test instead of each file
    importing tomllib + tomli individually.
    """
    try:
        import tomllib as _toml
    except ModuleNotFoundError:  # pragma: no cover - py39/py310
        import tomli as _toml
    with path.open("rb") as fh:
        return cast("dict[str, Any]", _toml.load(fh))


# ---------------------------------------------------------------------------
# LM Studio shared parametrisation.
# ---------------------------------------------------------------------------

#: Canonical model ids exercised by every LM-Studio-touching test. Adding a
#: third id here automatically parametrises every test using the fixture.
LMSTUDIO_SUPPORTED_MODEL_IDS: tuple[str, ...] = (
    "qwen2.5-14b-instruct",
    "google/gemma-4-e4b",
)


@pytest.fixture(
    params=LMSTUDIO_SUPPORTED_MODEL_IDS,
    ids=("qwen", "gemma"),
)
def lmstudio_model_id(request: pytest.FixtureRequest) -> str:
    """Model ids that LM Studio should round-trip without inspection.

    Provider behaviour MUST be identical across these — any test that
    diverges signals a model-specific code path that should not exist.
    """
    return cast("str", request.param)

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
