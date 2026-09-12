"""Pure Python Gymnasium wrapper for FORGE.

Wraps the native Rust ForgeEnv to provide full Gymnasium Env compliance
with proper Space objects from the gymnasium library.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None

__all__ = ["ForgeGymnasiumEnv", "build_forge_spaces", "coerce_observation"]

# Fallback defaults that mirror Rust-side constants (forge_types::constants).
# These are only used when the native observation_space dict does not provide
# the corresponding key — in normal operation the Rust side always sets them.
_DEFAULT_VISION_RADIUS = 5  # forge_types::constants::DEFAULT_VISION_RADIUS
_DEFAULT_VIEW_SIDE = 2 * _DEFAULT_VISION_RADIUS + 1
_DEFAULT_GRID_CHANNELS = 7  # forge_types::constants::OBS_FEATURES_PER_TILE
_DEFAULT_CARRY_CAPACITY = 10  # forge_types::constants::DEFAULT_CARRY_CAPACITY
_DEFAULT_NUM_DAY_PHASES = 4  # forge_types::constants::NUM_DAY_PHASES
_DEFAULT_ACTION_N = 40  # Action::space_size(0, false) base actions with no comm/drone
_UINT16_MAX = 65535  # Maximum value for uint16 observation ranges

_EnvBase: type = gym.Env if HAS_GYMNASIUM else object


def build_forge_spaces(
    native_obs_space: dict[str, Any],
    native_act_space: dict[str, Any],
) -> tuple[Any, Any]:
    """Build Gymnasium Dict / Discrete spaces from native space dicts.

    Args:
        native_obs_space: Observation-space description from ``ForgeEnv``.
        native_act_space: Action-space description from ``ForgeEnv``.

    Returns:
        ``(observation_space, action_space)`` Gymnasium Space objects.
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium not installed. Install with: pip install gymnasium")

    view_h = native_obs_space.get("grid_view_height", _DEFAULT_VIEW_SIDE)
    view_w = native_obs_space.get("grid_view_width", _DEFAULT_VIEW_SIDE)
    channels = native_obs_space.get("grid_view_channels", _DEFAULT_GRID_CHANNELS)
    inv_capacity = native_obs_space.get("inventory_capacity", _DEFAULT_CARRY_CAPACITY)
    messages_meta = native_obs_space.get("messages", {})
    messages_shape = messages_meta.get("shape", (0,)) if isinstance(messages_meta, dict) else (0,)

    observation_space = spaces.Dict(
        {
            "grid_view": spaces.Box(
                low=0, high=255, shape=(view_h, view_w, channels), dtype=np.uint8
            ),
            "inventory": spaces.Box(
                low=0, high=_UINT16_MAX, shape=(inv_capacity, 2), dtype=np.uint16
            ),
            "health": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "stamina": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "position": spaces.Box(low=0, high=_UINT16_MAX, shape=(2,), dtype=np.uint16),
            "messages": spaces.Box(
                low=0,
                high=_UINT16_MAX,
                shape=tuple(messages_shape),
                dtype=np.uint16,
            ),
            "day_phase": spaces.Discrete(_DEFAULT_NUM_DAY_PHASES),
        }
    )
    action_n = native_act_space.get("n", _DEFAULT_ACTION_N)
    action_space = spaces.Discrete(action_n)
    return observation_space, action_space


def coerce_observation(
    obs: dict[str, Any],
    observation_space: Any | None = None,
) -> dict[str, Any]:
    """Cast native observation values so they sit inside Gymnasium Box spaces.

    Args:
        obs: Observation dict from the native ``ForgeEnv``.
        observation_space: Optional Gymnasium Dict space used to pad
            ``messages`` to the declared shape.

    Returns:
        A new dict whose array fields have the dtypes declared by
        :func:`build_forge_spaces`.
    """
    if not HAS_GYMNASIUM:
        return dict(obs)
    coerced = dict(obs)
    if "health" in coerced:
        coerced["health"] = np.asarray(coerced["health"], dtype=np.float32)
    if "stamina" in coerced:
        coerced["stamina"] = np.asarray(coerced["stamina"], dtype=np.float32)
    if "position" in coerced:
        coerced["position"] = np.asarray(coerced["position"], dtype=np.uint16)
    if "messages" in coerced:
        messages = np.asarray(coerced["messages"], dtype=np.uint16).reshape(-1)
        if observation_space is not None and "messages" in observation_space.spaces:
            expected = int(np.prod(observation_space.spaces["messages"].shape))
            fitted = np.zeros(expected, dtype=np.uint16)
            n_copy = min(messages.size, expected)
            fitted[:n_copy] = messages[:n_copy]
            coerced["messages"] = fitted.reshape(observation_space.spaces["messages"].shape)
        else:
            coerced["messages"] = messages
    if "day_phase" in coerced:
        coerced["day_phase"] = int(coerced["day_phase"])
    return coerced


class ForgeGymnasiumEnv(_EnvBase):
    """Gymnasium ``Env`` wrapper around the native FORGE environment.

    Args:
        config: Optional configuration dict matching ForgeConfig structure.
        render_mode: Optional render mode ('ascii' or None).
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["ascii"]}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        if _NativeEnv is None:
            raise ImportError(
                "forge_env native module not found. "
                "Install with: pip install -e . (requires maturin)"
            )
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium not installed. Install with: pip install gymnasium")
        if HAS_GYMNASIUM:
            super().__init__()

        self._env = _NativeEnv(config=config)
        self.render_mode = render_mode
        self._config = config or {}
        self.observation_space, self.action_space = build_forge_spaces(
            self._env.observation_space,
            self._env.action_space,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment.

        Returns:
            (observation, info) tuple.
        """
        if HAS_GYMNASIUM:
            super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed, options=options)
        return coerce_observation(obs, self.observation_space), info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the environment with the given action.

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        return coerce_observation(obs, self.observation_space), reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render the environment."""
        if self.render_mode == "ascii":
            return self._env.render()  # type: ignore[no-any-return]
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    @property
    def native_env(self) -> Any:
        """Return the native PyO3 ForgeEnv."""
        return self._env
