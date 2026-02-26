"""Pure Python Gymnasium wrapper for FORGE.

Wraps the native Rust ForgeEnv to provide full Gymnasium Env compliance
with proper Space objects from the gymnasium library.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym  # noqa: F401
    import numpy as np
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None

__all__ = ["ForgeGymnasiumEnv"]


class ForgeGymnasiumEnv:
    """Gymnasium-compatible wrapper around the native FORGE environment.

    This wraps the Rust-backed ForgeEnv to provide proper gymnasium.Space
    objects for observation_space and action_space, and ensures all returned
    observations comply with the declared spaces.

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
            raise ImportError(
                "gymnasium not installed. Install with: pip install gymnasium"
            )

        self._env = _NativeEnv(config=config)
        self.render_mode = render_mode
        self._config = config or {}

        # Build proper Gymnasium spaces
        native_obs_space = self._env.observation_space
        native_act_space = self._env.action_space

        # Observation space is a Dict
        view_h = native_obs_space.get("grid_view_height", 11)
        view_w = native_obs_space.get("grid_view_width", 11)
        channels = native_obs_space.get("grid_view_channels", 7)
        inv_capacity = native_obs_space.get("inventory_capacity", 10)

        self.observation_space = spaces.Dict(
            {
                "grid_view": spaces.Box(
                    low=0, high=255, shape=(view_h, view_w, channels), dtype=np.uint8
                ),
                "inventory": spaces.Box(
                    low=0, high=65535, shape=(inv_capacity, 2), dtype=np.uint16
                ),
                "health": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "stamina": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "position": spaces.Box(
                    low=0, high=65535, shape=(2,), dtype=np.uint16
                ),
                "day_phase": spaces.Discrete(4),
            }
        )

        # Action space is Discrete
        action_n = native_act_space.get("n", 32)
        self.action_space = spaces.Discrete(action_n)

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
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the environment with the given action.

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
        """
        return self._env.step(action)  # type: ignore[no-any-return]

    def render(self) -> str | None:
        """Render the environment."""
        if self.render_mode == "ascii":
            return self._env.render()  # type: ignore[no-any-return]
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    @property
    def unwrapped(self) -> Any:
        """Return the native environment."""
        return self._env
