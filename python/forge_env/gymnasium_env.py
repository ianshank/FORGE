"""Pure Python Gymnasium wrapper for FORGE.

Wraps the native Rust ForgeEnv to provide full Gymnasium Env compliance
with proper Space objects from the gymnasium library.
"""

from typing import Any, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

try:
    from forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None


class ForgeGymnasiumEnv:
    """Gymnasium-compatible wrapper around the native FORGE environment.

    This wraps the Rust-backed ForgeEnv to provide proper gymnasium.Space
    objects for observation_space and action_space, and ensures all returned
    observations comply with the declared spaces.

    Args:
        config: Optional configuration dict matching ForgeConfig structure.
        render_mode: Optional render mode ('ascii' or None).
    """

    metadata = {"render_modes": ["ascii"]}

    def __init__(
        self,
        config: Optional[dict] = None,
        render_mode: Optional[str] = None,
    ):
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
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple:
        """Resets the environment.

        Returns:
            (observation, info) tuple.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: int) -> tuple:
        """Steps the environment with the given action.

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
        """
        return self._env.step(action)

    def render(self) -> Optional[str]:
        """Renders the environment."""
        if self.render_mode == "ascii":
            return self._env.render()
        return None

    def close(self):
        """Closes the environment."""
        self._env.close()

    @property
    def unwrapped(self):
        """Returns the native environment."""
        return self._env
