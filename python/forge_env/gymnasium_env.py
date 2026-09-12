"""Gymnasium wrapper for FORGE.

Wraps the native Rust ``ForgeEnv`` as a real :class:`gymnasium.Env` subclass, so
the environment passes ``gymnasium.utils.env_checker.check_env`` rather than
merely resembling the API. Two properties make that work and are worth calling
out because both were previously wrong:

* **It subclasses** :class:`gymnasium.Env`. The upstream checker asserts this
  outright, and inheriting also supplies ``np_random`` seeding, the ``unwrapped``
  contract, and wrapper interoperability for free.
* **Observations are fitted to the declared spaces.** A declared space is a
  promise the checker verifies with ``space.contains(obs)``. The native env
  hands back Python scalars, a position tuple, and a variable-length message
  list, none of which satisfy a fixed-shape ``Box``. The coercion lives in
  :mod:`forge_env.space_builder`, shared with the PettingZoo wrapper.

Spaces themselves are built from the descriptor the native env publishes, which
Rust derives from :class:`ForgeConfig` — so vision radius, carry capacity, comm
buffer size, and the enabled action families flow through from config instead of
being restated here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from forge_env.space_builder import (
    DEFAULT_ACTION_COUNT,
    DEFAULT_CARRY_CAPACITY,
    DEFAULT_GRID_CHANNELS,
    DEFAULT_NUM_DAY_PHASES,
    DEFAULT_VIEW_SIDE,
    DEFAULT_VISION_RADIUS,
    UINT16_MAX,
    build_action_space,
    build_observation_space,
    fit_observation,
)

logger = logging.getLogger(__name__)

try:
    from gymnasium import spaces  # noqa: F401  (re-exported for callers/tests)

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Always the real base for the type checker. At runtime the base falls back
    # to ``object`` when gymnasium is absent, so importing ``forge_env`` never
    # requires it; ``__init__`` raises a directive ImportError in that case.
    from gymnasium import Env as _EnvBase
elif HAS_GYMNASIUM:
    from gymnasium import Env as _EnvBase
else:
    _EnvBase = object

__all__ = ["ForgeGymnasiumEnv"]

# Backwards-compatible aliases. These names were this module's own constants
# before the space definitions moved to `forge_env.space_builder` to be shared
# with the PettingZoo wrapper; `tests/python/conftest.py` and downstream code
# import them from here, so they stay as re-exports of the single definition.
_DEFAULT_VISION_RADIUS = DEFAULT_VISION_RADIUS
_DEFAULT_VIEW_SIDE = DEFAULT_VIEW_SIDE
_DEFAULT_GRID_CHANNELS = DEFAULT_GRID_CHANNELS
_DEFAULT_CARRY_CAPACITY = DEFAULT_CARRY_CAPACITY
_DEFAULT_NUM_DAY_PHASES = DEFAULT_NUM_DAY_PHASES
_DEFAULT_ACTION_N = DEFAULT_ACTION_COUNT
_UINT16_MAX = UINT16_MAX


class ForgeGymnasiumEnv(_EnvBase):
    """Gymnasium-compatible environment backed by the native FORGE engine.

    Args:
        config: Optional configuration dict matching ``ForgeConfig``.
        render_mode: Optional render mode. Must be ``None`` or a member of
            ``metadata["render_modes"]``.

    Raises:
        ImportError: If the native extension or gymnasium is unavailable.
        ValueError: If ``render_mode`` is not a supported mode.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["ascii"]}

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

        supported_modes = self.metadata["render_modes"]
        if render_mode is not None and render_mode not in supported_modes:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}. "
                f"Supported modes: {supported_modes} (or None)."
            )

        self._env = _NativeEnv(config=config)
        self.render_mode = render_mode
        self._config = config or {}

        # Both spaces come from the native descriptor, which Rust derives from
        # the resolved ForgeConfig -- no shape or bound is restated here.
        self.observation_space = build_observation_space(self._env.observation_space)
        self.action_space = build_action_space(self._env.action_space)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Seed for both the native simulation and ``self.np_random``.
            options: Passed through to the native env.

        Returns:
            An ``(observation, info)`` tuple whose observation is contained in
            :attr:`observation_space`.
        """
        # Seeds `self.np_random`, which gymnasium's checker and every
        # `action_space.sample()`-driven rollout rely on.
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed, options=options)
        return fit_observation(obs, self.observation_space), info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the environment by one step.

        Args:
            action: Discrete action index. Accepts numpy integers, which is what
                ``action_space.sample()`` returns.

        Returns:
            An ``(observation, reward, terminated, truncated, info)`` tuple.
        """
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        return (
            fit_observation(obs, self.observation_space),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> str | None:
        """Render the environment, or return ``None`` when no mode is set."""
        if self.render_mode == "ascii":
            return self._env.render()  # type: ignore[no-any-return]
        return None

    def close(self) -> None:
        """Close the environment and release the native handle's resources."""
        self._env.close()

    @property
    def native(self) -> Any:
        """The wrapped native ``ForgeEnv`` handle.

        Use this to reach engine-only methods such as ``step_multi``.

        Note:
            This is where the native handle now lives. ``unwrapped`` previously
            returned it, which violated the Gymnasium contract that
            ``unwrapped`` yields the base :class:`gymnasium.Env` — the inherited
            behaviour (returning ``self``) is what wrapper chains and the
            upstream checker require.
        """
        return self._env
