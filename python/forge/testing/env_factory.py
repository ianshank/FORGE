"""Realistic fake environment and factory for FORGE testing.

Provides :class:`RealisticFakeEnv` — a drop-in Gymnasium-compatible
environment that generates physically plausible observations without
relying on MagicMock — and :func:`create_env`, a factory that returns
the real native FORGE env when available or the fake otherwise.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Detect whether the native Rust extension is actually compiled and loadable.
# The ``forge_env`` Python package is always importable, but the native
# ``_NativeEnv`` inside it may be ``None`` when ``maturin develop`` has not
# been run.
# ---------------------------------------------------------------------------
NATIVE_AVAILABLE = False
try:
    from forge_env.gymnasium_env import _NativeEnv

    NATIVE_AVAILABLE = _NativeEnv is not None
except (ImportError, AttributeError):
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FakeEnvConfig:
    """All tunable parameters for :class:`RealisticFakeEnv`.

    Attributes
    ----------
    grid_view_size:
        Side length of the square local grid window (must be odd).
    grid_channels:
        Number of feature channels per grid cell.
    inventory_slots:
        Number of inventory slots exposed in the observation.
    action_space_n:
        Cardinality of the discrete action space.
    max_episode_length:
        Maximum number of steps before truncation.
    seed:
        Default RNG seed used when ``reset`` is called without a seed.
    num_tiers:
        Number of task tiers for the info dict.
    resource_gather_reward:
        Reward signal for resource-gathering actions.
    exploration_reward:
        Reward signal for movement / exploration actions.
    combat_reward:
        Reward signal for combat actions.
    """

    grid_view_size: int = 11
    grid_channels: int = 7
    inventory_slots: int = 10
    action_space_n: int = 8
    max_episode_length: int = 50
    seed: int = 42
    num_tiers: int = 6
    resource_gather_reward: float = 0.5
    exploration_reward: float = 0.1
    combat_reward: float = 1.0


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

# Action indices — keep in sync with action_space_n default of 8.
_ACTION_NOOP = 0
_ACTION_MOVE_UP = 1
_ACTION_MOVE_DOWN = 2
_ACTION_MOVE_LEFT = 3
_ACTION_MOVE_RIGHT = 4
_ACTION_GATHER = 5
_ACTION_COMBAT = 6
_ACTION_INTERACT = 7


def _observation_size(cfg: FakeEnvConfig) -> int:
    """Return the flat observation vector length for *cfg*."""
    grid = cfg.grid_view_size * cfg.grid_view_size * cfg.grid_channels
    inventory = cfg.inventory_slots * 2  # (item_id, count) per slot
    health = 1
    stamina = 1
    position = 2
    day_phase = 1
    return grid + inventory + health + stamina + position + day_phase


def _build_grid_obs(
    rng: np.random.Generator,
    cfg: FakeEnvConfig,
    health: float,
) -> np.ndarray:
    """Generate a grid observation with fog-of-war distance attenuation.

    Noise amplitude falls off with distance from the centre cell so that the
    returned values look like a real partial-observability grid rather than
    uniform random numbers or zeros.
    """
    size = cfg.grid_view_size
    channels = cfg.grid_channels
    centre = size // 2

    # Build a distance-from-centre weight matrix.
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dist = np.sqrt((ys - centre) ** 2 + (xs - centre) ** 2)
    max_dist = np.sqrt(2) * centre
    # Attenuation: 1.0 at centre, close to 0 at corners.
    attenuation = np.clip(1.0 - dist / max_dist, 0.0, 1.0)  # (size, size)

    raw = rng.random((size, size, channels)).astype(np.float32)
    # Apply attenuation to every channel.
    attenuated = raw * attenuation[:, :, np.newaxis]
    # Scale channel 0 (terrain type proxy) by health to simulate awareness.
    attenuated[:, :, 0] *= float(health)
    return attenuated.reshape(-1).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# RealisticFakeEnv
# ---------------------------------------------------------------------------


class RealisticFakeEnv:
    """Gymnasium-compatible fake environment with physically plausible observations.

    Observation layout (flat float32 vector)::

        [grid (W*H*C)] [inventory (slots*2)] [health] [stamina] [x] [y] [day_phase]

    Parameters
    ----------
    config:
        :class:`FakeEnvConfig` instance controlling all parameters.
    """

    def __init__(self, config: FakeEnvConfig | None = None) -> None:
        self._cfg = config or FakeEnvConfig()
        self.action_space = types.SimpleNamespace(n=self._cfg.action_space_n)
        self._obs_size = _observation_size(self._cfg)

        # Internal state (initialised by reset).
        self._rng: np.random.Generator = np.random.default_rng(self._cfg.seed)
        self._tick: int = 0
        self._health: float = 1.0
        self._stamina: float = 1.0
        self._position: list[int] = [0, 0]
        self._inventory: np.ndarray = np.zeros((self._cfg.inventory_slots, 2), dtype=np.float32)
        self._task_tier: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> np.ndarray:
        """Build a flat float32 observation from current internal state."""
        cfg = self._cfg

        grid_part = _build_grid_obs(self._rng, cfg, self._health)

        inventory_part: np.ndarray = self._inventory.reshape(-1).astype(np.float32)

        scalar_part = np.array(
            [
                self._health,
                self._stamina,
                float(self._position[0]),
                float(self._position[1]),
                float(self._tick % 24),  # day_phase — 24 h cycle
            ],
            dtype=np.float32,
        )

        return np.concatenate([grid_part, inventory_part, scalar_part]).astype(  # type: ignore[no-any-return]
            np.float32
        )

    def _make_info(self) -> dict:
        return {
            "task_tier": self._task_tier,
            "task_success": False,
            "tick": self._tick,
            "health": self._health,
            "stamina": self._stamina,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial observation and info.

        Parameters
        ----------
        seed:
            Optional RNG seed.  When *None* the config default is used.
        options:
            Ignored; present for API compatibility.
        """
        effective_seed = seed if seed is not None else self._cfg.seed
        self._rng = np.random.default_rng(effective_seed)

        self._tick = 0
        self._health = 1.0
        self._stamina = 1.0
        self._position = [0, 0]
        self._inventory = np.zeros((self._cfg.inventory_slots, 2), dtype=np.float32)
        self._task_tier = int(self._rng.integers(0, self._cfg.num_tiers))

        return self._make_obs(), self._make_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment by one step.

        Parameters
        ----------
        action:
            Integer action index.  Must be in ``[0, action_space.n)``.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        cfg = self._cfg
        self._tick += 1

        # --- Apply action effects ------------------------------------------
        reward = 0.0
        terminated = False

        if action == _ACTION_NOOP:
            # Noop: small stamina recovery, no reward.
            self._stamina = min(1.0, self._stamina + 0.01)
            reward = 0.0

        elif action in (_ACTION_MOVE_UP, _ACTION_MOVE_DOWN, _ACTION_MOVE_LEFT, _ACTION_MOVE_RIGHT):
            # Movement: drains a little stamina, grants exploration reward.
            self._stamina = max(0.0, self._stamina - 0.02)
            dx, dy = {
                _ACTION_MOVE_UP: (0, -1),
                _ACTION_MOVE_DOWN: (0, 1),
                _ACTION_MOVE_LEFT: (-1, 0),
                _ACTION_MOVE_RIGHT: (1, 0),
            }[action]
            self._position[0] += dx
            self._position[1] += dy
            reward = cfg.exploration_reward

        elif action == _ACTION_GATHER:
            # Gather: drains stamina, fills a random inventory slot.
            self._stamina = max(0.0, self._stamina - 0.05)
            slot = int(self._rng.integers(0, cfg.inventory_slots))
            self._inventory[slot, 0] = float(self._rng.integers(1, 8))  # item id
            self._inventory[slot, 1] = min(self._inventory[slot, 1] + 1.0, 99.0)
            reward = cfg.resource_gather_reward

        elif action == _ACTION_COMBAT:
            # Combat: drains health and stamina, but yields high reward.
            damage = float(self._rng.uniform(0.05, 0.20))
            self._health = max(0.0, self._health - damage)
            self._stamina = max(0.0, self._stamina - 0.10)
            reward = cfg.combat_reward
            if self._health <= 0.0:
                terminated = True

        else:
            # Interact / unknown action: small positive reward.
            reward = cfg.exploration_reward * 0.5

        # --- Truncation -------------------------------------------------------
        truncated = self._tick >= cfg.max_episode_length

        obs = self._make_obs()
        return obs, reward, terminated, truncated, self._make_info()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_env(
    *,
    force_fake: bool = False,
    config: FakeEnvConfig | None = None,
    native_config: Any = None,
) -> Any:
    """Return a FORGE environment instance.

    When the native ``forge_env`` extension is available and *force_fake* is
    ``False``, the real :class:`forge_env.gymnasium_env.ForgeGymnasiumEnv` is
    returned.  Otherwise a :class:`RealisticFakeEnv` is returned.

    Parameters
    ----------
    force_fake:
        When ``True``, always return a :class:`RealisticFakeEnv` regardless
        of whether the native env is available.
    config:
        :class:`FakeEnvConfig` forwarded to :class:`RealisticFakeEnv`.  Only
        used when the fake env is selected.
    native_config:
        Passed through to ``ForgeGymnasiumEnv`` when the native env is used.
    """
    if NATIVE_AVAILABLE and not force_fake:
        from forge_env.gymnasium_env import (  # noqa: PLC0415
            ForgeGymnasiumEnv,
        )

        return ForgeGymnasiumEnv(config=native_config)
    return RealisticFakeEnv(config or FakeEnvConfig())
