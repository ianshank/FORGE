"""JAX-compatible vectorized environment wrapper for ForgeEnv.

This module provides ``ForgeJaxEnv``, a batched environment that wraps
multiple native ``ForgeEnv`` instances and exposes their observations,
rewards, and termination signals as JAX arrays.  The wrapper is designed
to be compatible with ``jax.vmap`` via ``jax.experimental.io_callback``,
allowing integration with JAX-based training pipelines while still
delegating the actual simulation to the native (non-JAX) environment.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from forge_env.forge_env import ForgeEnv as _NativeEnv
except ImportError:
    _NativeEnv = None

__all__ = ["ForgeJaxEnv"]


class ForgeJaxEnv:
    """Vectorized JAX wrapper around multiple :class:`ForgeEnv` instances.

    Each of the *n_envs* environments is stepped independently; the wrapper
    simply collects results and returns them as batched JAX arrays.  Because
    the underlying environments are opaque native code, interactions are
    performed through ``jax.experimental.io_callback`` so that the wrapper
    can still participate in ``jax.vmap`` / ``jax.jit`` transformations
    (the callback itself is **not** traced -- it executes eagerly).

    Parameters
    ----------
    n_envs : int
        Number of parallel environment instances to manage.
    config : dict or None, optional
        Configuration dictionary forwarded to every ``ForgeEnv`` constructor.
        When *None* the environments use their default settings.
    seed : int, optional
        Base random seed.  Environment *i* is reset with seed ``seed + i``
        so that each instance follows a distinct but reproducible trajectory.

    Raises
    ------
    RuntimeError
        If JAX or the native ``ForgeEnv`` module is not available.
    """

    def __init__(
        self,
        n_envs: int,
        config: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        if not HAS_JAX:
            raise RuntimeError(
                "JAX is required but not installed. Install it with: pip install jax jaxlib"
            )
        if _NativeEnv is None:
            raise RuntimeError(
                "Native ForgeEnv module is required but could not be imported. "
                "Make sure the forge_env native extension is built and installed."
            )
        if not HAS_NUMPY:
            raise RuntimeError(
                "NumPy is required but not installed. Install it with: pip install numpy"
            )

        self.n_envs: int = n_envs
        self.config = config
        self.seed: int = seed

        # Create the native environment instances.
        self._envs = [_NativeEnv(config=config) for _ in range(n_envs)]

        # Cache space descriptors from the first environment.
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_dict_to_arrays(
        obs: dict[str, Any],
    ) -> tuple[Any, Any, float, float, tuple[int, ...], int]:
        """Extract the numeric/array fields from an observation dict.

        Returns
        -------
        grid_view : np.ndarray
        inventory : np.ndarray
        health : float
        stamina : float
        position : tuple of int
        day_phase : int
        """
        return (
            np.asarray(obs["grid_view"]),
            np.asarray(obs["inventory"]),
            float(obs["health"]),
            float(obs["stamina"]),
            tuple(obs["position"]),
            int(obs["day_phase"]),
        )

    def _batch_observations(
        self,
        obs_list: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Stack a list of observation dicts into batched JAX arrays.

        Parameters
        ----------
        obs_list : list of dict
            One observation dictionary per environment.

        Returns
        -------
        dict of jnp.ndarray
            Each key maps to a JAX array whose leading dimension equals
            ``self.n_envs``.  The ``"messages"`` field is omitted because
            variable-length string lists are not representable as JAX
            arrays.
        """
        grid_views = []
        inventories = []
        healths = []
        staminas = []
        positions = []
        day_phases = []

        for obs in obs_list:
            gv, inv, hp, st, pos, dp = self._obs_dict_to_arrays(obs)
            grid_views.append(gv)
            inventories.append(inv)
            healths.append(hp)
            staminas.append(st)
            positions.append(pos)
            day_phases.append(dp)

        return {
            "grid_view": jnp.array(np.stack(grid_views)),
            "inventory": jnp.array(np.stack(inventories)),
            "health": jnp.array(healths, dtype=jnp.float32),
            "stamina": jnp.array(staminas, dtype=jnp.float32),
            "position": jnp.array(positions, dtype=jnp.int32),
            "day_phase": jnp.array(day_phases, dtype=jnp.int32),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[dict[str, Any], dict[str, list[Any]]]:
        """Reset every environment and return batched observations.

        Each environment *i* is reset with seed ``self.seed + i``.

        Returns
        -------
        obs : dict of jnp.ndarray
            Batched observations.  Each value has shape ``(n_envs, ...)``.
        info : dict
            Aggregated info dictionaries (list per key).
        """
        obs_list = []
        info_list = []
        for i, env in enumerate(self._envs):
            obs, info = env.reset(seed=self.seed + i)
            obs_list.append(obs)
            info_list.append(info)

        batched_obs = self._batch_observations(obs_list)

        # Merge info dicts: each key maps to a list of per-env values.
        merged_info: dict[str, list[Any]] = {}
        for info in info_list:
            for key, value in info.items():
                merged_info.setdefault(key, []).append(value)

        return batched_obs, merged_info

    def step(
        self,
        actions: Any,
    ) -> tuple[dict[str, Any], Any, Any, Any, dict[str, list[Any]]]:
        """Take one step in every environment.

        Parameters
        ----------
        actions : jnp.ndarray
            Integer action array of shape ``(n_envs,)``.  One action per
            environment.

        Returns
        -------
        obs : dict of jnp.ndarray
            Batched observations after the step.
        rewards : jnp.ndarray
            Reward array of shape ``(n_envs,)``.
        terminated : jnp.ndarray
            Boolean array of shape ``(n_envs,)`` indicating episode
            termination.
        truncated : jnp.ndarray
            Boolean array of shape ``(n_envs,)`` indicating episode
            truncation.
        info : dict
            Aggregated info dictionaries (list per key).
        """
        # Convert JAX actions to a plain Python/Numpy iterable so we can
        # index individual scalars for the native API.
        actions_np = np.asarray(actions) if HAS_JAX else actions

        obs_list = []
        rewards = []
        terminated_list = []
        truncated_list = []
        info_list = []

        for i, env in enumerate(self._envs):
            action_int = int(actions_np[i])
            obs, reward, terminated, truncated, info = env.step(action_int)
            obs_list.append(obs)
            rewards.append(float(reward))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))
            info_list.append(info)

        batched_obs = self._batch_observations(obs_list)
        batched_rewards = jnp.array(rewards, dtype=jnp.float32)
        batched_terminated = jnp.array(terminated_list, dtype=jnp.bool_)
        batched_truncated = jnp.array(truncated_list, dtype=jnp.bool_)

        merged_info: dict[str, list[Any]] = {}
        for info in info_list:
            for key, value in info.items():
                merged_info.setdefault(key, []).append(value)

        return (
            batched_obs,
            batched_rewards,
            batched_terminated,
            batched_truncated,
            merged_info,
        )

    # ------------------------------------------------------------------
    # vmap-friendly interface via io_callback
    # ------------------------------------------------------------------

    def jax_reset(self) -> dict[str, Any]:
        """Reset environments via ``jax.experimental.io_callback``.

        This method is safe to call inside ``jax.jit``-traced code.  The
        actual environment interaction happens outside of JAX tracing
        through an IO callback.

        Returns
        -------
        obs : dict of jnp.ndarray
            Batched observations with shape ``(n_envs, ...)``.
        """
        # Determine the output shapes/dtypes by doing a trial reset.
        trial_obs, _ = self.reset()

        result_shapes = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            trial_obs,
        )

        def _reset_callback() -> dict[str, Any]:
            obs, _ = self.reset()
            return obs

        return jax.experimental.io_callback(  # type: ignore[no-any-return]
            _reset_callback,
            result_shapes,
        )

    def jax_step(
        self,
        actions: Any,
    ) -> tuple[dict[str, Any], Any, Any, Any]:
        """Step environments via ``jax.experimental.io_callback``.

        This method is safe to call inside ``jax.jit``-traced code.  The
        info dict is **not** returned because it may contain variable-length
        Python objects that cannot be represented as JAX arrays.

        Parameters
        ----------
        actions : jnp.ndarray
            Integer action array of shape ``(n_envs,)``.

        Returns
        -------
        obs : dict of jnp.ndarray
            Batched observations after the step.
        rewards : jnp.ndarray
            Shape ``(n_envs,)``.
        terminated : jnp.ndarray
            Shape ``(n_envs,)``.
        truncated : jnp.ndarray
            Shape ``(n_envs,)``.
        """
        # We need concrete shapes for the callback's result_shape_dtypes.
        # Perform a probe step (with zero actions) to discover them.
        probe_actions = jnp.zeros((self.n_envs,), dtype=jnp.int32)
        probe_obs, probe_r, probe_t, probe_tr, _ = self.step(probe_actions)

        obs_shapes = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            probe_obs,
        )
        reward_shape = jax.ShapeDtypeStruct(probe_r.shape, probe_r.dtype)
        terminated_shape = jax.ShapeDtypeStruct(probe_t.shape, probe_t.dtype)
        truncated_shape = jax.ShapeDtypeStruct(probe_tr.shape, probe_tr.dtype)

        result_shapes = (obs_shapes, reward_shape, terminated_shape, truncated_shape)

        def _step_callback(
            acts: Any,
        ) -> tuple[dict[str, Any], Any, Any, Any]:
            obs, rewards, terminated, truncated, _ = self.step(acts)
            return obs, rewards, terminated, truncated

        # Reset environments to undo the probe step side effects.
        self.reset()

        return jax.experimental.io_callback(  # type: ignore[no-any-return]
            _step_callback,
            result_shapes,
            actions,
        )
