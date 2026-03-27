"""Vectorised environment wrappers for FORGE.

Provides synchronous and asynchronous vectorised environments that wrap
multiple :class:`~forge_env.gymnasium_env.ForgeGymnasiumEnv` instances so
that rollouts can be collected from N environments in parallel, improving
training throughput.

Classes
-------
ForgeSyncVecEnv
    Runs N environments sequentially in a single process.
ForgeAsyncVecEnv
    Runs N environments in separate sub-processes (multiprocessing).

Factory
-------
make_forge_vec_env
    Convenience factory; accepts a ``ForgeConfig``-compatible dict, the
    desired number of environments, and optional wrapper callables.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False

__all__ = [
    "ForgeAsyncVecEnv",
    "ForgeSyncVecEnv",
    "make_forge_vec_env",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of per-env observation dicts into batched arrays.

    Args:
        obs_list: List of observation dicts, one per environment.

    Returns:
        A single dict where each value is a numpy array with a leading
        batch dimension of size ``len(obs_list)``.
    """
    if not obs_list:
        return {}
    keys = list(obs_list[0].keys())
    batched: dict[str, Any] = {}
    for key in keys:
        values = [np.asarray(o[key]) for o in obs_list]
        batched[key] = np.stack(values, axis=0)
    return batched


# ---------------------------------------------------------------------------
# Worker function for ForgeAsyncVecEnv (module-level so it is picklable)
# ---------------------------------------------------------------------------

_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CLOSE = "close"
_CMD_GET_SPACES = "get_spaces"


def _worker(
    env_fn: Callable[[], Any],
    pipe: mp.connection.Connection,
    parent_pipe: mp.connection.Connection,
) -> None:
    """Worker loop executed in a subprocess.

    Receives ``(command, data)`` tuples from the parent and sends back
    results.  The loop exits when ``_CMD_CLOSE`` is received.

    Args:
        env_fn: Zero-argument callable that returns an env instance.
        pipe: Child end of the communication pipe.
        parent_pipe: Parent end (closed inside the worker to avoid fd leaks).
    """
    parent_pipe.close()
    env = env_fn()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == _CMD_RESET:
                seed, options = data
                result = env.reset(seed=seed, options=options)
                pipe.send(result)
            elif cmd == _CMD_STEP:
                result = env.step(data)
                pipe.send(result)
            elif cmd == _CMD_GET_SPACES:
                pipe.send((env.observation_space, env.action_space))
            elif cmd == _CMD_CLOSE:
                env.close()
                break
            else:  # pragma: no cover
                pipe.send(RuntimeError(f"Unknown command: {cmd}"))
    except Exception as exc:
        pipe.send(exc)
    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# ForgeSyncVecEnv
# ---------------------------------------------------------------------------


class ForgeSyncVecEnv:
    """Synchronous vectorised FORGE environment.

    Wraps *N* environment instances and steps them sequentially in the
    calling process.  Observations are batched into numpy arrays with a
    leading axis of size *N*.

    Args:
        env_fns: List of zero-argument callables, each returning a
            ``ForgeGymnasiumEnv`` (or compatible) instance.

    Raises:
        ImportError: If numpy is not installed.
        ValueError: If ``env_fns`` is empty.
    """

    def __init__(self, env_fns: list[Callable[[], Any]]) -> None:
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for ForgeSyncVecEnv. Install with: pip install numpy"
            )
        if not env_fns:
            raise ValueError("env_fns must contain at least one callable")

        self._envs = [fn() for fn in env_fns]
        self.num_envs: int = len(self._envs)

        # Expose spaces from the first env (all envs must be homogeneous).
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space
        logger.debug("ForgeSyncVecEnv created with %d envs", self.num_envs)

    # -- primary interface --------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Reset all environments.

        Args:
            seed: Base seed; env *i* receives ``seed + i`` for
                  reproducible but distinct initialisation.
            options: Optional options forwarded to every env.

        Returns:
            ``(batched_obs, infos)`` where ``infos`` is a list of per-env
            info dicts.
        """
        obs_list: list[dict[str, Any]] = []
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self._envs):
            env_seed = None if seed is None else seed + i
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            infos.append(info)
        return _stack_obs(obs_list), infos

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step all environments with the given per-env actions.

        Args:
            actions: Integer array of shape ``(num_envs,)`` with one
                     discrete action per environment.

        Returns:
            ``(batched_obs, rewards, terminated, truncated, infos)`` where
            each array has leading dimension ``num_envs``.
        """
        obs_list: list[dict[str, Any]] = []
        rewards: list[float] = []
        terminated_list: list[bool] = []
        truncated_list: list[bool] = []
        infos: list[dict[str, Any]] = []

        for env, action in zip(self._envs, actions):
            obs, reward, term, trunc, info = env.step(int(action))
            if term or trunc:
                # Auto-reset: start a new episode and store the terminal obs.
                info["terminal_observation"] = obs
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(reward)
            terminated_list.append(term)
            truncated_list.append(trunc)
            infos.append(info)

        return (
            _stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminated_list, dtype=bool),
            np.array(truncated_list, dtype=bool),
            infos,
        )

    def close(self) -> None:
        """Close all inner environments."""
        for env in self._envs:
            env.close()
        logger.debug("ForgeSyncVecEnv closed")

    def render(self) -> list[str | None]:
        """Render all environments and return a list of render outputs."""
        return [env.render() for env in self._envs]

    @property
    def envs(self) -> list[Any]:
        """Direct access to the underlying environment instances."""
        return self._envs


# ---------------------------------------------------------------------------
# ForgeAsyncVecEnv
# ---------------------------------------------------------------------------


class ForgeAsyncVecEnv:
    """Asynchronous vectorised FORGE environment using subprocesses.

    Each environment runs in a separate :mod:`multiprocessing` worker
    process and communicates through :class:`multiprocessing.Pipe`.  This
    releases the GIL during the Rust simulation step, allowing true
    parallelism on multi-core machines.

    Args:
        env_fns: List of zero-argument callables, each returning an env.
        context: Multiprocessing start method: ``"spawn"`` (default, safe
            on all platforms) or ``"fork"`` (faster on Linux).

    Raises:
        ImportError: If numpy is not installed.
        ValueError: If ``env_fns`` is empty.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], Any]],
        context: str = "spawn",
    ) -> None:
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for ForgeAsyncVecEnv. Install with: pip install numpy"
            )
        if not env_fns:
            raise ValueError("env_fns must contain at least one callable")

        self.num_envs: int = len(env_fns)
        ctx = mp.get_context(context)

        self._parent_pipes: list[mp.connection.Connection] = []
        self._processes: list[mp.Process] = []

        for fn in env_fns:
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(  # type: ignore[attr-defined]
                target=_worker,
                args=(fn, child_conn, parent_conn),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._parent_pipes.append(parent_conn)
            self._processes.append(process)

        # Retrieve spaces from the first worker.
        self._parent_pipes[0].send((_CMD_GET_SPACES, None))
        result = self._parent_pipes[0].recv()
        if isinstance(result, Exception):  # pragma: no cover
            raise result
        self.observation_space, self.action_space = result
        logger.debug("ForgeAsyncVecEnv created with %d subprocesses", self.num_envs)

    # -- primary interface --------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Reset all environments asynchronously.

        Args:
            seed: Base seed; worker *i* receives ``seed + i``.
            options: Optional options dict forwarded to every env.

        Returns:
            ``(batched_obs, infos)``
        """
        for i, pipe in enumerate(self._parent_pipes):
            env_seed = None if seed is None else seed + i
            pipe.send((_CMD_RESET, (env_seed, options)))

        obs_list: list[dict[str, Any]] = []
        infos: list[dict[str, Any]] = []
        for pipe in self._parent_pipes:
            result = pipe.recv()
            if isinstance(result, Exception):  # pragma: no cover
                raise result
            obs, info = result
            obs_list.append(obs)
            infos.append(info)
        return _stack_obs(obs_list), infos

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step all environments asynchronously.

        Args:
            actions: Integer array of shape ``(num_envs,)`` with one
                     discrete action per environment.

        Returns:
            ``(batched_obs, rewards, terminated, truncated, infos)``
        """
        for pipe, action in zip(self._parent_pipes, actions):
            pipe.send((_CMD_STEP, int(action)))

        obs_list: list[dict[str, Any]] = []
        rewards: list[float] = []
        terminated_list: list[bool] = []
        truncated_list: list[bool] = []
        infos: list[dict[str, Any]] = []

        for pipe in self._parent_pipes:
            result = pipe.recv()
            if isinstance(result, Exception):  # pragma: no cover
                raise result
            obs, reward, term, trunc, info = result
            obs_list.append(obs)
            rewards.append(reward)
            terminated_list.append(term)
            truncated_list.append(trunc)
            infos.append(info)

        # Auto-reset environments that have terminated or been truncated,
        # and store the terminal observation in the corresponding info
        # dict, matching ForgeSyncVecEnv semantics.
        for i, (term, trunc) in enumerate(zip(terminated_list, truncated_list)):
            if term or trunc:
                # Preserve the terminal observation before resetting.
                info = dict(infos[i]) if infos[i] is not None else {}
                info["terminal_observation"] = obs_list[i]
                infos[i] = info

                # Reset the finished environment in the worker process.
                pipe = self._parent_pipes[i]
                pipe.send((_CMD_RESET, (None, None)))
                reset_result = pipe.recv()
                if isinstance(reset_result, Exception):  # pragma: no cover
                    raise reset_result
                reset_obs, reset_info = reset_result

                # Replace observation with the reset observation. If the
                # reset returned additional info, merge it while preserving
                # the terminal_observation we already stored.
                obs_list[i] = reset_obs
                if isinstance(reset_info, dict) and reset_info:
                    merged_info = dict(reset_info)
                    merged_info.update(infos[i])
                    infos[i] = merged_info

        return (
            _stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminated_list, dtype=bool),
            np.array(truncated_list, dtype=bool),
            infos,
        )

    def close(self) -> None:
        """Send close command to all workers and join processes."""
        for pipe in self._parent_pipes:
            with contextlib.suppress(BrokenPipeError):
                pipe.send((_CMD_CLOSE, None))
        for pipe in self._parent_pipes:
            pipe.close()
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():  # pragma: no cover
                process.terminate()
        logger.debug("ForgeAsyncVecEnv closed")

    def render(self) -> list[None]:
        """Rendering is not supported in async mode; returns a list of None."""
        return [None] * self.num_envs


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def make_forge_vec_env(
    config: dict[str, Any] | None = None,
    n_envs: int = 1,
    seed: int = 0,
    asynchronous: bool = False,
    wrapper_fns: list[Callable[[Any], Any]] | None = None,
) -> ForgeSyncVecEnv | ForgeAsyncVecEnv:
    """Create a vectorised FORGE environment.

    Factory function that builds *n_envs* copies of a
    :class:`~forge_env.gymnasium_env.ForgeGymnasiumEnv`, optionally wraps
    each one with *wrapper_fns*, and returns a vectorised container.

    Args:
        config: Optional Rust-compatible config dict (e.g.
            ``{"world": {"width": 32, "height": 32}}``).  All envs share
            the same config; only the RNG seed differs.
        n_envs: Number of parallel environment copies to create.
        seed: Base seed.  Env *i* is seeded with ``seed + i``.
        asynchronous: If ``True`` use :class:`ForgeAsyncVecEnv`
            (subprocess-based); otherwise use :class:`ForgeSyncVecEnv`.
        wrapper_fns: Optional list of wrapper constructors applied to each
            env before it is placed into the vector.  Applied in order.

    Returns:
        A :class:`ForgeSyncVecEnv` or :class:`ForgeAsyncVecEnv` instance.

    Example::

        from forge_env.vecenv import make_forge_vec_env
        from forge_env.wrappers import FlattenObservationWrapper

        vec_env = make_forge_vec_env(
            config={"world": {"width": 32, "height": 32}},
            n_envs=4,
            seed=42,
            wrapper_fns=[FlattenObservationWrapper],
        )
        obs, infos = vec_env.reset()
        # obs["grid_view"].shape == (4, 11, 11, 7)
    """
    from forge_env.gymnasium_env import ForgeGymnasiumEnv  # noqa: PLC0415

    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    def _make_single(env_seed: int) -> Callable[[], Any]:
        """Return a zero-argument factory that creates one wrapped env."""
        _config = config
        _seed = env_seed
        _wrappers = wrapper_fns or []

        def _factory() -> Any:
            env: Any = ForgeGymnasiumEnv(config=_config)
            for wrap in _wrappers:
                env = wrap(env)
            env.reset(seed=_seed)
            return env

        return _factory

    env_fns = [_make_single(seed + i) for i in range(n_envs)]

    if asynchronous:
        logger.info("Creating ForgeAsyncVecEnv with %d envs (seed=%d)", n_envs, seed)
        return ForgeAsyncVecEnv(env_fns)

    logger.info("Creating ForgeSyncVecEnv with %d envs (seed=%d)", n_envs, seed)
    return ForgeSyncVecEnv(env_fns)
