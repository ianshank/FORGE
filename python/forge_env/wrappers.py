"""Observation and reward wrappers for the ForgeEnv.

Provides composable wrappers that modify observations, rewards, and episode
lifecycle behaviour. All wrappers delegate unknown attribute access to the
inner environment so they can be stacked freely.

Wrappers
--------
FlattenObservationWrapper
    Flattens a dict observation into a single 1-D numpy array.
NormalizeRewardWrapper
    Normalises rewards to zero mean / unit variance using running statistics.
TimeLimit
    Truncates an episode after a fixed number of steps.
RecordEpisodeStatistics
    Records cumulative return, episode length, and wall-clock time.
"""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

__all__ = [
    "BaseWrapper",
    "FlattenObservationWrapper",
    "NormalizeRewardWrapper",
    "RecordEpisodeStatistics",
    "RecordEpisodeWrapper",
    "TimeLimit",
]


# ---------------------------------------------------------------------------
# Base helper
# ---------------------------------------------------------------------------


class _BaseWrapper:
    """Minimal wrapper base that delegates attribute access to the inner env.

    All concrete wrappers inherit from this so that attributes such as
    ``observation_space``, ``action_space``, ``render``, and ``close`` are
    transparently forwarded when they are not overridden.

    Args:
        env: The environment to wrap.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    # Forward any attribute not found on the wrapper itself to the inner env.
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def __repr__(self) -> str:
        """Show the full wrapper chain for easier debugging."""
        return f"{type(self).__name__}({self.env!r})"

    @property
    def unwrapped(self) -> Any:
        """Return the innermost (unwrapped) environment.

        Delegates to the inner environment's ``unwrapped`` property when
        available (e.g. Gymnasium envs), otherwise traverses the wrapper
        chain via ``.env`` attributes.  This matches the
        :func:`gymnasium.Env.unwrapped` convention.
        """
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env


# Public alias so downstream code can reference the base class without the
# leading underscore convention.
BaseWrapper = _BaseWrapper


# ---------------------------------------------------------------------------
# FlattenObservationWrapper
# ---------------------------------------------------------------------------


class FlattenObservationWrapper(_BaseWrapper):
    """Flatten a dict observation into a single 1-D numpy array.

    Each value in the observation dict is cast to ``np.float32``, flattened,
    and then concatenated in *sorted key order* so that the result is
    deterministic.

    Args:
        env: A ForgeEnv-like environment whose ``reset`` and ``step`` methods
            return dict observations.

    Raises:
        ImportError: If numpy is not installed.
    """

    def __init__(self, env: Any) -> None:
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for FlattenObservationWrapper. "
                "Install with: pip install numpy"
            )
        super().__init__(env)

    # -- public helpers -----------------------------------------------------

    def flatten_obs(self, obs_dict: dict[str, Any]) -> np.ndarray:
        """Flatten a dict observation into a 1-D float32 numpy array.

        Args:
            obs_dict: Dictionary mapping observation keys to array-like values.

        Returns:
            A 1-D ``np.float32`` array containing all observation values
            concatenated in sorted-key order.
        """
        parts = [
            np.asarray(obs_dict[key], dtype=np.float32).ravel()
            for key in sorted(obs_dict.keys())
        ]
        return np.concatenate(parts)

    # -- env interface ------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the inner environment and flatten the observation.

        Returns:
            ``(flat_obs, info)`` tuple.
        """
        obs, info = self.env.reset(**kwargs)
        return self.flatten_obs(obs), info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the inner environment and flatten the observation.

        Returns:
            ``(flat_obs, reward, terminated, truncated, info)`` tuple.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.flatten_obs(obs), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# NormalizeRewardWrapper
# ---------------------------------------------------------------------------


class NormalizeRewardWrapper(_BaseWrapper):
    """Normalise rewards using a running mean and variance estimate.

    The wrapper maintains Welford-style running statistics and normalises
    each reward as ``(r - mean) / sqrt(var + 1e-8)``.  Normalised rewards
    are clipped to the range ``[-10, 10]`` to prevent extreme values.

    Args:
        env: The environment to wrap.
        clip: Maximum absolute value for the normalised reward.
        epsilon: Small constant added to the variance for numerical stability.

    Attributes:
        reward_mean: Running mean of observed rewards.
        reward_var: Running variance of observed rewards.
        count: Number of rewards observed so far.
    """

    def __init__(
        self,
        env: Any,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(env)
        self.reward_mean: float = 0.0
        self.reward_var: float = 1.0
        self.count: float = 0.0
        self._clip = clip
        self._epsilon = epsilon

    # -- internal -----------------------------------------------------------

    def _update_stats(self, reward: float) -> None:
        """Update running mean and variance with a new reward value."""
        self.count += 1.0
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.count
        delta2 = reward - self.reward_mean
        self.reward_var += (delta * delta2 - self.reward_var) / self.count

    def _normalize(self, reward: float) -> float:
        """Return the normalised and clipped reward."""
        std = (self.reward_var + self._epsilon) ** 0.5
        normed = (reward - self.reward_mean) / std
        # Clip to [-clip, clip]
        return float(max(-self._clip, min(self._clip, normed)))

    # -- env interface ------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the inner environment (statistics are *not* reset).

        Returns:
            ``(obs, info)`` tuple.
        """
        return self.env.reset(**kwargs)  # type: ignore[no-any-return]

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the inner environment and normalise the reward.

        Returns:
            ``(obs, normalised_reward, terminated, truncated, info)`` tuple.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_stats(reward)
        normalised_reward = self._normalize(reward)
        return obs, normalised_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# TimeLimit
# ---------------------------------------------------------------------------


class TimeLimit(_BaseWrapper):
    """Truncate an episode after a fixed number of steps.

    When the step count reaches ``max_steps`` the wrapper sets ``truncated``
    to ``True`` in the returned tuple.  The ``terminated`` flag from the
    inner environment is left unchanged.

    Args:
        env: The environment to wrap.
        max_steps: Maximum number of steps before truncation.

    Attributes:
        _current_step: Number of steps taken in the current episode.
    """

    def __init__(self, env: Any, max_steps: int) -> None:
        super().__init__(env)
        self.max_steps = max_steps
        self._current_step: int = 0

    # -- env interface ------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the inner environment and the step counter.

        Returns:
            ``(obs, info)`` tuple.
        """
        self._current_step = 0
        return self.env.reset(**kwargs)  # type: ignore[no-any-return]

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the inner environment and check the step limit.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` tuple where
            ``truncated`` is ``True`` when the step limit is reached.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_step += 1
        if self._current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# RecordEpisodeStatistics
# ---------------------------------------------------------------------------


class RecordEpisodeStatistics(_BaseWrapper):
    """Record episode return, length, and wall-clock duration.

    When an episode ends (``terminated or truncated``), the wrapper injects
    an ``"episode"`` key into the *info* dict with the following sub-keys:

    * ``"r"`` -- cumulative episode return (float).
    * ``"l"`` -- episode length in steps (int).
    * ``"t"`` -- elapsed wall-clock time in seconds (float).

    Args:
        env: The environment to wrap.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._episode_start: float = 0.0

    # -- env interface ------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the inner environment and start tracking a new episode.

        Returns:
            ``(obs, info)`` tuple.
        """
        obs, info = self.env.reset(**kwargs)
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_start = _time.time()
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the inner environment and update episode statistics.

        When the episode ends, the info dict is augmented with an
        ``"episode"`` entry containing ``"r"``, ``"l"``, and ``"t"`` keys.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` tuple.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += reward
        self._episode_length += 1

        if terminated or truncated:
            episode_info = {
                "r": self._episode_return,
                "l": self._episode_length,
                "t": _time.time() - self._episode_start,
            }
            info["episode"] = episode_info

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# RecordEpisodeWrapper — writes .forge replay files
# ---------------------------------------------------------------------------


class RecordEpisodeWrapper(_BaseWrapper):
    """Record a full episode to a ``.forge`` JSON replay file.

    The file is written when the episode ends (``terminated`` or ``truncated``).
    Target per-step overhead: ≤ 1%.

    Parameters
    ----------
    env:
        The environment to wrap.
    output_path:
        Path where the ``.forge`` file will be written.
    seed:
        Optional seed to embed in the replay header.
    config:
        Optional env config dict to embed in the replay header.
    store_observations:
        If ``True`` (default), observations are stored alongside actions.

    Examples
    --------
    >>> wrapper = RecordEpisodeWrapper(env, "replay.forge")
    >>> obs, _ = wrapper.reset(seed=42)
    """

    FORMAT_VERSION: int = 1

    def __init__(
        self,
        env: Any,
        output_path: str | Path,
        *,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        store_observations: bool = True,
    ) -> None:
        super().__init__(env)
        self._out_path = Path(output_path)
        self._seed = seed
        self._config: dict[str, Any] = config or {}
        self._store_obs = store_observations
        self._actions: list[int] = []
        self._observations: list[list[float]] = []
        self._rewards: list[float] = []
        self._timestamps_ms: list[float] = []
        self._ep_start_ms: float = 0.0
        self._recording: bool = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset and start a fresh recording."""
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._seed = seed
        self._actions = []
        self._observations = [self._to_list(obs)] if self._store_obs else []
        self._rewards = []
        self._timestamps_ms = []
        self._ep_start_ms = _time.monotonic() * 1000.0
        self._recording = True
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Record action/obs/reward, delegate to inner env, flush on done."""
        if not self._recording:
            raise RuntimeError("call reset() before step()")
        t_ms = _time.monotonic() * 1000.0 - self._ep_start_ms
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._timestamps_ms.append(round(t_ms, 3))
        if self._store_obs:
            self._observations.append(self._to_list(obs))
        if terminated or truncated:
            self._flush(len(self._actions))
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _to_list(obs: Any) -> list[float]:
        if HAS_NUMPY:
            import numpy as np  # noqa: PLC0415
            return np.asarray(obs).flatten().tolist()  # type: ignore[no-any-return]
        if hasattr(obs, "tolist"):
            return obs.tolist()  # type: ignore[no-any-return]
        return list(obs)

    def _flush(self, terminated_at: int) -> None:
        import json  # noqa: PLC0415

        self._out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415
            forge_version = version("forge-env")
        except PackageNotFoundError:
            forge_version = "dev"

        payload: dict[str, Any] = {
            "forge_version": forge_version,
            "format_version": self.FORMAT_VERSION,
            "seed": self._seed,
            "config": self._config,
            "actions": self._actions,
            "rewards": self._rewards,
            "terminated_at": terminated_at,
            "timestamps_ms": self._timestamps_ms,
        }
        if self._store_obs:
            payload["observations"] = self._observations

        self._out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Replay saved → %s (%d steps)", self._out_path, terminated_at)

