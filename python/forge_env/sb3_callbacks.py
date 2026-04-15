"""Stable Baselines 3 callbacks for FORGE training.

Provides ready-to-use SB3 callbacks that integrate FORGE-specific features
(curriculum learning, structured logging) into the standard SB3 training
loop without modifying any SB3 internals.

Classes
-------
ForgeCurriculumCallback
    Monitors episode success rate and increments the task tier when a
    rolling success window exceeds a configurable threshold.
ForgeMetricsCallback
    Collects episode-level metrics (return, length, task progress) and
    dispatches them to a pluggable :class:`~forge.training.loggers.ForgeLogger`.

Both callbacks require Stable Baselines 3 to be installed.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

try:
    from stable_baselines3.common.callbacks import BaseCallback

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    BaseCallback = object

if TYPE_CHECKING:
    from forge.training.loggers import ForgeLogger

__all__ = [
    "ForgeCurriculumCallback",
    "ForgeMetricsCallback",
]

# Key injected into the info dict by the FORGE environment on episode end.
_TASK_SUCCESS_KEY = "task_success"
_TASK_PROGRESS_KEY = "task_progress"
_EPISODE_KEY = "episode"


def _require_sb3() -> None:
    if not HAS_SB3:
        raise ImportError(
            "Stable Baselines 3 is required for FORGE callbacks. "
            "Install with: pip install stable-baselines3"
        )


# ---------------------------------------------------------------------------
# ForgeCurriculumCallback
# ---------------------------------------------------------------------------


class ForgeCurriculumCallback(BaseCallback):
    """Adaptive curriculum callback for FORGE task tiers.

    Monitors episode outcomes (read from ``info["task_success"]``) and
    increments the ``task_tier`` in the environment config whenever the
    rolling success rate surpasses *target_success_rate*.

    The callback respects the :class:`~forge_env.vecenv.ForgeSyncVecEnv`
    interface: it iterates over ``self.training_env.envs`` and calls
    ``env.set_config(new_config)`` on each underlying env.  If the env does
    not expose ``set_config`` the increment is skipped with a warning.

    Args:
        target_success_rate: Success rate threshold in ``[0, 1]`` that
            triggers a tier increment.
        window_size: Rolling window size (number of recent episodes).
        adjustment_rate: Number of tiers to increase per successful window.
        verbose: Verbosity level (0 = silent, 1 = log increments).

    Raises:
        ImportError: If Stable Baselines 3 is not installed.
    """

    def __init__(
        self,
        target_success_rate: float = 0.7,
        window_size: int = 100,
        adjustment_rate: int = 1,
        verbose: int = 1,
    ) -> None:
        _require_sb3()
        super().__init__(verbose=verbose)

        if not (0.0 < target_success_rate <= 1.0):
            raise ValueError(f"target_success_rate must be in (0, 1], got {target_success_rate}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if adjustment_rate < 1:
            raise ValueError(f"adjustment_rate must be >= 1, got {adjustment_rate}")

        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.adjustment_rate = adjustment_rate

        self._success_window: deque[bool] = deque(maxlen=window_size)
        self._current_tier: int = 1
        self._num_increments: int = 0
        self._training_env_override: Any | None = None

    @property
    def training_env(self) -> Any:
        """Return the SB3 training env, with optional test override support."""
        if self._training_env_override is not None:
            return self._training_env_override
        return super().training_env

    @training_env.setter
    def training_env(self, env: Any) -> None:
        self._training_env_override = env

    # -- BaseCallback hooks -------------------------------------------------

    def _on_step(self) -> bool:
        """Inspect ``infos`` for episode outcomes and update curriculum."""
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            if _EPISODE_KEY not in info:
                continue  # Episode still running
            success: bool = bool(info.get(_TASK_SUCCESS_KEY, False))
            self._success_window.append(success)

        if len(self._success_window) == self.window_size:
            rate = sum(self._success_window) / self.window_size
            if rate >= self.target_success_rate:
                self._increment_tier()
                self._success_window.clear()

        return True  # Continue training

    # -- helpers ------------------------------------------------------------

    def _increment_tier(self) -> None:
        """Increase task tier on all envs by *adjustment_rate*."""
        self._current_tier += self.adjustment_rate
        self._num_increments += 1

        if self.verbose >= 1:
            logger.info(
                "Curriculum: success rate reached threshold → tier %d",
                self._current_tier,
            )

        # Attempt to update each underlying env.
        try:
            envs = self.training_env.envs
        except AttributeError:
            logger.warning(
                "ForgeCurriculumCallback: training_env does not expose .envs — "
                "cannot update task tier."
            )
            return

        for env in envs:
            # Unwrap if needed
            unwrapped = getattr(env, "unwrapped", env)
            if callable(getattr(unwrapped, "set_config", None)):
                unwrapped.set_config({"task": {"max_tier": self._current_tier}})
            else:
                logger.debug(
                    "ForgeCurriculumCallback: env %r does not support set_config; "
                    "skipping tier update.",
                    env,
                )

    @property
    def current_tier(self) -> int:
        """Current task difficulty tier (read-only)."""
        return self._current_tier

    @property
    def num_tier_increments(self) -> int:
        """Total number of tier increments performed so far."""
        return self._num_increments


# ---------------------------------------------------------------------------
# ForgeMetricsCallback
# ---------------------------------------------------------------------------


class ForgeMetricsCallback(BaseCallback):
    """Episode-metrics logging callback for FORGE training.

    At the end of every episode the callback extracts episode return,
    length, and (when present) task progress from the ``info`` dict and
    dispatches them to a :class:`~forge.training.loggers.ForgeLogger`.

    If no logger is provided the metrics are emitted to the Python
    :mod:`logging` module at ``INFO`` level.

    Args:
        forge_logger: An optional :class:`~forge.training.loggers.ForgeLogger`
            instance (e.g. :class:`~forge.training.loggers.WandbLogger`).
        log_freq: Log metrics every *log_freq* timesteps (not episodes).
            Set to 1 to log every episode.
        verbose: Verbosity level (0 = silent, 1 = log each flush).

    Raises:
        ImportError: If Stable Baselines 3 is not installed.
    """

    def __init__(
        self,
        forge_logger: ForgeLogger | None = None,
        log_freq: int = 1000,
        verbose: int = 0,
    ) -> None:
        _require_sb3()
        super().__init__(verbose=verbose)

        if log_freq < 1:
            raise ValueError(f"log_freq must be >= 1, got {log_freq}")

        self._forge_logger = forge_logger
        self.log_freq = log_freq
        self._pending_metrics: dict[str, list[float]] = {}
        self._last_log_step: int = 0

    # -- BaseCallback hooks -------------------------------------------------

    def _on_step(self) -> bool:
        """Collect metrics from completed episodes."""
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get(_EPISODE_KEY)
            if ep_info is None:
                continue
            self._record("episode/return", float(ep_info.get("r", 0.0)))
            self._record("episode/length", float(ep_info.get("l", 0)))

            # Optional FORGE-specific keys
            if _TASK_PROGRESS_KEY in info:
                progress = info[_TASK_PROGRESS_KEY]
                if hasattr(progress, "__len__"):
                    mean_progress = sum(progress) / len(progress) if progress else 0.0
                else:
                    mean_progress = float(progress)
                self._record("episode/task_progress", mean_progress)

            if _TASK_SUCCESS_KEY in info:
                self._record("episode/task_success", float(info[_TASK_SUCCESS_KEY]))

        # Flush on schedule
        if (self.num_timesteps - self._last_log_step) >= self.log_freq:
            self._flush()

        return True  # Continue training

    def _on_training_end(self) -> None:
        """Flush any remaining metrics when training finishes."""
        self._flush()
        if self._forge_logger is not None:
            self._forge_logger.close()

    # -- helpers ------------------------------------------------------------

    def _record(self, key: str, value: float) -> None:
        """Buffer a metric value."""
        self._pending_metrics.setdefault(key, []).append(value)

    def _flush(self) -> None:
        """Compute per-key means and dispatch to the logger."""
        if not self._pending_metrics:
            return

        aggregated: dict[str, float] = {
            k: sum(vs) / len(vs) for k, vs in self._pending_metrics.items()
        }
        step = self.num_timesteps

        if self._forge_logger is not None:
            self._forge_logger.log(aggregated, step=step)
        elif self.verbose >= 1:
            for k, v in aggregated.items():
                logger.info("step=%d %s=%.4f", step, k, v)

        self._pending_metrics.clear()
        self._last_log_step = step
