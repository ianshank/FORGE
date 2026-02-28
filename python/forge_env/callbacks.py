"""forge_env/callbacks.py — Pluggable training callbacks for FORGE environments.

Provides abstract ``LoggingCallback`` base and ready-made implementations for
W&B (``WandbCallback``) and MLflow (``MLflowCallback``).  When neither library is
installed the callbacks emit warnings and become no-ops so training scripts remain
portable.

Usage::

    from forge_env.callbacks import WandbCallback, MetricsCallback
    from forge_env.wrappers import RecordEpisodeStatistics

    callback = WandbCallback(project="forge-ppo")
    env = RecordEpisodeStatistics(env, on_episode_end=callback)
"""

from __future__ import annotations

import csv
import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, IO, Any

if TYPE_CHECKING:
    import mlflow as mlflow_t
    import wandb as wandb_t

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional-library detection
# ---------------------------------------------------------------------------

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _HAS_WANDB = False

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _HAS_MLFLOW = False

# Runtime aliases used only inside type-annotated branches
_wandb: wandb_t | None = wandb
_mlflow: mlflow_t | None = mlflow


# ---------------------------------------------------------------------------
# EpisodeStats — passed to every callback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeStats:
    """Immutable snapshot of episode metrics fired when an episode ends."""

    episode: int
    """Zero-based episode index."""
    total_steps: int
    """Global step counter at episode end."""
    episode_length: int
    """Number of steps in this episode."""
    episode_return: float
    """Sum of rewards in this episode."""
    fps: float
    """Approximate frames-per-second over the episode."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Any additional metrics from the environment info dict."""

    def as_dict(self) -> dict[str, Any]:
        """Return all stats as a flat dict suitable for logging."""
        return {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "episode_length": self.episode_length,
            "episode_return": self.episode_return,
            "fps": self.fps,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LoggingCallback(ABC):
    """Abstract callback invoked at episode boundaries during training."""

    @abstractmethod
    def on_episode_end(self, stats: EpisodeStats) -> None:
        """Called when an episode finishes.

        Parameters
        ----------
        stats:
            Immutable snapshot of episode metrics.
        """

    def on_training_start(self) -> None:  # noqa: B027
        """Optional hook called before training begins."""

    def on_training_end(self) -> None:  # noqa: B027
        """Optional hook called after training concludes."""


# ---------------------------------------------------------------------------
# Composite callback (fan-out)
# ---------------------------------------------------------------------------


class CompositeCallback(LoggingCallback):
    """Fan-out wrapper: delegates to multiple callbacks in order.

    Parameters
    ----------
    callbacks:
        List of ``LoggingCallback`` instances to invoke.
    """

    def __init__(self, callbacks: list[LoggingCallback]) -> None:
        self._callbacks = list(callbacks)

    def on_episode_end(self, stats: EpisodeStats) -> None:
        for cb in self._callbacks:
            try:
                cb.on_episode_end(stats)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Callback %s raised: %s", cb, exc)

    def on_training_start(self) -> None:
        for cb in self._callbacks:
            cb.on_training_start()

    def on_training_end(self) -> None:
        for cb in self._callbacks:
            cb.on_training_end()


# ---------------------------------------------------------------------------
# CSV / console callbacks (zero-dep)
# ---------------------------------------------------------------------------


class ConsoleCallback(LoggingCallback):
    """Prints episode stats to stdout every ``log_every`` episodes.

    Parameters
    ----------
    log_every:
        Log frequency in episodes (default: 1).
    """

    def __init__(self, log_every: int = 1) -> None:
        self._log_every = max(1, log_every)

    def on_episode_end(self, stats: EpisodeStats) -> None:
        if stats.episode % self._log_every == 0:
            print(
                f"[ep {stats.episode:>5d}] "
                f"return={stats.episode_return:>8.3f}  "
                f"len={stats.episode_length:>5d}  "
                f"fps={stats.fps:>7.1f}  "
                f"steps={stats.total_steps:>8d}"
            )


class CsvCallback(LoggingCallback):
    """Writes one row per episode to a CSV file.

    Parameters
    ----------
    output_path:
        Path to write the CSV file.  Parent directories are created on demand.
    flush_every:
        Flush the file buffer every N episodes to reduce data loss risk.
    """

    def __init__(self, output_path: Path | str, flush_every: int = 10) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(1, flush_every)
        self._writer: csv.DictWriter[str] | None = None
        self._fh: IO[str] | None = None

    def on_training_start(self) -> None:
        self._fh = self._path.open("w", newline="", encoding="utf-8")
        # Writer initialised lazily on first episode (headers from EpisodeStats)
        logger.info("CsvCallback: writing metrics to %s", self._path)

    def on_episode_end(self, stats: EpisodeStats) -> None:
        row = stats.as_dict()
        if self._fh is None:
            self._fh = self._path.open("w", newline="", encoding="utf-8")

        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            self._writer.writeheader()

        self._writer.writerow(row)
        if stats.episode % self._flush_every == 0:
            self._fh.flush()

    def on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None
            logger.info("CsvCallback: closed %s", self._path)


# ---------------------------------------------------------------------------
# W&B callback
# ---------------------------------------------------------------------------


class WandbCallback(LoggingCallback):
    """Log episode metrics to Weights & Biases.

    Skips gracefully when ``wandb`` is not installed or ``WANDB_MODE=disabled``.

    Parameters
    ----------
    project:
        W&B project name.
    run_name:
        Optional run name.  Defaults to W&B auto-naming.
    config:
        Optional dict of hyperparameters to log to the run.
    """

    def __init__(
        self,
        project: str = "forge-rl",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._project = project
        self._run_name = run_name
        self._config = config or {}
        self._run = None

    def on_training_start(self) -> None:
        if not _HAS_WANDB:
            warnings.warn(
                "wandb is not installed — WandbCallback is inactive. "
                "Install it: pip install wandb",
                stacklevel=2,
            )
            return

        if os.environ.get("WANDB_MODE") == "disabled":
            logger.info("WandbCallback: WANDB_MODE=disabled, skipping init.")
            return

        self._run = wandb.init(
            project=self._project,
            name=self._run_name,
            config=self._config,
            resume="allow",
        )
        logger.info("WandbCallback: run %s started at %s", self._run.id, self._run.url)

    def on_episode_end(self, stats: EpisodeStats) -> None:
        if not _HAS_WANDB or self._run is None:
            return
        try:
            wandb.log(stats.as_dict(), step=stats.total_steps)
        except Exception as exc:
            logger.warning("WandbCallback.on_episode_end: %s", exc)

    def on_training_end(self) -> None:
        if not _HAS_WANDB or self._run is None:
            return
        wandb.finish()
        self._run = None
        logger.info("WandbCallback: run finished.")


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------


class MLflowCallback(LoggingCallback):
    """Log episode metrics to MLflow Tracking.

    Skips gracefully when ``mlflow`` is not installed.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name (created if it does not exist).
    run_name:
        Optional MLflow run name.
    tracking_uri:
        MLflow server URI.  Falls back to ``MLFLOW_TRACKING_URI`` env var, then
        local ``mlruns/`` directory.
    """

    def __init__(
        self,
        experiment_name: str = "forge-rl",
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        self._run = None

    def on_training_start(self) -> None:
        if not _HAS_MLFLOW:
            warnings.warn(
                "mlflow is not installed — MLflowCallback is inactive. "
                "Install it: pip install mlflow",
                stacklevel=2,
            )
            return

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)

        mlflow.set_experiment(self._experiment_name)
        self._run = mlflow.start_run(run_name=self._run_name)
        logger.info("MLflowCallback: run %s started.", self._run.info.run_id)

    def on_episode_end(self, stats: EpisodeStats) -> None:
        if not _HAS_MLFLOW or self._run is None:
            return
        try:
            mlflow.log_metrics(stats.as_dict(), step=stats.total_steps)
        except Exception as exc:
            logger.warning("MLflowCallback.on_episode_end: %s", exc)

    def on_training_end(self) -> None:
        if not _HAS_MLFLOW or self._run is None:
            return
        mlflow.end_run()
        self._run = None
        logger.info("MLflowCallback: run ended.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CompositeCallback",
    "ConsoleCallback",
    "CsvCallback",
    "EpisodeStats",
    "LoggingCallback",
    "MLflowCallback",
    "WandbCallback",
]
