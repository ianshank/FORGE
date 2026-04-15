"""Pluggable logging adapters for FORGE training.

Provides a thin, uniform interface over experiment-tracking backends
(Weights & Biases, MLflow, TensorBoard) so that training scripts and SB3
callbacks can log metrics without being coupled to a specific backend.

All external dependencies (``wandb``, ``mlflow``, ``torch.utils.tensorboard``)
are imported lazily and guarded by ``try/except ImportError``.  If a backend
is not installed, constructing its logger raises an :class:`ImportError` with
an actionable install hint.

Classes
-------
ForgeLogger
    Abstract base class.  Implement :meth:`log` and :meth:`close`.
WandbLogger
    Logs to Weights & Biases via ``wandb.log()``.
MLflowLogger
    Logs to MLflow via ``mlflow.log_metrics()``.
TensorBoardLogger
    Logs to TensorBoard via ``torch.utils.tensorboard.SummaryWriter``.
CompositeLogger
    Fan-out: dispatches every call to a list of child loggers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CompositeLogger",
    "ForgeLogger",
    "MLflowLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "make_logger",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ForgeLogger(ABC):
    """Abstract experiment-tracking logger.

    All concrete loggers implement exactly two methods so that training
    scripts can swap backends by changing a single constructor call.
    """

    @abstractmethod
    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics at a given training step.

        Args:
            metrics: Mapping from metric name to float value.
            step: Global training step (e.g. ``num_timesteps``).
        """

    @abstractmethod
    def close(self) -> None:
        """Flush buffers and release resources."""


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------


class WandbLogger(ForgeLogger):
    """Weights & Biases logger.

    Initialises a W&B run on construction (or attaches to an existing one
    if called inside an already-active run).

    Args:
        project: W&B project name.
        run_name: Optional run name shown in the W&B UI.
        config: Optional hyperparameter dict stored in the run's config.
        tags: Optional list of tags.
        kwargs: Additional keyword arguments forwarded to ``wandb.init()``.

    Raises:
        ImportError: If ``wandb`` is not installed.
    """

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import wandb

            self._wandb = wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required for WandbLogger. Install with: pip install wandb"
            ) from exc

        self._run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            **kwargs,
        )
        logger.info("WandbLogger: run %s initialised (project=%s)", run_name, project)

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dict of metric names → values.
            step: Global training step.
        """
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            self._run.finish()
            logger.info("WandbLogger: run finished")


# ---------------------------------------------------------------------------
# MLflowLogger
# ---------------------------------------------------------------------------


class MLflowLogger(ForgeLogger):
    """MLflow experiment logger.

    Args:
        experiment_name: MLflow experiment name (created if it does not
            already exist).
        run_name: Optional run name.
        tracking_uri: Optional MLflow tracking server URI.  If ``None``
            the default local ``./mlruns`` directory is used.
        params: Optional hyperparameter dict logged once at start.

    Raises:
        ImportError: If ``mlflow`` is not installed.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        try:
            import mlflow

            self._mlflow = mlflow
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MLflowLogger. Install with: pip install mlflow"
            ) from exc

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self._active_run = mlflow.start_run(run_name=run_name)

        if params:
            mlflow.log_params(params)

        logger.info(
            "MLflowLogger: run %s started (experiment=%s)",
            run_name,
            experiment_name,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dict of metric names → values.
            step: Global training step.
        """
        self._mlflow.log_metrics(metrics, step=step)

    def close(self) -> None:
        """End the active MLflow run."""
        self._mlflow.end_run()
        logger.info("MLflowLogger: run ended")


# ---------------------------------------------------------------------------
# TensorBoardLogger
# ---------------------------------------------------------------------------


class TensorBoardLogger(ForgeLogger):
    """TensorBoard logger backed by PyTorch's ``SummaryWriter``.

    Args:
        log_dir: Directory where TensorBoard event files are written.
        comment: Optional suffix appended to the auto-generated log dir.

    Raises:
        ImportError: If ``torch`` is not installed.
    """

    def __init__(
        self,
        log_dir: str,
        comment: str = "",
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=log_dir, comment=comment)
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for TensorBoardLogger. "
                "Install with: pip install torch"
            ) from exc

        logger.info("TensorBoardLogger: writing to %s", log_dir)

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Write scalar metrics to TensorBoard.

        Args:
            metrics: Dict of tag names → values.
            step: Global training step.
        """
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, global_step=step)

    def close(self) -> None:
        """Flush and close the SummaryWriter."""
        self._writer.flush()
        self._writer.close()
        logger.info("TensorBoardLogger: writer closed")


# ---------------------------------------------------------------------------
# CompositeLogger
# ---------------------------------------------------------------------------


class CompositeLogger(ForgeLogger):
    """Fan-out logger that dispatches to multiple child loggers.

    Args:
        loggers: List of :class:`ForgeLogger` instances.  Must contain at
            least one logger.

    Raises:
        ValueError: If *loggers* is empty.

    Example::

        from forge.training.loggers import CompositeLogger, WandbLogger, TensorBoardLogger

        combo = CompositeLogger([
            WandbLogger(project="my-project"),
            TensorBoardLogger(log_dir="runs/exp1"),
        ])
        combo.log({"loss": 0.42}, step=1000)
        combo.close()
    """

    def __init__(self, loggers: list[ForgeLogger]) -> None:
        if not loggers:
            raise ValueError("CompositeLogger requires at least one child logger")
        self._loggers = list(loggers)

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Dispatch metrics to all child loggers.

        Args:
            metrics: Dict of metric names → values.
            step: Global training step.
        """
        for child in self._loggers:
            child.log(metrics, step=step)

    def close(self) -> None:
        """Close all child loggers."""
        for child in self._loggers:
            child.close()

    @property
    def loggers(self) -> list[ForgeLogger]:
        """The list of child loggers (read-only view)."""
        return list(self._loggers)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_LOGGER_REGISTRY: dict[str, type[ForgeLogger]] = {
    "wandb": WandbLogger,
    "mlflow": MLflowLogger,
    "tensorboard": TensorBoardLogger,
}


def make_logger(
    backend: str,
    **kwargs: Any,
) -> ForgeLogger:
    """Construct a :class:`ForgeLogger` by name.

    Args:
        backend: One of ``"wandb"``, ``"mlflow"``, or ``"tensorboard"``.
        **kwargs: Passed directly to the logger's constructor.

    Returns:
        A :class:`ForgeLogger` instance.

    Raises:
        ValueError: If *backend* is not recognised.
        ImportError: If the required backend library is not installed.

    Example::

        logger = make_logger("tensorboard", log_dir="runs/experiment")
        logger.log({"loss": 0.5}, step=100)
    """
    if backend not in _LOGGER_REGISTRY:
        raise ValueError(
            f"Unknown logger backend '{backend}'. "
            f"Available: {sorted(_LOGGER_REGISTRY)}"
        )
    return _LOGGER_REGISTRY[backend](**kwargs)
