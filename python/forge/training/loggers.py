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
from typing import TYPE_CHECKING, Any

from forge.training.mlflow_config import MlflowSettings

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

logger = logging.getLogger(__name__)

__all__ = [
    "CompositeLogger",
    "ForgeLogger",
    "MLflowLogger",
    "MlflowSettings",
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
    """MLflow experiment logger with the full upstream surface.

    Backwards compatible with the original constructor signature:

    .. code-block:: python

        MLflowLogger(experiment_name="exp", run_name="run-1",
                     tracking_uri="file:./mlruns", params={"lr": 1e-3})

    For richer configuration, pass a :class:`MlflowSettings` instance via
    ``settings=``.  Explicit keyword args still win on a per-field basis
    so callers can mix both styles.

    The logger surfaces every method most training loops actually need:

    * :meth:`log` — scalar metrics (alias for ``mlflow.log_metrics``).
    * :meth:`log_params` — bulk parameter logging.
    * :meth:`log_artifact` / :meth:`log_artifacts` — files and directories.
    * :meth:`log_dict` / :meth:`log_text` — structured JSON / plain text
      blobs written into the artifact store.
    * :meth:`set_tags` / :meth:`set_tag` — per-run tags.

    All write paths are wrapped in best-effort error handling: an MLflow
    REST/IO exception is logged at ``WARNING`` and swallowed so a
    transient backend failure cannot crash a multi-hour training run.
    To opt into strict propagation set ``strict_errors=True``.

    The class is also a context manager — ``with MLflowLogger(...) as
    log:`` automatically calls :meth:`close` on exit, even on exception.

    Args:
        experiment_name: MLflow experiment name (created if missing).
            Optional when ``settings.experiment_name`` is provided.
        run_name: Optional run name.
        tracking_uri: Optional MLflow tracking server URI.  Falls back to
            ``settings.tracking_uri``, then to the MLflow library default.
        params: Optional hyperparameter dict logged once at start.
        tags: Optional tag dict applied at run creation.
        settings: Optional :class:`MlflowSettings` providing defaults for
            every other knob (tracking URI, registry URI, system metrics,
            artifact location).
        strict_errors: When ``True``, MLflow exceptions raised during
            ``log_*`` calls re-raise.  Defaults to ``False``.

    Raises:
        ImportError: If ``mlflow`` is not installed.
        ValueError: If neither ``experiment_name`` nor
            ``settings.experiment_name`` is supplied.
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        params: dict[str, Any] | None = None,
        *,
        tags: dict[str, str] | None = None,
        settings: MlflowSettings | None = None,
        strict_errors: bool = False,
    ) -> None:
        try:
            import mlflow

            self._mlflow = mlflow
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MLflowLogger. Install with: pip install mlflow"
            ) from exc

        # Resolve effective settings: explicit kwargs override settings fields.
        resolved = (settings or MlflowSettings()).merge(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
        )
        if not resolved.experiment_name:
            raise ValueError(
                "MLflowLogger requires an experiment_name (constructor arg or "
                "settings.experiment_name)."
            )
        self._settings = resolved
        self._strict = strict_errors

        resolved.apply_to(mlflow)

        if resolved.log_system_metrics:
            self._enable_system_metrics()

        experiment = self._resolve_experiment(mlflow, resolved)
        self._active_run = mlflow.start_run(
            run_name=resolved.run_name,
            tags=resolved.tags or None,
            nested=resolved.nested,
        )
        if params:
            self.log_params(params)

        logger.info(
            "MLflowLogger started: experiment=%s run_name=%s run_id=%s tracking_uri=%s",
            resolved.experiment_name,
            resolved.run_name,
            getattr(getattr(self._active_run, "info", None), "run_id", "<unknown>"),
            resolved.tracking_uri or "<library-default>",
        )
        logger.debug(
            "MLflowLogger settings: %s, experiment_id=%s",
            resolved.describe(),
            getattr(experiment, "experiment_id", "<unknown>"),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def settings(self) -> MlflowSettings:
        """Resolved settings used by this run (read-only)."""
        return self._settings

    @property
    def active_run(self) -> Any:
        """The underlying ``mlflow.ActiveRun`` handle, useful for tests."""
        return self._active_run

    @property
    def run_id(self) -> str | None:
        """MLflow run id, or ``None`` if the run has already been closed."""
        run = self._active_run
        if run is None:
            return None
        info = getattr(run, "info", None)
        return getattr(info, "run_id", None) if info is not None else None

    # ------------------------------------------------------------------
    # ForgeLogger interface
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at the given step.

        Args:
            metrics: Dict of metric names → values.
            step: Global training step.
        """
        self._safe_call("log_metrics", self._mlflow.log_metrics, metrics, step=step)

    def close(self) -> None:
        """End the active MLflow run (idempotent)."""
        if self._active_run is None:
            return
        try:
            self._mlflow.end_run()
        finally:
            self._active_run = None
            logger.info("MLflowLogger: run ended")

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a flat dict of hyperparameters (string-keyed)."""
        if not params:
            return
        self._safe_call("log_params", self._mlflow.log_params, params)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Log a single file artifact.

        Args:
            local_path: Path to the file on disk.
            artifact_path: Optional sub-directory inside the run's
                artifact root.
        """
        self._safe_call(
            "log_artifact",
            self._mlflow.log_artifact,
            str(local_path),
            artifact_path=artifact_path,
        )

    def log_artifacts(
        self,
        local_dir: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Recursively log every file under ``local_dir``."""
        self._safe_call(
            "log_artifacts",
            self._mlflow.log_artifacts,
            str(local_dir),
            artifact_path=artifact_path,
        )

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Log a dict as a JSON/YAML artifact (extension-driven)."""
        self._safe_call(
            "log_dict",
            self._mlflow.log_dict,
            dictionary,
            artifact_file,
        )

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log a plain-text artifact under ``artifact_file``."""
        self._safe_call("log_text", self._mlflow.log_text, text, artifact_file)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set or update multiple run tags atomically."""
        if not tags:
            return
        self._safe_call("set_tags", self._mlflow.set_tags, tags)

    def set_tag(self, key: str, value: str) -> None:
        """Set a single run tag."""
        self._safe_call("set_tag", self._mlflow.set_tag, key, value)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MLflowLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enable_system_metrics(self) -> None:
        """Best-effort system-metrics autologging (CPU/GPU/RAM)."""
        enable = getattr(self._mlflow, "enable_system_metrics_logging", None)
        if enable is None:
            logger.debug(
                "mlflow.enable_system_metrics_logging unavailable; "
                "skipping system metrics autologging"
            )
            return
        try:
            enable()
            logger.info("MLflow system-metrics logging enabled")
        except Exception:
            logger.warning("Failed to enable MLflow system-metrics logging", exc_info=True)

    def _safe_call(self, op: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke a logging op, swallowing errors unless ``strict_errors``."""
        try:
            return fn(*args, **kwargs)
        except Exception:
            if self._strict:
                raise
            logger.warning("MLflow %s failed (swallowed)", op, exc_info=True)
            return None

    @staticmethod
    def _resolve_experiment(mlflow_module: Any, resolved: MlflowSettings) -> Any:
        """Idempotently resolve (or create) an experiment by name.

        ``mlflow.set_experiment`` exists across all 2.x and 3.x releases but
        the accepted kwargs vary (``artifact_location`` is only honoured by
        the underlying ``create_experiment`` call).  Doing the lookup
        ourselves keeps the call signature stable across MLflow versions
        and lets us honour ``artifact_location`` only when the experiment
        is being created for the first time.
        """
        name = resolved.experiment_name
        if not name:
            msg = "experiment_name is required to resolve an MLflow experiment"
            raise ValueError(msg)
        existing = mlflow_module.get_experiment_by_name(name)
        if existing is None:
            try:
                experiment_id = mlflow_module.create_experiment(
                    name=name, artifact_location=resolved.artifact_location
                )
                logger.info(
                    "Created MLflow experiment '%s' (id=%s, artifact_location=%s)",
                    name,
                    experiment_id,
                    resolved.artifact_location or "<default>",
                )
            except Exception as exc:
                # TOCTOU: another process created the experiment between our
                # get_experiment_by_name check and create_experiment call.
                if "already exists" not in str(exc).lower():
                    raise
                logger.debug(
                    "Experiment '%s' created concurrently; proceeding normally",
                    name,
                )
        mlflow_module.set_experiment(experiment_name=name)
        return mlflow_module.get_experiment_by_name(name)


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
                "PyTorch is required for TensorBoardLogger. Install with: pip install torch"
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
            f"Unknown logger backend '{backend}'. Available: {sorted(_LOGGER_REGISTRY)}"
        )
    return _LOGGER_REGISTRY[backend](**kwargs)
