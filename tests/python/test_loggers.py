"""Tests for forge.training.loggers — ForgeLogger hierarchy and factory."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from forge.training.loggers import (
    CompositeLogger,
    ForgeLogger,
    MLflowLogger,
    TensorBoardLogger,
    WandbLogger,
    make_logger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingLogger(ForgeLogger):
    """Minimal concrete logger used to test the abstract interface."""

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int]] = []
        self.closed = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logged.append((metrics, step))

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# ForgeLogger (abstract contract)
# ---------------------------------------------------------------------------


class TestForgeLoggerContract:
    """The abstract ForgeLogger is correctly subclassable."""

    def test_concrete_subclass_works(self) -> None:
        logger = _RecordingLogger()
        logger.log({"loss": 0.5}, step=100)
        logger.close()
        assert logger.logged == [({"loss": 0.5}, 100)]
        assert logger.closed is True

    def test_multiple_log_calls(self) -> None:
        logger = _RecordingLogger()
        logger.log({"a": 1.0}, step=1)
        logger.log({"b": 2.0}, step=2)
        assert len(logger.logged) == 2

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            ForgeLogger()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------


class TestWandbLogger:
    """Tests for WandbLogger using a mocked wandb module."""

    def _make(self) -> tuple[WandbLogger, MagicMock]:
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            # Re-patch inside the import
            import forge.training.loggers as loggers_mod  # noqa: PLC0415
            loggers_mod_wandb = loggers_mod.WandbLogger.__module__

            logger = WandbLogger.__new__(WandbLogger)
            logger._wandb = mock_wandb
            logger._run = mock_run

        return logger, mock_wandb

    def test_log_calls_wandb_log(self) -> None:
        logger, mock_wandb = self._make()
        logger.log({"loss": 0.3}, step=50)
        mock_wandb.log.assert_called_once_with({"loss": 0.3}, step=50)

    def test_close_calls_run_finish(self) -> None:
        logger, _ = self._make()
        logger.close()
        logger._run.finish.assert_called_once()

    def test_missing_wandb_raises(self) -> None:
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises(ImportError, match="wandb"):
                WandbLogger(project="test")


# ---------------------------------------------------------------------------
# MLflowLogger
# ---------------------------------------------------------------------------


class TestMLflowLogger:
    """Tests for MLflowLogger using a mocked mlflow module."""

    def _make(self) -> tuple[MLflowLogger, MagicMock]:
        mock_mlflow = MagicMock()
        logger = MLflowLogger.__new__(MLflowLogger)
        logger._mlflow = mock_mlflow
        logger._active_run = MagicMock()
        return logger, mock_mlflow

    def test_log_calls_mlflow_log_metrics(self) -> None:
        logger, mock_mlflow = self._make()
        logger.log({"acc": 0.9}, step=200)
        mock_mlflow.log_metrics.assert_called_once_with({"acc": 0.9}, step=200)

    def test_close_calls_end_run(self) -> None:
        logger, mock_mlflow = self._make()
        logger.close()
        mock_mlflow.end_run.assert_called_once()

    def test_missing_mlflow_raises(self) -> None:
        with patch.dict(sys.modules, {"mlflow": None}):
            with pytest.raises(ImportError, match="mlflow"):
                MLflowLogger(experiment_name="test")


# ---------------------------------------------------------------------------
# TensorBoardLogger
# ---------------------------------------------------------------------------


class TestTensorBoardLogger:
    """Tests for TensorBoardLogger using a mocked SummaryWriter."""

    def _make(self) -> tuple[TensorBoardLogger, MagicMock]:
        mock_writer = MagicMock()
        logger = TensorBoardLogger.__new__(TensorBoardLogger)
        logger._writer = mock_writer
        return logger, mock_writer

    def test_log_calls_add_scalar_per_metric(self) -> None:
        logger, mock_writer = self._make()
        logger.log({"loss": 0.1, "reward": 2.5}, step=10)
        calls = mock_writer.add_scalar.call_args_list
        assert len(calls) == 2
        tags = {c[0][0] for c in calls}
        assert "loss" in tags
        assert "reward" in tags

    def test_log_passes_global_step(self) -> None:
        logger, mock_writer = self._make()
        logger.log({"x": 1.0}, step=42)
        _, kwargs = mock_writer.add_scalar.call_args
        assert kwargs.get("global_step") == 42 or mock_writer.add_scalar.call_args[0][2] == 42

    def test_close_flushes_and_closes_writer(self) -> None:
        logger, mock_writer = self._make()
        logger.close()
        mock_writer.flush.assert_called_once()
        mock_writer.close.assert_called_once()

    def test_missing_torch_raises(self) -> None:
        with patch.dict(sys.modules, {"torch": None, "torch.utils.tensorboard": None}):
            with pytest.raises(ImportError, match="PyTorch"):
                TensorBoardLogger(log_dir="/tmp/test")


# ---------------------------------------------------------------------------
# CompositeLogger
# ---------------------------------------------------------------------------


class TestCompositeLogger:
    """Tests for CompositeLogger fan-out behaviour."""

    def test_empty_loggers_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeLogger([])

    def test_log_dispatches_to_all(self) -> None:
        children = [_RecordingLogger(), _RecordingLogger()]
        composite = CompositeLogger(children)
        composite.log({"val": 0.5}, step=1)
        for child in children:
            assert child.logged == [({"val": 0.5}, 1)]

    def test_close_dispatches_to_all(self) -> None:
        children = [_RecordingLogger(), _RecordingLogger()]
        composite = CompositeLogger(children)
        composite.close()
        assert all(c.closed for c in children)

    def test_loggers_property(self) -> None:
        children = [_RecordingLogger()]
        composite = CompositeLogger(children)
        assert len(composite.loggers) == 1

    def test_multiple_log_calls(self) -> None:
        child = _RecordingLogger()
        composite = CompositeLogger([child])
        for i in range(5):
            composite.log({"step_val": float(i)}, step=i)
        assert len(child.logged) == 5

    def test_three_children(self) -> None:
        children = [_RecordingLogger() for _ in range(3)]
        composite = CompositeLogger(children)
        composite.log({"metric": 1.0}, step=10)
        assert all(len(c.logged) == 1 for c in children)


# ---------------------------------------------------------------------------
# make_logger factory
# ---------------------------------------------------------------------------


class TestMakeLogger:
    """Tests for the make_logger factory function."""

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown logger backend"):
            make_logger("foobar")

    def test_available_backends_listed_in_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            make_logger("bad_backend")
        msg = str(exc_info.value)
        for backend in ("wandb", "mlflow", "tensorboard"):
            assert backend in msg

    def test_tensorboard_backend_key(self) -> None:
        """'tensorboard' key should map to TensorBoardLogger."""
        mock_writer = MagicMock()
        mock_tb_module = MagicMock()
        mock_tb_module.SummaryWriter.return_value = mock_writer

        with patch.dict(sys.modules, {"torch.utils.tensorboard": mock_tb_module}):
            result = make_logger("tensorboard", log_dir="/tmp/tb")
        assert isinstance(result, TensorBoardLogger)
