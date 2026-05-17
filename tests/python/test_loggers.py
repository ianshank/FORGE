"""Tests for forge.training.loggers — ForgeLogger hierarchy and factory."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

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
            ForgeLogger()


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
        with patch.dict(sys.modules, {"wandb": None}), pytest.raises(ImportError, match="wandb"):
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
        with patch.dict(sys.modules, {"mlflow": None}), pytest.raises(ImportError, match="mlflow"):
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
        with (
            patch.dict(sys.modules, {"torch": None, "torch.utils.tensorboard": None}),
            pytest.raises(ImportError, match="PyTorch"),
        ):
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

    def test_mlflow_backend_key(self) -> None:
        """'mlflow' key should construct an MLflowLogger via the registry."""
        mock_mlflow = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"
        mock_mlflow.get_experiment_by_name.return_value = experiment
        mock_mlflow.set_experiment.return_value = experiment
        active = MagicMock()
        active.info.run_id = "run-1"
        mock_mlflow.start_run.return_value = active

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logger = make_logger("mlflow", experiment_name="from-factory")
        assert isinstance(logger, MLflowLogger)
        mock_mlflow.set_experiment.assert_called_once_with(experiment_name="from-factory")


# ---------------------------------------------------------------------------
# Extended MLflowLogger surface
# ---------------------------------------------------------------------------


class TestMLflowLoggerExtended:
    """Cover the params/artifact/dict/text/tag surface added on top of the
    backwards-compatible ``log()`` / ``close()`` interface."""

    def _make(self, *, strict: bool = False) -> tuple[MLflowLogger, MagicMock]:
        from forge.training.mlflow_config import MlflowSettings

        mock_mlflow = MagicMock()
        logger = MLflowLogger.__new__(MLflowLogger)
        logger._mlflow = mock_mlflow
        logger._active_run = MagicMock()
        logger._settings = MlflowSettings(experiment_name="t")
        logger._strict = strict
        return logger, mock_mlflow

    def test_log_params_forwards(self) -> None:
        logger, mock_mlflow = self._make()
        logger.log_params({"lr": 0.01, "bs": 32})
        mock_mlflow.log_params.assert_called_once_with({"lr": 0.01, "bs": 32})

    def test_log_params_empty_is_noop(self) -> None:
        logger, mock_mlflow = self._make()
        logger.log_params({})
        mock_mlflow.log_params.assert_not_called()

    def test_log_artifact_forwards(self) -> None:
        logger, mock_mlflow = self._make()
        logger.log_artifact("/tmp/a.txt", artifact_path="logs")
        mock_mlflow.log_artifact.assert_called_once_with("/tmp/a.txt", artifact_path="logs")

    def test_log_artifacts_forwards(self) -> None:
        logger, mock_mlflow = self._make()
        logger.log_artifacts("/tmp/dir", artifact_path="ckpt")
        mock_mlflow.log_artifacts.assert_called_once_with("/tmp/dir", artifact_path="ckpt")

    def test_log_dict_forwards(self) -> None:
        logger, mock_mlflow = self._make()
        logger.log_dict({"a": 1}, "config.json")
        mock_mlflow.log_dict.assert_called_once_with({"a": 1}, "config.json")

    def test_log_text_forwards(self) -> None:
        logger, mock_mlflow = self._make()
        logger.log_text("hi", "notes.txt")
        mock_mlflow.log_text.assert_called_once_with("hi", "notes.txt")

    def test_set_tags_forwards(self) -> None:
        logger, mock_mlflow = self._make()
        logger.set_tags({"env": "prod"})
        mock_mlflow.set_tags.assert_called_once_with({"env": "prod"})

    def test_set_tag_forwards(self) -> None:
        logger, mock_mlflow = self._make()
        logger.set_tag("k", "v")
        mock_mlflow.set_tag.assert_called_once_with("k", "v")

    def test_set_tags_empty_is_noop(self) -> None:
        logger, mock_mlflow = self._make()
        logger.set_tags({})
        mock_mlflow.set_tags.assert_not_called()

    def test_close_is_idempotent(self) -> None:
        logger, mock_mlflow = self._make()
        logger.close()
        logger.close()
        mock_mlflow.end_run.assert_called_once()

    def test_run_id_reflects_active_run(self) -> None:
        logger, _mock = self._make()
        logger._active_run.info.run_id = "abc123"
        assert logger.run_id == "abc123"
        logger.close()
        assert logger.run_id is None

    def test_settings_property_returns_resolved(self) -> None:
        logger, _mock = self._make()
        assert logger.settings.experiment_name == "t"

    def test_context_manager_closes_on_exit(self) -> None:
        logger, mock_mlflow = self._make()
        with logger as ctx:
            assert ctx is logger
        mock_mlflow.end_run.assert_called_once()

    def test_context_manager_closes_on_exception(self) -> None:
        logger, mock_mlflow = self._make()
        with pytest.raises(RuntimeError), logger:
            raise RuntimeError("boom")
        mock_mlflow.end_run.assert_called_once()

    def test_best_effort_swallows_exceptions(self) -> None:
        logger, mock_mlflow = self._make(strict=False)
        mock_mlflow.log_params.side_effect = RuntimeError("MLflow REST down")
        # Should not raise:
        logger.log_params({"a": 1})

    def test_strict_mode_re_raises(self) -> None:
        logger, mock_mlflow = self._make(strict=True)
        mock_mlflow.log_params.side_effect = RuntimeError("MLflow REST down")
        with pytest.raises(RuntimeError, match="REST"):
            logger.log_params({"a": 1})


class TestMLflowLoggerConstructor:
    """End-to-end constructor behaviour, with mlflow patched into sys.modules."""

    def _patch_mlflow(self) -> MagicMock:
        mock_mlflow = MagicMock()
        # set_experiment + get_experiment_by_name both return an experiment-like object
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"
        mock_mlflow.get_experiment_by_name.return_value = experiment
        mock_mlflow.set_experiment.return_value = experiment
        active_run = MagicMock()
        active_run.info.run_id = "run-1"
        mock_mlflow.start_run.return_value = active_run
        return mock_mlflow

    def test_constructor_requires_experiment_name(self) -> None:
        mock_mlflow = self._patch_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}), pytest.raises(
            ValueError, match="experiment_name"
        ):
            MLflowLogger()

    def test_constructor_accepts_settings_object(self) -> None:
        from forge.training.mlflow_config import MlflowSettings

        mock_mlflow = self._patch_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logger = MLflowLogger(
                settings=MlflowSettings(
                    experiment_name="exp",
                    tracking_uri="http://x",
                    tags={"env": "test"},
                )
            )
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://x")
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        assert logger.settings.experiment_name == "exp"

    def test_constructor_explicit_kwargs_override_settings(self) -> None:
        from forge.training.mlflow_config import MlflowSettings

        mock_mlflow = self._patch_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            logger = MLflowLogger(
                experiment_name="override",
                settings=MlflowSettings(experiment_name="base"),
            )
        assert logger.settings.experiment_name == "override"

    def test_constructor_logs_params_when_supplied(self) -> None:
        mock_mlflow = self._patch_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            MLflowLogger(experiment_name="exp", params={"lr": 0.01})
        mock_mlflow.log_params.assert_called_once_with({"lr": 0.01})

    def test_constructor_system_metrics_best_effort(self) -> None:
        from forge.training.mlflow_config import MlflowSettings

        mock_mlflow = self._patch_mlflow()
        # Older mlflow without the helper — getattr returns None, no crash:
        del mock_mlflow.enable_system_metrics_logging
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            MLflowLogger(
                settings=MlflowSettings(experiment_name="exp", log_system_metrics=True)
            )

    def test_constructor_creates_missing_experiment(self) -> None:
        mock_mlflow = self._patch_mlflow()
        # Simulate "not found" on the first lookup, then return on the second.
        mock_mlflow.get_experiment_by_name.side_effect = [
            None,
            mock_mlflow.get_experiment_by_name.return_value,
        ]
        mock_mlflow.create_experiment.return_value = "new-exp-id"
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            MLflowLogger(experiment_name="brand-new")
        mock_mlflow.create_experiment.assert_called_once_with(
            name="brand-new", artifact_location=None
        )
        mock_mlflow.set_experiment.assert_called_once_with(experiment_name="brand-new")

    def test_constructor_reuses_existing_experiment(self) -> None:
        mock_mlflow = self._patch_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            MLflowLogger(experiment_name="reused")
        mock_mlflow.create_experiment.assert_not_called()
        mock_mlflow.set_experiment.assert_called_once_with(experiment_name="reused")
