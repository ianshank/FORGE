"""Unit tests for the MLflow helper functions exposed by ``scripts/train.py``.

These cover the gap left by ``test_train_cli.py``: the CLI parser is
already exercised there, but the MLflow plumbing helpers
(``_build_mlflow_settings``, ``_params_for_run``, ``_flatten_for_params``,
``_maybe_make_mlflow_logger``) were untested.  All tests are pure-Python
and mock-driven so they run in the same sub-second envelope as the rest
of the Python suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from train import (
    _DEFAULT_MLFLOW_EXPERIMENT,
    _build_mlflow_settings,
    _flatten_for_params,
    _maybe_make_mlflow_logger,
    _params_for_run,
    parse_args,
)


def _ns(**overrides: Any) -> Any:
    """Build a CLI namespace with sane defaults for the MLflow helpers."""
    base = parse_args(["--agent", "random", "--episodes", "1"])
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class TestBuildMlflowSettings:
    """Tests for :func:`_build_mlflow_settings`."""

    def test_defaults_use_constant_experiment_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "MLFLOW_TRACKING_URI",
            "MLFLOW_REGISTRY_URI",
            "MLFLOW_EXPERIMENT_NAME",
            "MLFLOW_RUN_NAME",
            "MLFLOW_ARTIFACT_LOCATION",
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
            "FORGE_MLFLOW_TAGS",
        ):
            monkeypatch.delenv(var, raising=False)
        args = _ns(agent="random", seed=7)
        settings = _build_mlflow_settings(args)
        assert settings.experiment_name == _DEFAULT_MLFLOW_EXPERIMENT
        assert settings.run_name == "random-seed7"
        assert settings.tracking_uri is None
        assert settings.tags == {}

    def test_cli_flags_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-host:5000")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "env-exp")
        monkeypatch.setenv("FORGE_MLFLOW_TAGS", "from=env,shared=env-value")
        args = _ns(
            mlflow_tracking_uri="http://cli-host:5000",
            mlflow_experiment="cli-exp",
            mlflow_run_name="cli-run",
            mlflow_tags=["from=cli", "shared=cli-value"],
        )
        settings = _build_mlflow_settings(args)
        assert settings.tracking_uri == "http://cli-host:5000"
        assert settings.experiment_name == "cli-exp"
        assert settings.run_name == "cli-run"
        # env tag preserved, cli tag overrides on collision
        assert settings.tags == {"from": "cli", "shared": "cli-value"}

    def test_env_tags_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_MLFLOW_TAGS", "team=forge,phase=mlflow")
        args = _ns(mlflow_tags=[])
        settings = _build_mlflow_settings(args)
        assert settings.tags == {"team": "forge", "phase": "mlflow"}

    def test_system_metrics_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", raising=False)
        args = _ns(mlflow_system_metrics=True)
        assert _build_mlflow_settings(args).log_system_metrics is True

    def test_system_metrics_env_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true")
        args = _ns(mlflow_system_metrics=False)
        # CLI flag is False → None passed to merge → env-driven value (True) preserved
        assert _build_mlflow_settings(args).log_system_metrics is True


class TestFlattenForParams:
    """Tests for :func:`_flatten_for_params`."""

    def test_scalar_passthrough(self) -> None:
        assert _flatten_for_params({"a": 1, "b": "x"}) == {"a": "1", "b": "x"}

    def test_nested_dict(self) -> None:
        assert _flatten_for_params({"a": {"b": {"c": 3}}}) == {"a.b.c": "3"}

    def test_lists_joined(self) -> None:
        assert _flatten_for_params({"xs": [1, 2, 3]}) == {"xs": "1,2,3"}

    def test_empty_list_dropped(self) -> None:
        assert _flatten_for_params({"xs": []}) == {}

    def test_none_dropped(self) -> None:
        assert _flatten_for_params({"a": None, "b": 2}) == {"b": "2"}


class TestParamsForRun:
    """Tests for :func:`_params_for_run`."""

    def test_skips_none_and_mlflow_keys(self) -> None:
        args = _ns(mlflow_enabled=True, mlflow_tracking_uri="http://x")
        flat = _params_for_run(args, config=SimpleNamespace())
        assert not any(key.startswith("cli.mlflow") for key in flat)
        assert "cli.agent" in flat
        assert flat["cli.agent"] == "random"

    def test_lists_become_csv(self) -> None:
        args = _ns(scenario=["s1", "s2"])
        flat = _params_for_run(args, config=SimpleNamespace())
        assert flat["cli.scenario"] == "s1,s2"

    def test_config_to_dict_merged(self) -> None:
        cfg = SimpleNamespace(to_dict=lambda: {"forge": {"seed": 7, "name": "abc"}})
        flat = _params_for_run(_ns(), cfg)
        assert flat["config.forge.seed"] == "7"
        assert flat["config.forge.name"] == "abc"

    def test_config_to_dict_errors_swallowed(self) -> None:
        def boom() -> dict[str, Any]:
            raise RuntimeError("explode")

        cfg = SimpleNamespace(to_dict=boom)
        flat = _params_for_run(_ns(), cfg)
        assert "cli.agent" in flat
        assert not any(key.startswith("config.") for key in flat)


class TestMaybeMakeMlflowLogger:
    """Tests for :func:`_maybe_make_mlflow_logger`."""

    def test_disabled_returns_none(self) -> None:
        args = _ns(mlflow_enabled=False)
        assert _maybe_make_mlflow_logger(args, config=SimpleNamespace()) is None

    def test_logger_init_error_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        args = _ns(mlflow_enabled=True)
        broken = MagicMock(side_effect=RuntimeError("boom"))
        with patch("forge.training.loggers.MLflowLogger", broken):
            assert _maybe_make_mlflow_logger(args, config=SimpleNamespace()) is None
        broken.assert_called_once()

    def test_import_error_swallowed(self) -> None:
        args = _ns(mlflow_enabled=True)
        broken = MagicMock(side_effect=ImportError("no mlflow"))
        with patch("forge.training.loggers.MLflowLogger", broken):
            assert _maybe_make_mlflow_logger(args, config=SimpleNamespace()) is None

    def test_happy_path_logs_params_and_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        config_file = tmp_path / "forge.toml"
        config_file.write_text("# stub\n", encoding="utf-8")

        args = _ns(mlflow_enabled=True, config=str(config_file))
        fake_logger = MagicMock()
        fake_logger.settings = MagicMock()
        fake_logger.settings.describe.return_value = {"experiment_name": "x"}

        with patch("forge.training.loggers.MLflowLogger", return_value=fake_logger) as ctor:
            result = _maybe_make_mlflow_logger(
                args, config=SimpleNamespace(to_dict=lambda: {"k": 1})
            )

        assert result is fake_logger
        ctor.assert_called_once()
        # log_params received the flattened payload
        fake_logger.log_params.assert_called_once()
        params = fake_logger.log_params.call_args[0][0]
        assert "cli.agent" in params
        assert "config.k" in params
        # artifact upload of the config file
        fake_logger.log_artifact.assert_called_once_with(
            str(config_file), artifact_path="config"
        )
        fake_logger.log_dict.assert_called_once()
        target = fake_logger.log_dict.call_args[0][1]
        assert target == "config/mlflow_settings.json"

    def test_missing_config_skips_artifact_upload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        args = _ns(mlflow_enabled=True, config=str(tmp_path / "absent.toml"))
        fake_logger = MagicMock()
        fake_logger.settings.describe.return_value = {}
        with patch("forge.training.loggers.MLflowLogger", return_value=fake_logger):
            _maybe_make_mlflow_logger(args, config=SimpleNamespace())
        fake_logger.log_artifact.assert_not_called()
