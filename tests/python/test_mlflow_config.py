"""Unit tests for :mod:`forge.training.mlflow_config`.

These tests are entirely mock-driven — they never reach out to a real
MLflow server or touch ``os.environ`` globally.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from forge.training.mlflow_config import (
    ENV_ARTIFACT_LOCATION,
    ENV_EXPERIMENT_NAME,
    ENV_FORGE_TAGS,
    ENV_HTTP_TIMEOUT,
    ENV_REGISTRY_URI,
    ENV_RUN_NAME,
    ENV_SYSTEM_METRICS,
    ENV_TRACKING_URI,
    MlflowSettings,
    _coerce_bool,
    _coerce_float,
    parse_tag_string,
)


class TestParseTagString:
    """``parse_tag_string`` is the single tag-parsing entry point."""

    def test_none_returns_empty(self) -> None:
        assert parse_tag_string(None) == {}

    def test_empty_returns_empty(self) -> None:
        assert parse_tag_string("") == {}

    def test_single_pair(self) -> None:
        assert parse_tag_string("env=staging") == {"env": "staging"}

    def test_multiple_pairs(self) -> None:
        result = parse_tag_string("env=staging,owner=team-ml,suite=bringup")
        assert result == {"env": "staging", "owner": "team-ml", "suite": "bringup"}

    def test_whitespace_is_trimmed(self) -> None:
        assert parse_tag_string("  env = staging , owner=team ") == {
            "env": "staging",
            "owner": "team",
        }

    def test_malformed_pair_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="forge.training.mlflow_config"):
            result = parse_tag_string("env=staging,notapair,owner=team")
        assert result == {"env": "staging", "owner": "team"}
        assert any("malformed" in rec.message.lower() for rec in caplog.records)

    def test_empty_key_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="forge.training.mlflow_config"):
            result = parse_tag_string("=value,owner=team")
        assert result == {"owner": "team"}
        assert any("empty key" in rec.message.lower() for rec in caplog.records)

    def test_value_can_be_empty(self) -> None:
        assert parse_tag_string("key=") == {"key": ""}

    def test_blank_chunk_skipped(self) -> None:
        # `,,key=value,` produces three empty chunks alongside the real pair.
        assert parse_tag_string(",,key=value,") == {"key": "value"}


class TestCoerceHelpers:
    """Private helpers should never raise — they should fall back cleanly."""

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
    def test_bool_truthy(self, raw: str) -> None:
        assert _coerce_bool(raw) is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
    def test_bool_falsy(self, raw: str) -> None:
        assert _coerce_bool(raw) is False

    def test_bool_none_uses_default(self) -> None:
        assert _coerce_bool(None) is False
        assert _coerce_bool(None, default=True) is True

    def test_bool_unknown_falls_back(self) -> None:
        assert _coerce_bool("maybe") is False

    def test_float_valid(self) -> None:
        assert _coerce_float("3.5") == 3.5

    def test_float_none_or_empty(self) -> None:
        assert _coerce_float(None) is None
        assert _coerce_float("") is None

    def test_float_invalid_logs_and_returns_none(self) -> None:
        assert _coerce_float("not-a-number") is None


class TestMlflowSettingsFromEnv:
    """Settings should accept an injectable mapping (no global env touch)."""

    def test_empty_env_is_all_default(self) -> None:
        s = MlflowSettings.from_env(env={})
        assert s.tracking_uri is None
        assert s.registry_uri is None
        assert s.experiment_name is None
        assert s.run_name is None
        assert s.artifact_location is None
        assert s.tags == {}
        assert s.log_system_metrics is False
        assert s.http_request_timeout is None

    def test_full_env_propagates(self) -> None:
        env = {
            ENV_TRACKING_URI: "http://mlflow:5000",
            ENV_REGISTRY_URI: "http://registry:5000",
            ENV_EXPERIMENT_NAME: "exp",
            ENV_RUN_NAME: "run-1",
            ENV_ARTIFACT_LOCATION: "s3://bucket",
            ENV_HTTP_TIMEOUT: "120",
            ENV_SYSTEM_METRICS: "true",
            ENV_FORGE_TAGS: "owner=ml,env=prod",
        }
        s = MlflowSettings.from_env(env=env)
        assert s.tracking_uri == "http://mlflow:5000"
        assert s.registry_uri == "http://registry:5000"
        assert s.experiment_name == "exp"
        assert s.run_name == "run-1"
        assert s.artifact_location == "s3://bucket"
        assert s.http_request_timeout == 120.0
        assert s.log_system_metrics is True
        assert s.tags == {"owner": "ml", "env": "prod"}

    def test_blank_strings_become_none(self) -> None:
        env = {ENV_TRACKING_URI: "", ENV_EXPERIMENT_NAME: ""}
        s = MlflowSettings.from_env(env=env)
        assert s.tracking_uri is None
        assert s.experiment_name is None


class TestMlflowSettingsMerge:
    def test_merge_overrides_apply(self) -> None:
        s = MlflowSettings(experiment_name="base", run_name="r")
        out = s.merge(experiment_name="other", run_name=None, tracking_uri="http://x")
        # None-valued overrides do NOT clobber existing values.
        assert out.experiment_name == "other"
        assert out.run_name == "r"
        assert out.tracking_uri == "http://x"

    def test_merge_tags_are_combined(self) -> None:
        s = MlflowSettings(tags={"env": "prod", "owner": "ml"})
        out = s.merge(tags={"owner": "new", "suite": "bringup"})
        assert out.tags == {"env": "prod", "owner": "new", "suite": "bringup"}
        # Source is untouched (immutability of returned instance).
        assert s.tags == {"env": "prod", "owner": "ml"}

    def test_merge_returns_new_instance(self) -> None:
        s = MlflowSettings(experiment_name="x")
        out = s.merge(experiment_name="y")
        assert out is not s
        assert s.experiment_name == "x"
        assert out.experiment_name == "y"


class TestMlflowSettingsApplyTo:
    def test_apply_sets_uris_when_present(self) -> None:
        mock_mlflow = MagicMock()
        MlflowSettings(
            tracking_uri="http://t", registry_uri="http://r"
        ).apply_to(mock_mlflow)
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://t")
        mock_mlflow.set_registry_uri.assert_called_once_with("http://r")

    def test_apply_skips_when_unset(self) -> None:
        mock_mlflow = MagicMock()
        MlflowSettings().apply_to(mock_mlflow)
        mock_mlflow.set_tracking_uri.assert_not_called()
        mock_mlflow.set_registry_uri.assert_not_called()

    def test_apply_writes_timeout_to_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_HTTP_TIMEOUT, raising=False)
        MlflowSettings(http_request_timeout=42.0).apply_to(MagicMock())
        import os

        assert os.environ.get(ENV_HTTP_TIMEOUT) == "42.0"


class TestMlflowSettingsDescribe:
    def test_describe_excludes_credentials(self) -> None:
        s = MlflowSettings(
            tracking_uri="http://x",
            experiment_name="exp",
            tags={"k": "v"},
        )
        out = s.describe()
        assert out["tracking_uri"] == "http://x"
        assert out["experiment_name"] == "exp"
        assert out["tags"] == {"k": "v"}
        # No credential leakage:
        assert "password" not in out
        assert "token" not in out
        # Tags should be a copy:
        out["tags"]["mutated"] = "yes"
        assert "mutated" not in s.tags
