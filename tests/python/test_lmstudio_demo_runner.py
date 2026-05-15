"""Tests for scripts/run_lmstudio_demo.py — local LM Studio demo entry point.

The helper is intentionally narrow: it pings the configured LM Studio
endpoint with the preset's model and reports latency. Real collection
goes through scripts/train.py.
"""
from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from conftest import REPO_ROOT

SCRIPTS_DIR = REPO_ROOT / "scripts"
GEMMA_PRESET = REPO_ROOT / "configs" / "cognitive" / "gemma_e4b_teacher.toml"

# Make scripts/ importable as a flat module set.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture()
def fake_completion_response() -> MagicMock:
    resp = MagicMock()
    resp.text = '{"ok": true}'
    resp.input_tokens = 5
    resp.output_tokens = 3
    resp.latency_ms = 42.0
    return resp


def test_check_pings_provider_and_logs_latency(
    fake_completion_response: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    import run_lmstudio_demo

    fake_provider = MagicMock()
    fake_provider.complete.return_value = fake_completion_response
    with (
        patch.object(run_lmstudio_demo, "_build_provider", return_value=fake_provider),
        caplog.at_level(logging.INFO, logger="run_lmstudio_demo"),
    ):
        exit_code = run_lmstudio_demo.main(["--check", "--config", str(GEMMA_PRESET)])
    assert exit_code == 0
    assert fake_provider.complete.call_count == 1
    assert any("latency_ms=42.0" in rec.message for rec in caplog.records)


def test_check_returns_nonzero_on_provider_error() -> None:
    import run_lmstudio_demo

    fake_provider = MagicMock()
    fake_provider.complete.side_effect = RuntimeError("connection refused")
    with patch.object(run_lmstudio_demo, "_build_provider", return_value=fake_provider):
        exit_code = run_lmstudio_demo.main(["--check", "--config", str(GEMMA_PRESET)])
    assert exit_code == 1


def test_no_action_flag_is_a_parser_error() -> None:
    """Running without --check should be a usage error, not a silent no-op."""
    import run_lmstudio_demo

    with pytest.raises(SystemExit) as exc_info:
        run_lmstudio_demo.main(["--config", str(GEMMA_PRESET)])
    # argparse exits with code 2 on usage error.
    assert exc_info.value.code == 2


def test_default_config_path_loads_into_real_teacher_config() -> None:
    """Regression: the default --config must be a MangoMASBridgeConfig-shape
    TOML with a [teacher] section, NOT the cognitive-provider default.toml.

    Otherwise the demo helper silently loads an empty TeacherConfig and
    pings the mock provider instead of LM Studio. Caught in Gemini review.
    """
    import run_lmstudio_demo

    from forge.mangomas.config import MangoMASBridgeConfig

    cfg = MangoMASBridgeConfig.from_toml(run_lmstudio_demo.DEFAULT_CONFIG_PATH).teacher
    assert cfg.enabled is True, (
        f"default config {run_lmstudio_demo.DEFAULT_CONFIG_PATH} must have "
        "[teacher].enabled=true so --check actually pings LM Studio"
    )
    assert cfg.provider == "lmstudio", cfg.provider
    assert cfg.model, "[teacher].model must be set in the default preset"


def test_log_level_debug_emits_debug_records(
    fake_completion_response: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    import run_lmstudio_demo

    fake_provider = MagicMock()
    fake_provider.complete.return_value = fake_completion_response
    with (
        patch.object(run_lmstudio_demo, "_build_provider", return_value=fake_provider),
        caplog.at_level(logging.DEBUG, logger="run_lmstudio_demo"),
    ):
        exit_code = run_lmstudio_demo.main(
            ["--check", "--config", str(GEMMA_PRESET), "--log-level", "DEBUG"]
        )
    assert exit_code == 0
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, "expected at least one DEBUG record at --log-level DEBUG"
