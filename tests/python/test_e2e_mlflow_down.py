"""Negative-path harness: the e2e long run must survive an unreachable
MLflow tracking server.

Phase B exporters are best-effort by contract — a tracking-server outage
must NOT fail an otherwise-successful eval. This test points the orchestrator
at a guaranteed-unreachable URI (``http://127.0.0.1:1``, port 1 is reserved)
and asserts the subprocess still exits 0 plus emits *some* mlflow-related log
line so we can see in CI logs which retry path the exporter took.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
EVAL_BIN: Path = REPO_ROOT / "target" / "release" / "forge-eval-longrun"
PRESET_PATH: Path = REPO_ROOT / "configs" / "eval" / "e2e_long_preset.toml"
ORCHESTRATOR: Path = REPO_ROOT / "scripts" / "run_e2e_long.py"

pytestmark = [
    pytest.mark.e2e_long,
    pytest.mark.skipif(
        not EVAL_BIN.exists(),
        reason=(
            f"forge-eval-longrun binary missing at {EVAL_BIN}. Build with: "
            "cargo build -p forge-eval --bin forge-eval-longrun --features http-mlflow --release"
        ),
    ),
    pytest.mark.skipif(
        os.environ.get("FORGE_E2E_LONG", "") != "1",
        reason="opt-in: set FORGE_E2E_LONG=1 to run the mlflow-down e2e harness",
    ),
]


def test_e2e_run_completes_when_mlflow_unreachable(tmp_path: Path) -> None:
    # Override to an unreachable URI; port 1 is reserved and never bound.
    # Keep episode count tiny so the test doesn't pay for a real long run.
    env = {
        **os.environ,
        "FORGE_MLFLOW_TRACKING_URI": "http://127.0.0.1:1",
        "FORGE_E2E_OUTPUT_DIR": str(tmp_path),
        "FORGE_E2E_EPISODES": "4",
    }
    result = subprocess.run(
        [sys.executable, str(ORCHESTRATOR), "--config", str(PRESET_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "best-effort exporter contract: an unreachable MLflow URI must not fail the run. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    combined = (result.stdout + result.stderr).lower()
    # We don't assert a specific retry message — that would couple the test
    # to a particular log string. We just require *some* mlflow-related
    # line so CI logs surface which path the exporter took.
    assert "mlflow" in combined, (
        "expected at least one mlflow-related log line so the failure mode is observable; "
        f"got stdout={result.stdout!r} stderr={result.stderr!r}"
    )
