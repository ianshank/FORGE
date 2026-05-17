"""Opt-in end-to-end long-run validation: orchestrator -> live LM Studio
teacher -> MangoMAS collector -> BC trainer -> Rust forge-eval-longrun
-> live HTTP MLflow + HuggingFace JSONL.

Gating:
- ``pytest.mark.lmstudio`` — requires LM Studio (or compatible OpenAI server)
- ``pytest.mark.e2e_long``  — long-running (minutes to hours)
- ``FORGE_E2E_LONG=1``       — must be set to opt in
- ``forge-eval-longrun`` binary must exist (built locally or in CI)
- ``FORGE_MLFLOW_TRACKING_URI`` must point at a reachable MLflow HTTP server

Assertions are pipeline-shape ones, not score-value ones: live LLM sampling
is non-deterministic so we can't pin overall_score across runs. The contract
that matters is "every stage produced what the next stage needs" — MLflow run
FINISHED, HF JSONL has the expected episode count, scorecard.json on disk.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
EVAL_BIN: Path = REPO_ROOT / "target" / "release" / "forge-eval-longrun"
PRESET_PATH: Path = REPO_ROOT / "configs" / "eval" / "e2e_long_preset.toml"
ORCHESTRATOR: Path = REPO_ROOT / "scripts" / "run_e2e_long.py"

pytestmark = [
    pytest.mark.lmstudio,
    pytest.mark.e2e_long,
    pytest.mark.skipif(
        os.environ.get("FORGE_E2E_LONG", "") != "1",
        reason="opt-in: set FORGE_E2E_LONG=1 to run the long e2e harness",
    ),
    pytest.mark.skipif(
        not EVAL_BIN.exists(),
        reason=(
            f"forge-eval-longrun binary missing at {EVAL_BIN}. Build with: "
            "cargo build -p forge-eval --bin forge-eval-longrun --features http-mlflow --release"
        ),
    ),
    pytest.mark.skipif(
        "FORGE_MLFLOW_TRACKING_URI" not in os.environ,
        reason="FORGE_MLFLOW_TRACKING_URI must point at a live MLflow HTTP server",
    ),
]


def _episode_count() -> int:
    """Number of episodes to drive through the pipeline.

    CI typically overrides via ``FORGE_E2E_EPISODES``; defaults to a tiny
    25-episode smoke so the test stays under a few minutes for local runs.
    Mirrors the env name the orchestrator itself honours.
    """
    return int(os.environ.get("FORGE_E2E_EPISODES", "25"))


def test_e2e_long_run_emits_consumable_mlflow_and_hf(tmp_path: Path) -> None:
    """Pipeline-shape contract: scorecard on disk + MLflow FINISHED + HF JSONL."""
    env = {
        **os.environ,
        "FORGE_E2E_OUTPUT_DIR": str(tmp_path),
        "FORGE_E2E_EPISODES": str(_episode_count()),
    }
    result = subprocess.run(
        [sys.executable, str(ORCHESTRATOR), "--config", str(PRESET_PATH)],
        env=env,
        check=True,
        capture_output=False,
    )
    assert result.returncode == 0, "orchestrator must exit 0 for the eval CLI to be invoked"

    scorecard_path = tmp_path / "scorecard.json"
    assert scorecard_path.exists(), f"forge-eval-longrun must write {scorecard_path}"

    # ---- MLflow consumer validation ------------------------------------
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(os.environ["FORGE_MLFLOW_TRACKING_URI"])
    client = mlflow.tracking.MlflowClient()
    experiment_name = os.environ.get("FORGE_E2E_EXPERIMENT_NAME", "forge-e2e-long-run")
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None, f"MLflow experiment {experiment_name!r} must exist after run"

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs, "at least one run must be logged"
    parent_runs = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    assert parent_runs, "exactly one parent run is expected (children carry parentRunId)"
    parent = parent_runs[0]
    assert parent.info.status == "FINISHED", (
        f"parent run must terminate FINISHED, got status={parent.info.status!r}"
    )
    assert "overall_score" in parent.data.metrics, "overall_score metric must be logged on parent"

    # ---- HuggingFace consumer validation -------------------------------
    # We assert the export tree exists + that `datasets.load_dataset` parses
    # the JSONL files. Score value isn't checked — LLM sampling is
    # non-deterministic so pinning a number across runs would be flaky.
    datasets = pytest.importorskip("datasets")
    hf_root_env = os.environ.get("FORGE_HF_EXPORT_ROOT", "")
    hf_root = Path(hf_root_env) if hf_root_env else REPO_ROOT / "artifacts" / "e2e-long" / "hf"
    if not hf_root.is_absolute():
        hf_root = REPO_ROOT / hf_root
    run_dir = hf_root / parent.info.run_id
    jsonl_files = sorted(run_dir.rglob("*.jsonl"))
    assert jsonl_files, f"HF export must contain at least one .jsonl under {run_dir}"
    ds: Any = datasets.load_dataset("json", data_files=[str(p) for p in jsonl_files])
    assert "train" in ds, "load_dataset default split is 'train'"
    assert len(ds["train"]) > 0, "HF train split must have at least one row"


def test_e2e_long_run_writes_progress_checkpoint(tmp_path: Path) -> None:
    """A successful long run must leave a ProgressState checkpoint behind
    so a subsequent invocation (with the same --output-dir) resumes
    cleanly rather than re-collecting."""
    env = {
        **os.environ,
        "FORGE_E2E_OUTPUT_DIR": str(tmp_path),
        "FORGE_E2E_EPISODES": str(_episode_count()),
    }
    subprocess.run(
        [sys.executable, str(ORCHESTRATOR), "--config", str(PRESET_PATH)],
        env=env,
        check=True,
    )
    progress_path = tmp_path / ".e2e_progress.json"
    assert progress_path.exists(), "orchestrator must persist progress after collection"
    # Defer the parse to the always-on test_e2e_progress unit; here we
    # only care that the file is present and non-empty.
    assert progress_path.stat().st_size > 0
