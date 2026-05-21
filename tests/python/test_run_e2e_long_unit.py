"""Always-on unit tests for ``scripts/run_e2e_long.py``.

The e2e long-run pulls in LM Studio + the Rust eval binary, so its happy
path can't run in unit-test CI. These tests instead pin the *orchestrator
wiring* — env-takes-precedence, resume semantics, subprocess invocation —
by mocking ``run_collection`` / ``run_bc_training`` / ``run_eval_subprocess``
at the module boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ``run_e2e_long`` and ``_e2e_progress`` live in ``scripts/`` rather than an
# installed package; the root ``conftest.py`` puts that directory on ``sys.path``
# so these imports stay at the top of the module (no E402 noqa needed).
import run_e2e_long
from _e2e_progress import ProgressState
from _e2e_progress import load as load_progress
from _e2e_progress import save as save_progress

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PRESET_PATH: Path = REPO_ROOT / "configs" / "eval" / "e2e_long_preset.toml"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every FORGE_E2E_* env var so each test starts from defaults."""
    for key in run_e2e_long.FORGE_E2E_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# E2ELongConfig.from_toml_with_env_override
# ---------------------------------------------------------------------------


def test_config_loads_from_toml_defaults(tmp_path: Path, clean_env: None) -> None:
    cfg = run_e2e_long.E2ELongConfig.from_toml_with_env_override(
        PRESET_PATH,
        output_dir_override=tmp_path,
    )
    assert cfg.total_episodes == 1000
    assert cfg.experiment_name == "forge-e2e-long-run"
    assert cfg.teacher_preset == "gemma_e4b_teacher"
    assert cfg.mlflow_tracking_uri == "http://localhost:5000"
    assert cfg.output_root == tmp_path
    assert len(cfg.scenario_refs) == 11
    assert cfg.run_id  # fresh UUID, non-empty


def test_config_env_overrides_toml(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_E2E_EPISODES", "42")
    monkeypatch.setenv("FORGE_MLFLOW_TRACKING_URI", "https://mlflow.example.org")
    monkeypatch.setenv("FORGE_E2E_EXPERIMENT_NAME", "ci-experiment")
    monkeypatch.setenv("FORGE_E2E_TEACHER_PRESET", "qwen14b_teacher")
    cfg = run_e2e_long.E2ELongConfig.from_toml_with_env_override(
        PRESET_PATH,
        output_dir_override=tmp_path,
    )
    assert cfg.total_episodes == 42
    assert cfg.mlflow_tracking_uri == "https://mlflow.example.org"
    assert cfg.experiment_name == "ci-experiment"
    assert cfg.teacher_preset == "qwen14b_teacher"


def test_config_cli_flag_overrides_env(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_E2E_EPISODES", "100")
    cfg = run_e2e_long.E2ELongConfig.from_toml_with_env_override(
        PRESET_PATH,
        output_dir_override=tmp_path,
        episodes_override=7,
    )
    assert cfg.total_episodes == 7


def test_config_explicit_run_id_via_env(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_E2E_RUN_ID", "ci-pin-001")
    cfg = run_e2e_long.E2ELongConfig.from_toml_with_env_override(
        PRESET_PATH,
        output_dir_override=tmp_path,
    )
    assert cfg.run_id == "ci-pin-001"


def test_config_run_id_resumes_from_progress(tmp_path: Path, clean_env: None) -> None:
    cfg = run_e2e_long.E2ELongConfig.from_toml_with_env_override(
        PRESET_PATH,
        output_dir_override=tmp_path,
        progress_run_id="resumed-run-zzz",
    )
    assert cfg.run_id == "resumed-run-zzz"


def test_config_relative_hf_root_resolves_under_repo(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_HF_EXPORT_ROOT", "artifacts/relative-hf")
    cfg = run_e2e_long.E2ELongConfig.from_toml_with_env_override(
        PRESET_PATH,
        output_dir_override=tmp_path,
    )
    assert cfg.hf_export_root.is_absolute()
    assert cfg.hf_export_root == REPO_ROOT / "artifacts" / "relative-hf"


def test_config_absolute_hf_root_is_preserved(
    tmp_path: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    abs_hf = tmp_path / "abs-hf"
    monkeypatch.setenv("FORGE_HF_EXPORT_ROOT", str(abs_hf))
    cfg = run_e2e_long.E2ELongConfig.from_toml_with_env_override(
        PRESET_PATH,
        output_dir_override=tmp_path,
    )
    assert cfg.hf_export_root == abs_hf


# ---------------------------------------------------------------------------
# main() — orchestrator control flow with collector + subprocess mocked
# ---------------------------------------------------------------------------


def _fake_collection_result() -> Any:
    """A minimal stand-in for ScenarioCollectionResult that satisfies
    ``run_bc_training`` when that path is also mocked (it shouldn't be
    invoked directly by these tests)."""
    fake = MagicMock()
    fake.total_episodes.return_value = 4
    fake.total_steps.return_value = 32
    return fake


def test_main_skips_collection_when_already_complete(tmp_path: Path, clean_env: None) -> None:
    progress_path = tmp_path / ".e2e_progress.json"
    # Pre-seed a checkpoint whose episodes_completed == requested episodes.
    save_progress(
        progress_path,
        ProgressState(run_id="r1", episodes_completed=4, scenario_cursor=11, last_seed=42),
    )
    weights_path = tmp_path / "bc_weights.npz"
    weights_path.write_bytes(b"\x00")  # avoid the "weights missing" warning path

    with (
        patch.object(run_e2e_long, "run_collection") as mock_collect,
        patch.object(run_e2e_long, "run_bc_training") as mock_bc,
        patch.object(run_e2e_long, "run_eval_subprocess") as mock_eval,
    ):
        rc = run_e2e_long.main(
            [
                "--config",
                str(PRESET_PATH),
                "--output-dir",
                str(tmp_path),
                "--episodes",
                "4",
            ]
        )
    assert rc == 0
    mock_collect.assert_not_called()
    mock_bc.assert_not_called()
    mock_eval.assert_called_once()
    # The resumed run_id must flow into the eval subprocess via the config.
    invoked_cfg, invoked_weights = mock_eval.call_args.args
    assert invoked_cfg.run_id == "r1"
    assert invoked_weights == weights_path


def test_main_runs_collection_then_bc_then_eval_on_fresh_run(
    tmp_path: Path,
    clean_env: None,
) -> None:
    fake_weights = tmp_path / "bc_weights.npz"

    with (
        patch.object(
            run_e2e_long, "run_collection", return_value=_fake_collection_result()
        ) as mock_collect,
        patch.object(run_e2e_long, "run_bc_training", return_value=fake_weights) as mock_bc,
        patch.object(run_e2e_long, "run_eval_subprocess") as mock_eval,
    ):
        rc = run_e2e_long.main(
            [
                "--config",
                str(PRESET_PATH),
                "--output-dir",
                str(tmp_path),
                "--episodes",
                "4",
            ]
        )

    assert rc == 0
    mock_collect.assert_called_once()
    mock_bc.assert_called_once()
    mock_eval.assert_called_once()
    invoked_cfg, invoked_weights = mock_eval.call_args.args
    assert invoked_weights == fake_weights
    assert isinstance(invoked_cfg, run_e2e_long.E2ELongConfig)

    # Progress must be checkpointed after a successful collection.
    state = load_progress(tmp_path / ".e2e_progress.json")
    assert state is not None
    assert state.episodes_completed == 4
    assert state.scenario_cursor == 11  # all preset scenarios consumed


def test_main_collection_seed_offset_by_progress(tmp_path: Path, clean_env: None) -> None:
    # Partial-progress resume: collector should be called with the
    # remainder + a seed offset by the already-completed episodes.
    save_progress(
        tmp_path / ".e2e_progress.json",
        ProgressState(run_id="rA", episodes_completed=3, scenario_cursor=2, last_seed=42),
    )
    with (
        patch.object(
            run_e2e_long, "run_collection", return_value=_fake_collection_result()
        ) as mock_collect,
        patch.object(run_e2e_long, "run_bc_training", return_value=tmp_path / "bc.npz"),
        patch.object(run_e2e_long, "run_eval_subprocess"),
    ):
        run_e2e_long.main(
            [
                "--config",
                str(PRESET_PATH),
                "--output-dir",
                str(tmp_path),
                "--episodes",
                "10",
            ]
        )
    _, kwargs = mock_collect.call_args
    assert kwargs["remaining_episodes"] == 7  # 10 requested - 3 completed
    assert kwargs["start_seed"] == 42 + 3  # base_seed + already_done


def test_main_eval_subprocess_receives_resolved_env(
    tmp_path: Path,
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_MLFLOW_TRACKING_URI", "https://mlflow.example.org")
    fake_weights = tmp_path / "bc_weights.npz"

    with (
        patch.object(run_e2e_long, "run_collection", return_value=_fake_collection_result()),
        patch.object(run_e2e_long, "run_bc_training", return_value=fake_weights),
        patch.object(run_e2e_long, "run_eval_subprocess") as mock_eval,
    ):
        run_e2e_long.main(
            [
                "--config",
                str(PRESET_PATH),
                "--output-dir",
                str(tmp_path),
                "--episodes",
                "2",
            ]
        )
    cfg_passed: run_e2e_long.E2ELongConfig = mock_eval.call_args.args[0]
    assert cfg_passed.mlflow_tracking_uri == "https://mlflow.example.org"
    assert cfg_passed.output_root == tmp_path
    assert cfg_passed.total_episodes == 2


# ---------------------------------------------------------------------------
# run_eval_subprocess: env contract + missing-bin handling
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path, *, eval_cli_bin: Path) -> run_e2e_long.E2ELongConfig:
    return run_e2e_long.E2ELongConfig(
        scenario_refs=("patrol",),
        total_episodes=2,
        base_seed=1,
        bc_epochs=1,
        experiment_name="exp-x",
        run_id="rid-x",
        eval_cli_bin=eval_cli_bin,
        mlflow_tracking_uri="http://m:5000",
        mlflow_batch_size=100,
        hf_export_root=tmp_path / "hf",
        output_root=tmp_path,
        teacher_preset="gemma_e4b_teacher",
    )


def test_run_eval_subprocess_raises_when_bin_missing(tmp_path: Path, clean_env: None) -> None:
    cfg = _make_cfg(tmp_path, eval_cli_bin=tmp_path / "does" / "not" / "exist")
    with pytest.raises(FileNotFoundError, match="forge-eval-longrun"):
        run_e2e_long.run_eval_subprocess(cfg, tmp_path / "bc.npz")


def test_run_eval_subprocess_passes_required_env(tmp_path: Path, clean_env: None) -> None:
    fake_bin = tmp_path / "forge-eval-longrun"
    fake_bin.write_bytes(b"#!/bin/sh\nexit 0\n")
    cfg = _make_cfg(tmp_path, eval_cli_bin=fake_bin)

    with patch.object(run_e2e_long.subprocess, "run") as mock_run:
        run_e2e_long.run_eval_subprocess(cfg, tmp_path / "bc.npz")

    args, kwargs = mock_run.call_args
    assert args[0] == [str(fake_bin)]
    invoked_env = kwargs["env"]
    assert invoked_env["FORGE_MLFLOW_TRACKING_URI"] == "http://m:5000"
    assert invoked_env["FORGE_E2E_EXPERIMENT_NAME"] == "exp-x"
    assert invoked_env["FORGE_E2E_RUN_ID"] == "rid-x"
    assert invoked_env["FORGE_E2E_EPISODES"] == "2"
    assert invoked_env["FORGE_E2E_OUTPUT_DIR"] == str(tmp_path)
    assert invoked_env["FORGE_E2E_BC_WEIGHTS"] == str(tmp_path / "bc.npz")
    assert kwargs["check"] is True


# ---------------------------------------------------------------------------
# FORGE_E2E_ENV_VARS registry is exhaustive
# ---------------------------------------------------------------------------


def test_env_var_registry_covers_every_env_lookup() -> None:
    """Any FORGE_E2E_* / FORGE_MLFLOW_* / MLFLOW_TRACKING_* env key the
    orchestrator reads must also appear in FORGE_E2E_ENV_VARS, so the
    "drop the registry into a single env scrub call" pattern stays correct.

    Catches drift: adding a new ``os.environ.get("FORGE_E2E_NEW_KNOB")``
    without registering it would silently break CI env hygiene.

    Implemented as an AST walk (not a regex) so all three access shapes are
    covered uniformly:
      * ``os.environ.get("KEY", ...)``
      * ``os.environ["KEY"]``
      * ``"KEY" in os.environ``
    """
    import ast
    import inspect
    import re

    # inspect.getsourcefile handles edge cases where __file__ may be None
    # (e.g. namespace packages, frozen modules) better than __file__ directly.
    source_path = inspect.getsourcefile(run_e2e_long)
    assert source_path, "could not locate source file for run_e2e_long"
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))

    key_pattern = re.compile(r"^(FORGE_[A-Z0-9_]+|MLFLOW_[A-Z0-9_]+)$")
    found: set[str] = set()

    def _is_os_environ(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        )

    def _record_if_key(node: ast.AST) -> None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and key_pattern.match(node.value)
        ):
            found.add(node.value)

    for node in ast.walk(tree):
        # os.environ.get("KEY", ...) / os.environ.get(key=..., ...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_os_environ(node.func.value)
            and node.args
        ):
            _record_if_key(node.args[0])
        # os.environ["KEY"]
        elif isinstance(node, ast.Subscript) and _is_os_environ(node.value):
            _record_if_key(node.slice)
        # "KEY" in os.environ
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and node.comparators
            and _is_os_environ(node.comparators[0])
        ):
            _record_if_key(node.left)

    missing = found - set(run_e2e_long.FORGE_E2E_ENV_VARS)
    assert not missing, f"orchestrator reads env vars not in FORGE_E2E_ENV_VARS: {missing}"
