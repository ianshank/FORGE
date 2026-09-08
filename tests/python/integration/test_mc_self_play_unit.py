"""Unit tests for `scripts/mc_self_play.sh` (T6 orchestrator).

Drives the script in `--dry-run` mode so no Docker / bash subprocess
beyond the script itself is needed. NOT gated behind the
`minecraft_e2e` marker — runs on every PR CI invocation so a typo
in the orchestrator surfaces immediately.

Each test asserts that the expected argv sequence is printed (the
script's `run_or_echo` / `capture_or_echo` helpers emit
`DRY-RUN: <argv>` lines to stderr so callers capturing stdout don't
swallow them). The synthetic schema_id placeholder
`deadbeef` + 56 zeros is hard-coded into the dry-run path so
downstream `bootstrap` / `up` calls have a well-formed value to
interpolate.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ._helpers import lf_normalized_script


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """The FORGE repo root, anchored to this test file."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def script_path(repo_root: Path) -> Path:
    """The orchestrator script's absolute path."""
    return repo_root / "scripts" / "mc_self_play.sh"


def _run_dry(script_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke `mc_self_play.sh` in dry-run mode and capture both
    stdout + stderr. bash is required (skip on hosts without it)."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH; mc_self_play.sh unit tests need POSIX shell")
    with lf_normalized_script(script_path) as posix_script:
        return subprocess.run(
            [bash, posix_script, "--dry-run", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )


def test_script_exists_and_is_executable(script_path: Path) -> None:
    assert script_path.exists(), f"orchestrator script missing: {script_path}"
    # On Windows the executable bit doesn't survive git checkout; we
    # rely on `bash script.sh` to run it, so just verify it parses
    # as bash (no syntax errors).
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    with lf_normalized_script(script_path) as posix_script:
        rc = subprocess.run([bash, "-n", posix_script], check=False, capture_output=True, text=True)
    assert rc.returncode == 0, f"bash -n failed: {rc.stderr}"


def test_dry_run_emits_compute_schema_id_call(script_path: Path) -> None:
    """The pipeline MUST start with `compute-schema-id --quiet` so
    the captured schema_id is hash-only (no log noise)."""
    result = _run_dry(script_path)
    combined = result.stdout + result.stderr
    assert "compute-schema-id" in combined
    assert "--quiet" in combined


def test_dry_run_emits_bootstrap_call_when_manifest_missing(script_path: Path) -> None:
    """In dry-run mode the script assumes the manifest is missing
    (it can't run `docker run` to actually check). The bootstrap
    one-shot MUST be invoked with the captured schema_id + default
    obs-dim/action-dim."""
    result = _run_dry(script_path)
    combined = result.stdout + result.stderr
    # The bootstrap sub-command lands in argv right after `--rm trainer-bootstrap`.
    assert "trainer-bootstrap" in combined
    assert "bootstrap" in combined
    # v0.5 Phase 1: `--obs-dim 920` and `--action-dim 12` are the
    # defaults pinned in the script's env-var ladder (OBS_DIM /
    # ACTION_DIM).  T2 flipped OBS_DIM from 31 → 920 to match the
    # block-grid observation contract; operators can re-set to 31
    # for the legacy `include_block_grid = false` path.
    assert "--obs-dim 920" in combined
    assert "--action-dim 12" in combined


def test_dry_run_forwards_env_file_to_mc_run(script_path: Path) -> None:
    """`mc_self_play.sh` MUST propagate `--env-file` to the final
    `mc_run.sh` invocation so `MC_EULA=TRUE` reaches every compose
    service (peer-review IMPORTANT fix)."""
    result = _run_dry(script_path)
    combined = result.stdout + result.stderr
    assert "--env-file" in combined
    # Must mention the resolved env file path (default or example).
    assert "compose.minecraft.env" in combined


def test_dry_run_exports_forge_mc_schema_id(script_path: Path) -> None:
    """The final compose-up invocation MUST carry
    `FORGE_MC_SCHEMA_ID=<hash>` so the runner's env-var ladder
    (T3) picks it up over the static runner.toml."""
    result = _run_dry(script_path)
    combined = result.stdout + result.stderr
    assert "FORGE_MC_SCHEMA_ID=" in combined
    # The synthetic dry-run placeholder must appear (proves the
    # capture+propagation chain works end-to-end).
    assert "deadbeef" in combined


def test_dry_run_exports_trained_identity(script_path: Path) -> None:
    """Default self-play MUST flip trained identity via the env ladder
    (`RUNNER_RANDOM_ACTIONS=false`) and request the bundled image
    (`RUNNER_FEATURES=mc-live-bundled`), plus `--build` so a stale
    `mc-live` image is not reused.
    """
    result = _run_dry(script_path)
    combined = result.stdout + result.stderr
    assert "RUNNER_RANDOM_ACTIONS=false" in combined
    assert "RUNNER_FEATURES=mc-live-bundled" in combined
    assert "--build" in combined
    assert "--profile" in combined
    assert "self-play" in combined


def test_dry_run_baseline_only_skips_trainer_and_does_not_force_trained(script_path: Path) -> None:
    """`--baseline-only` omits the self-play profile and must not
    export trained identity (shipped runner.toml stays random).
    """
    result = _run_dry(script_path, "--baseline-only")
    combined = result.stdout + result.stderr
    assert "compute-schema-id" not in combined
    assert "RUNNER_RANDOM_ACTIONS=false" not in combined
    assert "mc-live-bundled" not in combined
    assert "--profile self-play" not in combined
    assert "baseline-only" in combined


def test_dry_run_gpu_flag_includes_overlay(script_path: Path) -> None:
    """With `--gpu`, the compose-up argv must include
    `-f docker/compose.minecraft.gpu.yml` AFTER the base file."""
    result = _run_dry(script_path, "--gpu")
    combined = result.stdout + result.stderr
    assert "compose.minecraft.gpu.yml" in combined
    assert "--gpu" in combined


def test_dry_run_without_gpu_omits_overlay(script_path: Path) -> None:
    """Without `--gpu`, the GPU overlay file MUST NOT appear in argv."""
    result = _run_dry(script_path)
    combined = result.stdout + result.stderr
    assert "compose.minecraft.gpu.yml" not in combined


def test_dry_run_detach_passes_through_to_mc_run(script_path: Path) -> None:
    result = _run_dry(script_path, "--detach")
    combined = result.stdout + result.stderr
    assert "--detach" in combined


def test_dry_run_down_only_short_circuits(script_path: Path) -> None:
    """`--down` skips the schema-id / bootstrap / up steps and goes
    straight to a teardown invocation."""
    result = _run_dry(script_path, "--down")
    combined = result.stdout + result.stderr
    # Teardown step must be present.
    assert "--down" in combined
    # Schema-id computation MUST NOT run on a teardown-only path.
    assert "compute-schema-id" not in combined
    assert "bootstrap " not in combined
