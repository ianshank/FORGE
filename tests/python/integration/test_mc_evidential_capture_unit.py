"""Unit tests for ``scripts/mc_evidential_capture.sh``.

Dry-run only — no Docker. Mirrors ``test_mc_self_play_unit.py``.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from ._helpers import lf_normalized_script


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def script_path(repo_root: Path) -> Path:
    return repo_root / "scripts" / "mc_evidential_capture.sh"


def _run_dry(script_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    with lf_normalized_script(script_path) as posix_script:
        return subprocess.run(
            [bash, posix_script, "--dry-run", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )


def test_script_is_executable_and_parses(script_path: Path) -> None:
    assert script_path.is_file()
    import sys
    mode = script_path.stat().st_mode
    if sys.platform != "win32":
        assert mode & stat.S_IXUSR
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    with lf_normalized_script(script_path) as posix_script:
        rc = subprocess.run([bash, "-n", posix_script], check=False, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


def test_dry_run_prints_random_then_trained_capture(script_path: Path) -> None:
    result = _run_dry(script_path)
    combined = result.stdout + result.stderr
    assert "--baseline-only" in combined
    assert "capture-baseline" in combined
    assert "--variant random" in combined or "random" in combined
    assert "--variant trained" in combined or "trained" in combined
    assert "mc_plot_baseline.py" in combined
    assert "DRY-RUN:" in combined


def test_dry_run_skip_trained_omits_plotter(script_path: Path) -> None:
    result = _run_dry(script_path, "--skip-trained")
    combined = result.stdout + result.stderr
    assert "capture-baseline" in combined
    assert "mc_plot_baseline.py" not in combined


def test_episodes_below_floor_exits_2(script_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    with lf_normalized_script(script_path) as posix_script:
        rc = subprocess.run(
            [bash, posix_script, "--dry-run", "--episodes", "1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert rc.returncode == 2
    assert "evidential floor" in (rc.stdout + rc.stderr)


def test_unknown_flag_exits_2(script_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    with lf_normalized_script(script_path) as posix_script:
        rc = subprocess.run(
            [bash, posix_script, "--not-a-real-flag"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert rc.returncode == 2
    assert "unknown flag" in (rc.stdout + rc.stderr)


def test_live_without_docker_exits_3(script_path: Path, tmp_path: Path) -> None:
    bash = shutil.which("bash")
    date = shutil.which("date")
    if bash is None or date is None:
        pytest.skip("bash/date not on PATH")
    isolated = tmp_path / "bin"
    isolated.mkdir()
    (isolated / "date").symlink_to(date)
    env = os.environ.copy()
    env["PATH"] = str(isolated)
    with lf_normalized_script(script_path) as posix_script:
        rc = subprocess.run(
            [bash, posix_script, "--episodes", "3"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    assert rc.returncode == 3
    assert "docker is not on PATH" in (rc.stdout + rc.stderr)
