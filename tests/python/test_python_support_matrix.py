"""Tests for scripts/check_python_support_matrix.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "check_python_support_matrix.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_python_support_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repo_matrix_is_honest() -> None:
    mod = _load()
    requires = mod.parse_requires_floor(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    ci = mod.parse_ci_minors(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    assert not mod.evaluate(requires, ci)


def test_detects_fail_open_floor() -> None:
    mod = _load()
    assert mod.evaluate((3, 9), {(3, 11)})


def test_script_exits_zero_on_repo() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
