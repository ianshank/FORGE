"""Tests for ``scripts/check_local_ci_parity.py``.

The check exists to stop ``make verify`` passing while CI goes red on a job
with no local counterpart. A parity checker that cannot itself fail would
reintroduce exactly that class of problem, so every one of its four failure
directions is exercised here against synthetic inputs, plus the parsers
against the real repository files and an end-to-end subprocess run.

No test writes to a tracked file; the synthetic cases are plain strings and
the on-disk cases are read-only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_local_ci_parity as parity

REPO_ROOT = Path(__file__).resolve().parents[2]

# A workflow fragment shaped like ci.yml: two-space-indented job ids under a
# top-level `jobs:` key, with nested keys indented deeper so they must not be
# mistaken for jobs.
SYNTHETIC_WORKFLOW = """\
name: CI
on: [push]
env:
  SOME_PIN: "1.0.0"
jobs:
  alpha:
    name: Alpha
    steps:
      - run: echo alpha
  beta:
    name: Beta
    runs-on: ubuntu-latest
    steps:
      - run: echo beta
"""

SYNTHETIC_MAKEFILE = """\
.PHONY: alpha-target beta-target verify

VAR := not-a-target

alpha-target: ## does alpha
\techo alpha

beta-target: ## does beta
\techo beta

verify: alpha-target ## the gate
\t@echo ok
"""


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parses_job_ids_and_ignores_nested_keys() -> None:
    """Only two-space-indented keys under `jobs:` are jobs."""
    assert parity.parse_ci_jobs(SYNTHETIC_WORKFLOW) == {"alpha", "beta"}


def test_parses_make_targets_and_ignores_variable_assignments() -> None:
    """`VAR :=` is an assignment, not a target, and must not be collected."""
    targets = parity.parse_make_targets(SYNTHETIC_MAKEFILE)
    assert targets == {"alpha-target", "beta-target", "verify"}
    assert "VAR" not in targets


def test_parses_verify_prerequisites() -> None:
    """`verify:`'s dependency list is what decides the VERIFY_REQUIRED check."""
    assert parity.parse_verify_prerequisites(SYNTHETIC_MAKEFILE) == {"alpha-target"}


def test_rejects_a_workflow_with_no_jobs_mapping() -> None:
    """A workflow this cannot parse must raise, never return an empty set.

    Returning empty would make every downstream comparison vacuously pass.
    """
    with pytest.raises(ValueError, match="no top-level `jobs:` mapping"):
        parity.parse_ci_jobs("name: CI\non: [push]\n")


# ---------------------------------------------------------------------------
# The four failure directions
# ---------------------------------------------------------------------------


def test_fails_on_a_ci_job_with_no_target_and_no_exception() -> None:
    """The primary case: a new CI job nobody wired a local path for."""
    problems = parity.evaluate(
        ci_jobs={"alpha", "brand-new-job"},
        make_targets={"alpha-target"},
        verify_deps={"alpha-target"},
    )
    joined = "\n".join(problems)
    assert "brand-new-job" in joined
    assert "no local counterpart" in joined


def test_fails_when_a_mapping_names_a_deleted_make_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renaming a target must not silently orphan the mapping that claims it."""
    monkeypatch.setattr(parity, "JOB_TO_MAKE_TARGET", {"alpha": "target-that-was-renamed"})
    monkeypatch.setattr(parity, "JOB_EXCEPTIONS", {})
    monkeypatch.setattr(parity, "VERIFY_REQUIRED_JOBS", frozenset())
    problems = parity.evaluate(
        ci_jobs={"alpha"}, make_targets={"alpha-target"}, verify_deps=set()
    )
    joined = "\n".join(problems)
    assert "target-that-was-renamed" in joined
    assert "no longer exist" in joined


def test_fails_on_an_entry_for_a_deleted_ci_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """The list must not accumulate entries for jobs that were removed."""
    monkeypatch.setattr(parity, "JOB_TO_MAKE_TARGET", {"alpha": "alpha-target"})
    monkeypatch.setattr(parity, "JOB_EXCEPTIONS", {"job-deleted-last-year": "reason"})
    monkeypatch.setattr(parity, "VERIFY_REQUIRED_JOBS", frozenset())
    problems = parity.evaluate(
        ci_jobs={"alpha"}, make_targets={"alpha-target"}, verify_deps=set()
    )
    joined = "\n".join(problems)
    assert "job-deleted-last-year" in joined
    assert "delete them" in joined


def test_fails_when_a_required_job_drops_out_of_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    """A target existing is not enough if `make verify` never calls it."""
    monkeypatch.setattr(parity, "JOB_TO_MAKE_TARGET", {"alpha": "alpha-target"})
    monkeypatch.setattr(parity, "JOB_EXCEPTIONS", {})
    monkeypatch.setattr(parity, "VERIFY_REQUIRED_JOBS", frozenset({"alpha"}))
    problems = parity.evaluate(
        ci_jobs={"alpha"},
        make_targets={"alpha-target"},
        verify_deps=set(),  # verify does not depend on it
    )
    joined = "\n".join(problems)
    assert "alpha-target" in joined
    assert "can pass while CI goes red" in joined


def test_passes_when_everything_lines_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path must be reachable, or the failure tests prove nothing."""
    monkeypatch.setattr(parity, "JOB_TO_MAKE_TARGET", {"alpha": "alpha-target"})
    monkeypatch.setattr(parity, "JOB_EXCEPTIONS", {"beta": "needs credentials"})
    monkeypatch.setattr(parity, "VERIFY_REQUIRED_JOBS", frozenset({"alpha"}))
    assert (
        parity.evaluate(
            ci_jobs={"alpha", "beta"},
            make_targets={"alpha-target"},
            verify_deps={"alpha-target"},
        )
        == []
    )


# ---------------------------------------------------------------------------
# Against the real repository
# ---------------------------------------------------------------------------


def test_the_real_repository_is_in_parity() -> None:
    """The committed mapping must match the committed ci.yml and Makefile.

    This is the assertion that keeps the mapping a live description of the
    repo rather than documentation that drifted.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_local_ci_parity.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f"local/CI parity drift:\n{result.stdout}\n{result.stderr}"
    )


def test_exits_nonzero_and_reports_when_a_workflow_gains_an_unmapped_job(
    tmp_path: Path,
) -> None:
    """End-to-end: a real subprocess run over a workflow with an extra job."""
    workflow = tmp_path / "ci.yml"
    real = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow.write_text(real + "\n  a-brand-new-unmapped-job:\n    name: New\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_local_ci_parity.py"),
            "--workflow",
            str(workflow),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == parity.EXIT_DRIFT
    assert "a-brand-new-unmapped-job" in result.stderr


def test_exits_with_input_error_on_a_missing_file(tmp_path: Path) -> None:
    """A missing input is distinguishable from drift, so CI can tell them apart."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_local_ci_parity.py"),
            "--workflow",
            str(tmp_path / "does-not-exist.yml"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == parity.EXIT_INPUT_ERROR


def test_every_exception_carries_a_nonempty_reason() -> None:
    """An exception without a reason is indistinguishable from an oversight."""
    for job, reason in parity.JOB_EXCEPTIONS.items():
        assert reason.strip(), f"JOB_EXCEPTIONS[{job!r}] has no stated reason"
        assert len(reason.strip()) > 20, (
            f"JOB_EXCEPTIONS[{job!r}] reason is too terse to be useful: {reason!r}"
        )
