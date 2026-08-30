"""Unit tests for the `FORGE_MC_REQUIRE_E2E` guard in :mod:`conftest`.

The guard exists so a scheduled job cannot report success while
executing nothing: `pytest.skip` exits ``0``, so a suite that skips
every test looks identical to one that passed. These tests pin the
guard's own logic, which the docker-gated E2E tests cannot reach.

They carry no `minecraft_e2e` marker and need no Docker, so they run on
every PR alongside the rest of the fast suite.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path
from typing import Any

import pytest

from . import conftest as guard
from ._helpers import REQUIRE_E2E_ENV_VAR

#: The E2E module whose test ordering is guarded below.
_E2E_MODULE = "test_minecraft_e2e.py"

#: Fixture name that marks a test as depending on the shared stack.
_STACK_FIXTURE = "compose_up_minecraft_stack"

#: Flag that makes a test destructive — it stops the shared stack.
_TEARDOWN_FLAG = "--down"


class _Report:
    """Minimal stand-in for ``pytest.TestReport``.

    Only the four attributes the hook reads are modelled; constructing a
    real ``TestReport`` would couple these tests to pytest internals for
    no added signal.
    """

    def __init__(
        self,
        *,
        when: str = "call",
        skipped: bool = False,
        keywords: dict[str, Any] | None = None,
        nodeid: str = "tests/python/integration/test_minecraft_e2e.py::test_x",
    ) -> None:
        self.when = when
        self.skipped = skipped
        self.keywords = keywords if keywords is not None else {guard._E2E_MARKER: 1}
        self.nodeid = nodeid


@pytest.fixture(autouse=True)
def _isolated_guard_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a private `_e2e_executed`, so ordering cannot
    leak recorded node ids between them."""
    monkeypatch.setattr(guard, "_e2e_executed", set())


def _session(exitstatus: int = 0) -> Any:
    return types.SimpleNamespace(exitstatus=exitstatus)


def test_only_marked_tests_count_towards_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The marker filter is load-bearing.

    Without it, ANY test reaching its call phase — including these unit
    tests — would satisfy the guard, silently reopening the green-hole
    the guard exists to close.
    """
    monkeypatch.setenv(REQUIRE_E2E_ENV_VAR, "1")

    guard.pytest_runtest_logreport(_Report(keywords={"some_other_test": 1}))
    assert guard._e2e_executed == set(), "an unmarked test must not count as E2E execution"

    session = _session()
    guard.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_a_marked_test_that_ran_satisfies_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REQUIRE_E2E_ENV_VAR, "1")

    guard.pytest_runtest_logreport(_Report())
    assert len(guard._e2e_executed) == 1

    session = _session()
    guard.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0, "a session that really ran an E2E test must not be failed"


def test_skipped_and_non_call_phases_do_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A skip is exactly the case the guard must catch, and setup/
    teardown reports for a passing test must not double-count."""
    monkeypatch.setenv(REQUIRE_E2E_ENV_VAR, "1")

    guard.pytest_runtest_logreport(_Report(skipped=True))
    guard.pytest_runtest_logreport(_Report(when="setup"))
    guard.pytest_runtest_logreport(_Report(when="teardown"))
    assert guard._e2e_executed == set()

    session = _session()
    guard.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_guard_is_inert_when_the_flag_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset must preserve today's behaviour exactly: skip and exit 0."""
    monkeypatch.delenv(REQUIRE_E2E_ENV_VAR, raising=False)

    session = _session()
    guard.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0


def test_a_more_specific_failure_code_is_not_clobbered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `exitstatus == 0` guard matters: a collection error (2) or a
    usage error (4) is more informative than TESTS_FAILED, and pytest
    has already failed the run, so the hook must leave it alone."""
    monkeypatch.setenv(REQUIRE_E2E_ENV_VAR, "1")

    for original in (pytest.ExitCode.USAGE_ERROR, pytest.ExitCode.INTERNAL_ERROR):
        session = _session(int(original))
        guard.pytest_sessionfinish(session, int(original))
        assert session.exitstatus == int(original), f"{original!r} must survive the guard"


def test_skip_or_fail_raises_skip_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REQUIRE_E2E_ENV_VAR, raising=False)
    with pytest.raises(BaseException) as excinfo:
        guard._skip_or_fail("docker missing")
    assert excinfo.typename == "Skipped"


def test_skip_or_fail_raises_failure_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: the same missing prerequisite must be fatal for
    a job that exists to run the suite."""
    monkeypatch.setenv(REQUIRE_E2E_ENV_VAR, "1")
    with pytest.raises(BaseException) as excinfo:
        guard._skip_or_fail("docker missing")
    assert excinfo.typename == "Failed"
    assert REQUIRE_E2E_ENV_VAR in str(excinfo.value)


def test_stack_destroying_tests_run_last_in_the_e2e_module() -> None:
    """A test that tears the stack down must be the last one that uses it.

    The E2E stack fixture is session-scoped and pytest runs a module's
    tests in source order, so a test that runs ``mc_run.sh --down``
    leaves every later stack-dependent test with a dead stack. The
    symptom is a poll timeout against an endpoint that no longer
    answers — with nothing pointing at the real cause — and it is
    deterministic, not flaky, which makes it worse: the suite would
    never pass.

    Detected structurally rather than by name: any test taking the
    stack fixture whose body mentions the teardown flag is destructive.
    Parsed with :mod:`ast` so this needs neither Docker nor the
    module's own imports.
    """
    module_path = Path(__file__).with_name(_E2E_MODULE)
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))

    stack_dependent: list[str] = []
    destructive: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        params = {arg.arg for arg in node.args.args}
        if _STACK_FIXTURE not in params:
            continue
        stack_dependent.append(node.name)
        source = ast.unparse(node)
        if _TEARDOWN_FLAG in source:
            destructive.append(node.name)

    assert stack_dependent, (
        f"no test in {_E2E_MODULE} takes the {_STACK_FIXTURE!r} fixture — "
        "this guard has drifted from the module it protects"
    )
    assert destructive, (
        f"no test in {_E2E_MODULE} runs {_TEARDOWN_FLAG!r} — as above, "
        "the guard is no longer watching anything"
    )
    for name in destructive:
        assert name == stack_dependent[-1], (
            f"{name} tears the shared stack down but is not the last "
            f"stack-dependent test in {_E2E_MODULE}; the tests after it "
            f"({stack_dependent[stack_dependent.index(name) + 1 :]}) would "
            "run against a stopped stack. Move it to the end of the file."
        )
