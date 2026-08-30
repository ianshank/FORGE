"""Unit tests for the `FORGE_MC_REQUIRE_E2E` guard in :mod:`conftest`.

The guard exists so a scheduled job cannot report success while
executing nothing: `pytest.skip` exits ``0``, so a suite that skips
every test looks identical to one that passed. These tests pin the
guard's own logic, which the docker-gated E2E tests cannot reach.

They carry no `minecraft_e2e` marker and need no Docker, so they run on
every PR alongside the rest of the fast suite.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from . import conftest as guard
from ._helpers import REQUIRE_E2E_ENV_VAR


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
