"""Unit tests for :mod:`tests.python.integration._helpers`.

These tests do NOT carry the ``minecraft_e2e`` marker — they exercise
the polling / docker-shim helpers via mocks and run on every PR CI
invocation, ensuring the shared helpers stay coverage-tracked even
when the opt-in compose-stack E2E doesn't run.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator  # noqa: TC003 — runtime use in _FakeMonotonic.__init__
from typing import Any

import pytest

from . import _helpers
from ._helpers import (
    COMPOSE_DOWN_TIMEOUT_SECS,
    COMPOSE_UP_TIMEOUT_SECS,
    DEFAULT_DOCKER_LOGS_TAIL,
    DEFAULT_RUNNER_CONTAINER,
    DOCKER_INTROSPECT_TIMEOUT_SECS,
    DOCKER_LOGS_TIMEOUT_SECS,
    POLL_INTERVAL_SECS,
    POLL_TIMEOUT_SECS,
    docker_compose_available,
    docker_logs,
    runner_container_state,
    wait_until,
)

_HELPERS_MOD = _helpers.__name__

# ---------- constants -------------------------------------------------


def test_constants_are_positive() -> None:
    """Sanity-pin every timeout / interval / tail constant. Catches
    accidental drift to zero or negative."""
    assert POLL_TIMEOUT_SECS > 0
    assert POLL_INTERVAL_SECS > 0
    assert DOCKER_INTROSPECT_TIMEOUT_SECS > 0
    assert DOCKER_LOGS_TIMEOUT_SECS > 0
    assert COMPOSE_UP_TIMEOUT_SECS > 0
    assert COMPOSE_DOWN_TIMEOUT_SECS > 0
    assert DEFAULT_DOCKER_LOGS_TAIL > 0
    assert DEFAULT_RUNNER_CONTAINER


# ---------- wait_until ------------------------------------------------


class _FakeMonotonic:
    """Hand-rolled monotonic clock for deterministic timeout tests."""

    def __init__(self, ticks: Iterator[float]) -> None:
        self._ticks = iter(ticks)

    def __call__(self) -> float:
        return next(self._ticks)


def test_wait_until_returns_on_first_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predicate true on the very first call → immediate return, no
    sleep, no extra clock reads beyond the loop guard."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        f"{_HELPERS_MOD}.time.monotonic",
        _FakeMonotonic(iter([0.0, 0.0])),
    )
    calls: list[int] = []

    def predicate() -> bool:
        calls.append(1)
        return True

    wait_until(predicate, timeout_secs=10, interval_secs=0.01)
    assert calls == [1]


def test_wait_until_propagates_health_check_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """The health-check is called BEFORE the predicate's try/except, so
    a ``pytest.fail`` from the health check propagates immediately and
    is not swallowed as a transient predicate error."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        f"{_HELPERS_MOD}.time.monotonic",
        _FakeMonotonic(iter([0.0, 0.0, 0.0])),
    )

    def health() -> None:
        raise RuntimeError("container crashed")

    def predicate() -> bool:
        return False

    with pytest.raises(RuntimeError, match="container crashed"):
        wait_until(predicate, timeout_secs=10, health_check=health)


def test_wait_until_swallows_predicate_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predicate raising Exception is recorded as the last_error and
    retried until either it returns True or the timeout fires."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.time.sleep", lambda *_: None)
    # Tick sequence: deadline check returns 0.0 every iteration until
    # we've made 3 predicate calls, then jump past 10.
    ticks = iter([0.0] * 6 + [100.0])
    monkeypatch.setattr(
        f"{_HELPERS_MOD}.time.monotonic",
        lambda: next(ticks),
    )
    n = {"calls": 0}

    def predicate() -> bool:
        n["calls"] += 1
        if n["calls"] < 3:
            raise ValueError(f"transient {n['calls']}")
        return True

    wait_until(predicate, timeout_secs=10)
    assert n["calls"] == 3


def test_wait_until_times_out_with_last_error_in_msg(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the deadline fires before the predicate flips true, the
    raised TimeoutError contains the last predicate exception's repr."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.time.sleep", lambda *_: None)
    # Two iterations then deadline elapsed.
    ticks = iter([0.0, 0.0, 0.0, 100.0])
    monkeypatch.setattr(
        f"{_HELPERS_MOD}.time.monotonic",
        lambda: next(ticks),
    )

    def predicate() -> bool:
        raise RuntimeError("predicate-broken")

    with pytest.raises(TimeoutError) as exc_info:
        wait_until(predicate, timeout_secs=10, description="dummy")
    assert "timed out after 10s" in str(exc_info.value)
    assert "predicate-broken" in str(exc_info.value)


def test_wait_until_times_out_without_predicate_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeout path when the predicate never raises (just returns
    False). The error message must NOT include the
    'last predicate error' suffix in that case."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.time.sleep", lambda *_: None)
    ticks = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(
        f"{_HELPERS_MOD}.time.monotonic",
        lambda: next(ticks),
    )

    with pytest.raises(TimeoutError) as exc_info:
        wait_until(lambda: False, timeout_secs=10, description="never-true")
    assert "timed out after 10s" in str(exc_info.value)
    assert "last predicate error" not in str(exc_info.value)


# ---------- docker shim functions -------------------------------------


def test_docker_compose_available_returns_false_when_docker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No docker CLI on PATH → short-circuit False (no subprocess call)."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: None)
    assert docker_compose_available() is False


def test_docker_compose_available_returns_false_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess timeout → False (no propagation)."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: "/usr/bin/docker")

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

    monkeypatch.setattr(f"{_HELPERS_MOD}.subprocess.run", fake_run)
    assert docker_compose_available() is False


def test_docker_compose_available_returns_true_on_zero_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: "/usr/bin/docker")

    class _Completed:
        returncode = 0

    monkeypatch.setattr(
        f"{_HELPERS_MOD}.subprocess.run",
        lambda *a, **k: _Completed(),
    )
    assert docker_compose_available() is True


def test_runner_container_state_returns_none_when_docker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: None)
    assert runner_container_state("any") is None


def test_runner_container_state_returns_status_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: "/usr/bin/docker")

    class _Completed:
        returncode = 0
        stdout = "running\n"

    monkeypatch.setattr(
        f"{_HELPERS_MOD}.subprocess.run",
        lambda *a, **k: _Completed(),
    )
    assert runner_container_state("foo") == "running"


def test_runner_container_state_returns_none_on_nonzero_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`docker inspect` returns non-zero when the container doesn't
    exist; the helper must surface that as None, not a crash."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: "/usr/bin/docker")

    class _Completed:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(
        f"{_HELPERS_MOD}.subprocess.run",
        lambda *a, **k: _Completed(),
    )
    assert runner_container_state("nope") is None


def test_runner_container_state_returns_none_on_empty_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty `.State.Status` (e.g. a fresh `created` container with no
    state yet) maps to None rather than an empty string."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: "/usr/bin/docker")

    class _Completed:
        returncode = 0
        stdout = "   \n"

    monkeypatch.setattr(
        f"{_HELPERS_MOD}.subprocess.run",
        lambda *a, **k: _Completed(),
    )
    assert runner_container_state("foo") is None


def test_docker_logs_returns_marker_when_docker_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: None)
    assert docker_logs("foo") == "<docker CLI not available>"


def test_docker_logs_returns_concatenated_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: "/usr/bin/docker")

    class _Completed:
        returncode = 0
        stdout = "stdout-line\n"
        stderr = "stderr-line\n"

    monkeypatch.setattr(
        f"{_HELPERS_MOD}.subprocess.run",
        lambda *a, **k: _Completed(),
    )
    out = docker_logs("foo", tail=5)
    assert "stdout-line" in out
    assert "stderr-line" in out


def test_docker_logs_returns_marker_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken docker engine surface (`OSError`) returns a diagnostic
    marker, NOT a propagated exception — the caller is usually a test
    failure-reporting path and shouldn't be derailed by a docker
    failure."""
    monkeypatch.setattr(f"{_HELPERS_MOD}.shutil.which", lambda _: "/usr/bin/docker")

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("engine down")

    monkeypatch.setattr(f"{_HELPERS_MOD}.subprocess.run", fake_run)
    assert "docker logs failed" in docker_logs("foo")
