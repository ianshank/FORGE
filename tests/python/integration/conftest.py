"""Shared fixtures for the opt-in Minecraft E2E pytest suite.

The :func:`compose_up_minecraft_stack` fixture brings up the
`docker/compose.minecraft.yml` stack via `scripts/mc_run.sh`, polls
the runner's `/metrics` endpoint for the first
`forge_mc_episode_total` increment with a **runner-health fail-fast**
guard (so a runner crash surfaces as `pytest.fail(<docker logs>)`
within seconds rather than the test hanging until the CI timeout),
and tears down on exit.

All numbers, ports, and paths flow through the
`docker/compose.minecraft.env.example` file the script defaults to
when the operator-edited `compose.minecraft.env` is absent. The
tests themselves read env vars to discover the actual runtime ports
— **no hard-coded values** at the test level.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator  # noqa: TC003 — runtime use in fixture return type
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ._helpers import (
    COMPOSE_DOWN_TIMEOUT_SECS,
    DEFAULT_DOCKER_LOGS_TAIL,
    DEFAULT_RUNNER_CONTAINER,
    REQUIRE_E2E_ENV_VAR,
    compose_up_timeout_secs,
    docker_compose_available,
    docker_logs,
    lf_normalized_script,
    require_e2e_execution,
    runner_container_state,
    use_prebuilt_images,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Marker identifying the docker-gated Minecraft E2E tests. The
#: session guard below only counts tests carrying it.
_E2E_MARKER = "minecraft_e2e"

#: Node ids of `minecraft_e2e` tests that reached their call phase
#: (pass or fail, not skip). A mutable set rather than a bool so the
#: hook mutates instead of rebinding, and so the failure message can
#: say how many tests actually ran.
_e2e_executed: set[str] = set()


def _skip_or_fail(reason: str) -> None:
    """Skip, unless this job exists to run the E2E suite.

    ``pytest.skip`` exits the process ``0``. A scheduled job that skips
    every test therefore reports success while having verified nothing.
    When :data:`REQUIRE_E2E_ENV_VAR` is set, a missing prerequisite is
    a hard failure instead.
    """
    if require_e2e_execution():
        pytest.fail(f"{reason}\n({REQUIRE_E2E_ENV_VAR} is set: this job must run the E2E suite)")
    pytest.skip(reason)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record whether any Minecraft E2E test reached its call phase."""
    if report.when == "call" and not report.skipped and _E2E_MARKER in report.keywords:
        _e2e_executed.add(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail a `require-E2E` session in which nothing actually ran.

    Belt-and-braces companion to :func:`_skip_or_fail`: it catches the
    case where the tests are deselected or skipped by something other
    than the fixtures (a marker filter that matches nothing, a
    collection error swallowed upstream), which would otherwise still
    exit ``0``.
    """
    if not require_e2e_execution() or _e2e_executed:
        return
    if exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    logger.error(
        "%s is set but no %r test reached its call phase — "
        "the suite verified nothing. Failing the session.",
        REQUIRE_E2E_ENV_VAR,
        _E2E_MARKER,
    )


def _repo_root() -> Path:
    """The FORGE repo root, discovered relative to this conftest.

    Mirrors the discipline `tests/python/conftest.py` uses — anchors
    paths to a file the test layout owns, not to the caller's CWD.
    """
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def compose_up_minecraft_stack() -> Iterator[dict[str, Any]]:
    """Bring up the Minecraft + mc-bot + runner stack via
    `scripts/mc_run.sh --build --detach`. Tears down on exit.

    Yields a dict carrying the resolved env values (so tests can
    discover ports + container names without re-reading the env file).
    """
    if not docker_compose_available():
        _skip_or_fail("docker compose unavailable; opt-in E2E job needs Docker")

    repo_root = _repo_root()
    script = repo_root / "scripts" / "mc_run.sh"
    if not script.exists():
        _skip_or_fail(f"{script} not found (Phase 6 orchestration script missing)")

    env = os.environ.copy()
    runner_container = env.get("FORGE_MC_RUNNER_CONTAINER", DEFAULT_RUNNER_CONTAINER)

    # `--build` forces a cold `cargo build --release` of the runner
    # inside Docker. When the images are already published, skip it so
    # the bring-up is a pull.
    up_args = ["--detach"] if use_prebuilt_images() else ["--build", "--detach"]
    timeout_secs = compose_up_timeout_secs()
    logger.info("compose up (%s), timeout=%ds", " ".join(up_args), timeout_secs)
    with lf_normalized_script(script) as posix_script:
        up = subprocess.run(
            ["bash", posix_script, *up_args],
            cwd=repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
    if up.returncode != 0:
        _skip_or_fail(
            f"compose up failed (rc={up.returncode}); skipping E2E. stderr:\n{up.stderr[-800:]}"
        )

    try:
        yield {
            "repo_root": repo_root,
            "script": script,
            "runner_container": runner_container,
            "metrics_url": env.get(
                "FORGE_MC_METRICS_URL",
                "http://127.0.0.1:9090/metrics",
            ),
        }
    finally:
        logger.info("compose down")
        with lf_normalized_script(script) as posix_script:
            subprocess.run(
                ["bash", posix_script, "--down"],
                cwd=repo_root,
                env=env,
                check=False,
                capture_output=True,
                timeout=COMPOSE_DOWN_TIMEOUT_SECS,
            )


@pytest.fixture
def runner_health_check(compose_up_minecraft_stack: dict[str, Any]) -> Callable[[], None]:
    """Per-test convenience: a callable that raises ``pytest.fail``
    when the runner container is no longer in the ``running`` state.

    Tests pass this to ``wait_until(..., health_check=health)`` so a
    runner crash mid-poll surfaces as a clear failure with the tail
    of the container logs, not as a poll-timeout 120 seconds later.
    """
    container = compose_up_minecraft_stack["runner_container"]

    def _check() -> None:
        status = runner_container_state(container)
        if status is None:
            pytest.fail(
                f"runner container {container!r} not found via docker inspect; "
                f"compose stack may have failed to start."
            )
        if status != "running":
            logs = docker_logs(container, tail=DEFAULT_DOCKER_LOGS_TAIL)
            pytest.fail(
                f"runner container {container!r} state = {status!r} "
                f"(expected 'running'). "
                f"Last {DEFAULT_DOCKER_LOGS_TAIL} log lines:\n{logs}"
            )

    return _check
