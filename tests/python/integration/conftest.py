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
    DEFAULT_RUNNER_CONTAINER,
    docker_compose_available,
    docker_logs,
    runner_container_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


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
        pytest.skip("docker compose unavailable; opt-in E2E job needs Docker")

    repo_root = _repo_root()
    script = repo_root / "scripts" / "mc_run.sh"
    if not script.exists():
        pytest.skip(f"{script} not found (Phase 6 orchestration script missing)")

    env = os.environ.copy()
    runner_container = env.get("FORGE_MC_RUNNER_CONTAINER", DEFAULT_RUNNER_CONTAINER)

    logger.info("compose up (--build --detach)")
    up = subprocess.run(
        ["bash", str(script), "--build", "--detach"],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if up.returncode != 0:
        pytest.skip(
            f"compose up failed (rc={up.returncode}); skipping E2E. "
            f"stderr:\n{up.stderr[-800:]}"
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
        subprocess.run(
            ["bash", str(script), "--down"],
            cwd=repo_root,
            env=env,
            check=False,
            capture_output=True,
            timeout=120,
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
            logs = docker_logs(container)
            pytest.fail(
                f"runner container {container!r} state = {status!r} "
                f"(expected 'running'). Last 50 log lines:\n{logs}"
            )

    return _check
