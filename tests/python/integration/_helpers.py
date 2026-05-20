"""Shared helpers for the opt-in Minecraft E2E suite.

Lives next to ``conftest.py`` so both the fixture surface and the
individual test files can import it directly without the pytest /
mypy cross-package-name confusion (the test directory has no
``__init__.py`` ancestors, so `tests.python.integration.conftest`
and `integration.conftest` are seen as duplicate modules by
strict mypy).

All numbers and container-name defaults are exposed as
module-level constants — single source of truth.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

#: Maximum seconds to wait for a polling predicate to flip true.
#: The longest realistic compose-stack startup we've seen is ~60s
#: (Minecraft world-gen on first boot); we double it for safety.
POLL_TIMEOUT_SECS: int = 120

#: Per-iteration sleep between predicate evaluations.
POLL_INTERVAL_SECS: float = 2.0

#: Default name of the runner container, as docker compose
#: constructs it from `compose.minecraft.yml`. Overridable via the
#: ``FORGE_MC_RUNNER_CONTAINER`` env var so a custom compose project
#: name (`docker compose -p foo up`) still works.
DEFAULT_RUNNER_CONTAINER: str = "forge-mc-runner"


def docker_compose_available() -> bool:
    """True iff ``docker compose version`` returns 0."""
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "compose", "version"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def runner_container_state(container: str) -> str | None:
    """Return the docker container's ``.State.Status`` or ``None``
    if the container is absent.
    """
    if shutil.which("docker") is None:
        return None
    try:
        completed = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def docker_logs(container: str, tail: int = 50) -> str:
    """Best-effort tail of the runner container logs for failure
    messages.
    """
    if shutil.which("docker") is None:
        return "<docker CLI not available>"
    try:
        completed = subprocess.run(
            ["docker", "logs", f"--tail={tail}", container],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"<docker logs failed: {e}>"
    return completed.stdout + completed.stderr


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_secs: int = POLL_TIMEOUT_SECS,
    interval_secs: float = POLL_INTERVAL_SECS,
    health_check: Callable[[], None] | None = None,
    description: str = "predicate",
) -> None:
    """Poll ``predicate`` until it returns True or ``timeout_secs``
    elapses. Calls ``health_check`` on every iteration; any exception
    raised by the health check propagates immediately (used by the
    runner-container guard to fail fast on a crashed runner).
    """
    deadline = time.monotonic() + timeout_secs
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if health_check is not None:
            health_check()
        try:
            if predicate():
                return
        except Exception as exc:
            last_error = exc
        time.sleep(interval_secs)
    msg = f"wait_until({description}) timed out after {timeout_secs}s"
    if last_error is not None:
        msg += f"; last predicate error: {last_error!r}"
    raise TimeoutError(msg)
