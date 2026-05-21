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

import contextlib
import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

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

#: Subprocess timeout for `docker compose version` / `docker inspect`
#: introspection calls. These calls are local and should return
#: within a couple of seconds; ten is generous.
DOCKER_INTROSPECT_TIMEOUT_SECS: int = 10

#: Subprocess timeout for `docker logs --tail=N`. Higher than the
#: introspection timeout because docker may stream from a remote
#: engine.
DOCKER_LOGS_TIMEOUT_SECS: int = 30

#: Subprocess timeout for the `scripts/mc_run.sh --build --detach`
#: stack bring-up. The realistic worst case (cold image pull +
#: Minecraft world-gen + runner build) lands well under 5 minutes;
#: ten is the safety ceiling.
COMPOSE_UP_TIMEOUT_SECS: int = 600

#: Subprocess timeout for `scripts/mc_run.sh --down` stack teardown.
COMPOSE_DOWN_TIMEOUT_SECS: int = 120

#: Default tail length for :func:`docker_logs` and the failure paths
#: in :mod:`conftest`. Pinned so all error messages quote the same
#: log volume.
DEFAULT_DOCKER_LOGS_TAIL: int = 50


def docker_compose_available() -> bool:
    """True iff ``docker compose version`` returns 0."""
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "compose", "version"],
            check=False,
            capture_output=True,
            timeout=DOCKER_INTROSPECT_TIMEOUT_SECS,
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
            timeout=DOCKER_INTROSPECT_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def docker_logs(container: str, tail: int = DEFAULT_DOCKER_LOGS_TAIL) -> str:
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
            timeout=DOCKER_LOGS_TIMEOUT_SECS,
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


def get_posix_path(path: Path) -> str:
    """Convert a pathlib.Path to a POSIX path format suitable for the shell.

    On Windows, if WSL bash is used, converts C:\\foo to /mnt/c/foo.
    Otherwise, returns the standard path using forward slashes.
    """
    if os.name != "nt":
        return path.as_posix()

    bash = shutil.which("bash")
    if not bash:
        return path.as_posix()

    # Check if the bash is WSL bash (usually C:\\WINDOWS\\system32\\bash.exe or Microsoft store)
    is_wsl = "system32\\bash.exe" in bash.lower() or "windowsapps" in bash.lower()

    if is_wsl:
        parts = path.resolve().parts
        if parts and len(parts[0]) >= 2 and parts[0][1] == ":":
            drive = parts[0][0].lower()
            return f"/mnt/{drive}/" + "/".join(parts[1:])

    return path.as_posix()


@contextlib.contextmanager
def lf_normalized_script(script_path: Path) -> Iterator[str]:
    """Context manager to ensure a bash script has LF line endings.

    Creates a temporary script in the same folder if on Windows and using WSL bash.
    Yields the POSIX-compliant path of the script to execute.
    """
    if os.name != "nt":
        yield script_path.as_posix()
        return

    bash = shutil.which("bash")
    if not bash:
        yield script_path.as_posix()
        return

    # Check if WSL bash is used
    is_wsl = "system32\\bash.exe" in bash.lower() or "windowsapps" in bash.lower()

    if not is_wsl:
        yield script_path.as_posix()
        return

    content = script_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    temp_script = script_path.parent / f".tmp_{script_path.name}"
    temp_script.write_bytes(content.encode("utf-8"))
    try:
        yield get_posix_path(temp_script)
    finally:
        if temp_script.exists():
            temp_script.unlink()
