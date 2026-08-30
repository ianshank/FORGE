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
import logging
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

logger = logging.getLogger(__name__)

#: Env var overriding :data:`COMPOSE_UP_TIMEOUT_SECS`.
COMPOSE_UP_TIMEOUT_ENV_VAR: str = "FORGE_MC_COMPOSE_UP_TIMEOUT"

#: Env var that, when truthy, makes the stack bring-up reuse prebuilt
#: images instead of passing ``--build``. See :func:`use_prebuilt_images`.
USE_PREBUILT_ENV_VAR: str = "FORGE_MC_USE_PREBUILT"

#: Env var that, when truthy, turns environment-related skips into hard
#: failures and requires at least one Minecraft E2E test to actually
#: execute. See :func:`require_e2e_execution`.
REQUIRE_E2E_ENV_VAR: str = "FORGE_MC_REQUIRE_E2E"

#: Budget for a bring-up that only PULLS: a pull of the published
#: images plus Minecraft world-gen. Ten minutes, unchanged from the
#: value this constant has always carried.
#:
#: This is **not** the default for either path — see
#: :func:`compose_up_timeout_secs`, which cannot know that a build is
#: impossible. It is the value an operator sets via
#: :data:`COMPOSE_UP_TIMEOUT_ENV_VAR` when they know every image is
#: already present and want a missing one to fail fast rather than
#: quietly compile.
#:
#: NOTE: this constant previously claimed the *build* path also landed
#: "well under 5 minutes". It does not. `--build` triggers a cold
#: `cargo build --release` of forge-mc-runner inside Docker, and GitHub
#: Actions does not persist BuildKit cache mounts across runners, so
#: every CI build starts from scratch.
COMPOSE_UP_TIMEOUT_SECS: int = 600

#: Subprocess timeout for a bring-up in which a BUILD can happen.
#: Sized for a cold release compile of the Rust runner plus a ~200 MB
#: ONNX Runtime download, and deliberately kept below the
#: `timeout-minutes: 60` cap on the `python-test-minecraft-e2e` job so
#: the suite fails with a diagnosable message rather than the runner
#: killing the job mid-build. Keep the two in step.
COMPOSE_UP_BUILD_TIMEOUT_SECS: int = 2700

#: Subprocess timeout for `scripts/mc_run.sh --down` stack teardown.
COMPOSE_DOWN_TIMEOUT_SECS: int = 120

#: Default tail length for :func:`docker_logs` and the failure paths
#: in :mod:`conftest`. Pinned so all error messages quote the same
#: log volume.
DEFAULT_DOCKER_LOGS_TAIL: int = 50


def _env_flag(name: str) -> bool:
    """True iff env var ``name`` is set to a recognised truthy value.

    Accepts ``1``/``true``/``yes``/``on`` case-insensitively so the
    flag behaves the same from a shell, a compose file, and a GitHub
    Actions ``env:`` block.
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def use_prebuilt_images() -> bool:
    """True iff the stack should reuse already-published images.

    When true, the bring-up drops ``--build``. A Compose service that
    declares both ``build:`` and ``image:`` and leaves ``pull_policy``
    unset already "attempts to pull the image first and falls back to
    building from source if the image is not found" (Compose spec), so
    omitting ``--build`` is by itself enough to prefer a published image
    — and still degrades to a build when there is none.
    """
    return _env_flag(USE_PREBUILT_ENV_VAR)


def require_e2e_execution() -> bool:
    """True iff a missing prerequisite must fail rather than skip.

    The Minecraft E2E fixtures ``pytest.skip`` when Docker is absent or
    the stack fails to come up. Skipped tests exit pytest ``0``, so a
    scheduled job would report success while executing nothing. Jobs
    that exist *to run* this suite set
    :data:`REQUIRE_E2E_ENV_VAR` to convert those skips into failures.
    """
    return _env_flag(REQUIRE_E2E_ENV_VAR)


def compose_up_timeout_secs() -> int:
    """Resolve the bring-up timeout.

    Precedence: :data:`COMPOSE_UP_TIMEOUT_ENV_VAR` if set to a positive
    integer, else :data:`COMPOSE_UP_BUILD_TIMEOUT_SECS`. A non-numeric
    or non-positive override falls back to the default rather than
    raising, so a typo cannot abort a long run before it starts.

    The default does **not** depend on :func:`use_prebuilt_images`,
    even though it once did. Dropping ``--build`` does not remove the
    possibility of a build: a Compose service with ``build:`` and
    ``image:`` and no ``pull_policy`` falls back to building from
    source when the image is not found. So a prebuilt run whose image
    tag is wrong, or whose registry is unreachable, silently enters
    exactly the cold `cargo build --release` this module budgets 2700s
    for — and giving it the 600s pull budget would kill it mid-compile
    with a timeout that looks like a hung stack.

    A budget can only be too small in one direction. Over-budgeting
    costs a slower failure and is still bounded by the job's own
    ``timeout-minutes``; under-budgeting kills a legitimate run. When
    the operator knows every image is present, they say so explicitly
    with ``FORGE_MC_COMPOSE_UP_TIMEOUT`` — :data:`COMPOSE_UP_TIMEOUT_SECS`
    is the value to use.
    """
    override = compose_up_timeout_override()
    return COMPOSE_UP_BUILD_TIMEOUT_SECS if override is None else override


def compose_up_timeout_override() -> int | None:
    """The operator's explicit bring-up budget, or ``None`` for the default.

    Split out of :func:`compose_up_timeout_secs` so a caller can tell
    "the operator chose this budget" from "nobody chose, so we
    defaulted", without re-reading and re-validating the variable itself.
    :mod:`conftest` uses it to decide whether its pull-only hint is worth
    printing — telling someone to export a variable they have already
    exported is noise, and slightly wrong.

    A typo or a non-positive value is deliberately **not** an override.
    Both degrade to the default, so a caller offering guidance about the
    default should still offer it.
    """
    raw = os.environ.get(COMPOSE_UP_TIMEOUT_ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            "ignoring non-integer %s=%r; using the default",
            COMPOSE_UP_TIMEOUT_ENV_VAR,
            raw,
        )
        return None
    if parsed > 0:
        return parsed
    logger.warning(
        "ignoring non-positive %s=%d; using the default",
        COMPOSE_UP_TIMEOUT_ENV_VAR,
        parsed,
    )
    return None


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
