"""Pin the forge-server container bind contract.

After ``22d0fb80`` the binary defaults to loopback (``127.0.0.1:<port>``) so
a bare ``cargo run`` is not reachable off-host. Docker port publishing
arrives on the container's own interface, which a loopback-bound process
refuses — compose already sets ``FORGE_SERVER_BIND``, but a bare
``docker run -p`` (CI GHCR smoke) does not inherit compose env.

The simulation image therefore opts in via runtime ENV. This file is the
PR-CI gate: the ``docker`` job's ``/health`` probe only runs on the
repository default branch / ``v*`` tags, so a drifted Dockerfile or smoke
step would otherwise merge green and fail after merge.

No forge / native imports: read-only text contract over Dockerfile,
compose, ``ci.yml``, and the Rust default. Comment-stripped matching
follows ``scripts/check_pinned_config_consistency.py`` so a commented-out
pin is not a live occurrence. This pytest is the wired gate — a PreToolUse
hook would duplicate it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DISTRIBUTED = REPO_ROOT / "docker" / "docker-compose.distributed.yml"
GCP_OVERLAY = REPO_ROOT / "docker" / "docker-compose.gcp.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CONFIG_RS = REPO_ROOT / "crates" / "forge-server" / "src" / "config.rs"
CONSTANTS_RS = REPO_ROOT / "crates" / "forge-types" / "src" / "constants.rs"

HISTORY_DIR = "/home/forge/forge-history"
_PORT_RE = re.compile(r"pub const DEFAULT_SERVER_PORT: u16 = (\d+);")
_SMOKE_STEP = "Smoke-pull GHCR image and /health probe"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing {path.relative_to(REPO_ROOT)}"
    return path.read_text(encoding="utf-8")


def _strip_comment(line: str) -> str:
    """Truncate at the first ``#`` — same contract as the pin-check helper."""
    return line.split("#", 1)[0]


def _active_lines(text: str) -> list[str]:
    return [_strip_comment(line).rstrip() for line in text.splitlines()]


def _has_active_line(text: str, exact: str) -> bool:
    needle = exact.strip()
    return any(line.strip() == needle for line in _active_lines(text))


def _has_active_substring(text: str, needle: str) -> bool:
    return any(needle in _strip_comment(line) for line in text.splitlines())


def _default_server_port() -> str:
    for line in _active_lines(_read(CONSTANTS_RS)):
        hit = _PORT_RE.search(line)
        if hit:
            return hit.group(1)
    raise AssertionError("DEFAULT_SERVER_PORT not found in constants.rs")


def _container_bind(port: str) -> str:
    return f"0.0.0.0:{port}"


def _loopback_bind(port: str) -> str:
    return f"127.0.0.1:{port}"


def _runtime_stage(dockerfile: str) -> str:
    parts = re.split(r"(?m)^FROM ", dockerfile)
    assert len(parts) >= 2, "Dockerfile has no FROM"
    return "FROM " + parts[-1]


def _compose_service_block(text: str, service: str) -> str:
    """Indented block under a top-level compose service key."""
    header = f"  {service}:"
    lines = text.splitlines(keepends=True)
    start: int | None = None
    for index, line in enumerate(lines):
        if _strip_comment(line).rstrip() == header:
            start = index + 1
            break
    assert start is not None, f"compose service {service!r} not found"
    collected: list[str] = []
    for line in lines[start:]:
        active = _strip_comment(line).rstrip("\n")
        if active and not active.startswith(" ") and active.endswith(":"):
            break
        if (
            active.startswith("  ")
            and not active.startswith("    ")
            and active.endswith(":")
            and not active.lstrip().startswith("-")
        ):
            break
        collected.append(line)
    return "".join(collected)


def _live_env_assignments(block: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in _active_lines(block):
        stripped = line.strip()
        if not stripped.startswith("- ") or "=" not in stripped:
            continue
        item = stripped[2:].strip()
        key, _, value = item.partition("=")
        found[key.strip()] = value
    return found


def _rust_fn_body(src: str, name: str) -> str:
    match = re.search(
        rf"fn {re.escape(name)}\([^)]*\)[^{{]*\{{([^}}]*)\}}",
        src,
    )
    assert match is not None, f"fn {name} not found"
    return match.group(1)


def _ci_smoke_script(ci_text: str) -> str:
    idx = ci_text.find(_SMOKE_STEP)
    assert idx != -1, "CI smoke step missing"
    rest = ci_text[idx:]
    nxt = re.search(r"\n      - name:", rest[1:])
    return rest if nxt is None else rest[: nxt.start() + 1]


# ---------------------------------------------------------------------------
# Helper unit tests (comment-blindness / mutation survivors)
# ---------------------------------------------------------------------------


def test_commented_bind_env_is_not_live() -> None:
    fake = "# ENV FORGE_SERVER_BIND=0.0.0.0:8080\n"
    assert not _has_active_line(fake, "ENV FORGE_SERVER_BIND=0.0.0.0:8080")


def test_live_bind_env_is_detected() -> None:
    fake = "ENV FORGE_SERVER_BIND=0.0.0.0:8080\n"
    assert _has_active_line(fake, "ENV FORGE_SERVER_BIND=0.0.0.0:8080")


# ---------------------------------------------------------------------------
# Image / CI / compose / binary contract
# ---------------------------------------------------------------------------


def test_dockerfile_sets_container_bind_env() -> None:
    port = _default_server_port()
    runtime = _runtime_stage(_read(DOCKERFILE))
    expected = f"ENV FORGE_SERVER_BIND={_container_bind(port)}"
    assert _has_active_line(runtime, expected), (
        "simulation image must opt into 0.0.0.0 via live ENV so a bare "
        f"`docker run -p` reaches /health; expected {expected}"
    )


def test_dockerfile_history_dir_writable_for_forge_user() -> None:
    runtime = _runtime_stage(_read(DOCKERFILE))
    assert _has_active_line(
        runtime, f"ENV FORGE_SERVER_HISTORY_DIR={HISTORY_DIR}"
    )
    mkdir_idx = runtime.find(f"mkdir -p {HISTORY_DIR}")
    user_idx = runtime.rfind("USER forge")
    assert mkdir_idx != -1, f"runtime stage must mkdir {HISTORY_DIR}"
    assert user_idx != -1, "runtime stage must switch to USER forge"
    assert mkdir_idx < user_idx, (
        "mkdir/chown of the history dir must run as root before USER forge"
    )
    window = runtime[mkdir_idx:user_idx]
    chown_lines = [
        line.strip()
        for line in _active_lines(window)
        if "chown" in line and HISTORY_DIR in line
    ]
    assert chown_lines, (
        f"live chown of {HISTORY_DIR} required between mkdir and USER forge"
    )


def test_dockerfile_expose_and_healthcheck_match_port() -> None:
    port = _default_server_port()
    runtime = _runtime_stage(_read(DOCKERFILE))
    assert _has_active_line(runtime, f"EXPOSE {port}")
    assert _has_active_substring(runtime, f"http://localhost:{port}/health")


def test_ci_smoke_passes_bind_and_history_env() -> None:
    port = _default_server_port()
    smoke = _ci_smoke_script(_read(CI_WORKFLOW))
    bind = _container_bind(port)
    assert _has_active_substring(smoke, f"-e FORGE_SERVER_BIND={bind}"), (
        "CI smoke docker run must pass FORGE_SERVER_BIND "
        f"(belt-and-suspenders with Dockerfile ENV {bind})"
    )
    assert _has_active_substring(
        smoke, f"-e FORGE_SERVER_HISTORY_DIR={HISTORY_DIR}"
    )
    published = (
        _has_active_substring(smoke, f"-p {port}:{port}")
        or _has_active_substring(smoke, f"-p 127.0.0.1:{port}:{port}")
    )
    assert published, f"CI smoke must publish container port {port}"


def test_compose_simulation_interpolates_bind() -> None:
    port = _default_server_port()
    block = _compose_service_block(_read(COMPOSE), "simulation")
    env = _live_env_assignments(block)
    expected = f"0.0.0.0:${{FORGE_SERVER_PORT:-{port}}}"
    assert env.get("FORGE_SERVER_BIND") == expected, env


def test_distributed_coordinator_and_worker_binds() -> None:
    """Image ENV is 0.0.0.0; workers that publish no port keep loopback."""
    port = _default_server_port()
    text = _read(DISTRIBUTED)
    coordinator = _live_env_assignments(
        _compose_service_block(text, "coordinator")
    )
    worker = _live_env_assignments(_compose_service_block(text, "worker"))
    assert coordinator.get("FORGE_SERVER_BIND") == _container_bind(port), (
        coordinator
    )
    assert worker.get("FORGE_SERVER_BIND") == _loopback_bind(port), worker
    assert coordinator.get("FORGE_SERVER_BIND") != worker.get(
        "FORGE_SERVER_BIND"
    )


def test_gcp_overlay_does_not_clobber_bind() -> None:
    text = _read(GCP_OVERLAY)
    assert not _has_active_substring(text, "FORGE_SERVER_BIND")


def test_binary_default_bind_addr_is_loopback() -> None:
    body = _rust_fn_body(_read(CONFIG_RS), "default_bind_addr")
    assert "Ipv4Addr::LOCALHOST" in body, body
    assert "DEFAULT_SERVER_PORT" in body, body
    assert "UNSPECIFIED" not in body, body
