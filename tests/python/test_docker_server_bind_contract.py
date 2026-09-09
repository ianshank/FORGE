"""Pin the forge-server container bind contract.

After ``22d0fb80`` the binary defaults to loopback (``127.0.0.1:8080``) so
a bare ``cargo run`` is not reachable off-host. Docker port publishing
arrives on the container's own interface, which a loopback-bound process
refuses — compose already sets ``FORGE_SERVER_BIND``, but a bare
``docker run -p 8080:8080`` (CI GHCR smoke) does not inherit compose env.

The simulation image therefore opts in via runtime ENV. This file is the
PR-CI gate: the ``docker`` job's ``/health`` probe only runs on the
repository default branch / ``v*`` tags, so a drifted Dockerfile or smoke
step would otherwise merge green and fail after merge.

No forge / native imports: this is a read-only text contract over
Dockerfile, compose, and ``ci.yml``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Same string the image, CI smoke, and compose's 8080 default must share.
# Compose interpolates FORGE_SERVER_PORT; Dockerfile/CI cannot, and pin
# DEFAULT_SERVER_PORT (8080) which matches EXPOSE / HEALTHCHECK.
CONTAINER_BIND = "0.0.0.0:8080"
HISTORY_DIR = "/home/forge/forge-history"

DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DISTRIBUTED = REPO_ROOT / "docker" / "docker-compose.distributed.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CONFIG_RS = REPO_ROOT / "crates" / "forge-server" / "src" / "config.rs"
CONSTANTS_RS = REPO_ROOT / "crates" / "forge-types" / "src" / "constants.rs"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing {path.relative_to(REPO_ROOT)}"
    return path.read_text(encoding="utf-8")


def test_dockerfile_sets_container_bind_env() -> None:
    text = _read(DOCKERFILE)
    assert f"ENV FORGE_SERVER_BIND={CONTAINER_BIND}" in text, (
        "simulation image must opt into 0.0.0.0 via ENV so a bare "
        f"`docker run -p 8080:8080` reaches /health; expected "
        f"ENV FORGE_SERVER_BIND={CONTAINER_BIND}"
    )


def test_dockerfile_history_dir_writable_for_forge_user() -> None:
    text = _read(DOCKERFILE)
    assert f"ENV FORGE_SERVER_HISTORY_DIR={HISTORY_DIR}" in text
    mkdir_idx = text.find(f"mkdir -p {HISTORY_DIR}")
    user_idx = text.rfind("USER forge")
    assert mkdir_idx != -1, f"runtime stage must mkdir {HISTORY_DIR}"
    assert user_idx != -1, "runtime stage must switch to USER forge"
    assert mkdir_idx < user_idx, (
        "mkdir/chown of the history dir must run as root before USER forge"
    )
    assert "chown" in text[mkdir_idx:user_idx]


def test_ci_smoke_passes_bind_env() -> None:
    text = _read(CI_WORKFLOW)
    assert "Smoke-pull GHCR image and /health probe" in text
    assert f"-e FORGE_SERVER_BIND={CONTAINER_BIND}" in text, (
        "CI smoke docker run must pass FORGE_SERVER_BIND "
        f"(belt-and-suspenders with Dockerfile ENV {CONTAINER_BIND})"
    )


def test_compose_interpolates_bind_for_published_port() -> None:
    text = _read(COMPOSE)
    assert "FORGE_SERVER_BIND=0.0.0.0:${FORGE_SERVER_PORT:-8080}" in text


def test_distributed_workers_keep_loopback_override() -> None:
    """Image ENV would otherwise bind workers on 0.0.0.0; they publish no port."""
    text = _read(DISTRIBUTED)
    assert "FORGE_SERVER_BIND=0.0.0.0:8080" in text
    assert "FORGE_SERVER_BIND=127.0.0.1:8080" in text


def test_binary_default_remains_loopback() -> None:
    config = _read(CONFIG_RS)
    constants = _read(CONSTANTS_RS)
    assert "Ipv4Addr::LOCALHOST" in config
    assert "pub const DEFAULT_SERVER_PORT: u16 = 8080;" in constants
    assert "fn default_bind_addr()" in config
