"""Validation tests for `docker/compose.minecraft.yml` + the v0.4
`compose.minecraft.gpu.yml` overlay.

Pure-YAML asserts — no Docker required. CI runs these on every PR
(NOT gated behind the `minecraft_e2e` marker) so a compose-config
typo surfaces immediately rather than at `docker compose up` time.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """The FORGE repo root, anchored to this test file."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def compose_data(repo_root: Path) -> dict:
    """Parsed `docker/compose.minecraft.yml`."""
    yaml = pytest.importorskip("yaml")
    text = (repo_root / "docker" / "compose.minecraft.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def gpu_overlay(repo_root: Path) -> dict:
    """Parsed `docker/compose.minecraft.gpu.yml`."""
    yaml = pytest.importorskip("yaml")
    text = (repo_root / "docker" / "compose.minecraft.gpu.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return data


# --- Base compose ----------------------------------------------------


def test_compose_has_core_services(compose_data: dict) -> None:
    """The v0.3-pre core (minecraft + mc-bot + runner) MUST still
    exist. Pinned so a refactor that accidentally drops a service
    fails this test."""
    services = compose_data["services"]
    assert "minecraft" in services
    assert "mc-bot" in services
    assert "runner" in services


def test_compose_has_v04_trainer_services(compose_data: dict) -> None:
    """v0.4 adds `trainer` + `trainer-bootstrap` under the
    `self-play` profile."""
    services = compose_data["services"]
    assert "trainer" in services
    assert "trainer-bootstrap" in services
    assert services["trainer"]["profiles"] == ["self-play"]
    assert services["trainer-bootstrap"]["profiles"] == ["self-play"]


def test_trainer_service_mounts_models_and_trajectories(compose_data: dict) -> None:
    """The trainer MUST be able to read trajectories + read/write
    the models dir. Order-independent assert via substring match
    (compose `volumes:` entries are `host:container[:ro]` strings)."""
    volumes = compose_data["services"]["trainer"]["volumes"]
    assert any(":/app/models" in v for v in volumes), (
        "trainer must mount the models dir (rw) to write ONNX bundles + manifest"
    )
    assert any(":/app/trajectories" in v for v in volumes), (
        "trainer must mount the trajectories dir to read runner output"
    )
    # Trajectories MUST be read-only on the trainer side — the
    # runner owns writes; the trimmer's `unlink` requires rw so
    # this is a deliberate read-only mount on the consumer side.
    traj_mount = next(v for v in volumes if ":/app/trajectories" in v)
    assert traj_mount.endswith(":ro"), f"trajectories mount must be read-only: {traj_mount}"


def test_trainer_command_carries_continuous_flag(compose_data: dict) -> None:
    """The trainer's default command MUST invoke `train --continuous`
    so the self-play loop self-improves. Plain `train` (fixed-iters
    mode) would exit after the cap is reached."""
    cmd = compose_data["services"]["trainer"]["command"]
    assert "--continuous" in cmd
    assert any(c == "train" or c.endswith("train") for c in cmd), (
        "trainer command must dispatch to the `train` subcommand"
    )


def test_runner_service_carries_schema_id_env_passthrough(compose_data: dict) -> None:
    """The runner MUST read `FORGE_MC_SCHEMA_ID` from the env so
    `mc_self_play.sh` can populate it after running `compute-schema-id`.
    """
    env = compose_data["services"]["runner"]["environment"]
    # compose env can be a dict or a list of `KEY=VALUE` strings;
    # this file uses the dict form.
    assert "FORGE_MC_SCHEMA_ID" in env, (
        "runner env must include FORGE_MC_SCHEMA_ID for the v0.4 schema-id ladder"
    )


def test_trainer_bootstrap_is_one_shot_no_restart(compose_data: dict) -> None:
    """The bootstrap one-shot service MUST NOT have a `restart` policy
    — it runs once via `compose run --rm` and exits."""
    svc = compose_data["services"]["trainer-bootstrap"]
    assert "restart" not in svc


def test_trainer_torch_variant_buildarg_default_is_cpu(compose_data: dict) -> None:
    """The default `TORCH_VARIANT` build-arg MUST be `cpu` so the
    base image stays runnable on any host. GPU images opt in via
    the env file or the GPU overlay."""
    args = compose_data["services"]["trainer"]["build"]["args"]
    # Compose interpolates `${TRAINER_TORCH_VARIANT:-cpu}` here;
    # the literal "cpu" is the fallback after `:-`.
    assert "cpu" in str(args.get("TORCH_VARIANT", "")), args


# --- GPU overlay -----------------------------------------------------


def test_gpu_overlay_only_modifies_trainer(gpu_overlay: dict) -> None:
    """The GPU overlay MUST be additive — only touches the `trainer`
    service. The base runner + mc-bot + minecraft services stay
    CPU-bound regardless of the overlay."""
    services = gpu_overlay["services"]
    assert set(services.keys()) == {"trainer"}, (
        f"GPU overlay should only modify `trainer`, got {set(services.keys())}"
    )


def test_gpu_overlay_adds_nvidia_device_reservation(gpu_overlay: dict) -> None:
    """Verify the canonical `deploy.resources.reservations.devices`
    block with `driver: nvidia` is in place. Compose v2 ≥ 2.20
    interprets this; older versions silently ignore it (operator
    must run the preflight in `mc_self_play.sh`)."""
    deploy = gpu_overlay["services"]["trainer"]["deploy"]
    devices = deploy["resources"]["reservations"]["devices"]
    assert isinstance(devices, list) and devices
    assert devices[0]["driver"] == "nvidia"
    assert "gpu" in devices[0]["capabilities"]


def test_gpu_overlay_sets_default_device_to_cuda(gpu_overlay: dict) -> None:
    """The GPU overlay MUST override `TRAINER_DEVICE` so the trainer
    picks `cuda` even if the base .env file left it on `cpu`."""
    env = gpu_overlay["services"]["trainer"]["environment"]
    assert "TRAINER_DEVICE" in env
    # Compose-interpolated literal `${TRAINER_DEVICE:-cuda}` — `cuda`
    # is the fallback after `:-`.
    assert "cuda" in str(env["TRAINER_DEVICE"])


# --- env.example pin -------------------------------------------------


def test_env_example_documents_all_trainer_vars(repo_root: Path) -> None:
    """The shipped `compose.minecraft.env.example` MUST document
    every `TRAINER_*` var the compose file consumes. Catches docs
    drift when a new var is added to the compose without the
    matching .env line."""
    env_text = (repo_root / "docker" / "compose.minecraft.env.example").read_text(
        encoding="utf-8"
    )
    required_vars = {
        "TRAINER_IMAGE",
        "TRAINER_TORCH_VARIANT",
        "TRAINER_DEVICE",
        "TRAINER_ROUND_ITERS",
        "TRAINER_POLL_SLEEP",
        "TRAINER_MAX_TRAJECTORIES",
        "TRAINER_MAX_BUNDLE_VERSIONS",
        "TRAINER_EXPORT_EVERY",
        "TRAINER_OBS_DIM",
        "TRAINER_ACTION_DIM",
        "TRAINER_GPU_COUNT",
        "TRAINER_GPU_DEVICES",
    }
    missing = {v for v in required_vars if v not in env_text}
    assert not missing, f"env.example missing v0.4 trainer vars: {sorted(missing)}"
