"""Self-improvement smoke test (T7 PR-CI variant).

Drives the v0.4 trainer in continuous mode against a pre-seeded
trajectory directory + verifies that:

1. The trainer's cold-start poll loop completes when trajectories
   are present.
2. The atomic versioned-bundle layout (T4a) emits
   ``<out>/v00000001/``, ``v00000002/``, ... subdirs as exports
   proceed.
3. The manifest's `version` field bumps with each export.
4. The replay buffer trimmer (T4) respects the floor (newest N
   always kept).

Runs on every PR CI under the `minecraft_e2e_smoke` marker (NOT
behind `minecraft_e2e` — that's docker-gated). Skips on hosts
missing `torch` / `onnx` / `onnxscript` extras (`torch.onnx.export`
requires `onnxscript` ≥ torch 2.4).

**Network shape pin**: `MuZeroConfig` requires
``obs_dim == grid_h * grid_w * grid_channels + vector_dim``. The
default is ``11 * 11 * 7 + 73 = 920``. We use that value verbatim
with a tiny CNN override (`cnn_channels=(8,)`) so the network
trains in ~1s on CPU. The production runner would override
`grid_*` / `vector_dim` to match the mc-bot's observation layout;
the smoke validates the LOOP, not the production network shape.

Sized to fit comfortably in CI's 2-minute time-box: `round_iters=2`,
`export_every_n_iters=1`, `max_bundle_versions=3`, two rounds total
→ ~10-15s wall time on CPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path  # noqa: TC003 — runtime use in _populate_trajectory
from typing import Any

import pytest

pytestmark = pytest.mark.minecraft_e2e_smoke

# CNN-required obs dimensionality — must match
# `grid_h * grid_w * grid_channels + vector_dim` from
# `MuZeroConfig.__post_init__`. The smoke uses the default values
# (11 * 11 * 7 + 73 = 920) so the existing tiny-network CNN config
# applies without further overrides.
_SMOKE_OBS_DIM = 11 * 11 * 7 + 73
_SMOKE_ACTION_DIM = 5


def _populate_trajectory(path: Path, *, obs_dim: int, action_count: int, steps: int) -> None:
    """Write a minimal valid `TrajectoryV2` file the trainer can load."""
    trajectory = {
        "format_version": 2,
        "schema_id": "smoke-test",
        "env_id": "smoke",
        "episode_id": path.stem,
        "obs_dim": obs_dim,
        "action_count": action_count,
        "started_at": "2026-05-20T00:00:00Z",
        "ended_at": None,
        "steps": [
            {
                "obs": [0.1] * obs_dim,
                "action_id": 0,
                "policy_target": [1.0 / action_count] * action_count,
                "reward": 0.1,
                "value_target": 0.0,
                "terminated": False,
                "truncated": False,
            }
            for _ in range(steps)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory), encoding="utf-8")


def _make_smoke_model() -> Any:
    """Build a tiny MuZero model sized for the smoke's CPU budget.
    Mirrors `_make_tiny_model` in `test_muzero_mc_trainer.py` so both
    tests exercise the same network shape that
    `train_with_gradients` was validated against.
    """
    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel

    return MuZeroWorldModel(
        MuZeroConfig(
            obs_dim=_SMOKE_OBS_DIM,
            action_dim=_SMOKE_ACTION_DIM,
            latent_dim=16,
            hidden_dim=16,
            num_blocks=1,
            num_unroll_steps=2,
            reward_support_size=11,
            value_support_size=11,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
        )
    )


def test_trainer_continuous_bumps_manifest_with_atomic_bundles(tmp_path: Path) -> None:
    """End-to-end smoke for the v0.4 self-improvement loop (minus the
    Rust runner, which needs Docker / ONNX Runtime).

    Pre-seeds a few trajectory files, runs `train_continuous` for two
    rounds with `export_every_n_iters=1`, then asserts:

    - The manifest exists and `version >= 2`.
    - Two `v{NNNNNNNN}/` subdirs exist (`v00000001` and `v00000002`).
    - Each subdir contains the three ONNX files.
    - The manifest's per-role path points at the LATEST versioned subdir.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    # `torch.onnx.export` requires `onnxscript` from torch >= 2.4. If
    # the host's CI install lacks it, skip — covered by the CI fix
    # in commit `c4e73dd` once ONNX wheels are installed.
    pytest.importorskip("onnxscript")
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import MuzeroMcTrainer, MuZeroMcTrainerConfig

    traj_dir = tmp_path / "trajectories"
    out_dir = tmp_path / "models"

    # Pre-seed three trajectories so the cold-start poll loop
    # completes immediately on the first round.
    for i in range(3):
        _populate_trajectory(
            traj_dir / f"ep-{i:06d}.json",
            obs_dim=_SMOKE_OBS_DIM,
            action_count=_SMOKE_ACTION_DIM,
            steps=5,
        )

    model = _make_smoke_model()
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="smoke-test",
        output_dir=out_dir,
        device="cpu",
        round_poll_sleep_s=0.05,  # near-instant cold-start poll
        max_bundle_versions=3,
        export_every_n_iters=1,  # CRITICAL: ensure manifest bumps every iter
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)

    # Run exactly two rounds, then stop. Each round runs
    # `round_iters=2` gradient steps and (with export_every_n_iters=1)
    # exports two bundles per round.
    rounds_seen = {"n": 0}

    def stop_after_two_rounds() -> bool:
        return rounds_seen["n"] >= 2

    start = time.monotonic()
    for round_summary in trainer.train_continuous(
        round_iters=2, stop=stop_after_two_rounds
    ):
        rounds_seen["n"] += 1
        # Pin the per-round invariants right where they're produced.
        assert round_summary["exports"] >= 1, round_summary
        assert round_summary["last_manifest_version"] >= rounds_seen["n"]
    elapsed = time.monotonic() - start
    # Fast-path assertion: smoke MUST finish well under the 2-minute
    # CI budget. A regression that introduces a per-iter blocking
    # I/O hang surfaces here.
    assert elapsed < 60, f"smoke loop took {elapsed:.1f}s (budget 60s)"

    # Manifest pin: file exists, version >= 2, points at the newest
    # versioned subdir.
    manifest_path = out_dir / "model_manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["version"] >= 2, data
    rep_path = data["files"]["representation"]["path"]
    assert rep_path.startswith(f"v{data['version']:08d}/"), (
        f"manifest's per-role path must point at the newest versioned subdir; "
        f"got {rep_path!r} for version {data['version']}"
    )

    # Versioned subdirs exist + carry the three ONNX files.
    versioned_subdirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
    assert len(versioned_subdirs) >= 2, (
        f"expected at least 2 versioned subdirs, found {len(versioned_subdirs)}: "
        f"{[p.name for p in versioned_subdirs]}"
    )
    for subdir in versioned_subdirs:
        assert (subdir / "representation.onnx").is_file()
        assert (subdir / "dynamics.onnx").is_file()
        assert (subdir / "prediction.onnx").is_file()


def test_trainer_continuous_respects_max_bundle_versions_floor(tmp_path: Path) -> None:
    """When `max_bundle_versions=2`, only the two newest versioned
    subdirs survive after multiple exports. Pins the T4a GC contract.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import MuzeroMcTrainer, MuZeroMcTrainerConfig

    traj_dir = tmp_path / "trajectories"
    out_dir = tmp_path / "models"
    _populate_trajectory(
        traj_dir / "ep-000001.json",
        obs_dim=_SMOKE_OBS_DIM,
        action_count=_SMOKE_ACTION_DIM,
        steps=5,
    )

    model = _make_smoke_model()
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="smoke-test",
        output_dir=out_dir,
        device="cpu",
        round_poll_sleep_s=0.05,
        max_bundle_versions=2,
        export_every_n_iters=1,
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)

    # Three rounds of 2 iters each = 6 exports. With max_bundle_versions=2,
    # only the newest 2 subdirs should survive.
    rounds_seen = {"n": 0}

    def stop_after_three() -> bool:
        return rounds_seen["n"] >= 3

    for _ in trainer.train_continuous(round_iters=2, stop=stop_after_three):
        rounds_seen["n"] += 1

    surviving = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert len(surviving) == 2, surviving
    # Surviving subdirs MUST be the two HIGHEST versions.
    expected_top_versions = {
        f"v{trainer.last_manifest_version:08d}",
        f"v{trainer.last_manifest_version - 1:08d}",
    }
    assert set(surviving) == expected_top_versions, (
        f"GC kept wrong subdirs: surviving={surviving!r}, "
        f"expected={expected_top_versions!r}"
    )
