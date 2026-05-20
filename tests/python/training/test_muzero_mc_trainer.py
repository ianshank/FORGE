"""Tests for :mod:`forge.training.muzero_mc.trainer`.

Covers:

- :func:`build_batch_from_trajectory` shape + content invariants.
- :class:`MuzeroMcTrainer` deterministic loss-decrease on a fixed
  seed + synthetic trajectory (regression for the trainer's
  forward / backward path).
- Periodic ONNX export + manifest version bump.

All trainer tests gate on ``torch`` availability via
``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path  # noqa: TC003 — used as a runtime fixture type.
from typing import Any

import numpy as np
import pytest

from forge.training.muzero_mc.trainer import build_batch_from_trajectory

# Match the muzero_step test fixture exactly so we exercise the same
# network shape the shared `train_with_gradients` was validated against.
_OBS_DIM = 11 * 11 * 7 + 73
_ACTION_DIM = 5
_LATENT_DIM = 16
_HIDDEN_DIM = 16


def _make_tiny_model() -> Any:
    pytest.importorskip("torch")
    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel

    return MuZeroWorldModel(
        MuZeroConfig(
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            latent_dim=_LATENT_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_blocks=1,
            num_unroll_steps=2,
            reward_support_size=11,
            value_support_size=11,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
        )
    )


def _write_trajectory(
    dir_: Path,
    *,
    episode_id: str,
    steps: int,
    obs_dim: int,
    action_count: int,
    schema_id: str = "stub-sid",
    gzip_it: bool = False,
) -> Path:
    """Write a synthetic TrajectoryV2 file the trainer can consume."""
    rng = np.random.default_rng(0)
    payload: dict[str, Any] = {
        "format_version": 2,
        "env_id": "stub-env",
        "schema_id": schema_id,
        "episode_id": episode_id,
        "seed": 42,
        "obs_dim": obs_dim,
        "action_count": action_count,
        "steps": [
            {
                "tick": t,
                "obs": rng.standard_normal(obs_dim).astype(np.float32).tolist(),
                "action_id": int(rng.integers(0, action_count)),
                "policy_target": [1.0 / action_count] * action_count,
                "value_target": float(rng.standard_normal()),
                "reward": float(rng.standard_normal()),
                "terminated": t == steps - 1,
                "truncated": False,
            }
            for t in range(steps)
        ],
        "final_reward": 0.0,
        "started_at": "2026-05-21T00:00:00Z",
        "ended_at": "2026-05-21T00:00:01Z",
    }
    if gzip_it:
        p = dir_ / f"{episode_id}.json.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
    else:
        p = dir_ / f"{episode_id}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------
# build_batch_from_trajectory — pure-Python, no torch needed
# ---------------------------------------------------------------------


def test_build_batch_shapes_match_muzero_buffer_contract(tmp_path: Path) -> None:
    """The batch dict shape (observations / actions / target_*) must
    match the shape :func:`forge.training._muzero_step.train_with_gradients`
    consumes — same keys, same per-row dims.
    """
    p = _write_trajectory(
        tmp_path, episode_id="ep-1", steps=8, obs_dim=4, action_count=3
    )
    trajectory = json.loads(p.read_text(encoding="utf-8"))
    batch = build_batch_from_trajectory(
        trajectory,
        indices=[0, 3],
        num_unroll_steps=2,
        td_steps=3,
        discount=0.9,
    )
    assert set(batch) >= {
        "observations",
        "actions",
        "target_values",
        "target_rewards",
        "target_policies",
    }
    assert len(batch["observations"]) == 2
    # actions: 2 rows of 2-step unrolls each.
    assert all(len(row) == 2 for row in batch["actions"])
    # target_values + target_policies have num_unroll_steps + 1 rows each
    # (one extra for the initial inference target).
    assert all(len(row) == 3 for row in batch["target_values"])
    assert all(len(row) == 3 for row in batch["target_policies"])


def test_build_batch_pads_past_end_of_trajectory(tmp_path: Path) -> None:
    """Starting near the trajectory tail must pad missing rows with
    zero rewards + uniform policies rather than raise IndexError.
    """
    p = _write_trajectory(
        tmp_path, episode_id="ep-tail", steps=3, obs_dim=2, action_count=4
    )
    trajectory = json.loads(p.read_text(encoding="utf-8"))
    batch = build_batch_from_trajectory(
        trajectory,
        indices=[2],  # last step
        num_unroll_steps=3,
        td_steps=2,
        discount=0.99,
    )
    # 3 unroll steps + 1 initial = 4 policy rows.
    assert len(batch["target_policies"][0]) == 4
    # The padded policy entries are uniform.
    for row in batch["target_policies"][0][1:]:
        for p_val in row:
            assert math.isclose(p_val, 0.25, abs_tol=1e-9)
    # Padded actions / rewards are 0.
    assert batch["actions"][0][-1] == 0
    assert batch["target_rewards"][0][-1] == 0.0


def test_build_batch_raises_for_index_past_steps(tmp_path: Path) -> None:
    p = _write_trajectory(
        tmp_path, episode_id="ep-1", steps=2, obs_dim=2, action_count=2
    )
    trajectory = json.loads(p.read_text(encoding="utf-8"))
    with pytest.raises(IndexError):
        build_batch_from_trajectory(
            trajectory,
            indices=[5],
            num_unroll_steps=1,
            td_steps=1,
            discount=0.99,
        )


# ---------------------------------------------------------------------
# MuzeroMcTrainer — gated on torch
# ---------------------------------------------------------------------


def test_trainer_decreases_loss_on_fixed_seed(tmp_path: Path) -> None:
    """Deterministic loss-decrease check. Pinned baseline: with
    `torch.manual_seed(0)` + `np.random.seed(0)` + 5 train steps
    against a fixed synthetic trajectory, the loss reliably drops
    by at least 1 % vs the initial value. We assert a conservative
    threshold (final < initial * 0.99) — well-below the locally-
    observed reduction, so this is robust against minor
    numpy/torch version drift.
    """
    torch = pytest.importorskip("torch")
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import (
        MuzeroMcTrainer,
        MuZeroMcTrainerConfig,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    _write_trajectory(
        tmp_path,
        episode_id="ep-1",
        steps=8,
        obs_dim=_OBS_DIM,
        action_count=_ACTION_DIM,
    )

    model = _make_tiny_model()
    reader = TrajectoryReader(tmp_path, batch_size=4)
    cfg = MuZeroMcTrainerConfig(
        train_iters=0,  # we'll drive train_step manually
        output_dir=tmp_path / "out",
        manifest_path=tmp_path / "out" / "model_manifest.json",
        schema_id="stub-sid",
        batch_size=4,
        seed=0,
        export_every_n_iters=0,
        log_every_n_iters=0,
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)

    initial = trainer.train_step()
    for _ in range(4):
        last = trainer.train_step()

    assert last["loss"] < initial["loss"] * 0.99, (
        f"loss did not drop ≥1%; initial={initial['loss']:.4f} "
        f"final={last['loss']:.4f}"
    )
    for k in ("loss", "policy_loss", "value_loss", "reward_loss", "l2_reg"):
        assert math.isfinite(last[k]), f"{k} = {last[k]}"


def test_trainer_consumes_jsongz_input(tmp_path: Path) -> None:
    """Smoke-test that the trainer happily loads .json.gz trajectory
    files via the reader's auto-detect path.
    """
    pytest.importorskip("torch")
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import (
        MuzeroMcTrainer,
        MuZeroMcTrainerConfig,
    )

    _write_trajectory(
        tmp_path,
        episode_id="ep-gz",
        steps=5,
        obs_dim=_OBS_DIM,
        action_count=_ACTION_DIM,
        gzip_it=True,
    )
    model = _make_tiny_model()
    reader = TrajectoryReader(tmp_path, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        train_iters=0,
        output_dir=tmp_path / "out",
        manifest_path=tmp_path / "out" / "model_manifest.json",
        schema_id="stub-sid",
        batch_size=2,
        seed=0,
        export_every_n_iters=0,
        log_every_n_iters=0,
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)
    metrics = trainer.train_step()
    assert math.isfinite(metrics["loss"])


def test_trainer_config_rejects_invalid_values(tmp_path: Path) -> None:
    from forge.training.muzero_mc.trainer import MuZeroMcTrainerConfig

    with pytest.raises(ValueError, match="train_iters"):
        MuZeroMcTrainerConfig(
            train_iters=-1, schema_id="x", output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="batch_size"):
        MuZeroMcTrainerConfig(batch_size=0, schema_id="x", output_dir=tmp_path)
    with pytest.raises(ValueError, match="schema_id"):
        MuZeroMcTrainerConfig(schema_id="", output_dir=tmp_path)


def test_trainer_config_device_default_is_cpu(tmp_path: Path) -> None:
    """The default `MuZeroMcTrainerConfig.device` MUST be `'cpu'` for
    backwards-compat with v0.3-pre (which had no device knob). Operators
    must opt into GPU explicitly via `--device cuda` or `--device auto`.
    """
    from forge.training.muzero_mc.trainer import DEFAULT_DEVICE, MuZeroMcTrainerConfig

    cfg = MuZeroMcTrainerConfig(schema_id="x", output_dir=tmp_path)
    assert DEFAULT_DEVICE == "cpu"
    assert cfg.device == "cpu"


def test_trainer_config_rejects_invalid_device(tmp_path: Path) -> None:
    """Typos at config-time MUST surface in `__post_init__` validation
    rather than as a confusing torch error later."""
    from forge.training.muzero_mc.trainer import MuZeroMcTrainerConfig

    with pytest.raises(ValueError, match="device"):
        MuZeroMcTrainerConfig(device="mps", schema_id="x", output_dir=tmp_path)
    with pytest.raises(ValueError, match="device"):
        MuZeroMcTrainerConfig(device="CUDA", schema_id="x", output_dir=tmp_path)


def test_resolve_device_auto_picks_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """`device='auto'` resolves to `cuda` when `torch.cuda.is_available()`
    returns True. Monkeypatched to make the test runnable on CPU-only
    CI hosts."""
    torch = pytest.importorskip("torch")
    from forge.training.muzero_mc.trainer import _resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    resolved = _resolve_device("auto")
    assert resolved.type == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """`device='auto'` falls back to `cpu` when CUDA is unavailable."""
    torch = pytest.importorskip("torch")
    from forge.training.muzero_mc.trainer import _resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    resolved = _resolve_device("auto")
    assert resolved.type == "cpu"


def test_resolve_device_explicit_cpu_does_not_query_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `device='cpu'` MUST NOT call `torch.cuda.is_available()`.
    Pins the lazy-resolve discipline so a hostile CUDA driver can't
    affect a CPU-only training run."""
    torch = pytest.importorskip("torch")
    from forge.training.muzero_mc.trainer import _resolve_device

    queried = {"n": 0}

    def fake_is_available() -> bool:
        queried["n"] += 1
        return True

    monkeypatch.setattr(torch.cuda, "is_available", fake_is_available)
    resolved = _resolve_device("cpu")
    assert resolved.type == "cpu"
    assert queried["n"] == 0


def test_trainer_moves_model_to_configured_device(tmp_path: Path) -> None:
    """End-to-end: `MuzeroMcTrainer.__init__` should move the model
    to the resolved device. Asserts via the `trainer.device` property
    + a model parameter's `.device.type`. CPU-only assertion (skip if
    we ever extend this to assert against CUDA)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import MuzeroMcTrainer, MuZeroMcTrainerConfig

    # Empty dir is fine — we never call train_step in this test.
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    model = MuZeroWorldModel(MuZeroConfig(obs_dim=4, action_dim=3))
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="x", output_dir=tmp_path / "out", device="cpu"
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)
    assert trainer.device.type == "cpu"
    params = trainer._model.all_parameters()
    assert params, "model has no parameters"
    assert params[0].device.type == "cpu"
    # Silence the unused-import warning the unused torch reference triggers.
    _ = torch


# --- T4: continuous-mode + replay-buffer hygiene ----------------------


def test_trainer_config_rejects_invalid_round_poll_sleep(tmp_path: Path) -> None:
    from forge.training.muzero_mc.trainer import MuZeroMcTrainerConfig

    with pytest.raises(ValueError, match="round_poll_sleep_s"):
        MuZeroMcTrainerConfig(
            round_poll_sleep_s=0.0, schema_id="x", output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="round_poll_sleep_s"):
        MuZeroMcTrainerConfig(
            round_poll_sleep_s=-1.0, schema_id="x", output_dir=tmp_path
        )


def test_trainer_config_rejects_zero_max_trajectories(tmp_path: Path) -> None:
    from forge.training.muzero_mc.trainer import MuZeroMcTrainerConfig

    with pytest.raises(ValueError, match="max_trajectories"):
        MuZeroMcTrainerConfig(
            max_trajectories=0, schema_id="x", output_dir=tmp_path
        )


def test_trainer_config_max_trajectories_default_is_none(tmp_path: Path) -> None:
    """v0.3-pre callers MUST see no disk-cleanup behaviour by default
    (backwards-compat). Explicit pin against the default value.
    """
    from forge.training.muzero_mc.trainer import MuZeroMcTrainerConfig

    cfg = MuZeroMcTrainerConfig(schema_id="x", output_dir=tmp_path)
    assert cfg.max_trajectories is None


def test_trim_replay_buffer_none_is_noop(tmp_path: Path) -> None:
    """When `max_trajectories=None`, `_trim_replay_buffer` MUST NOT
    touch the disk regardless of how many files are present.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import MuzeroMcTrainer, MuZeroMcTrainerConfig

    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    # Plant 5 dummy trajectory files.
    for i in range(5):
        (traj_dir / f"ep-{i:06d}.json").write_text("{}", encoding="utf-8")

    model = MuZeroWorldModel(MuZeroConfig(obs_dim=4, action_dim=3))
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="x", output_dir=tmp_path / "out", device="cpu"
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)
    deleted = trainer._trim_replay_buffer(None)
    assert deleted == 0
    assert len(list(traj_dir.glob("ep-*.json"))) == 5


def test_trim_replay_buffer_keeps_newest_when_cap_below_floor(tmp_path: Path) -> None:
    """The trimmer MUST keep at least
    `DEFAULT_TRIM_KEEP_NEWEST` files even when the cap would otherwise
    require deleting more. Guards against unlinking an in-flight
    runner write.
    """
    import time

    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import (
        DEFAULT_TRIM_KEEP_NEWEST,
        MuzeroMcTrainer,
        MuZeroMcTrainerConfig,
    )

    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    # Plant 8 trajectory files with staggered mtimes so the trimmer
    # can sort them.
    for i in range(8):
        p = traj_dir / f"ep-{i:06d}.json"
        p.write_text("{}", encoding="utf-8")
        # Touch the older files to set their mtime — Windows-safe.
        old_time = time.time() - (10 - i)  # i=0 oldest, i=7 newest
        import os

        os.utime(p, (old_time, old_time))

    model = MuZeroWorldModel(MuZeroConfig(obs_dim=4, action_dim=3))
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="x",
        output_dir=tmp_path / "out",
        device="cpu",
        max_trajectories=1,  # Below the floor.
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)
    deleted = trainer._trim_replay_buffer(1)
    remaining = sorted(traj_dir.glob("ep-*.json"))
    assert len(remaining) == DEFAULT_TRIM_KEEP_NEWEST
    assert deleted == 8 - DEFAULT_TRIM_KEEP_NEWEST
    # The newest N files MUST be the survivors (highest indices).
    surviving_indices = sorted(int(p.stem.removeprefix("ep-")) for p in remaining)
    expected_indices = list(range(8 - DEFAULT_TRIM_KEEP_NEWEST, 8))
    assert surviving_indices == expected_indices


def test_format_bundle_version_dir_matches_pad_width() -> None:
    """Pins `format_bundle_version_dir` against
    `BUNDLE_VERSION_PAD_WIDTH = 8` + `BUNDLE_VERSION_PREFIX = "v"`.
    Drift on either side fails this test.
    """
    from forge.training.muzero_mc.trainer import (
        BUNDLE_VERSION_PAD_WIDTH,
        BUNDLE_VERSION_PREFIX,
        format_bundle_version_dir,
    )

    assert BUNDLE_VERSION_PREFIX == "v"
    assert BUNDLE_VERSION_PAD_WIDTH == 8
    assert format_bundle_version_dir(1) == "v00000001"
    assert format_bundle_version_dir(42) == "v00000042"
    assert format_bundle_version_dir(99_999_999) == "v99999999"
    # Overflow grows the field (matches `format_episode_id`).
    assert format_bundle_version_dir(100_000_000) == "v100000000"


def test_export_bundle_writes_versioned_subdir(tmp_path: Path) -> None:
    """Pin T4a's atomicity contract: `_export_bundle` MUST write
    the three ONNX files into `output_dir/vNNNNNNNN/` (NEW subdir),
    NOT in-place into `output_dir/`. The manifest's per-role `path`
    field carries the versioned prefix.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    # `torch.onnx.export` (torch >= 2.4) imports `onnxscript` lazily.
    # Skip cleanly if absent rather than surfacing a confusing
    # ModuleNotFoundError mid-test.
    pytest.importorskip("onnxscript")
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import MuzeroMcTrainer, MuZeroMcTrainerConfig

    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    out_dir = tmp_path / "models"
    # Reuses `_make_tiny_model()` so the CNN shape requirement
    # (`obs_dim == grid_h * grid_w * grid_channels + vector_dim`)
    # is satisfied. Plain `obs_dim=4` would fail the reshape in
    # the representation network.
    model = _make_tiny_model()
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="x", output_dir=out_dir, device="cpu", max_bundle_versions=0
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)
    trainer._export_bundle()
    # Bundle subdir exists; flat onnx files do NOT exist at top level.
    assert (out_dir / "v00000001").is_dir()
    assert (out_dir / "v00000001" / "representation.onnx").is_file()
    assert (out_dir / "v00000001" / "dynamics.onnx").is_file()
    assert (out_dir / "v00000001" / "prediction.onnx").is_file()
    assert not (out_dir / "representation.onnx").exists()
    # Manifest's per-role path uses the versioned prefix.
    import json

    manifest_data = json.loads((out_dir / "model_manifest.json").read_text())
    assert manifest_data["files"]["representation"]["path"].startswith("v00000001/")
    assert manifest_data["version"] == 1


def test_export_bundle_subsequent_versions_dont_overwrite(tmp_path: Path) -> None:
    """The active bundle (`v{N}/`) MUST never be overwritten in place.
    Pinned via writing two bundles and asserting BOTH subdirs survive.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import MuzeroMcTrainer, MuZeroMcTrainerConfig

    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    out_dir = tmp_path / "models"
    model = _make_tiny_model()
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="x", output_dir=out_dir, device="cpu", max_bundle_versions=0
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)
    trainer._export_bundle()
    trainer._export_bundle()
    assert (out_dir / "v00000001").is_dir()
    assert (out_dir / "v00000002").is_dir()
    import json

    manifest_data = json.loads((out_dir / "model_manifest.json").read_text())
    assert manifest_data["version"] == 2
    assert "v00000002/" in manifest_data["files"]["representation"]["path"]


def test_export_bundle_gc_removes_old_versions(tmp_path: Path) -> None:
    """With `max_bundle_versions=2`, the trainer keeps only the two
    newest bundles; older `v{N}/` subdirs are removed after the
    manifest swap.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import MuzeroMcTrainer, MuZeroMcTrainerConfig

    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    out_dir = tmp_path / "models"
    model = _make_tiny_model()
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="x", output_dir=out_dir, device="cpu", max_bundle_versions=2
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)
    for _ in range(4):
        trainer._export_bundle()
    surviving = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert surviving == ["v00000003", "v00000004"]


def test_train_continuous_stops_on_flag(tmp_path: Path) -> None:
    """`train_continuous` yields summaries until `stop()` returns True.
    Cold-start phase: empty trajectory dir → trainer sleeps in the
    poll loop, then the stop flag interrupts.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.training.muzero_mc.replay import TrajectoryReader
    from forge.training.muzero_mc.trainer import MuzeroMcTrainer, MuZeroMcTrainerConfig

    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()  # Empty — cold-start.

    model = MuZeroWorldModel(MuZeroConfig(obs_dim=4, action_dim=3))
    reader = TrajectoryReader(traj_dir, batch_size=2)
    cfg = MuZeroMcTrainerConfig(
        schema_id="x",
        output_dir=tmp_path / "out",
        device="cpu",
        round_poll_sleep_s=0.01,  # Short sleep so the test runs fast.
    )
    trainer = MuzeroMcTrainer(model, reader, cfg)

    # Stop flag: True after first call, but the trainer never runs
    # `train_step` because the trajectory dir stays empty.
    call_count = {"n": 0}

    def stop_now() -> bool:
        call_count["n"] += 1
        # First few calls return False so we hit the cold-start sleep,
        # then return True to break out cleanly.
        return call_count["n"] >= 3

    summaries = list(trainer.train_continuous(round_iters=1, stop=stop_now))
    # Should yield 0 summaries because we never had a trajectory.
    assert summaries == []
    # Trainer's gradient counter must NOT have advanced.
    assert trainer.iter == 0
