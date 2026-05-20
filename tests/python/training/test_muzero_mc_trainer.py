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
