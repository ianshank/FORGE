"""Tests for ``forge.training.muzero_mc.replay``.

Builds synthetic ``TrajectoryV2`` JSONL files matching the shape the
Rust ``TrajectoryWriter`` produces, then iterates with
:class:`TrajectoryReader` and asserts batch shapes / validation /
ordering.

Synthetic fixtures (rather than running the Rust runner here) keep the
test isolated to the Python side; a separate Phase 5 cross-runtime
round-trip test exercises the runner ↔ reader contract.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — used as a runtime fixture type.
from typing import Any

import pytest

from forge.training.muzero_mc.replay import (
    DEFAULT_TRAJECTORY_GLOB,
    TRAJECTORY_FORMAT_VERSION,
    StepBatch,
    TrajectoryError,
    TrajectoryReader,
    load_trajectory,
)


def _write_trajectory(
    dir_: Path,
    *,
    episode_id: str,
    steps: int,
    obs_dim: int = 4,
    action_count: int = 3,
    schema_id: str = "schema-sid",
    env_id: str = "stub-env",
    format_version: int = TRAJECTORY_FORMAT_VERSION,
) -> Path:
    payload: dict[str, Any] = {
        "format_version": format_version,
        "env_id": env_id,
        "schema_id": schema_id,
        "episode_id": episode_id,
        "seed": 42,
        "obs_dim": obs_dim,
        "action_count": action_count,
        "steps": [
            {
                "tick": t,
                "obs": [float(t)] * obs_dim,
                "action_id": t % action_count,
                "policy_target": [1.0 / action_count] * action_count,
                "value_target": float(t) * 0.5,
                "reward": 1.0,
                "terminated": t == steps - 1,
                "truncated": False,
            }
            for t in range(steps)
        ],
        "final_reward": float(steps),
        "started_at": "2026-05-20T00:00:00Z",
        "ended_at": "2026-05-20T00:00:01Z",
    }
    p = dir_ / f"{episode_id}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_default_glob_matches_runner_output() -> None:
    assert DEFAULT_TRAJECTORY_GLOB == "ep-*.json"


def test_load_trajectory_returns_full_dict(tmp_path: Path) -> None:
    p = _write_trajectory(tmp_path, episode_id="ep-000001", steps=3)
    data = load_trajectory(p)
    assert data["episode_id"] == "ep-000001"
    assert data["format_version"] == TRAJECTORY_FORMAT_VERSION
    assert len(data["steps"]) == 3


def test_load_rejects_wrong_format_version(tmp_path: Path) -> None:
    p = _write_trajectory(
        tmp_path,
        episode_id="ep-000001",
        steps=1,
        format_version=TRAJECTORY_FORMAT_VERSION + 1,
    )
    with pytest.raises(TrajectoryError, match="format_version mismatch"):
        load_trajectory(p)


def test_reader_yields_batches_of_configured_size(tmp_path: Path) -> None:
    _write_trajectory(tmp_path, episode_id="ep-000001", steps=5)
    _write_trajectory(tmp_path, episode_id="ep-000002", steps=4)

    reader = TrajectoryReader(tmp_path, batch_size=4)
    batches: list[StepBatch] = list(reader)
    # 9 total steps → batches of 4,4,1
    assert [len(b) for b in batches] == [4, 4, 1]
    # Concatenated batch lengths should align with field shapes.
    for b in batches:
        assert len(b.obs) == len(b.action_id) == len(b.policy_target)
        assert len(b.value_target) == len(b.reward)
        assert len(b.terminated) == len(b.truncated)
        for obs in b.obs:
            assert len(obs) == 4  # obs_dim default in fixture
        for p in b.policy_target:
            assert len(p) == 3


def test_reader_validates_obs_dim_when_expected_supplied(tmp_path: Path) -> None:
    _write_trajectory(tmp_path, episode_id="ep-000001", steps=1, obs_dim=4)
    reader = TrajectoryReader(tmp_path, batch_size=2, expected_obs_dim=8)
    with pytest.raises(TrajectoryError, match="obs_dim"):
        list(reader)


def test_reader_validates_action_count_when_expected_supplied(tmp_path: Path) -> None:
    _write_trajectory(tmp_path, episode_id="ep-000001", steps=1, action_count=3)
    reader = TrajectoryReader(tmp_path, batch_size=2, expected_action_count=10)
    with pytest.raises(TrajectoryError, match="action_count"):
        list(reader)


def test_reader_validates_schema_id_when_expected_supplied(tmp_path: Path) -> None:
    _write_trajectory(tmp_path, episode_id="ep-000001", steps=1, schema_id="real-sid")
    reader = TrajectoryReader(tmp_path, batch_size=2, expected_schema_id="wrong-sid")
    with pytest.raises(TrajectoryError, match="schema_id"):
        list(reader)


def test_reader_iter_episodes_yields_one_dict_per_file(tmp_path: Path) -> None:
    _write_trajectory(tmp_path, episode_id="ep-000001", steps=2)
    _write_trajectory(tmp_path, episode_id="ep-000002", steps=3)
    reader = TrajectoryReader(tmp_path)
    eps = list(reader.iter_episodes())
    assert len(eps) == 2
    assert {e["episode_id"] for e in eps} == {"ep-000001", "ep-000002"}


def test_episode_paths_are_sorted_by_default(tmp_path: Path) -> None:
    # Create files in non-alphabetical order; sorted() must put them
    # back so the trainer sees them deterministically.
    _write_trajectory(tmp_path, episode_id="ep-000003", steps=1)
    _write_trajectory(tmp_path, episode_id="ep-000001", steps=1)
    _write_trajectory(tmp_path, episode_id="ep-000002", steps=1)
    reader = TrajectoryReader(tmp_path)
    names = [p.name for p in reader.episode_paths()]
    assert names == ["ep-000001.json", "ep-000002.json", "ep-000003.json"]


def test_shuffle_with_seed_is_deterministic(tmp_path: Path) -> None:
    for i in range(4):
        _write_trajectory(tmp_path, episode_id=f"ep-{i:06d}", steps=1)

    r1 = TrajectoryReader(tmp_path, shuffle=True, seed=7).episode_paths()
    r2 = TrajectoryReader(tmp_path, shuffle=True, seed=7).episode_paths()
    assert [p.name for p in r1] == [p.name for p in r2]


def test_batch_size_zero_rejected() -> None:
    with pytest.raises(ValueError):
        TrajectoryReader(".", batch_size=0)


def test_step_batch_as_torch_returns_expected_tensor_shapes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _write_trajectory(tmp_path, episode_id="ep-000001", steps=3, obs_dim=5, action_count=4)
    reader = TrajectoryReader(tmp_path, batch_size=3)
    (batch,) = list(reader)
    tensors = batch.as_torch()
    assert tuple(tensors["obs"].shape) == (3, 5)
    assert tuple(tensors["policy_target"].shape) == (3, 4)
    assert tuple(tensors["action_id"].shape) == (3,)
    assert tensors["action_id"].dtype.is_signed
    assert tensors["terminated"].dtype.is_floating_point is False


def test_missing_required_top_level_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "ep-broken.json"
    p.write_text(
        json.dumps(
            {
                "format_version": TRAJECTORY_FORMAT_VERSION,
                # missing env_id
                "schema_id": "x",
                "episode_id": "ep-broken",
                "obs_dim": 2,
                "action_count": 2,
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrajectoryError, match="missing key"):
        load_trajectory(p)
