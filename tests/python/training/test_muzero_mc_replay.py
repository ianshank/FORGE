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


# --- gzip auto-detect + bomb-cap tests (Track 4) -------------------------


def _write_gzip_trajectory(
    dir_: Path,
    *,
    episode_id: str,
    steps: int,
    obs_dim: int = 4,
    action_count: int = 3,
    schema_id: str = "schema-sid",
) -> Path:
    """Write a gzip-compressed trajectory that mirrors the Rust
    ``save_json_gz`` on-disk shape.
    """
    import gzip

    payload: dict[str, Any] = {
        "format_version": TRAJECTORY_FORMAT_VERSION,
        "env_id": "stub-env",
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
        "started_at": "2026-05-21T00:00:00Z",
        "ended_at": "2026-05-21T00:00:01Z",
    }
    p = dir_ / f"{episode_id}.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return p


def test_load_trajectory_auto_detects_gzip_by_extension(tmp_path: Path) -> None:
    """Mirrors the Rust ``load_json_auto_detects_gzip_by_extension``
    cross-language test.
    """
    plain = _write_trajectory(tmp_path, episode_id="ep-000001", steps=3, obs_dim=4)
    gz = _write_gzip_trajectory(tmp_path, episode_id="ep-000002", steps=3, obs_dim=4)

    plain_data = load_trajectory(plain)
    gz_data = load_trajectory(gz)
    # Bodies are independent (different episode_ids) but shape +
    # schema must match.
    assert plain_data["obs_dim"] == gz_data["obs_dim"]
    assert plain_data["action_count"] == gz_data["action_count"]
    assert len(plain_data["steps"]) == len(gz_data["steps"]) == 3


def test_load_trajectory_rejects_gzip_bomb(tmp_path: Path) -> None:
    """Pathological gzip-bomb (decompresses past the
    MAX_DECOMPRESSED_TRAJECTORY_BYTES cap) surfaces as
    :class:`TrajectoryError`, not an OOM.
    """
    import gzip

    from forge.training.muzero_mc.replay import MAX_DECOMPRESSED_TRAJECTORY_BYTES

    p = tmp_path / "bomb.json.gz"
    # Write zeros that decompress to MAX+1 bytes — flate2 / Python's
    # gzip both compress this to a tiny on-disk size.
    chunk = b"\x00" * (1024 * 1024)
    written = 0
    with gzip.open(p, "wb") as f:
        while written <= MAX_DECOMPRESSED_TRAJECTORY_BYTES:
            n = len(chunk) if (written + len(chunk)) < (MAX_DECOMPRESSED_TRAJECTORY_BYTES + 1) else (MAX_DECOMPRESSED_TRAJECTORY_BYTES + 1 - written)
            f.write(chunk[:n])
            written += n
    with pytest.raises(TrajectoryError, match=r"byte cap|exceeds"):
        load_trajectory(p)


def test_format_episode_id_matches_rust_runner() -> None:
    """Cross-language pin against ``forge_mc_runner::format_episode_id``
    and ``EPISODE_ID_{PREFIX,PAD_WIDTH}``. Drift on either side fails
    both this test and the Rust-side
    ``episode_id_constants_match_python_side`` test.
    """
    from forge.training.muzero_mc.replay import (
        EPISODE_ID_PAD_WIDTH,
        EPISODE_ID_PREFIX,
        format_episode_id,
    )

    # Constant values pinned to the Rust side.
    assert EPISODE_ID_PREFIX == "ep-"
    assert EPISODE_ID_PAD_WIDTH == 6
    # Functional pins.
    assert format_episode_id(1) == "ep-000001"
    assert format_episode_id(42) == "ep-000042"
    assert format_episode_id(999_999) == "ep-999999"
    # Pad-width is a minimum, not a max — overflow grows the field.
    assert format_episode_id(1_000_000) == "ep-1000000"


def test_load_trajectory_ignores_non_json_gzip_extensions(tmp_path: Path) -> None:
    """A stray ``.tar.gz`` (or any other ``*.gz`` that isn't
    ``.json.gz``) must NOT be auto-decompressed. The loader falls
    back to a plain-JSON read, which surfaces a clean
    :class:`TrajectoryError` rather than feeding gzipped bytes to
    ``GzDecoder`` and producing confusing diagnostics. Pins the
    ``final_ext_is_gz && stem_ext_is_json`` discipline against
    future "simplification" of the extension check. Mirrors the
    Rust ``load_json_ignores_non_json_gzip_extensions`` regression
    test.
    """
    import gzip

    p = tmp_path / "archive.tar.gz"
    with gzip.open(p, "wb") as f:
        f.write(b'{"format_version":2}')
    # Plain-JSON path attempts UTF-8 decode on gzip bytes → fails.
    with pytest.raises(TrajectoryError):
        load_trajectory(p)


def test_load_trajectory_corrupt_gzip_returns_trajectory_error(tmp_path: Path) -> None:
    p = tmp_path / "corrupt.json.gz"
    p.write_bytes(b"this is not a gzip stream")
    with pytest.raises(TrajectoryError, match="gzip decode"):
        load_trajectory(p)


def test_reader_iterates_mixed_json_and_jsongz(tmp_path: Path) -> None:
    """A directory containing both ``.json`` and ``.json.gz`` files
    yields all steps from both formats through the default reader
    glob.
    """
    _write_trajectory(tmp_path, episode_id="ep-000001", steps=2, obs_dim=4)
    _write_gzip_trajectory(tmp_path, episode_id="ep-000002", steps=3, obs_dim=4)

    reader = TrajectoryReader(tmp_path, batch_size=10)
    paths = [p.name for p in reader.episode_paths()]
    assert "ep-000001.json" in paths
    assert "ep-000002.json.gz" in paths

    batches = list(reader)
    total = sum(len(b) for b in batches)
    assert total == 5  # 2 + 3 steps across the two files


def test_max_decompressed_size_matches_documented_constant() -> None:
    """The Python cap must equal 512 MiB (the documented Rust
    constant). Catches accidental drift between the two sides; the
    Rust gate is the authoritative source.
    """
    from forge.training.muzero_mc.replay import MAX_DECOMPRESSED_TRAJECTORY_BYTES

    assert MAX_DECOMPRESSED_TRAJECTORY_BYTES == 512 * 1024 * 1024


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
