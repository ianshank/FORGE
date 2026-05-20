"""Streaming reader for ``TrajectoryV2`` JSONL files.

The Rust ``forge_mc_runner::TrajectoryWriter`` writes one JSON document
per episode (``<episode_id>.json``). This module iterates over those
files and yields tensor batches the trainer can consume directly.

Memory discipline:

- Files are read one episode at a time (no whole-directory buffering).
- Steps within an episode are concatenated lazily; the iterator never
  holds more than ``batch_size`` rows in memory.
- The optional ``torch`` import is deferred so module import does not
  require torch (the ``minecraft`` optional-deps group pulls it in,
  but downstream tools that only need the manifest do not).

No hard-coded values: batch size, ordering policy, and the directory
glob pattern are all constructor arguments on
:class:`TrajectoryReader`.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_TRAJECTORY_GLOB",
    "TRAJECTORY_FORMAT_VERSION",
    "StepBatch",
    "TrajectoryError",
    "TrajectoryReader",
    "load_trajectory",
]

import json
import logging
import os  # noqa: TC003 — runtime use (PathLike).
import random
from collections.abc import (  # noqa: TC003 — used in annotations only but `from __future__ import annotations` makes runtime import safe + cheap.
    Iterable,
    Iterator,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Pinned trajectory format version. Mirrors
#: ``forge_replay::v2::TRAJECTORY_FORMAT_VERSION``. Bumping is a
#: breaking change.
TRAJECTORY_FORMAT_VERSION: int = 2

#: Default glob the reader uses to discover episode files inside the
#: trajectory directory.
DEFAULT_TRAJECTORY_GLOB: str = "ep-*.json"


class TrajectoryError(Exception):
    """Raised on malformed trajectory files (wrong version, missing
    keys, dimension mismatches, ...).
    """


@dataclass(frozen=True)
class StepBatch:
    """A minibatch of MuZero training examples.

    Stored as plain Python lists / nested floats so this module does
    not require torch at import time. Consumers convert to tensors
    via :meth:`as_torch` (see ``muzero_mc.bootstrap`` /
    ``muzero_trainer`` callers).

    Fields are aligned by index: ``obs[i]`` is the observation that
    produced ``action_id[i]``, with target ``policy_target[i]`` and
    bootstrap ``value_target[i]``.
    """

    obs: list[list[float]]
    action_id: list[int]
    policy_target: list[list[float]]
    value_target: list[float]
    reward: list[float]
    terminated: list[bool]
    truncated: list[bool]

    def __len__(self) -> int:
        return len(self.action_id)

    def as_torch(self) -> dict[str, Any]:
        """Convert to a dict of torch tensors. Lazy ``torch`` import."""
        import torch

        return {
            "obs": torch.tensor(self.obs, dtype=torch.float32),
            "action_id": torch.tensor(self.action_id, dtype=torch.int64),
            "policy_target": torch.tensor(self.policy_target, dtype=torch.float32),
            "value_target": torch.tensor(self.value_target, dtype=torch.float32),
            "reward": torch.tensor(self.reward, dtype=torch.float32),
            "terminated": torch.tensor(self.terminated, dtype=torch.bool),
            "truncated": torch.tensor(self.truncated, dtype=torch.bool),
        }


def load_trajectory(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a single ``TrajectoryV2`` JSON file and validate its
    invariants.

    Raises :class:`TrajectoryError` on schema drift (wrong
    ``format_version``, missing keys, dim mismatch). Returns the raw
    dict — callers iterate ``steps`` themselves.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TrajectoryError(f"{p}: root must be JSON object")
    fmt = data.get("format_version")
    if fmt != TRAJECTORY_FORMAT_VERSION:
        raise TrajectoryError(
            f"{p}: format_version mismatch (expected "
            f"{TRAJECTORY_FORMAT_VERSION}, got {fmt!r})"
        )
    for key in (
        "env_id",
        "schema_id",
        "episode_id",
        "obs_dim",
        "action_count",
        "steps",
    ):
        if key not in data:
            raise TrajectoryError(f"{p}: missing key '{key}'")
    return data


class TrajectoryReader:
    """Streams ``TrajectoryV2`` episodes from a directory.

    Construct with the directory the Rust writer is writing into; the
    reader picks up any file matching :data:`DEFAULT_TRAJECTORY_GLOB`
    (overridable via ``glob``).

    Args:
        directory: Where ``ep-*.json`` files live.
        batch_size: Number of steps per :class:`StepBatch`. Must be
            ``>= 1``.
        glob: File pattern (default: ``"ep-*.json"``).
        shuffle: If true, the episode-file list is shuffled before
            iteration (within the constructor scope, so each fresh
            ``__iter__`` call sees the same order).
        seed: Optional RNG seed for shuffling. Required when ``shuffle``
            is true and reproducibility matters.
        expected_obs_dim: Optional; if set, every loaded trajectory
            must match this dim (cross-checked against the file
            header).
        expected_action_count: Optional; if set, every loaded
            trajectory must match.
        expected_schema_id: Optional; if set, every loaded trajectory
            must have a matching ``schema_id``. Detects model/env
            drift.

    Iteration yields :class:`StepBatch` instances. The final batch may
    be smaller than ``batch_size`` (last batch is **not** padded).
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        batch_size: int = 32,
        glob: str = DEFAULT_TRAJECTORY_GLOB,
        shuffle: bool = False,
        seed: int | None = None,
        expected_obs_dim: int | None = None,
        expected_action_count: int | None = None,
        expected_schema_id: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._dir = Path(directory)
        self._batch_size = batch_size
        self._glob = glob
        self._shuffle = shuffle
        self._seed = seed
        self._expected_obs_dim = expected_obs_dim
        self._expected_action_count = expected_action_count
        self._expected_schema_id = expected_schema_id

    @property
    def directory(self) -> Path:
        """The directory being read."""
        return self._dir

    @property
    def batch_size(self) -> int:
        """Configured batch size."""
        return self._batch_size

    def episode_paths(self) -> list[Path]:
        """Sorted list of episode file paths the reader will visit."""
        paths = sorted(self._dir.glob(self._glob))
        if self._shuffle:
            rng = random.Random(self._seed)
            rng.shuffle(paths)
        return paths

    def __iter__(self) -> Iterator[StepBatch]:
        return self._iter_batches()

    def _iter_batches(self) -> Iterator[StepBatch]:
        buf: list[dict[str, Any]] = []
        for ep_path in self.episode_paths():
            trajectory = load_trajectory(ep_path)
            self._validate_header(trajectory, ep_path)
            for step in trajectory["steps"]:
                buf.append(step)
                if len(buf) >= self._batch_size:
                    yield _batch_from_steps(buf[: self._batch_size])
                    del buf[: self._batch_size]
        if buf:
            yield _batch_from_steps(buf)

    def iter_episodes(self) -> Iterable[dict[str, Any]]:
        """Yield raw trajectory dicts, one per file. Validation
        happens before each is yielded.
        """
        for ep_path in self.episode_paths():
            trajectory = load_trajectory(ep_path)
            self._validate_header(trajectory, ep_path)
            yield trajectory

    def _validate_header(self, trajectory: dict[str, Any], path: Path) -> None:
        if (
            self._expected_obs_dim is not None
            and trajectory["obs_dim"] != self._expected_obs_dim
        ):
            raise TrajectoryError(
                f"{path}: obs_dim {trajectory['obs_dim']!r} != "
                f"expected {self._expected_obs_dim!r}"
            )
        if (
            self._expected_action_count is not None
            and trajectory["action_count"] != self._expected_action_count
        ):
            raise TrajectoryError(
                f"{path}: action_count {trajectory['action_count']!r} != "
                f"expected {self._expected_action_count!r}"
            )
        if (
            self._expected_schema_id is not None
            and trajectory["schema_id"] != self._expected_schema_id
        ):
            raise TrajectoryError(
                f"{path}: schema_id {trajectory['schema_id']!r} != "
                f"expected {self._expected_schema_id!r}"
            )


def _batch_from_steps(steps: list[dict[str, Any]]) -> StepBatch:
    return StepBatch(
        obs=[list(step["obs"]) for step in steps],
        action_id=[int(step["action_id"]) for step in steps],
        policy_target=[list(step["policy_target"]) for step in steps],
        value_target=[float(step["value_target"]) for step in steps],
        reward=[float(step["reward"]) for step in steps],
        terminated=[bool(step["terminated"]) for step in steps],
        truncated=[bool(step["truncated"]) for step in steps],
    )
