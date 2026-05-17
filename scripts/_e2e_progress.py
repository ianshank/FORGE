"""Orchestrator-resume progress state for ``run_e2e_long.py``.

A tiny JSON checkpoint that survives a crash mid-run and lets the orchestrator
skip already-completed work on the next invocation. Atomic save via
``tempfile + os.replace`` so a partial write can never leave a half-flushed
JSON on disk.

This is **deliberately separate** from
:class:`forge.training.checkpointing.CheckpointManager`. ``CheckpointManager``
serialises training state (model weights, optimiser state, epoch counters).
``ProgressState`` records *which orchestrator stage we got to* — episodes
collected so far, scenario cursor, last seed used. Mixing the two would
conflate "where in the long run we are" with "what does the model know".

Format is stable + minimal so a human can ``cat .e2e_progress.json`` and
debug a stuck run without unpickling anything. All fields are scalars only;
no numpy / torch / forge imports here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressState:
    """One-shot snapshot of orchestrator progress between stage boundaries.

    Parameters
    ----------
    run_id:
        Stable identifier shared with the Rust ``forge-eval-longrun`` invocation
        and surfaced on the MLflow run as the canonical id. Generated once at
        first launch; preserved across resumes.
    episodes_completed:
        Number of episodes already produced by the collector + persisted to
        teacher_trace shards. The next collection batch starts at this offset.
    scenario_cursor:
        Index into the configured ``scenario_refs`` list pointing at the next
        scenario to collect from. Matches ``len(scenario_refs)`` when the run
        has cycled through every scenario.
    last_seed:
        Last seed handed to the collector. The next collection call uses
        ``last_seed + episodes_completed`` so re-runs stay deterministic
        relative to the original seed schedule.
    """

    run_id: str
    episodes_completed: int
    scenario_cursor: int
    last_seed: int


def load(path: Path) -> ProgressState | None:
    """Return the parsed state, or ``None`` if no checkpoint exists yet.

    A missing file is the normal "first run" case and must not raise — the
    orchestrator decides whether to start fresh or fail based on the return
    value, not on a caught exception.
    """
    if not path.exists():
        logger.debug("progress checkpoint missing at %s; starting fresh", path)
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        # A corrupted checkpoint is louder than a missing one — the caller
        # almost certainly wants to know rather than silently restart.
        raise ValueError(f"progress checkpoint at {path} is not valid JSON: {err}") from err
    return ProgressState(
        run_id=str(raw["run_id"]),
        episodes_completed=int(raw["episodes_completed"]),
        scenario_cursor=int(raw["scenario_cursor"]),
        last_seed=int(raw["last_seed"]),
    )


def save(path: Path, state: ProgressState) -> None:
    """Atomically write the checkpoint via tempfile + os.replace.

    ``os.replace`` is atomic across both POSIX and Windows when the source
    and destination live on the same filesystem (the tempfile is created in
    ``path.parent`` to guarantee that). A crash mid-write therefore leaves
    either the previous checkpoint intact or the new one fully flushed —
    never a truncated JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".e2e_progress.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(asdict(state), fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
        logger.debug(
            "progress checkpoint saved: run_id=%s episodes=%d cursor=%d seed=%d -> %s",
            state.run_id,
            state.episodes_completed,
            state.scenario_cursor,
            state.last_seed,
            path,
        )
    except BaseException:
        # Best-effort cleanup of the orphan tempfile; the exception still
        # surfaces to the caller.
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
