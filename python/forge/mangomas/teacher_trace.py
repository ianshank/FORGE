"""Teacher decision traces and JSONL writer.

Records emitted by the LM Studio / Qwen teacher are persisted to disk in
JSONL (optionally gzip-compressed) shards. ``TeacherTraceWriter`` builds
on top of :class:`forge.traces.trace_logger.TraceLogger` — the latter was
widened to accept any ``to_dict()``-serialisable record via the
``TraceRecord`` protocol, so we get gzip + size-limit handling for free
and only add typed record + shard-rollover semantics here.

Shard naming::

    {output_root}/{scenario_id}/ep{episode:06d}-{shard:04d}.jsonl[.gz]
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge.traces.trace_logger import TraceLogger

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

DEFAULT_SHARD_SIZE: int = 1000
DEFAULT_COMPRESS: bool = True
DEFAULT_SCHEMA_VERSION: str = "1.0"
# Per-shard byte cap (MB) before the writer rotates to a fresh file.
# Override via TeacherTraceWriter(..., max_file_size_mb=N) or by plumbing
# through TeacherConfig in callers that already own a config struct.
DEFAULT_MAX_FILE_SIZE_MB: int = 100


@dataclass
class TeacherDecisionTrace:
    """One teacher decision record."""

    scenario_id: str = ""
    episode_index: int = 0
    step_index: int = 0
    observation: dict[str, Any] = field(default_factory=dict)
    legal_actions: list[int] = field(default_factory=list)
    action_id: int = 0
    intention: int | None = None
    subgoals: list[str] = field(default_factory=list)
    rationale: str = ""
    value_hat: float | None = None
    constraint_critique: dict[str, bool] = field(default_factory=dict)
    top_k_probs: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    schema_version: str = DEFAULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict view."""
        return asdict(self)


class TeacherTraceWriter:
    """Sharded JSONL writer for ``TeacherDecisionTrace`` records.

    Writes are append-only within an episode shard. When ``shard_size``
    records have been written a new shard is opened automatically.
    Use as a context manager to guarantee shards are flushed/closed::

        with TeacherTraceWriter(root, "hex_patrol", 0) as w:
            for trace in traces:
                w.log(trace)
    """

    def __init__(
        self,
        output_root: str | Path,
        scenario_id: str,
        episode_index: int,
        *,
        shard_size: int = DEFAULT_SHARD_SIZE,
        compress: bool = DEFAULT_COMPRESS,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> None:
        self._output_root = Path(output_root)
        self._scenario_id = scenario_id
        self._episode_index = episode_index
        self._shard_size = max(1, int(shard_size))
        self._compress = compress
        self._max_file_size_mb = max_file_size_mb
        self._records_in_shard: int = 0
        self._shard_index: int = 0
        self._total_records: int = 0
        self._logger: TraceLogger | None = None
        self._open_shard()
        logger.info(
            "TeacherTraceWriter open scenario=%s episode=%d shard_size=%d compress=%s",
            scenario_id,
            episode_index,
            self._shard_size,
            compress,
        )

    def _shard_path(self) -> Path:
        suffix = ".jsonl.gz" if self._compress else ".jsonl"
        return (
            self._output_root
            / self._scenario_id
            / f"ep{self._episode_index:06d}-{self._shard_index:04d}{suffix}"
        )

    def _open_shard(self) -> None:
        path = self._shard_path()
        self._logger = TraceLogger(
            str(path),
            max_file_size_mb=self._max_file_size_mb,
            compress=self._compress,
        )
        self._records_in_shard = 0

    def _rollover(self) -> None:
        if self._logger is not None:
            self._logger.close()
        self._shard_index += 1
        self._open_shard()

    def log(self, trace: TeacherDecisionTrace) -> None:
        """Write a single record, rolling over the shard when full.

        Counters (``records_in_shard``, ``total_records``) only advance
        when the underlying ``TraceLogger`` actually persists the record.
        If the write is skipped (max-file-size hit on the current shard),
        the writer rolls over to a fresh shard and retries once before
        raising — this keeps counters honest and avoids silently dropping
        teacher decisions.
        """
        if self._logger is None:
            msg = "TeacherTraceWriter is closed"
            raise RuntimeError(msg)
        if self._records_in_shard >= self._shard_size:
            self._rollover()
        assert self._logger is not None
        written = self._logger.log(trace)
        if not written:
            # Underlying writer rejected the write because the current
            # shard exceeded max_file_size_mb. Rollover to a fresh shard
            # and retry exactly once.
            self._rollover()
            assert self._logger is not None
            written = self._logger.log(trace)
            if not written:
                msg = (
                    "TeacherTraceWriter: TraceLogger refused write twice in a row "
                    f"(shard_size={self._shard_size}, "
                    f"max_file_size_mb={self._max_file_size_mb}). "
                    "Either lower shard_size or raise max_file_size_mb."
                )
                raise RuntimeError(msg)
        self._records_in_shard += 1
        self._total_records += 1

    def flush(self) -> None:
        """Flush the current shard to disk."""
        if self._logger is not None:
            self._logger.flush()

    def close(self) -> None:
        """Close the writer and underlying shard file."""
        if self._logger is not None:
            self._logger.close()
            self._logger = None
            logger.info(
                "TeacherTraceWriter close scenario=%s episode=%d records=%d shards=%d",
                self._scenario_id,
                self._episode_index,
                self._total_records,
                self._shard_index + 1,
            )

    @property
    def total_records(self) -> int:
        return self._total_records

    @property
    def shard_count(self) -> int:
        return self._shard_index + 1

    def __enter__(self) -> TeacherTraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


class TeacherTraceReader:
    """Iterate ``TeacherDecisionTrace`` records back from a writer output.

    Walks all shards under ``{root}/{scenario_id}/`` in sorted order.
    """

    def __init__(
        self,
        output_root: str | Path,
        scenario_id: str,
        *,
        compress: bool = DEFAULT_COMPRESS,
    ) -> None:
        self._dir = Path(output_root) / scenario_id
        self._compress = compress

    def __iter__(self) -> Any:
        suffix = ".jsonl.gz" if self._compress else ".jsonl"
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob(f"*{suffix}")):
            yield from self._read_file(path)

    def _read_file(self, path: Path) -> Any:
        import gzip

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                yield TeacherDecisionTrace(**json.loads(line))
