"""Tests for ``forge.mangomas.teacher_trace``."""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING

from forge.mangomas.teacher_trace import (
    DEFAULT_SCHEMA_VERSION,
    TeacherDecisionTrace,
    TeacherTraceReader,
    TeacherTraceWriter,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_trace(step: int = 0) -> TeacherDecisionTrace:
    return TeacherDecisionTrace(
        scenario_id="hex_patrol",
        episode_index=0,
        step_index=step,
        observation={"x": float(step)},
        legal_actions=[0, 1, 2],
        action_id=step % 3,
        intention=1,
        subgoals=["move"],
        rationale="ok",
        value_hat=0.5,
        constraint_critique={"violates_cooldown": False},
        top_k_probs=[],
        provider="lmstudio",
        model="qwen",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=42.0,
    )


def test_writer_roundtrip_jsonl(tmp_path: Path) -> None:
    with TeacherTraceWriter(tmp_path, "hex_patrol", 0, compress=False) as w:
        w.log(_make_trace(0))
        w.log(_make_trace(1))
        assert w.total_records == 2

    files = sorted((tmp_path / "hex_patrol").glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["scenario_id"] == "hex_patrol"
    assert record["step_index"] == 0
    assert record["schema_version"] == DEFAULT_SCHEMA_VERSION


def test_compressed_roundtrip(tmp_path: Path) -> None:
    with TeacherTraceWriter(tmp_path, "hex_patrol", 0, compress=True) as w:
        w.log(_make_trace(0))
    files = sorted((tmp_path / "hex_patrol").glob("*.jsonl.gz"))
    assert len(files) == 1
    with gzip.open(files[0], "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 1
    assert records[0]["step_index"] == 0


def test_shard_rollover_on_record_count(tmp_path: Path) -> None:
    with TeacherTraceWriter(tmp_path, "hex_patrol", 0, compress=False, shard_size=2) as w:
        for step in range(5):
            w.log(_make_trace(step))
        assert w.shard_count == 3
        assert w.total_records == 5

    files = sorted((tmp_path / "hex_patrol").glob("*.jsonl"))
    assert len(files) == 3
    counts = [len(f.read_text(encoding="utf-8").splitlines()) for f in files]
    assert counts == [2, 2, 1]


def test_reader_yields_identical_records(tmp_path: Path) -> None:
    written: list[TeacherDecisionTrace] = []
    with TeacherTraceWriter(tmp_path, "hex_patrol", 0, compress=False, shard_size=2) as w:
        for step in range(3):
            trace = _make_trace(step)
            written.append(trace)
            w.log(trace)

    reader = TeacherTraceReader(tmp_path, "hex_patrol", compress=False)
    read_back = list(reader)
    assert len(read_back) == 3
    for original, recovered in zip(written, read_back, strict=True):
        assert original.to_dict() == recovered.to_dict()


def test_writer_log_after_close_raises(tmp_path: Path) -> None:
    w = TeacherTraceWriter(tmp_path, "hex_patrol", 0, compress=False)
    w.close()
    import pytest

    with pytest.raises(RuntimeError, match="closed"):
        w.log(_make_trace(0))


def test_writer_shard_naming(tmp_path: Path) -> None:
    with TeacherTraceWriter(tmp_path, "hex_patrol", 7, compress=True, shard_size=1) as w:
        w.log(_make_trace(0))
        w.log(_make_trace(1))
    files = sorted((tmp_path / "hex_patrol").glob("*.jsonl.gz"))
    assert [f.name for f in files] == [
        "ep000007-0000.jsonl.gz",
        "ep000007-0001.jsonl.gz",
    ]


def test_trace_record_protocol_satisfied() -> None:
    from forge.traces.trace_logger import TraceRecord

    trace = _make_trace(0)
    assert isinstance(trace, TraceRecord)


def test_writer_rolls_over_when_underlying_logger_skips(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Regression: TraceLogger.log returns bool; writer must rollover on skip.

    Previously the writer would increment counters even when the
    underlying TraceLogger silently skipped a write because
    max_file_size_mb was exceeded — losing records and reporting wrong
    totals. The writer now retries exactly once after a forced rollover.
    """
    from forge.mangomas import teacher_trace as tt

    skip_count = {"n": 0}

    class _SkippingLogger:
        def __init__(self, output_path: str, *args: object, **kwargs: object) -> None:
            self.output_path = output_path
            self.calls = 0
            self.closed = False

        def log(self, _trace: tt.TeacherDecisionTrace) -> bool:
            self.calls += 1
            # Skip on the first call to the first shard only.
            if self.output_path.endswith("ep000000-0000.jsonl") and self.calls == 1:
                skip_count["n"] += 1
                return False
            return True

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(tt, "TraceLogger", _SkippingLogger)
    with tt.TeacherTraceWriter(tmp_path, "scenario", 0, compress=False, shard_size=10) as w:
        w.log(_make_trace(0))
        # First write was skipped → writer rolled over. After retry on
        # shard 1 the record is persisted and counters are incremented.
        assert w.shard_count == 2
        assert w.total_records == 1
    assert skip_count["n"] == 1
