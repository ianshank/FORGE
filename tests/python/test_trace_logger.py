"""Tests for JSONL trace logger."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock

from forge.traces.trace_logger import TraceLogger


def _make_trace(data: dict | None = None) -> MagicMock:
    """Create a mock DecisionTrace with a to_dict method."""
    trace = MagicMock()
    trace.to_dict.return_value = data or {"action": "move", "tick": 1, "reward": 0.5}
    return trace


class TestLogWritesJsonl:
    """Each trace should be written as a separate JSONL line."""

    def test_log_writes_jsonl(self, tmp_path):
        path = str(tmp_path / "traces.jsonl")
        logger = TraceLogger(path, compress=False)

        trace1 = _make_trace({"action": "move", "tick": 1})
        trace2 = _make_trace({"action": "craft", "tick": 2})

        logger.log(trace1)
        logger.log(trace2)
        logger.close()

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["action"] == "move"
        assert json.loads(lines[1])["action"] == "craft"


class TestFlushPersists:
    """Data should be visible on disk after calling flush."""

    def test_flush_persists(self, tmp_path):
        path = str(tmp_path / "traces.jsonl")
        logger = TraceLogger(path, compress=False)

        logger.log(_make_trace({"tick": 1}))
        logger.flush()

        content = Path(path).read_text()
        assert "tick" in content

        logger.close()


class TestContextManager:
    """TraceLogger should work as a context manager."""

    def test_context_manager(self, tmp_path):
        path = str(tmp_path / "traces.jsonl")

        with TraceLogger(path, compress=False) as tl:
            tl.log(_make_trace({"action": "mine"}))

        # File should be closed and readable after exiting context
        content = Path(path).read_text()
        parsed = json.loads(content.strip())
        assert parsed["action"] == "mine"


class TestCloseIdempotent:
    """Calling close multiple times should not raise."""

    def test_close_idempotent(self, tmp_path):
        path = str(tmp_path / "traces.jsonl")
        logger = TraceLogger(path, compress=False)

        logger.log(_make_trace())
        logger.close()
        logger.close()  # Second close should be a no-op


class TestCompressedOutput:
    """When compress=True, output should be gzip-compressed."""

    def test_compressed_output(self, tmp_path):
        path = str(tmp_path / "traces.jsonl.gz")
        logger = TraceLogger(path, compress=True)

        trace_data = {"action": "build", "tick": 5}
        logger.log(_make_trace(trace_data))
        logger.close()

        # Should be readable as gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            line = f.readline().strip()

        parsed = json.loads(line)
        assert parsed["action"] == "build"
        assert parsed["tick"] == 5


class TestMultipleTraces:
    """Multiple log calls should produce multiple JSONL lines."""

    def test_multiple_traces(self, tmp_path):
        path = str(tmp_path / "traces.jsonl")
        logger = TraceLogger(path, compress=False)

        for i in range(20):
            logger.log(_make_trace({"tick": i}))
        logger.close()

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 20

        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["tick"] == i
