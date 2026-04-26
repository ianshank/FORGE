"""Tests for FORGE trace logging."""

from __future__ import annotations

import gzip
import json

from forge.traces.decision_trace import DecisionTrace
from forge.traces.trace_logger import TraceLogger


class TestDecisionTraceDefaults:
    """Tests for DecisionTrace default values."""

    def test_defaults(self) -> None:
        """DecisionTrace should have sensible defaults."""
        trace = DecisionTrace()
        assert trace.tick == 0
        assert trace.agent_id == ""
        assert trace.action == 0
        assert trace.confidence == 0.0
        assert trace.search_depth == 0
        assert trace.ucb1_score == 0.0
        assert trace.intent_label == ""
        assert trace.preconditions == []
        assert trace.expected_outcome == ""
        assert trace.schema_version == 1


class TestTraceLogger:
    """Tests for TraceLogger."""

    def test_writes_jsonl(self, tmp_path: object) -> None:
        """TraceLogger should write valid JSONL lines."""
        # tmp_path is a pathlib.Path provided by pytest
        import pathlib

        path = pathlib.Path(str(tmp_path))
        output = str(path / "traces.jsonl")
        logger = TraceLogger(output, compress=False)
        trace = DecisionTrace(tick=1, agent_id="a1", action=2)
        logger.log(trace)
        logger.close()

        with pathlib.Path(output).open() as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["tick"] == 1
        assert data["agent_id"] == "a1"
        assert data["action"] == 2

    def test_context_manager(self, tmp_path: object) -> None:
        """TraceLogger should work as a context manager."""
        import pathlib

        path = pathlib.Path(str(tmp_path))
        output = str(path / "traces_cm.jsonl")
        with TraceLogger(output, compress=False) as tl:
            tl.log(DecisionTrace(tick=10))
            tl.log(DecisionTrace(tick=20))

        with pathlib.Path(output).open() as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["tick"] == 10
        assert json.loads(lines[1])["tick"] == 20

    def test_compressed_output(self, tmp_path: object) -> None:
        """TraceLogger should support gzip compression."""
        import pathlib

        path = pathlib.Path(str(tmp_path))
        output = str(path / "traces.jsonl.gz")
        with TraceLogger(output, compress=True) as tl:
            tl.log(DecisionTrace(tick=5, agent_id="compressed"))

        with gzip.open(output, "rt") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["tick"] == 5
