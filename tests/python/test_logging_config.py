"""Tests for forge.utils.logging_config module."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from forge.utils.logging_config import JsonFormatter, setup_logging

logger = logging.getLogger(__name__)


class TestJsonFormatter:
    """Tests for the JsonFormatter class."""

    def test_json_formatter_output(self) -> None:
        """Formatted output is valid JSON with expected keys."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Hello world"
        assert "timestamp" in parsed

    def test_json_formatter_with_exception(self) -> None:
        """Exception info is included in the JSON output."""
        import sys

        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="failure",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestSetupLogging:
    """Tests for the setup_logging function."""

    @pytest.fixture(autouse=True)
    def _restore_root_logger(self) -> None:
        """Save and restore root logger state around each test."""
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        yield
        root.handlers = original_handlers
        root.setLevel(original_level)

    def test_setup_logging_default(self) -> None:
        """Default setup sets INFO level."""
        setup_logging()

        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1  # console only

    def test_setup_logging_debug(self) -> None:
        """Debug level is applied correctly."""
        setup_logging(level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_logging_json_format(self) -> None:
        """JSON format attaches a JsonFormatter to the console handler."""
        setup_logging(json_format=True)

        root = logging.getLogger()
        assert len(root.handlers) >= 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)

    def test_setup_logging_log_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """File handler is added when log_file is specified."""
        log_file = str(tmp_path / "test.log")
        setup_logging(log_file=log_file)

        root = logging.getLogger()
        # Should have console + file handlers
        assert len(root.handlers) == 2

        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

        # Verify the file is writable
        logging.getLogger("test_file").info("test message")
        file_handlers[0].flush()

        with Path(log_file).open() as f:
            content = f.read()
        assert "test message" in content

    def test_setup_logging_clears_existing_handlers(self) -> None:
        """Existing handlers are removed before adding new ones."""
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) >= 2

        setup_logging()

        # Should have exactly 1 handler (console only)
        assert len(root.handlers) == 1
