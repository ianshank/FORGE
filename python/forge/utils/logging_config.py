"""Logging configuration for the FORGE framework."""

from __future__ import annotations

import json
import logging
import sys

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
JSON_FORMAT_KEYS = ["asctime", "levelname", "name", "message"]


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure the root logger for the FORGE framework.

    Args:
        level: Logging level string (e.g. "INFO", "DEBUG").
        json_format: If True, output logs as JSON.
        log_file: Optional path to a log file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    if json_format:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
