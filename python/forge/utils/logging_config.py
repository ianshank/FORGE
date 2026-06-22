"""Logging configuration for the FORGE framework."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import TextIO

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
JSON_FORMAT_KEYS = ["asctime", "levelname", "name", "message"]

# Environment variables mirroring the Rust `forge-observability` crate so the
# whole stack shares one structured-logging switch. Keep `LOG_FORMAT_ENV` in
# lock-step with `crates/forge-observability/src/lib.rs::LOG_FORMAT_ENV`.
LOG_FORMAT_ENV = "FORGE_LOG_FORMAT"
LOG_LEVEL_ENV = "FORGE_LOG_LEVEL"


def json_format_from_env(default: bool = False) -> bool:
    """Resolve whether JSON logging is requested from ``FORGE_LOG_FORMAT``.

    Returns ``True`` when the env var equals ``"json"`` (case-insensitive),
    ``False`` for ``"text"``, and ``default`` when the var is unset. Unknown
    values fall back to ``default`` so a typo never crashes startup.
    """
    raw = os.environ.get(LOG_FORMAT_ENV)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "json":
        return True
    if value == "text":
        return False
    return default


def setup_logging_from_env(
    level: str | None = None,
    *,
    default_level: str = DEFAULT_LOG_LEVEL,
    default_json: bool = False,
    log_file: str | None = None,
    stream: TextIO | None = None,
    clear_existing: bool = False,
) -> None:
    """Configure logging using environment-driven defaults.

    Reuses :func:`setup_logging`. The effective level is ``level`` when given,
    else ``FORGE_LOG_LEVEL``, else ``default_level``. The format is JSON when
    ``FORGE_LOG_FORMAT=json`` (falling back to ``default_json``). ``stream``
    optionally overrides the console destination (e.g. ``sys.stderr``) so
    callers that must keep ``stdout`` clean can route logs elsewhere.

    ``clear_existing`` defaults to ``False`` here (unlike :func:`setup_logging`)
    because this helper targets CLI entrypoints: a fresh process has no root
    handlers, and preserving any pre-installed handler keeps test capture
    (``caplog``) working.
    """
    effective_level = level or os.environ.get(LOG_LEVEL_ENV) or default_level
    setup_logging(
        level=effective_level,
        json_format=json_format_from_env(default_json),
        log_file=log_file,
        stream=stream,
        clear_existing=clear_existing,
    )


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
    stream: TextIO | None = None,
    clear_existing: bool = True,
) -> None:
    """Configure the root logger for the FORGE framework.

    Args:
        level: Logging level string (e.g. "INFO", "DEBUG").
        json_format: If True, output logs as JSON.
        log_file: Optional path to a log file.
        stream: Optional console stream (defaults to ``sys.stderr``). Lets
            callers that must keep ``stdout`` clean route logs explicitly.
        clear_existing: When True (default) remove pre-existing root handlers
            before installing ours. CLI entrypoints pass ``False`` so they do
            not tear down externally-installed handlers (e.g. pytest's
            ``caplog``); in a fresh process the root logger has no handlers, so
            this is behaviourally identical to clearing.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers unless the caller opts to preserve them.
    if clear_existing:
        root_logger.handlers.clear()

    if json_format:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
