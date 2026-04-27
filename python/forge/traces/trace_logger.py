"""JSONL trace logger with optional compression and size limits."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

    from forge.traces.decision_trace import DecisionTrace

logger = logging.getLogger(__name__)

DEFAULT_MAX_FILE_SIZE_MB = 100
BYTES_PER_MB = 1024 * 1024


class TraceLogger:
    """Writes DecisionTrace records as JSONL, with optional gzip compression."""

    def __init__(
        self,
        output_path: str,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        compress: bool = True,
    ) -> None:
        self.output_path = output_path
        self.max_file_size_mb = max_file_size_mb
        self.compress = compress
        self._file: IO[str] | None = None
        self._open()

    def _open(self) -> None:
        """Open the output file for writing."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        # The handle is owned by `self._file` and explicitly closed in
        # `close()` / `__exit__` — a `with`-block here would close the file
        # before any caller could log anything. SIM115 suppression is therefore
        # intentional, not accidental.
        if self.compress:
            self._file = gzip.open(self.output_path, "wt", encoding="utf-8")  # noqa: SIM115
        else:
            self._file = Path(self.output_path).open("w", encoding="utf-8")  # noqa: SIM115
        logger.info("TraceLogger opened %s (compress=%s)", self.output_path, self.compress)

    def log(self, trace: DecisionTrace) -> None:
        """Write a single trace as a JSONL line."""
        if self._file is None:
            msg = "TraceLogger is closed"
            raise RuntimeError(msg)
        if self._exceeds_size_limit():
            logger.warning("File size limit reached (%d MB), skipping write", self.max_file_size_mb)
            return
        line = json.dumps(trace.to_dict())
        self._file.write(line + "\n")

    def flush(self) -> None:
        """Flush the output buffer."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the output file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info("TraceLogger closed %s", self.output_path)

    def _exceeds_size_limit(self) -> bool:
        """Check if the file exceeds the configured size limit."""
        try:
            size = Path(self.output_path).stat().st_size
        except OSError:
            return False
        return size > self.max_file_size_mb * BYTES_PER_MB

    def __enter__(self) -> TraceLogger:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and close the file."""
        self.close()
