"""Metrics tracking with windowed statistics."""
from __future__ import annotations

import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 100


class MetricsTracker:
    """Tracks named metrics with a sliding window for computing statistics."""

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self._data: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def record(self, name: str, value: float) -> None:
        """Record a single metric value."""
        self._data[name].append(value)

    def mean(self, name: str) -> float:
        """Compute the mean of a metric over the current window."""
        values = self._data.get(name)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def latest(self, name: str) -> float:
        """Return the most recent value of a metric, or 0.0 if empty."""
        values = self._data.get(name)
        if not values:
            return 0.0
        return values[-1]

    def count(self, name: str) -> int:
        """Return the number of recorded values for a metric."""
        values = self._data.get(name)
        return len(values) if values else 0

    def all_metrics(self) -> dict[str, float]:
        """Return the mean of all tracked metrics."""
        return {name: self.mean(name) for name in self._data}

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._data.clear()
