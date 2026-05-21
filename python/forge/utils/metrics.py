"""Metrics tracking with windowed statistics, plus Prometheus scrape helpers.

The Prometheus helpers below were hoisted out of
``tests/python/integration/test_minecraft_e2e.py`` in v0.5 Phase 1 so
both the existing E2E test and the new
``forge.training.muzero_mc.cli capture-baseline`` subcommand consume
one canonical implementation. The test-side ``_fetch_metrics`` /
``_scrape_counter`` symbols are now thin wrappers around these.
"""

from __future__ import annotations

import builtins
import logging
import math
import re
import urllib.request
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 100

#: HTTP timeout (seconds) for the Prometheus scrape helper. Aligned
#: with the runner's metrics-endpoint defaults: scrapes are local
#: traffic so anything beyond a couple of seconds is wedged.
DEFAULT_METRICS_FETCH_TIMEOUT_SECS: float = 5.0


class MetricsTracker:
    """Tracks named metrics with a sliding window for computing statistics."""

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self._data: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

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

    def std(self, name: str) -> float:
        """Compute standard deviation of a metric over the window."""
        values = self._data.get(name)
        if not values or len(values) < 2:
            return 0.0
        mu = sum(values) / len(values)
        variance = sum((v - mu) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def min(self, name: str) -> float:
        """Return the minimum value in the window."""
        values = self._data.get(name)
        if not values:
            return 0.0
        return builtins.min(values)

    def max(self, name: str) -> float:
        """Return the maximum value in the window."""
        values = self._data.get(name)
        if not values:
            return 0.0
        return builtins.max(values)

    def summary(self, name: str) -> dict[str, float]:
        """Return a full summary dict: mean, std, min, max, count."""
        return {
            "mean": self.mean(name),
            "std": self.std(name),
            "min": self.min(name),
            "max": self.max(name),
            "count": float(self.count(name)),
        }

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._data.clear()


def fetch_prometheus_metrics(
    url: str,
    *,
    timeout_secs: float = DEFAULT_METRICS_FETCH_TIMEOUT_SECS,
) -> str:
    """GET the Prometheus text-format scrape body from ``url``.

    Raises ``urllib.error.URLError`` (or subclass) on connection
    failure / non-2xx status. Callers that want a soft-failure path
    should wrap this in their own ``try``.
    """
    with urllib.request.urlopen(url, timeout=timeout_secs) as resp:
        raw = resp.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[\-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$",
)


def _iter_metric_values(metrics_text: str, name: str) -> list[float]:
    """All numeric values for ``name`` in the scrape body (ignores labels)."""
    out: list[float] = []
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if match is None:
            continue
        if match.group("name") != name:
            continue
        raw_value = match.group("value")
        try:
            out.append(float(raw_value))
        except ValueError:
            continue
    return out


def scrape_counter(metrics_text: str, name: str) -> float:
    """Sum every value of counter ``name`` in the scrape body.

    Sums across labeled variants (the canonical Prometheus way to
    aggregate a labeled counter). Returns ``0.0`` if the counter
    isn't present.
    """
    return sum(_iter_metric_values(metrics_text, name))


def scrape_gauge(metrics_text: str, name: str) -> float | None:
    """Latest value of gauge ``name`` in the scrape body.

    Returns ``None`` (NOT 0.0) if the gauge is absent, so callers can
    distinguish "metric not yet emitted" from "metric is zero".
    """
    values = _iter_metric_values(metrics_text, name)
    return values[-1] if values else None
