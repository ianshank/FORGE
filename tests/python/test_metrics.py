"""Tests for forge.utils.metrics module."""
from __future__ import annotations

import logging

import pytest
from forge.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class TestMetricsTracker:
    """Tests for the MetricsTracker class."""

    @pytest.fixture()
    def tracker(self) -> MetricsTracker:
        """Return a default MetricsTracker."""
        return MetricsTracker()

    def test_record_and_mean(self, tracker: MetricsTracker) -> None:
        """Basic record and mean computation."""
        tracker.record("reward", 1.0)
        tracker.record("reward", 3.0)

        assert tracker.mean("reward") == pytest.approx(2.0)

    def test_windowed_mean(self) -> None:
        """Window size limits the number of values used for mean."""
        tracker = MetricsTracker(window_size=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record("x", v)

        # Only last 3 values (3, 4, 5) should be in the window
        assert tracker.mean("x") == pytest.approx(4.0)

    def test_latest_value(self, tracker: MetricsTracker) -> None:
        """latest() returns the most recently recorded value."""
        tracker.record("loss", 0.5)
        tracker.record("loss", 0.3)
        tracker.record("loss", 0.1)

        assert tracker.latest("loss") == pytest.approx(0.1)

    def test_count(self, tracker: MetricsTracker) -> None:
        """count() returns the number of recorded values."""
        tracker.record("steps", 10.0)
        tracker.record("steps", 20.0)

        assert tracker.count("steps") == 2

    def test_all_metrics(self, tracker: MetricsTracker) -> None:
        """all_metrics() returns a dict mapping names to means."""
        tracker.record("a", 1.0)
        tracker.record("a", 3.0)
        tracker.record("b", 10.0)

        result = tracker.all_metrics()

        assert result == {"a": pytest.approx(2.0), "b": pytest.approx(10.0)}

    def test_reset_clears(self, tracker: MetricsTracker) -> None:
        """reset() removes all tracked metrics."""
        tracker.record("x", 1.0)
        tracker.reset()

        assert tracker.count("x") == 0
        assert tracker.all_metrics() == {}

    def test_missing_metric(self, tracker: MetricsTracker) -> None:
        """Mean of a nonexistent metric returns 0.0."""
        assert tracker.mean("nonexistent") == 0.0

    def test_single_value(self, tracker: MetricsTracker) -> None:
        """Mean of a single value equals that value."""
        tracker.record("solo", 42.0)

        assert tracker.mean("solo") == pytest.approx(42.0)

    def test_custom_window_size(self) -> None:
        """Non-default window size is respected."""
        tracker = MetricsTracker(window_size=2)
        tracker.record("val", 1.0)
        tracker.record("val", 2.0)
        tracker.record("val", 100.0)

        # Only last 2 values (2.0, 100.0) in window
        assert tracker.mean("val") == pytest.approx(51.0)
        assert tracker.count("val") == 2

    def test_latest_missing_metric(self, tracker: MetricsTracker) -> None:
        """latest() on nonexistent metric returns 0.0."""
        assert tracker.latest("missing") == 0.0
