"""Tests for forge.utils.metrics module."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from forge.utils.metrics import (
    DEFAULT_METRICS_FETCH_TIMEOUT_SECS,
    MetricsTracker,
    fetch_prometheus_metrics,
    scrape_counter,
    scrape_gauge,
)

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


class TestMetricsExtended:
    def test_std(self) -> None:
        tracker = MetricsTracker()
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            tracker.record("reward", v)
        assert abs(tracker.std("reward") - 2.0) < 0.01

    def test_min_max(self) -> None:
        tracker = MetricsTracker()
        for v in [3.0, 1.0, 4.0, 1.0, 5.0]:
            tracker.record("ep_len", v)
        assert tracker.min("ep_len") == 1.0
        assert tracker.max("ep_len") == 5.0

    def test_summary(self) -> None:
        tracker = MetricsTracker()
        for v in [10.0, 20.0, 30.0]:
            tracker.record("reward", v)
        s = tracker.summary("reward")
        assert s["mean"] == 20.0
        assert s["min"] == 10.0
        assert s["max"] == 30.0
        assert s["count"] == 3
        assert "std" in s

    def test_empty_std_returns_zero(self) -> None:
        tracker = MetricsTracker()
        assert tracker.std("missing") == 0.0

    def test_summary_per_tier(self) -> None:
        tracker = MetricsTracker()
        tracker.record("tier_1/success", 1.0)
        tracker.record("tier_1/success", 0.0)
        assert tracker.mean("tier_1/success") == 0.5


class TestPrometheusScrapeHelpers:
    """Tests for the v0.5 Prometheus scrape helpers hoisted from
    `tests/python/integration/test_minecraft_e2e.py` into
    `forge.utils.metrics`. The shim functions in the E2E test file
    delegate here, so a regression here silently inverts the
    `scrape_gauge` returns-None-vs-0.0 contract that callers like
    `capture_baseline.py::_summary_gauges` and the E2E test's
    `_scrape_gauge` shim rely on.
    """

    SAMPLE_SCRAPE = "\n".join(
        [
            "# HELP forge_mc_episode_total Episodes finished",
            "# TYPE forge_mc_episode_total counter",
            "forge_mc_episode_total 7",
            "# HELP forge_mc_model_version Manifest version currently loaded",
            "# TYPE forge_mc_model_version gauge",
            "forge_mc_model_version 3",
            "# Labeled counter — gets summed across variants",
            'forge_mc_protocol_errors_total{kind="planner"} 2',
            'forge_mc_protocol_errors_total{kind="env"} 5',
        ]
    )

    def test_scrape_counter_returns_unlabeled_value(self) -> None:
        assert scrape_counter(self.SAMPLE_SCRAPE, "forge_mc_episode_total") == 7.0

    def test_scrape_counter_sums_labeled_variants(self) -> None:
        # Canonical Prometheus way to aggregate a labeled counter
        # — drift here would silently undercount kind-specific errors.
        assert scrape_counter(self.SAMPLE_SCRAPE, "forge_mc_protocol_errors_total") == 7.0

    def test_scrape_counter_returns_zero_for_absent(self) -> None:
        # Absent counter is Prometheus's "never incremented" signal;
        # 0.0 is the correct default.
        assert scrape_counter(self.SAMPLE_SCRAPE, "forge_mc_nonexistent_total") == 0.0

    def test_scrape_gauge_returns_value_when_present(self) -> None:
        assert scrape_gauge(self.SAMPLE_SCRAPE, "forge_mc_model_version") == 3.0

    def test_scrape_gauge_returns_none_when_absent(self) -> None:
        # The KEY contract: absent gauge is `None`, NOT `0.0`. The
        # capture-baseline + E2E backwards-compat shim relies on
        # this so callers can distinguish "metric not yet emitted"
        # from "metric is zero".
        assert scrape_gauge(self.SAMPLE_SCRAPE, "forge_mc_nonexistent_gauge") is None

    def test_scrape_gauge_returns_zero_when_value_is_zero(self) -> None:
        # Reciprocal of the above: a gauge with value 0.0 must NOT
        # collapse to None.
        zero_gauge = "forge_mc_model_version 0"
        assert scrape_gauge(zero_gauge, "forge_mc_model_version") == 0.0

    def test_scrape_gauge_returns_latest_value_for_repeated_lines(self) -> None:
        # The bot's exporter occasionally writes a gauge multiple
        # times within a single scrape; the LAST value is the
        # current one per Prometheus convention.
        body = "\n".join(
            [
                "forge_mc_model_version 1",
                "forge_mc_model_version 2",
                "forge_mc_model_version 5",
            ]
        )
        assert scrape_gauge(body, "forge_mc_model_version") == 5.0

    def test_scrape_handles_special_float_values(self) -> None:
        # Prometheus text format allows NaN/+Inf/-Inf for gauges
        # tracking floating-point quantities. The scrape function
        # must parse them as Python floats without raising.
        body = "\n".join(
            [
                "forge_mc_latency NaN",
                "forge_mc_latency +Inf",
                "forge_mc_latency -Inf",
            ]
        )
        last = scrape_gauge(body, "forge_mc_latency")
        assert last is not None
        # `-Inf` is the most recent line; verify it round-tripped.
        import math

        assert math.isinf(last) and last < 0

    def test_fetch_prometheus_metrics_round_trips_response_body(self) -> None:
        """Smoke test against a mocked `urllib.request.urlopen`. No
        real HTTP — we just verify the body decoding and the
        timeout-arg threading.
        """
        body = "forge_mc_episode_total 42\n"

        class _FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def __enter__(self) -> _FakeResponse:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def read(self) -> bytes:
                return self._payload

        with patch(
            "forge.utils.metrics.urllib.request.urlopen",
            return_value=_FakeResponse(body.encode("utf-8")),
        ) as opener:
            result = fetch_prometheus_metrics("http://localhost:9090/metrics")
        assert result == body
        # Default timeout flowed through to urlopen.
        opener.assert_called_once()
        _args, kwargs = opener.call_args
        assert kwargs.get("timeout") == DEFAULT_METRICS_FETCH_TIMEOUT_SECS

    def test_fetch_prometheus_metrics_honours_explicit_timeout(self) -> None:
        with patch(
            "forge.utils.metrics.urllib.request.urlopen",
            return_value=MagicMock(
                __enter__=MagicMock(
                    return_value=MagicMock(
                        read=MagicMock(return_value=b""),
                    )
                ),
                __exit__=MagicMock(return_value=None),
            ),
        ) as opener:
            fetch_prometheus_metrics("http://localhost:9090/metrics", timeout_secs=2.5)
        _args, kwargs = opener.call_args
        assert kwargs.get("timeout") == 2.5
