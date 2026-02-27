"""test_bench_compare.py — Unit tests for scripts/bench_compare.py.

Tests are pure-Python and have zero external dependencies beyond the
standard library and pytest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make the scripts/ directory importable regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from bench_compare import (  # noqa: E402 — after sys.path setup
    BenchResult,
    _load_bench_results,
    compare,
    main,
    write_markdown_report,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal Criterion estimates.json format
# ---------------------------------------------------------------------------

_SAMPLE_ESTIMATES: dict = {
    "mean": {"point_estimate": 7500.0, "standard_error": 50.0},
    "std_dev": {"point_estimate": 120.0, "standard_error": 5.0},
    "median": {"point_estimate": 7480.0, "standard_error": 45.0},
    "median_abs_dev": {"point_estimate": 80.0, "standard_error": 3.0},
}


def _write_estimates(
    criterion_dir: Path,
    bench_name: str,
    mean_ns: float,
    *,
    folder: str = "base",
) -> Path:
    """Create a minimal Criterion estimates.json for a given benchmark."""
    out = criterion_dir / bench_name / folder / "estimates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {**_SAMPLE_ESTIMATES}
    data["mean"]["point_estimate"] = mean_ns
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# BenchResult
# ---------------------------------------------------------------------------


class TestBenchResult:
    def test_from_estimates_extracts_mean(self) -> None:
        result = BenchResult.from_estimates("step_throughput", _SAMPLE_ESTIMATES)
        assert result.mean_ns == pytest.approx(7500.0)

    def test_mean_us_conversion(self) -> None:
        result = BenchResult("step", mean_ns=7500.0, std_ns=120.0)
        assert result.mean_us == pytest.approx(7.5)

    def test_repr_contains_name(self) -> None:
        result = BenchResult("my_bench", mean_ns=1000.0, std_ns=10.0)
        assert "my_bench" in repr(result)


# ---------------------------------------------------------------------------
# _load_bench_results
# ---------------------------------------------------------------------------


class TestLoadBenchResults:
    def test_loads_single_benchmark(self, tmp_path: Path) -> None:
        _write_estimates(tmp_path, "step_throughput/32x32", 7500.0)
        results = _load_bench_results(tmp_path)
        assert len(results) == 1
        assert "step_throughput/32x32" in results

    def test_loads_multiple_benchmarks(self, tmp_path: Path) -> None:
        for name, mean in [("bench_a", 1000.0), ("bench_b", 2000.0), ("bench_c", 3000.0)]:
            _write_estimates(tmp_path, name, mean)
        results = _load_bench_results(tmp_path)
        assert len(results) == 3

    def test_ignores_malformed_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "broken" / "base" / "estimates.json"
        bad_file.parent.mkdir(parents=True)
        bad_file.write_text("not json", encoding="utf-8")
        results = _load_bench_results(tmp_path)
        assert len(results) == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        results = _load_bench_results(tmp_path)
        assert results == {}

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        results = _load_bench_results(tmp_path / "does_not_exist")
        assert results == {}


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def _make_results(self, entries: dict[str, float]) -> dict[str, BenchResult]:
        return {k: BenchResult(k, v, std_ns=10.0) for k, v in entries.items()}

    def test_no_regression(self) -> None:
        current = self._make_results({"bench_a": 1000.0})
        baseline = self._make_results({"bench_a": 1000.0})
        records = compare(current, baseline, threshold_pct=10.0)
        assert len(records) == 1
        assert not records[0]["regressed"]

    def test_regression_detected(self) -> None:
        current = self._make_results({"bench_a": 1200.0})  # +20%
        baseline = self._make_results({"bench_a": 1000.0})
        records = compare(current, baseline, threshold_pct=10.0)
        assert records[0]["regressed"] is True
        assert records[0]["delta_pct"] == pytest.approx(20.0)

    def test_improvement_not_regression(self) -> None:
        current = self._make_results({"bench_a": 800.0})  # -20%
        baseline = self._make_results({"bench_a": 1000.0})
        records = compare(current, baseline, threshold_pct=10.0)
        assert not records[0]["regressed"]
        assert records[0]["delta_pct"] == pytest.approx(-20.0)

    def test_new_benchmark_skipped(self) -> None:
        """New benchmarks without a baseline entry should not appear in results."""
        current = self._make_results({"new_bench": 1000.0})
        baseline: dict[str, BenchResult] = {}
        records = compare(current, baseline, threshold_pct=10.0)
        assert records == []

    def test_missing_current_benchmark_skipped(self) -> None:
        """Baseline benchmarks with no current counterpart are ignored."""
        current: dict[str, BenchResult] = {}
        baseline = self._make_results({"old_bench": 1000.0})
        records = compare(current, baseline, threshold_pct=10.0)
        assert records == []

    def test_exact_threshold_not_regression(self) -> None:
        """Exactly AT the threshold should not be a regression (> not >=)."""
        current = self._make_results({"bench_a": 1100.0})  # exactly +10%
        baseline = self._make_results({"bench_a": 1000.0})
        records = compare(current, baseline, threshold_pct=10.0)
        assert not records[0]["regressed"]

    def test_just_above_threshold_is_regression(self) -> None:
        current = self._make_results({"bench_a": 1101.0})  # +10.1%
        baseline = self._make_results({"bench_a": 1000.0})
        records = compare(current, baseline, threshold_pct=10.0)
        assert records[0]["regressed"] is True

    def test_multiple_benchmarks_partial_regression(self) -> None:
        current = self._make_results({"ok": 1000.0, "bad": 2000.0})
        baseline = self._make_results({"ok": 1000.0, "bad": 1000.0})
        records = compare(current, baseline, threshold_pct=10.0)
        by_name = {r["name"]: r for r in records}
        assert not by_name["ok"]["regressed"]
        assert by_name["bad"]["regressed"]


# ---------------------------------------------------------------------------
# write_markdown_report
# ---------------------------------------------------------------------------


class TestWriteMarkdownReport:
    def _records(self, delta_pct: float, name: str = "bench_a") -> list[dict]:
        return [
            {
                "name": name,
                "baseline_us": 7.5,
                "current_us": 7.5 * (1 + delta_pct / 100),
                "delta_pct": delta_pct,
                "regressed": delta_pct > 10.0,
            }
        ]

    def test_writes_file(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        write_markdown_report([], threshold_pct=10.0, output_path=report)
        assert report.exists()

    def test_no_regression_summary(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        write_markdown_report(self._records(delta_pct=5.0), 10.0, report)
        content = report.read_text(encoding="utf-8")
        assert "No regressions" in content

    def test_regression_summary(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        write_markdown_report(self._records(delta_pct=25.0), 10.0, report)
        content = report.read_text(encoding="utf-8")
        assert "regression" in content.lower()
        assert "bench_a" in content

    def test_markdown_table_header_present(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        write_markdown_report(self._records(5.0), 10.0, report)
        content = report.read_text(encoding="utf-8")
        assert "| Benchmark |" in content

    def test_empty_records_no_error(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        write_markdown_report([], 10.0, report)
        assert report.read_text(encoding="utf-8")  # non-empty


# ---------------------------------------------------------------------------
# main() integration (CLI)
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_no_baseline_exits_zero(self, tmp_path: Path) -> None:
        rc = main(
            [
                "--current",
                str(tmp_path / "criterion"),
                "--baseline",
                str(tmp_path / "no-baseline"),
                "--report",
                str(tmp_path / "report.md"),
            ]
        )
        assert rc == 0

    def test_no_current_exits_one(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        rc = main(
            [
                "--current",
                str(tmp_path / "criterion"),
                "--baseline",
                str(baseline),
                "--report",
                str(tmp_path / "report.md"),
            ]
        )
        assert rc == 1

    def test_regression_exits_one(self, tmp_path: Path) -> None:
        current_dir = tmp_path / "current"
        baseline_dir = tmp_path / "baseline"
        _write_estimates(current_dir, "step", 2000.0)  # +100%
        _write_estimates(baseline_dir, "step", 1000.0)
        rc = main(
            [
                "--current",
                str(current_dir),
                "--baseline",
                str(baseline_dir),
                "--threshold",
                "10",
                "--report",
                str(tmp_path / "report.md"),
            ]
        )
        assert rc == 1

    def test_within_threshold_exits_zero(self, tmp_path: Path) -> None:
        current_dir = tmp_path / "current"
        baseline_dir = tmp_path / "baseline"
        _write_estimates(current_dir, "step", 1050.0)  # +5%, within 10%
        _write_estimates(baseline_dir, "step", 1000.0)
        rc = main(
            [
                "--current",
                str(current_dir),
                "--baseline",
                str(baseline_dir),
                "--threshold",
                "10",
                "--report",
                str(tmp_path / "report.md"),
            ]
        )
        assert rc == 0
