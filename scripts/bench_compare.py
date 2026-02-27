#!/usr/bin/env python3
"""bench_compare.py — Criterion benchmark regression gate.

Compares the current Criterion JSON output against a saved baseline,
fails (exit code 1) if any benchmark has regressed beyond the threshold,
and writes a Markdown report to /tmp/bench_report.md for CI comments.

Usage::

    python scripts/bench_compare.py \\
        --current  target/criterion \\
        --baseline .bench-baseline/ \\
        --threshold 10 \\
        [--github-output]

Exit codes:
    0 — No regression or no baseline available
    1 — One or more benchmarks regressed beyond threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REPORT_PATH = Path("/tmp/bench_report.md")
_ESTIMATES_FILE = "estimates.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class BenchResult:
    """Parsed result from a single Criterion `estimates.json`."""

    def __init__(self, bench_name: str, mean_ns: float, std_ns: float) -> None:
        self.bench_name = bench_name
        self.mean_ns = mean_ns
        self.std_ns = std_ns

    @classmethod
    def from_estimates(cls, bench_name: str, data: dict[str, Any]) -> BenchResult:
        """Create from Criterion's `estimates.json` structure."""
        mean_ns: float = data["mean"]["point_estimate"]
        std_ns: float = data["std_dev"]["point_estimate"]
        return cls(bench_name=bench_name, mean_ns=mean_ns, std_ns=std_ns)

    @property
    def mean_us(self) -> float:
        """Mean in microseconds."""
        return self.mean_ns / 1_000.0

    def __repr__(self) -> str:
        return f"BenchResult({self.bench_name!r}, mean={self.mean_us:.2f}µs)"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _load_bench_results(criterion_dir: Path) -> dict[str, BenchResult]:
    """Walk a Criterion output directory and return {bench_name: BenchResult}."""
    results: dict[str, BenchResult] = {}
    for estimates_file in criterion_dir.rglob(_ESTIMATES_FILE):
        # Criterion layout: criterion/<group>/<bench>/base/estimates.json
        # We skip the "new" folder; only look at "base" (persisted reports)
        if estimates_file.parent.name not in {"base", "new"}:
            continue
        try:
            data: dict[str, Any] = json.loads(estimates_file.read_text(encoding="utf-8"))
            # Derive a human-readable name from directory structure
            parts = estimates_file.relative_to(criterion_dir).parts
            bench_name = "/".join(parts[:-2])  # strip base/estimates.json
            results[bench_name] = BenchResult.from_estimates(bench_name, data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping malformed estimates file %s: %s", estimates_file, exc)
    return results


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare(
    current: dict[str, BenchResult],
    baseline: dict[str, BenchResult],
    threshold_pct: float,
) -> list[dict[str, Any]]:
    """Compare current vs baseline; return list of regression records."""
    regressions: list[dict[str, Any]] = []
    for name, curr in current.items():
        if name not in baseline:
            logger.debug("New benchmark (no baseline): %s", name)
            continue
        base = baseline[name]
        delta_pct = (curr.mean_ns - base.mean_ns) / base.mean_ns * 100.0
        record: dict[str, Any] = {
            "name": name,
            "baseline_us": base.mean_us,
            "current_us": curr.mean_us,
            "delta_pct": delta_pct,
            "regressed": delta_pct > threshold_pct,
        }
        if record["regressed"]:
            logger.warning(
                "REGRESSION: %s %.2f µs → %.2f µs (+%.1f%% > %.1f%% threshold)",
                name,
                base.mean_us,
                curr.mean_us,
                delta_pct,
                threshold_pct,
            )
        regressions.append(record)
    return regressions


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _format_delta(delta_pct: float) -> str:
    """Return a coloured markdown badge for the delta percentage."""
    icon = "🔴" if delta_pct > 0 else "🟢"
    sign = "+" if delta_pct >= 0 else ""
    return f"{icon} `{sign}{delta_pct:.1f}%`"


def write_markdown_report(
    records: list[dict[str, Any]],
    threshold_pct: float,
    output_path: Path = _REPORT_PATH,
) -> None:
    """Write a Markdown table summarising benchmark comparisons."""
    lines: list[str] = [
        "| Benchmark | Baseline | Current | Delta |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| `{rec['name']}` "
        f"| {rec['baseline_us']:.2f} µs "
        f"| {rec['current_us']:.2f} µs "
        f"| {_format_delta(rec['delta_pct'])} |"
        for rec in sorted(records, key=lambda r: -abs(r["delta_pct"]))
    )

    regressions = [r for r in records if r["regressed"]]
    summary = (
        f"\n✅ **No regressions** (threshold: {threshold_pct:.0f}%)"
        if not regressions
        else (
            f"\n❌ **{len(regressions)} regression(s) detected** "
            f"(threshold: {threshold_pct:.0f}%)\n\n"
            + "\n".join(f"- `{r['name']}` (+{r['delta_pct']:.1f}%)" for r in regressions)
        )
    )

    output_path.write_text("\n".join(lines) + summary + "\n", encoding="utf-8")
    logger.info("Benchmark report written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("target/criterion"),
        help="Path to current Criterion output dir (default: target/criterion)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(".bench-baseline"),
        help="Path to saved baseline dir (default: .bench-baseline)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold in percent (default: 10.0)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_REPORT_PATH,
        help=f"Path to write Markdown report (default: {_REPORT_PATH})",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write ::error:: annotations for GitHub Actions",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns exit code."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(name)s: %(message)s",
    )

    if not args.baseline.exists():
        logger.info("No baseline found at %s — skipping regression check.", args.baseline)
        return 0

    if not args.current.exists():
        logger.error("Current Criterion output not found at %s", args.current)
        return 1

    current = _load_bench_results(args.current)
    baseline = _load_bench_results(args.baseline)

    if not current:
        logger.warning("No benchmark results parsed from %s", args.current)
        return 0

    logger.info(
        "Comparing %d benchmark(s) against baseline (%d available)",
        len(current),
        len(baseline),
    )

    records = compare(current, baseline, threshold_pct=args.threshold)
    write_markdown_report(records, threshold_pct=args.threshold, output_path=args.report)

    regressions = [r for r in records if r["regressed"]]
    if regressions and args.github_output:
        for reg in regressions:
            print(
                f"::error file=crates/forge-bench/benches/::Benchmark regression: "
                f"{reg['name']} regressed by {reg['delta_pct']:.1f}% "
                f"(baseline {reg['baseline_us']:.2f}µs → current {reg['current_us']:.2f}µs)"
            )

    if regressions:
        logger.error(
            "%d benchmark(s) regressed beyond %.1f%% threshold.",
            len(regressions),
            args.threshold,
        )
        return 1

    logger.info("All benchmarks within %.1f%% threshold.", args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
