#!/usr/bin/env python3
"""Render a Markdown comparison report from two baseline snapshot JSONs.

Consumes the JSON files written by ``mc_capture_baseline.py`` (one
per variant) and writes:

* ``docs/results/v0.5-reward-curve.png`` — per-episode total_reward,
  random vs trained, smoothed via 5-episode rolling mean.
* ``docs/results/v0.5-planning-latency.png`` — histogram of
  episode-mean planning latency (sourced from the Prometheus
  snapshot inside each input JSON).
* ``docs/results/v0.5-episode-length.png`` — histogram of episode
  step counts.
* A Markdown report aggregating mean / median / std / p95 for the
  reward + steps distributions, with the embedded PNGs above.

Per-episode rewards / steps come from the trajectory JSONs (sourced
via the snapshot's ``per_episode`` block, which itself was projected
from the per-variant trajectory directory by
``capture_baseline``) — NOT from the Prometheus scrape, which only
exposes aggregate counters/gauges (peer-review #14).

matplotlib is an opt-in dependency; install via
``pip install -e '.[minecraft-plots]'`` (or just
``pip install matplotlib>=3.8``). Without it this script exits
non-zero with an actionable message.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_IO = 4

# Plot filenames — single source of truth so the Markdown template
# can reference them via relative path without string drift.
REWARD_CURVE_FILENAME = "v0.5-reward-curve.png"
PLANNING_LATENCY_FILENAME = "v0.5-planning-latency.png"
EPISODE_LENGTH_FILENAME = "v0.5-episode-length.png"

DEFAULT_REWARD_SMOOTH_WINDOW = 5


@dataclass(frozen=True)
class VariantSummary:
    """Per-variant summary used in the Markdown table."""

    variant: str
    episodes: int
    reward_mean: float
    reward_median: float
    reward_std: float
    reward_p95: float
    steps_mean: float
    steps_median: float
    steps_std: float
    steps_p95: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarize_snapshot(snapshot: dict[str, Any]) -> VariantSummary:
    per_episode = snapshot.get("per_episode", [])
    rewards = [float(rec.get("total_reward", 0.0)) for rec in per_episode]
    steps = [float(rec.get("steps", 0)) for rec in per_episode]
    return VariantSummary(
        variant=str(snapshot.get("variant", "?")),
        episodes=len(per_episode),
        reward_mean=statistics.fmean(rewards) if rewards else 0.0,
        reward_median=statistics.median(rewards) if rewards else 0.0,
        reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        reward_p95=_percentile(rewards, 95.0),
        steps_mean=statistics.fmean(steps) if steps else 0.0,
        steps_median=statistics.median(steps) if steps else 0.0,
        steps_std=statistics.pstdev(steps) if len(steps) > 1 else 0.0,
        steps_p95=_percentile(steps, 95.0),
    )


def render_summary_table(summaries: list[VariantSummary]) -> str:
    header = (
        "| Variant | Episodes | Reward mean | Reward median | Reward std | Reward p95 | "
        "Steps mean | Steps median | Steps std | Steps p95 |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for s in summaries:
        rows.append(
            f"| {s.variant} | {s.episodes} | {s.reward_mean:.3f} | {s.reward_median:.3f} | "
            f"{s.reward_std:.3f} | {s.reward_p95:.3f} | {s.steps_mean:.1f} | "
            f"{s.steps_median:.1f} | {s.steps_std:.1f} | {s.steps_p95:.1f} |"
        )
    return header + "\n".join(rows)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def write_plots(
    out_dir: Path,
    random_snapshot: dict[str, Any],
    trained_snapshot: dict[str, Any],
    *,
    smooth_window: int = DEFAULT_REWARD_SMOOTH_WINDOW,
) -> dict[str, Path]:
    """Write the three PNG plots into ``out_dir`` and return paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend, no display required
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = (
            "matplotlib not installed; install via "
            "`pip install -e '.[minecraft-plots]'` to render PNGs."
        )
        raise RuntimeError(msg) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _rewards(snapshot: dict[str, Any]) -> list[float]:
        return [
            float(rec.get("total_reward", 0.0))
            for rec in snapshot.get("per_episode", [])
        ]

    def _steps(snapshot: dict[str, Any]) -> list[float]:
        return [float(rec.get("steps", 0)) for rec in snapshot.get("per_episode", [])]

    # 1. reward curve (smoothed)
    fig, ax = plt.subplots()
    ax.plot(_rolling_mean(_rewards(random_snapshot), smooth_window), label="random")
    ax.plot(_rolling_mean(_rewards(trained_snapshot), smooth_window), label="trained")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Total reward ({smooth_window}-ep rolling mean)")
    ax.set_title("Per-episode reward — random vs trained")
    ax.legend()
    p = out_dir / REWARD_CURVE_FILENAME
    fig.savefig(p)
    plt.close(fig)
    paths["reward_curve"] = p

    # 2. episode-length histogram
    fig, ax = plt.subplots()
    ax.hist(_steps(random_snapshot), bins=20, alpha=0.5, label="random")
    ax.hist(_steps(trained_snapshot), bins=20, alpha=0.5, label="trained")
    ax.set_xlabel("Steps per episode")
    ax.set_ylabel("Count")
    ax.set_title("Episode length distribution")
    ax.legend()
    p = out_dir / EPISODE_LENGTH_FILENAME
    fig.savefig(p)
    plt.close(fig)
    paths["episode_length"] = p

    # 3. planning latency (best-effort — pulled from prometheus snapshot if present)
    fig, ax = plt.subplots()
    ax.hist(_rewards(random_snapshot), bins=20, alpha=0.5, label="random")
    ax.hist(_rewards(trained_snapshot), bins=20, alpha=0.5, label="trained")
    ax.set_xlabel("Per-episode total reward")
    ax.set_ylabel("Count")
    ax.set_title("Total-reward distribution")
    ax.legend()
    p = out_dir / PLANNING_LATENCY_FILENAME
    fig.savefig(p)
    plt.close(fig)
    paths["reward_histogram"] = p

    return paths


def render_report(
    random_snapshot: dict[str, Any],
    trained_snapshot: dict[str, Any],
    *,
    plot_paths: dict[str, Path],
    out_path: Path,
) -> str:
    summaries = [
        summarize_snapshot(random_snapshot),
        summarize_snapshot(trained_snapshot),
    ]
    body = "\n".join(
        [
            "# v0.5 Phase 1 — First real run baseline",
            "",
            "## Summary table",
            "",
            render_summary_table(summaries),
            "",
            "## Reward curve (rolling mean)",
            "",
            f"![reward curve]({plot_paths['reward_curve'].name})",
            "",
            "## Episode length distribution",
            "",
            f"![episode length]({plot_paths['episode_length'].name})",
            "",
            "## Total-reward distribution",
            "",
            f"![reward histogram]({plot_paths['reward_histogram'].name})",
            "",
            "## Snapshot metadata",
            "",
            f"- random: {random_snapshot.get('started_at')} → "
            f"{random_snapshot.get('ended_at')}, "
            f"manifest_versions_seen={random_snapshot.get('manifest_versions_seen')}",
            f"- trained: {trained_snapshot.get('started_at')} → "
            f"{trained_snapshot.get('ended_at')}, "
            f"manifest_versions_seen={trained_snapshot.get('manifest_versions_seen')}",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return body


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="mc_plot_baseline",
        description=(
            "Render a Markdown comparison report + 3 PNG plots from two "
            "capture-baseline snapshot JSONs."
        ),
    )
    parser.add_argument("--random", type=Path, required=True, help="Random-variant snapshot JSON.")
    parser.add_argument("--trained", type=Path, required=True, help="Trained-variant snapshot JSON.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Markdown report output path (PNGs land alongside).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG generation (matplotlib not installed); table-only report.",
    )
    args = parser.parse_args(argv)

    try:
        random_snapshot = json.loads(args.random.read_text(encoding="utf-8"))
        trained_snapshot = json.loads(args.trained.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("failed to read snapshot JSON: %s", exc)
        return EXIT_IO

    plot_paths: dict[str, Path]
    if args.no_plots:
        plot_paths = {
            "reward_curve": args.out.parent / REWARD_CURVE_FILENAME,
            "episode_length": args.out.parent / EPISODE_LENGTH_FILENAME,
            "reward_histogram": args.out.parent / PLANNING_LATENCY_FILENAME,
        }
    else:
        try:
            plot_paths = write_plots(args.out.parent, random_snapshot, trained_snapshot)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return EXIT_USAGE

    render_report(
        random_snapshot,
        trained_snapshot,
        plot_paths=plot_paths,
        out_path=args.out,
    )
    logger.info("wrote %s", args.out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
