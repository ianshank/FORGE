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

Only evidential episodes feed the aggregates below — the summary
table AND the three plots share the same filter via
``_evidential_records``, so a record excluded from one cannot still
shift the other. A record whose protocol-error count is non-zero,
whose observation dimension disagrees with the handshake, or — for
any record that is not a confirmed natural terminal, truncated or
ambiguous alike — whose step count falls short of
``MIN_EVIDENTIAL_STEPS_IF_TRUNCATED``, did not measure the system and
is excluded before any mean, median, deviation, curve, or histogram
is computed. See ``openspec/changes/refuse-non-evidential-aggregates/``.

matplotlib is an opt-in dependency; install via
``pip install -e '.[minecraft-plots]'`` (or just
``pip install matplotlib>=3.8``). Without it this script exits
non-zero with an actionable message.
"""

from __future__ import annotations

import argparse
import importlib
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
EXIT_INSUFFICIENT_EVIDENCE = 3
EXIT_IO = 4

# Plot filenames — single source of truth so the Markdown template
# can reference them via relative path without string drift.
REWARD_CURVE_FILENAME = "v0.5-reward-curve.png"
REWARD_HISTOGRAM_FILENAME = "v0.5-reward-histogram.png"
EPISODE_LENGTH_FILENAME = "v0.5-episode-length.png"
# Backwards-compat alias for the misnamed v0.5-Phase-1-pre-review
# filename. Kept so a future Prometheus-sourced planning-latency
# plot can re-use the symbol; the actual file written today is the
# reward histogram (see write_plots()).
PLANNING_LATENCY_FILENAME = REWARD_HISTOGRAM_FILENAME

DEFAULT_REWARD_SMOOTH_WINDOW = 5

# Pinned floors for evidential-record filtering. Lowering either one
# changes what counts as evidence, so each is pinned by a paired test
# in tests/python/test_mc_plot_baseline_unit.py that must be edited in
# the same change — the review sees the lowering rather than it
# passing silently. See openspec/changes/refuse-non-evidential-aggregates/.
MIN_EVIDENTIAL_STEPS_IF_TRUNCATED = 5
MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON = 3


@dataclass(frozen=True)
class VariantSummary:
    """Per-variant summary used in the Markdown table.

    ``reward_*``/``steps_*`` are computed over evidential episodes
    only and are ``None`` when there are zero of them — a mean of no
    episodes is not a number and must not be silently reported as 0.0.
    """

    variant: str
    episodes: int
    evidential_episodes: int
    excluded_episodes: int
    missing_evidentiality_signals: bool
    reward_mean: float | None
    reward_median: float | None
    reward_std: float | None
    reward_p95: float | None
    steps_mean: float | None
    steps_median: float | None
    steps_std: float | None
    steps_p95: float | None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _is_evidential(record: dict[str, Any], hello_obs_dim: Any) -> bool:
    """Whether ``record`` measured the system rather than a broken run.

    A record is evidential only when its protocol-error count is
    zero, its observation dimension equals the handshake's, and — for
    any record that is not a *confirmed* natural terminal — its
    realised step count meets ``MIN_EVIDENTIAL_STEPS_IF_TRUNCATED``.
    The floor is keyed on ``terminated`` rather than ``truncated``:
    the producer's own truncation loop can exit with both flags false
    when its driver-side step budget runs out before the environment
    ever reports either outcome, and that ambiguous case must not
    bypass the floor just because it isn't explicitly ``truncated``. A
    confirmed natural terminal (``terminated`` true) is evidential
    regardless of length — a twelve-step terminal is as much evidence
    as a two-hundred-step one. A record missing the signals needed to
    decide any of this is treated as non-evidential rather than
    assumed clean, matching the documented CLI producer whose records
    carry no protocol-error count at all.
    """
    protocol_errors = record.get("protocol_errors")
    if protocol_errors is None or protocol_errors != 0:
        return False
    obs_dim = record.get("obs_dim")
    if obs_dim is None or obs_dim != hello_obs_dim:
        return False
    if not record.get("terminated"):
        steps = record.get("steps")
        if steps is None or steps < MIN_EVIDENTIAL_STEPS_IF_TRUNCATED:
            return False
    return True


def _evidential_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """The subset of ``snapshot["per_episode"]`` that measured the system.

    Shared by ``summarize_snapshot`` and ``write_plots`` so a record
    excluded from the summary table cannot still slip into a plotted
    mean, curve, or histogram.
    """
    hello_obs_dim = (snapshot.get("hello") or {}).get("obs_dim")
    return [
        rec for rec in snapshot.get("per_episode", []) if _is_evidential(rec, hello_obs_dim)
    ]


def _evidential_rewards(snapshot: dict[str, Any]) -> list[float]:
    return [float(rec.get("total_reward", 0.0)) for rec in _evidential_records(snapshot)]


def _evidential_steps(snapshot: dict[str, Any]) -> list[float]:
    return [float(rec.get("steps", 0)) for rec in _evidential_records(snapshot)]


def _missing_evidentiality_signals(snapshot: dict[str, Any]) -> bool:
    """Whether this snapshot's own schema cannot support evidentiality.

    The documented ``capture_baseline.py`` CLI path's ``BaselineRecord``
    carries no ``protocol_errors`` field and its snapshot carries no
    ``hello`` block at all, so every one of its records is excluded
    regardless of how many real episodes were captured. That is a
    schema gap in the producer, not evidence the run was broken, and
    the refusal message should say which one it is rather than
    reporting a bare exclusion count that reads as "every episode
    failed."
    """
    per_episode = snapshot.get("per_episode", [])
    if not per_episode:
        return False
    if (snapshot.get("hello") or {}).get("obs_dim") is None:
        return True
    return all(rec.get("protocol_errors") is None for rec in per_episode)


def summarize_snapshot(snapshot: dict[str, Any]) -> VariantSummary:
    per_episode = snapshot.get("per_episode", [])
    rewards = _evidential_rewards(snapshot)
    steps = _evidential_steps(snapshot)
    evidential_count = len(_evidential_records(snapshot))
    return VariantSummary(
        variant=str(snapshot.get("variant", "?")),
        episodes=len(per_episode),
        evidential_episodes=evidential_count,
        excluded_episodes=len(per_episode) - evidential_count,
        missing_evidentiality_signals=_missing_evidentiality_signals(snapshot),
        reward_mean=statistics.fmean(rewards) if rewards else None,
        reward_median=statistics.median(rewards) if rewards else None,
        reward_std=statistics.pstdev(rewards) if rewards else None,
        reward_p95=_percentile(rewards, 95.0),
        steps_mean=statistics.fmean(steps) if steps else None,
        steps_median=statistics.median(steps) if steps else None,
        steps_std=statistics.pstdev(steps) if steps else None,
        steps_p95=_percentile(steps, 95.0),
    )


def _evidential_floor_violation(summaries: list[VariantSummary]) -> str | None:
    """Error message naming the shortfall, or ``None`` if every summary
    clears ``MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON``."""
    shortfalls = []
    for s in summaries:
        if s.evidential_episodes >= MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON:
            continue
        note = f"{s.variant}: {s.evidential_episodes} evidential, {s.excluded_episodes} excluded"
        if s.missing_evidentiality_signals:
            note += (
                " (its records carry none of the required evidentiality fields — "
                "this looks like a producer schema gap, not necessarily a broken run)"
            )
        shortfalls.append(note)
    if not shortfalls:
        return None
    return (
        "refusing to render a comparison: need at least "
        f"{MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON} evidential episodes per variant "
        f"({'; '.join(shortfalls)})"
    )


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def render_summary_table(summaries: list[VariantSummary]) -> str:
    header = (
        "| Variant | Episodes | Evidential | Excluded | Reward mean | Reward median | "
        "Reward std | Reward p95 | Steps mean | Steps median | Steps std | Steps p95 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = [
        (
            f"| {s.variant} | {s.episodes} | {s.evidential_episodes} | "
            f"{s.excluded_episodes} | {_fmt(s.reward_mean, '.3f')} | "
            f"{_fmt(s.reward_median, '.3f')} | {_fmt(s.reward_std, '.3f')} | "
            f"{_fmt(s.reward_p95, '.3f')} | {_fmt(s.steps_mean, '.1f')} | "
            f"{_fmt(s.steps_median, '.1f')} | {_fmt(s.steps_std, '.1f')} | "
            f"{_fmt(s.steps_p95, '.1f')} |"
        )
        for s in summaries
    ]
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


def write_plots(  # noqa: PLR0915
    out_dir: Path,
    random_snapshot: dict[str, Any],
    trained_snapshot: dict[str, Any],
    *,
    smooth_window: int = DEFAULT_REWARD_SMOOTH_WINDOW,
) -> dict[str, Path]:
    """Write the three PNG plots into ``out_dir`` and return paths."""
    try:
        # matplotlib is an opt-in dependency (declared under the
        # `minecraft-plots` extra in pyproject.toml). The dual
        # `import-not-found, unused-ignore` suppression covers both
        # cases — CI lint job runs without stubs (import-not-found
        # fires), local-dev with `pip install matplotlib` HAS stubs
        # (unused-ignore would fire without the second code). The
        # `Any` annotations keep mypy quiet about the dynamic
        # `matplotlib.use` + `plt.subplots()` return types.
        matplotlib: Any = importlib.import_module("matplotlib")
        matplotlib.use("Agg")  # headless backend, no display required
        plt: Any = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        msg = (
            "matplotlib not installed; install via "
            "`pip install -e '.[minecraft-plots]'` to render PNGs."
        )
        raise RuntimeError(msg) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. reward curve (smoothed)
    fig, ax = plt.subplots()
    random_rewards = _evidential_rewards(random_snapshot)
    trained_rewards = _evidential_rewards(trained_snapshot)

    random_rolling = _rolling_mean(random_rewards, smooth_window)
    trained_rolling = _rolling_mean(trained_rewards, smooth_window)

    ax.plot(random_rolling, label="random (rolling)", color="#1f77b4", alpha=0.6)
    ax.plot(trained_rolling, label="trained (rolling)", color="#ff7f0e", linewidth=2.5)

    if random_rewards:
        rand_mean = statistics.fmean(random_rewards)
        rand_std = statistics.pstdev(random_rewards) if len(random_rewards) > 1 else 0.0
        ax.axhline(rand_mean, color="#1f77b4", linestyle="--", label="random mean", alpha=0.8)

        # Shade random baseline standard deviation range
        max_len = max(len(random_rewards), len(trained_rewards))
        ax.fill_between(
            range(max_len),
            rand_mean - rand_std,
            rand_mean + rand_std,
            color="#1f77b4",
            alpha=0.1,
            label="random baseline ±1σ"  # noqa: RUF001
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Total reward ({smooth_window}-ep rolling mean)")
    ax.set_title("Per-episode reward — random vs trained progression")
    ax.legend(loc="best")
    p = out_dir / REWARD_CURVE_FILENAME
    fig.savefig(p)
    plt.close(fig)
    paths["reward_curve"] = p

    # 2. episode-length histogram
    fig, ax = plt.subplots()
    ax.hist(
        _evidential_steps(random_snapshot), bins=20, alpha=0.5, label="random", color="#1f77b4"
    )
    ax.hist(
        _evidential_steps(trained_snapshot), bins=20, alpha=0.5, label="trained", color="#ff7f0e"
    )
    ax.set_xlabel("Steps per episode")
    ax.set_ylabel("Count")
    ax.set_title("Episode length distribution")
    ax.legend()
    p = out_dir / EPISODE_LENGTH_FILENAME
    fig.savefig(p)
    plt.close(fig)
    paths["episode_length"] = p

    # 3. reward histogram
    fig, ax = plt.subplots()
    ax.hist(random_rewards, bins=20, alpha=0.5, label="random", color="#1f77b4")
    ax.hist(trained_rewards, bins=20, alpha=0.5, label="trained", color="#ff7f0e")
    ax.set_xlabel("Per-episode total reward")
    ax.set_ylabel("Count")
    ax.set_title("Total-reward distribution")
    ax.legend()
    p = out_dir / REWARD_HISTOGRAM_FILENAME
    fig.savefig(p)
    plt.close(fig)
    paths["reward_histogram"] = p

    return paths


def render_report(
    random_snapshot: dict[str, Any],
    trained_snapshot: dict[str, Any],
    summaries: list[VariantSummary],
    *,
    plot_paths: dict[str, Path],
    out_path: Path,
) -> str:
    body = "\n".join(
        [
            "# FORGE v0.5 Trained vs. Random Comparative Report",
            "",
            "This comparative report evaluates the learning progression and performance of a trained **MuZero MC** agent against a **Random Action Baseline** in Minecraft.",
            "",
            "## Summary table",
            "",
            render_summary_table(summaries),
            "",
            "> [!TIP]",
            "> **Learning Active**: A successful training run is demonstrated when the trained agent's average reward rises significantly above the random baseline's flat mean.",
            "",
            "## Reward progression curve",
            "",
            f"![reward curve]({plot_paths['reward_curve'].name})",
            "",
            "> [!IMPORTANT]",
            "> The trained agent curve (orange) represents the agent's reward progression over successive self-play training episodes. The dashed blue line represents the flat average reward of the random action baseline, with the shaded light-blue region representing the standard deviation of random rollouts.",
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
    parser.add_argument(
        "--trained", type=Path, required=True, help="Trained-variant snapshot JSON."
    )
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

    summaries = [summarize_snapshot(random_snapshot), summarize_snapshot(trained_snapshot)]
    floor_violation = _evidential_floor_violation(summaries)
    if floor_violation is not None:
        logger.error("%s", floor_violation)
        return EXIT_INSUFFICIENT_EVIDENCE

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
        summaries,
        plot_paths=plot_paths,
        out_path=args.out,
    )
    logger.info("wrote %s", args.out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
