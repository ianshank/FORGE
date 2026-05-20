"""Unit tests for the v0.5 baseline plotter.

Lives flat under ``tests/python/`` (no `tests/python/scripts/`
subdir — peer-review #13: that directory doesn't exist in the
repo, and other script-coverage tests like
``tests/python/test_check_zero_alloc.py`` follow the flat layout).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.mc_plot_baseline import (
    EPISODE_LENGTH_FILENAME,
    PLANNING_LATENCY_FILENAME,
    REWARD_CURVE_FILENAME,
    VariantSummary,
    main,
    render_report,
    render_summary_table,
    summarize_snapshot,
)


def _snapshot(variant: str, rewards: list[float], steps: list[int]) -> dict[str, Any]:
    return {
        "variant": variant,
        "started_at": "2026-05-20T00:00:00+00:00",
        "ended_at": "2026-05-20T00:10:00+00:00",
        "manifest_versions_seen": [0] if variant == "random" else [1, 2, 3],
        "per_episode": [
            {
                "episode_id": f"ep-{i:06d}",
                "total_reward": rewards[i - 1],
                "steps": steps[i - 1],
                "terminated": False,
                "truncated": True,
                "obs_dim": 920,
                "action_dim": 12,
                "schema_id": "abc",
            }
            for i in range(1, len(rewards) + 1)
        ],
    }


def test_summarize_snapshot_computes_stats() -> None:
    snap = _snapshot("random", [1.0, 2.0, 3.0, 4.0, 5.0], [10, 20, 30, 40, 50])
    summary = summarize_snapshot(snap)
    assert summary.variant == "random"
    assert summary.episodes == 5
    assert summary.reward_mean == pytest.approx(3.0)
    assert summary.reward_median == 3.0
    assert summary.steps_mean == pytest.approx(30.0)


def test_summarize_snapshot_handles_empty_per_episode() -> None:
    snap = {"variant": "random", "per_episode": []}
    summary = summarize_snapshot(snap)
    assert summary.episodes == 0
    assert summary.reward_mean == 0.0
    assert summary.reward_std == 0.0


def test_render_summary_table_pins_columns() -> None:
    summaries = [
        VariantSummary(
            variant="random",
            episodes=10,
            reward_mean=1.0,
            reward_median=1.0,
            reward_std=0.5,
            reward_p95=2.0,
            steps_mean=50.0,
            steps_median=48.0,
            steps_std=10.0,
            steps_p95=70.0,
        ),
    ]
    table = render_summary_table(summaries)
    assert "| Variant |" in table
    assert "| random | 10 |" in table
    assert "Reward mean" in table


def test_render_report_emits_markdown_with_plot_references(tmp_path: Path) -> None:
    random_snap = _snapshot("random", [1.0, 2.0, 3.0], [10, 20, 30])
    trained_snap = _snapshot("trained", [5.0, 7.0, 9.0], [50, 60, 70])
    out = tmp_path / "report.md"
    plot_paths = {
        "reward_curve": tmp_path / REWARD_CURVE_FILENAME,
        "episode_length": tmp_path / EPISODE_LENGTH_FILENAME,
        "reward_histogram": tmp_path / PLANNING_LATENCY_FILENAME,
    }
    body = render_report(random_snap, trained_snap, plot_paths=plot_paths, out_path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == body
    assert "# v0.5 Phase 1 — First real run baseline" in body
    assert "## Summary table" in body
    assert REWARD_CURVE_FILENAME in body
    assert EPISODE_LENGTH_FILENAME in body
    assert PLANNING_LATENCY_FILENAME in body
    assert "| random |" in body
    assert "| trained |" in body


def test_main_no_plots_path_writes_table_only_report(tmp_path: Path) -> None:
    random_snap_path = tmp_path / "baseline_random.json"
    trained_snap_path = tmp_path / "baseline_trained.json"
    out_path = tmp_path / "report.md"
    random_snap_path.write_text(json.dumps(_snapshot("random", [1.0, 2.0], [10, 20])))
    trained_snap_path.write_text(json.dumps(_snapshot("trained", [5.0, 6.0], [50, 60])))

    rc = main(
        [
            "--random",
            str(random_snap_path),
            "--trained",
            str(trained_snap_path),
            "--out",
            str(out_path),
            "--no-plots",
        ]
    )
    assert rc == 0
    assert out_path.exists()
    body = out_path.read_text()
    assert "## Summary table" in body


def test_main_with_plots_generates_pngs(tmp_path: Path) -> None:
    # Skip when matplotlib isn't installed — the test still exercises
    # the importable-script surface but a PNG-free assertion path.
    pytest.importorskip("matplotlib")

    random_snap_path = tmp_path / "baseline_random.json"
    trained_snap_path = tmp_path / "baseline_trained.json"
    out_path = tmp_path / "report.md"
    random_snap_path.write_text(
        json.dumps(_snapshot("random", [1.0, 2.0, 3.0, 4.0, 5.0], [10, 20, 30, 40, 50]))
    )
    trained_snap_path.write_text(
        json.dumps(_snapshot("trained", [5.0, 6.0, 7.0, 8.0, 9.0], [50, 60, 70, 80, 90]))
    )

    rc = main(
        [
            "--random",
            str(random_snap_path),
            "--trained",
            str(trained_snap_path),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    for fname in (REWARD_CURVE_FILENAME, EPISODE_LENGTH_FILENAME, PLANNING_LATENCY_FILENAME):
        assert (tmp_path / fname).exists()
