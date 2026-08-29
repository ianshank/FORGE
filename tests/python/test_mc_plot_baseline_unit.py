"""Unit tests for the v0.5 baseline plotter.

Lives flat under ``tests/python/`` (no `tests/python/scripts/`
subdir — peer-review #13: that directory doesn't exist in the
repo, and other script-coverage tests like
``tests/python/test_check_zero_alloc.py`` follow the flat layout).
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — runtime use in pytest tmp_path fixture
from typing import Any

import pytest

from scripts.mc_plot_baseline import (
    EPISODE_LENGTH_FILENAME,
    EXIT_INSUFFICIENT_EVIDENCE,
    MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON,
    MIN_EVIDENTIAL_STEPS_IF_TRUNCATED,
    PLANNING_LATENCY_FILENAME,
    REWARD_CURVE_FILENAME,
    VariantSummary,
    _evidential_rewards,
    _evidential_steps,
    main,
    render_report,
    render_summary_table,
    summarize_snapshot,
)

_OBS_DIM = 920


def _episode(
    idx: int,
    reward: float,
    steps: int,
    *,
    obs_dim: int = _OBS_DIM,
    protocol_errors: int | None = 0,
    terminated: bool = True,
    truncated: bool = False,
) -> dict[str, Any]:
    """One per-episode record.

    Defaults produce an evidential record: no protocol error, an
    observation dimension matching the handshake, and a natural
    terminal (not truncated, so the step floor doesn't apply).
    """
    record: dict[str, Any] = {
        "episode_id": f"ep-{idx:06d}",
        "total_reward": reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "obs_dim": obs_dim,
        "action_dim": 12,
        "schema_id": "abc",
    }
    if protocol_errors is not None:
        record["protocol_errors"] = protocol_errors
    return record


def _snapshot(
    variant: str,
    rewards: list[float],
    steps: list[int],
    *,
    obs_dim: int = _OBS_DIM,
    protocol_errors: int | list[int] = 0,
    terminated: bool | list[bool] = True,
    truncated: bool | list[bool] = False,
    hello_obs_dim: int = _OBS_DIM,
) -> dict[str, Any]:
    n = len(rewards)
    pe_list = protocol_errors if isinstance(protocol_errors, list) else [protocol_errors] * n
    te_list = terminated if isinstance(terminated, list) else [terminated] * n
    tr_list = truncated if isinstance(truncated, list) else [truncated] * n
    return {
        "variant": variant,
        "started_at": "2026-05-20T00:00:00+00:00",
        "ended_at": "2026-05-20T00:10:00+00:00",
        "manifest_versions_seen": [0] if variant == "random" else [1, 2, 3],
        "hello": {"obs_dim": hello_obs_dim},
        "per_episode": [
            _episode(
                i,
                rewards[i - 1],
                steps[i - 1],
                obs_dim=obs_dim,
                protocol_errors=pe_list[i - 1],
                terminated=te_list[i - 1],
                truncated=tr_list[i - 1],
            )
            for i in range(1, n + 1)
        ],
    }


def test_summarize_snapshot_computes_stats_over_evidential_records() -> None:
    snap = _snapshot("random", [1.0, 2.0, 3.0, 4.0, 5.0], [10, 20, 30, 40, 50])
    summary = summarize_snapshot(snap)
    assert summary.variant == "random"
    assert summary.episodes == 5
    assert summary.evidential_episodes == 5
    assert summary.excluded_episodes == 0
    assert summary.reward_mean == pytest.approx(3.0)
    assert summary.reward_median == 3.0
    assert summary.steps_mean == pytest.approx(30.0)


def test_summarize_snapshot_handles_empty_per_episode() -> None:
    snap = {"variant": "random", "hello": {"obs_dim": _OBS_DIM}, "per_episode": []}
    summary = summarize_snapshot(snap)
    assert summary.episodes == 0
    assert summary.evidential_episodes == 0
    assert summary.excluded_episodes == 0
    assert summary.reward_mean is None
    assert summary.reward_std is None


def test_summarize_snapshot_excludes_records_with_protocol_errors() -> None:
    # Reproduces the shape of the committed v0.5 baseline artifacts:
    # every record reports a protocol error, so none measured the
    # system even though the snapshot has 30 rows.
    snap = _snapshot(
        "random",
        rewards=[0.0] * 30,
        steps=[1] * 30,
        protocol_errors=1,
        terminated=False,
        truncated=True,
    )
    summary = summarize_snapshot(snap)
    assert summary.episodes == 30
    assert summary.evidential_episodes == 0
    assert summary.excluded_episodes == 30
    assert summary.reward_mean is None


def test_summarize_snapshot_excludes_obs_dim_mismatch() -> None:
    snap = _snapshot("random", [1.0, 2.0, 3.0], [10, 20, 30], obs_dim=0)
    summary = summarize_snapshot(snap)
    assert summary.evidential_episodes == 0
    assert summary.excluded_episodes == 3


def test_summarize_snapshot_excludes_short_truncated_episodes() -> None:
    below_floor = MIN_EVIDENTIAL_STEPS_IF_TRUNCATED - 1
    snap = _snapshot("random", [1.0], [below_floor], truncated=True)
    summary = summarize_snapshot(snap)
    assert summary.evidential_episodes == 0
    assert summary.excluded_episodes == 1


def test_summarize_snapshot_includes_truncated_episode_at_step_floor() -> None:
    snap = _snapshot("random", [1.0], [MIN_EVIDENTIAL_STEPS_IF_TRUNCATED], truncated=True)
    summary = summarize_snapshot(snap)
    assert summary.evidential_episodes == 1
    assert summary.excluded_episodes == 0


def test_summarize_snapshot_includes_short_natural_terminal() -> None:
    # A non-truncated (natural terminal) episode is evidential
    # regardless of length — the step floor applies only to
    # truncated outcomes.
    short = MIN_EVIDENTIAL_STEPS_IF_TRUNCATED - 1
    snap = _snapshot("random", [1.0], [short], terminated=True, truncated=False)
    summary = summarize_snapshot(snap)
    assert summary.evidential_episodes == 1


def test_summarize_snapshot_excludes_record_missing_protocol_error_count() -> None:
    # The documented CLI producer projects BaselineRecord from a
    # trajectory and carries no protocol_errors field at all. A
    # record whose evidentiality cannot be established from its own
    # contents fails closed rather than being assumed clean.
    snap = _snapshot("random", [1.0, 2.0], [10, 20], protocol_errors=None)
    summary = summarize_snapshot(snap)
    assert summary.evidential_episodes == 0
    assert summary.excluded_episodes == 2


def test_render_summary_table_pins_columns() -> None:
    summaries = [
        VariantSummary(
            variant="random",
            episodes=10,
            evidential_episodes=8,
            excluded_episodes=2,
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
    assert "| Evidential |" in table
    assert "| Excluded |" in table
    assert "| random | 10 | 8 | 2 |" in table
    assert "Reward mean" in table


def test_render_summary_table_renders_na_for_missing_stats() -> None:
    summaries = [
        VariantSummary(
            variant="random",
            episodes=5,
            evidential_episodes=0,
            excluded_episodes=5,
            reward_mean=None,
            reward_median=None,
            reward_std=None,
            reward_p95=None,
            steps_mean=None,
            steps_median=None,
            steps_std=None,
            steps_p95=None,
        ),
    ]
    table = render_summary_table(summaries)
    assert "| random | 5 | 0 | 5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in table


def test_render_report_emits_markdown_with_plot_references(tmp_path: Path) -> None:
    random_snap = _snapshot("random", [1.0, 2.0, 3.0], [10, 20, 30])
    trained_snap = _snapshot("trained", [5.0, 7.0, 9.0], [50, 60, 70])
    summaries = [summarize_snapshot(random_snap), summarize_snapshot(trained_snap)]
    out = tmp_path / "report.md"
    plot_paths = {
        "reward_curve": tmp_path / REWARD_CURVE_FILENAME,
        "episode_length": tmp_path / EPISODE_LENGTH_FILENAME,
        "reward_histogram": tmp_path / PLANNING_LATENCY_FILENAME,
    }
    body = render_report(random_snap, trained_snap, summaries, plot_paths=plot_paths, out_path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == body
    assert "# FORGE v0.5 Trained vs. Random Comparative Report" in body
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
    # Exactly MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON per variant — at
    # the floor should pass, not just above it.
    n = MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON
    random_snap_path.write_text(json.dumps(_snapshot("random", [1.0] * n, [10] * n)))
    trained_snap_path.write_text(json.dumps(_snapshot("trained", [5.0] * n, [50] * n)))

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


def test_main_refuses_below_evidential_floor_and_writes_nothing(tmp_path: Path) -> None:
    random_snap_path = tmp_path / "baseline_random.json"
    trained_snap_path = tmp_path / "baseline_trained.json"
    out_path = tmp_path / "report.md"
    below_floor = MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON - 1
    # Reproduces the committed-artifact defect: every record carries a
    # protocol error, so nothing is evidential regardless of count.
    random_snap_path.write_text(
        json.dumps(
            _snapshot(
                "random",
                [0.0] * 30,
                [1] * 30,
                protocol_errors=1,
                terminated=False,
                truncated=True,
            )
        )
    )
    trained_snap_path.write_text(
        json.dumps(_snapshot("trained", [5.0] * below_floor, [50] * below_floor))
    )

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
    assert rc == EXIT_INSUFFICIENT_EVIDENCE
    assert not out_path.exists()


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


def test_min_evidential_steps_if_truncated_is_pinned() -> None:
    assert MIN_EVIDENTIAL_STEPS_IF_TRUNCATED == 5, (
        "Lowering this floor changes which truncated episodes count as "
        "evidence. If intentional, update this pin in the same change "
        "and say why in the PR description."
    )


def test_min_evidential_episodes_for_comparison_is_pinned() -> None:
    assert MIN_EVIDENTIAL_EPISODES_FOR_COMPARISON == 3, (
        "Lowering this floor changes how few real episodes are needed "
        "before a comparison report is considered meaningful. Update "
        "this pin deliberately and say why in the PR description."
    )


def test_evidential_rewards_excludes_non_evidential_records() -> None:
    # write_plots() feeds this into the reward curve, the rolling
    # mean/std shading, and the reward histogram. If it read
    # unfiltered per_episode records, a broken run would still shift
    # the plots even though summarize_snapshot()'s table excludes it.
    snap = _snapshot(
        "random",
        rewards=[1.0, 2.0, 999.0],
        steps=[10, 20, 1],
        protocol_errors=[0, 0, 1],
        terminated=[True, True, False],
        truncated=[False, False, True],
    )
    assert _evidential_rewards(snap) == [1.0, 2.0]


def test_evidential_steps_excludes_non_evidential_records() -> None:
    snap = _snapshot(
        "random",
        rewards=[1.0, 2.0, 999.0],
        steps=[10, 20, 1],
        protocol_errors=[0, 0, 1],
        terminated=[True, True, False],
        truncated=[False, False, True],
    )
    assert _evidential_steps(snap) == [10.0, 20.0]
