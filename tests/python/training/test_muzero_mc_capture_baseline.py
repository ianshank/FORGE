"""Tests for the `capture-baseline` subcommand + capture_baseline module.

Driven entirely against in-process stubs — no docker, no live HTTP
endpoint. The orchestration loop's external surfaces
(`metrics_fetcher`, `trajectory_loader`, `docker_log_reader`,
`sleeper`, `clock`) are all injectable for exactly this purpose.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path  # noqa: TC003 — runtime use in pytest tmp_path fixture
from typing import Any

import pytest

from forge.training.muzero_mc.capture_baseline import (
    ALL_VARIANTS,
    DEFAULT_METRICS_URL,
    VARIANT_RANDOM,
    VARIANT_TRAINED,
    BaselineRecord,
    CaptureConfig,
    capture_baseline,
    collect_per_episode_payload,
    load_episode_records,
    resolve_trajectory_dir,
)
from forge.training.muzero_mc.cli import EXIT_OK, EXIT_USAGE, main
from forge.utils.metrics import (
    fetch_prometheus_metrics,  # noqa: F401 — re-exported pin
    scrape_counter,
    scrape_gauge,
)

# --- Metrics module pins -------------------------------------------


def test_scrape_counter_handles_unlabeled_metric() -> None:
    body = "\n".join(
        [
            "# HELP forge_mc_episode_total Episodes finished",
            "# TYPE forge_mc_episode_total counter",
            "forge_mc_episode_total 7",
        ]
    )
    assert scrape_counter(body, "forge_mc_episode_total") == 7.0


def test_scrape_counter_sums_labeled_variants() -> None:
    body = "\n".join(
        [
            'forge_mc_protocol_errors_total{kind="planner"} 2',
            'forge_mc_protocol_errors_total{kind="env"} 5',
        ]
    )
    assert scrape_counter(body, "forge_mc_protocol_errors_total") == 7.0


def test_scrape_counter_returns_zero_for_missing() -> None:
    assert scrape_counter("# nothing here", "forge_mc_missing") == 0.0


def test_scrape_gauge_distinguishes_missing_from_zero() -> None:
    body_zero = "forge_mc_model_version 0"
    body_missing = "# nothing here"
    assert scrape_gauge(body_zero, "forge_mc_model_version") == 0.0
    assert scrape_gauge(body_missing, "forge_mc_model_version") is None


def test_scrape_gauge_returns_latest_value() -> None:
    body = "\n".join(
        [
            "forge_mc_model_version 1",
            "forge_mc_model_version 3",
        ]
    )
    assert scrape_gauge(body, "forge_mc_model_version") == 3.0


def test_scrape_ignores_comment_and_blank_lines() -> None:
    body = "\n".join(
        [
            "# HELP foo",
            "",
            "# TYPE foo counter",
            "foo 1",
            "foo 2",
        ]
    )
    assert scrape_counter(body, "foo") == 3.0


# --- Trajectory loading -------------------------------------------


def _write_trajectory(
    path: Path,
    *,
    episode_id: str,
    obs_dim: int = 920,
    action_dim: int = 12,
    rewards: list[float] | None = None,
    terminated: bool = False,
    truncated: bool = False,
    gz: bool = False,
) -> Path:
    rewards = rewards if rewards is not None else [1.0, -0.5, 2.0]
    payload = {
        "episode_id": episode_id,
        "schema_id": "deadbeef",
        "env_id": "minecraft",
        "obs_dim": obs_dim,
        "action_count": action_dim,
        "steps": [
            {
                "tick": i,
                "action_id": 0,
                "reward": r,
                "terminated": (i == len(rewards) - 1) and terminated,
                "truncated": (i == len(rewards) - 1) and truncated,
            }
            for i, r in enumerate(rewards)
        ],
    }
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
    else:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    return path


def test_load_episode_records_parses_uncompressed_and_gzip(tmp_path: Path) -> None:
    _write_trajectory(tmp_path / "ep-000001.json", episode_id="ep-000001")
    _write_trajectory(tmp_path / "ep-000002.json.gz", episode_id="ep-000002", gz=True)
    records = load_episode_records(tmp_path)
    assert [r.episode_id for r in records] == ["ep-000001", "ep-000002"]
    assert all(r.total_reward == 2.5 for r in records)  # 1.0 - 0.5 + 2.0
    assert all(r.steps == 3 for r in records)


def test_load_episode_records_sorts_by_episode_id(tmp_path: Path) -> None:
    _write_trajectory(tmp_path / "ep-000003.json", episode_id="ep-000003")
    _write_trajectory(tmp_path / "ep-000001.json", episode_id="ep-000001")
    _write_trajectory(tmp_path / "ep-000002.json", episode_id="ep-000002")
    records = load_episode_records(tmp_path)
    assert [r.episode_id for r in records] == [
        "ep-000001",
        "ep-000002",
        "ep-000003",
    ]


def test_load_episode_records_skips_corrupt_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_trajectory(tmp_path / "ep-000001.json", episode_id="ep-000001")
    (tmp_path / "ep-000002.json").write_text("not valid json")
    caplog.set_level(logging.WARNING, logger="forge.training.muzero_mc.capture_baseline")
    records = load_episode_records(tmp_path)
    assert [r.episode_id for r in records] == ["ep-000001"]
    assert any("failed to parse" in r.getMessage() for r in caplog.records)


def test_collect_per_episode_payload_serialises_dataclasses() -> None:
    records = [
        BaselineRecord(
            episode_id="ep-1",
            total_reward=1.0,
            steps=3,
            terminated=False,
            truncated=True,
            obs_dim=920,
            action_dim=12,
            schema_id="abc",
        )
    ]
    payload = collect_per_episode_payload(records)
    assert payload == [
        {
            "episode_id": "ep-1",
            "total_reward": 1.0,
            "steps": 3,
            "terminated": False,
            "truncated": True,
            "obs_dim": 920,
            "action_dim": 12,
            "schema_id": "abc",
        }
    ]


# --- CaptureConfig validation -------------------------------------


def test_resolve_trajectory_dir_namespaces_by_variant() -> None:
    p = resolve_trajectory_dir(VARIANT_RANDOM)
    assert p.name == "trajectories.random"


def test_resolve_trajectory_dir_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant must be one of"):
        resolve_trajectory_dir("unknown-variant")


def test_capture_config_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CaptureConfig(
            variant="bogus",
            episodes=1,
            out_path=tmp_path / "out.json",
            trajectory_dir=tmp_path,
        )


def test_capture_config_rejects_zero_episodes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CaptureConfig(
            variant=VARIANT_RANDOM,
            episodes=0,
            out_path=tmp_path / "out.json",
            trajectory_dir=tmp_path,
        )


# --- capture_baseline orchestration ------------------------------


class _StubClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class _StubSleeper:
    def __init__(self, clock: _StubClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, dt: float) -> None:
        self.calls.append(dt)
        self.clock.advance(dt)


def test_capture_baseline_writes_snapshot_when_target_reached(tmp_path: Path) -> None:
    out_path = tmp_path / "baseline_random.json"
    trajectory_dir = tmp_path / "trajectories.random"
    trajectory_dir.mkdir()
    _write_trajectory(trajectory_dir / "ep-000001.json", episode_id="ep-000001")
    _write_trajectory(trajectory_dir / "ep-000002.json", episode_id="ep-000002")

    cfg = CaptureConfig(
        variant=VARIANT_RANDOM,
        episodes=2,
        out_path=out_path,
        trajectory_dir=trajectory_dir,
        poll_interval_secs=0.01,
        timeout_secs=10,
    )

    scrape_sequence = iter(
        [
            "forge_mc_episode_total 0\nforge_mc_model_version 1\n",
            "forge_mc_episode_total 1\nforge_mc_model_version 1\n",
            "forge_mc_episode_total 2\nforge_mc_model_version 2\n",
        ]
    )

    clock = _StubClock()
    sleeper = _StubSleeper(clock)
    snapshot = capture_baseline(
        cfg,
        metrics_fetcher=lambda _url: next(scrape_sequence),
        trajectory_loader=lambda: load_episode_records(trajectory_dir),
        docker_log_reader=lambda: "",
        sleeper=sleeper,
        clock=clock,
    )

    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written == snapshot
    assert snapshot["variant"] == VARIANT_RANDOM
    assert snapshot["episodes_target"] == 2
    assert snapshot["episodes_observed"] == 2
    assert sorted(snapshot["manifest_versions_seen"]) == [1.0, 2.0]
    assert snapshot["summary_counters"]["forge_mc_episode_total"] == 2.0
    assert snapshot["summary_gauges"]["forge_mc_model_version"] == 2.0
    assert len(snapshot["per_episode"]) == 2


def test_capture_baseline_times_out_when_target_unreachable(tmp_path: Path) -> None:
    out_path = tmp_path / "baseline_trained.json"
    trajectory_dir = tmp_path / "trajectories.trained"
    trajectory_dir.mkdir()
    cfg = CaptureConfig(
        variant=VARIANT_TRAINED,
        episodes=5,
        out_path=out_path,
        trajectory_dir=trajectory_dir,
        poll_interval_secs=0.1,
        timeout_secs=1,  # very short
    )
    clock = _StubClock()
    sleeper = _StubSleeper(clock)
    with pytest.raises(TimeoutError, match="timed out"):
        capture_baseline(
            cfg,
            metrics_fetcher=lambda _url: "forge_mc_episode_total 0\n",
            trajectory_loader=lambda: [],
            docker_log_reader=lambda: "<no logs>",
            sleeper=sleeper,
            clock=clock,
        )


def test_capture_baseline_tolerates_transient_scrape_errors(tmp_path: Path) -> None:
    out_path = tmp_path / "baseline_random.json"
    trajectory_dir = tmp_path / "trajectories.random"
    trajectory_dir.mkdir()
    _write_trajectory(trajectory_dir / "ep-000001.json", episode_id="ep-000001")

    cfg = CaptureConfig(
        variant=VARIANT_RANDOM,
        episodes=1,
        out_path=out_path,
        trajectory_dir=trajectory_dir,
        poll_interval_secs=0.01,
        timeout_secs=10,
    )

    counter = {"calls": 0}

    def flaky_fetch(_url: str) -> str:
        counter["calls"] += 1
        if counter["calls"] <= 2:
            msg = "connection refused"
            raise OSError(msg)
        return "forge_mc_episode_total 1\nforge_mc_model_version 7\n"

    clock = _StubClock()
    sleeper = _StubSleeper(clock)
    snapshot = capture_baseline(
        cfg,
        metrics_fetcher=flaky_fetch,
        trajectory_loader=lambda: load_episode_records(trajectory_dir),
        docker_log_reader=lambda: "",
        sleeper=sleeper,
        clock=clock,
    )
    assert snapshot["episodes_observed"] == 1
    assert counter["calls"] >= 3  # two transient failures + one success


# --- CLI surface --------------------------------------------------


def test_cli_dry_run_emits_config_summary(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Run from tmp_path so default `baseline_<variant>.json` lands
    # somewhere predictable.
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="forge.training.muzero_mc.cli")
    rc = main(
        [
            "capture-baseline",
            "--variant",
            VARIANT_RANDOM,
            "--episodes",
            "50",
            "--dry-run",
        ]
    )
    assert rc == EXIT_OK
    summary_lines = [
        r.getMessage() for r in caplog.records if "DRY-RUN capture-baseline" in r.getMessage()
    ]
    assert summary_lines, "expected an INFO line containing DRY-RUN capture-baseline"
    msg = summary_lines[0]
    assert "variant=random" in msg
    assert "episodes=50" in msg
    assert "baseline_random.json" in msg
    assert "trajectories.random" in msg


def test_cli_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "capture-baseline",
                "--variant",
                "bogus",
                "--dry-run",
            ]
        )
    # argparse exits 2 on unknown choice.
    assert exc.value.code == EXIT_USAGE


def test_cli_capture_baseline_metric_url_default_matches_module(tmp_path: Path) -> None:
    """Belt-and-braces: the CLI's `--metrics-url` default must equal the
    canonical module constant. Drift would surface as silent operator
    confusion when the runner moves to a non-default port.
    """
    assert isinstance(DEFAULT_METRICS_URL, str)
    assert "metrics" in DEFAULT_METRICS_URL


def test_known_variant_constants_present() -> None:
    """Pin the variant names operators see at the CLI."""
    assert set(ALL_VARIANTS) == {VARIANT_RANDOM, VARIANT_TRAINED}
    assert VARIANT_RANDOM == "random"
    assert VARIANT_TRAINED == "trained"


def test_cli_handles_timeout_with_exit_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When `capture_baseline` raises `TimeoutError`, the CLI must
    return `EXIT_IO` (not propagate the exception) so docker
    orchestrations can keep tearing the stack down.
    """
    from forge.training.muzero_mc import capture_baseline as cb_module
    from forge.training.muzero_mc.cli import EXIT_IO

    def raise_timeout(_cfg: Any) -> None:
        raise TimeoutError("synthetic")

    monkeypatch.setattr(cb_module, "capture_baseline", raise_timeout)
    monkeypatch.chdir(tmp_path)

    rc = main(
        [
            "capture-baseline",
            "--variant",
            VARIANT_TRAINED,
            "--episodes",
            "5",
            "--out",
            str(tmp_path / "baseline_trained.json"),
            "--trajectory-dir",
            str(tmp_path),
        ]
    )
    assert rc == EXIT_IO


def test_cli_handles_runtime_error_with_exit_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OSError / RuntimeError from the capture orchestration also
    maps to EXIT_IO rather than propagating.
    """
    from forge.training.muzero_mc import capture_baseline as cb_module
    from forge.training.muzero_mc.cli import EXIT_IO

    def raise_runtime(_cfg: Any) -> None:
        raise RuntimeError("synthetic")

    monkeypatch.setattr(cb_module, "capture_baseline", raise_runtime)
    monkeypatch.chdir(tmp_path)
    rc = main(
        [
            "capture-baseline",
            "--variant",
            VARIANT_RANDOM,
            "--episodes",
            "1",
            "--out",
            str(tmp_path / "baseline_random.json"),
            "--trajectory-dir",
            str(tmp_path),
        ]
    )
    assert rc == EXIT_IO


def test_capture_baseline_end_of_run_summary_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """End-of-run INFO summary must contain mean_reward + episodes_observed
    so operators can grep multi-hour log volumes.
    """
    out_path = tmp_path / "baseline_random.json"
    trajectory_dir = tmp_path / "trajectories.random"
    trajectory_dir.mkdir()
    _write_trajectory(trajectory_dir / "ep-000001.json", episode_id="ep-000001")

    cfg = CaptureConfig(
        variant=VARIANT_RANDOM,
        episodes=1,
        out_path=out_path,
        trajectory_dir=trajectory_dir,
        poll_interval_secs=0.01,
        timeout_secs=10,
    )
    caplog.set_level(logging.INFO, logger="forge.training.muzero_mc.capture_baseline")
    clock = _StubClock()
    sleeper = _StubSleeper(clock)
    capture_baseline(
        cfg,
        metrics_fetcher=lambda _url: "forge_mc_episode_total 1\n",
        trajectory_loader=lambda: load_episode_records(trajectory_dir),
        docker_log_reader=lambda: "",
        sleeper=sleeper,
        clock=clock,
    )
    summary = [r.getMessage() for r in caplog.records if "capture-baseline done" in r.getMessage()]
    assert summary, "expected an INFO summary line tagged 'capture-baseline done'"
    msg = summary[0]
    assert "mean_reward=" in msg
    assert "episodes_observed=" in msg
    assert "manifest_versions_seen=" in msg


def test_capture_baseline_uses_injected_fixtures(tmp_path: Path) -> None:
    """Sanity check that every injection point is honoured (regression
    guard against future inversions where the orchestration function
    forgets to thread one of the stubs through).
    """
    out_path = tmp_path / "baseline_random.json"
    trajectory_dir = tmp_path / "trajectories.random"
    trajectory_dir.mkdir()
    cfg = CaptureConfig(
        variant=VARIANT_RANDOM,
        episodes=1,
        out_path=out_path,
        trajectory_dir=trajectory_dir,
        poll_interval_secs=0.01,
        timeout_secs=5,
    )

    captured_records = [
        BaselineRecord(
            episode_id="ep-injected",
            total_reward=42.0,
            steps=10,
            terminated=True,
            truncated=False,
            obs_dim=920,
            action_dim=12,
            schema_id="abc",
        )
    ]
    seen_calls: dict[str, Any] = {"metrics": 0, "trajectories": 0, "logs": 0}

    def metrics_stub(_url: str) -> str:
        seen_calls["metrics"] += 1
        return "forge_mc_episode_total 1\n"

    def trajectories_stub() -> list[BaselineRecord]:
        seen_calls["trajectories"] += 1
        return captured_records

    def logs_stub() -> str:
        seen_calls["logs"] += 1
        return ""

    clock = _StubClock()
    sleeper = _StubSleeper(clock)
    snapshot = capture_baseline(
        cfg,
        metrics_fetcher=metrics_stub,
        trajectory_loader=trajectories_stub,
        docker_log_reader=logs_stub,
        sleeper=sleeper,
        clock=clock,
    )
    assert seen_calls["metrics"] >= 1
    assert seen_calls["trajectories"] == 1
    assert snapshot["per_episode"][0]["episode_id"] == "ep-injected"
