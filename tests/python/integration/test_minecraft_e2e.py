"""Opt-in pytest E2E driving the Minecraft compose stack.

Skipped by default — the `minecraft_e2e` marker keeps these tests
out of the standard PR CI. The opt-in `python-test-minecraft-e2e`
workflow_dispatch CI job runs them with `MC_EULA=TRUE` against the
ubuntu-latest runner's docker engine.

Local development: ``pytest tests/python/integration/test_minecraft_e2e.py -m minecraft_e2e -v``
with docker running.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable  # noqa: TC003 — used in test signatures
from pathlib import Path
from typing import Any

import pytest

from ._helpers import POLL_TIMEOUT_SECS, wait_until

pytestmark = pytest.mark.minecraft_e2e


def _fetch_metrics(url: str) -> str:
    """Return the body of the `/metrics` endpoint.

    Uses the stdlib `urllib` so the test suite doesn't pull `httpx`
    or `requests` as a hard dep for the cheap GET case.
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body: bytes = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"metrics fetch failed: {e}") from e
    return body.decode("utf-8", errors="replace")


def _scrape_counter(metrics_text: str, name: str) -> float:
    """Extract a counter value from the Prometheus text format.

    Matches `forge_mc_episode_total 7` lines (counter without labels);
    returns 0.0 if the counter hasn't been incremented yet (Prometheus
    convention).
    """
    pattern = re.compile(rf"^{re.escape(name)}\s+(\S+)\s*$", re.MULTILINE)
    m = pattern.search(metrics_text)
    if m is None:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def _scrape_gauge(metrics_text: str, name: str) -> float:
    """Same as :func:`_scrape_counter` but for gauges — semantically
    identical extraction (Prometheus text format treats counters and
    gauges identically wire-wise; the type is metadata only).
    """
    return _scrape_counter(metrics_text, name)


def test_two_episode_loop_writes_trajectories(
    compose_up_minecraft_stack: dict[str, Any],
    runner_health_check: Callable[[], None],
) -> None:
    """Stack-level smoke: bring the compose stack up, wait for the
    first two episodes to complete (via the `forge_mc_episode_total`
    counter), assert trajectory files appear on disk.

    Runner-health check on every poll iteration: a crash surfaces
    immediately as `pytest.fail` with the tail of the container
    logs, NOT as a 120s poll timeout.
    """
    metrics_url = compose_up_minecraft_stack["metrics_url"]
    repo_root: Path = compose_up_minecraft_stack["repo_root"]
    traj_dir_env = os.environ.get("FORGE_MC_TRAJECTORIES_DIR")
    traj_dir = Path(traj_dir_env) if traj_dir_env else repo_root / "trajectories"

    def two_episodes_done() -> bool:
        try:
            text = _fetch_metrics(metrics_url)
        except RuntimeError:
            return False
        return _scrape_counter(text, "forge_mc_episode_total") >= 2.0

    wait_until(
        two_episodes_done,
        timeout_secs=POLL_TIMEOUT_SECS,
        health_check=runner_health_check,
        description="forge_mc_episode_total >= 2",
    )

    # Assert at least two trajectory files on disk. Accepts either
    # the plain `.json` or the opt-in `.json.gz` form.
    trajectories = sorted(traj_dir.glob("ep-*.json")) + sorted(traj_dir.glob("ep-*.json.gz"))
    assert len(trajectories) >= 2, (
        f"expected at least 2 trajectory files in {traj_dir}, got {len(trajectories)}; "
        f"directory listing: {[p.name for p in traj_dir.iterdir()] if traj_dir.exists() else 'missing'}"
    )


def test_metrics_endpoint_reports_runner_invariants(
    compose_up_minecraft_stack: dict[str, Any],
    runner_health_check: Callable[[], None],
) -> None:
    """Sanity check the `/metrics` endpoint exposes the five
    v2-plan §3.6 signals once at least one episode has finished.
    """
    metrics_url = compose_up_minecraft_stack["metrics_url"]

    def one_episode_done() -> bool:
        try:
            text = _fetch_metrics(metrics_url)
        except RuntimeError:
            return False
        return _scrape_counter(text, "forge_mc_episode_total") >= 1.0

    wait_until(
        one_episode_done,
        timeout_secs=POLL_TIMEOUT_SECS,
        health_check=runner_health_check,
        description="forge_mc_episode_total >= 1",
    )

    text = _fetch_metrics(metrics_url)
    for name in (
        "forge_mc_episode_total",
        "forge_mc_episode_reward_sum",
        "forge_mc_planning_latency_seconds",
        "forge_mc_model_version",
        "forge_mc_protocol_error_total",
    ):
        assert name in text, f"metric {name!r} missing from /metrics output:\n{text[:1000]}"


def test_compose_down_is_idempotent(
    compose_up_minecraft_stack: dict[str, Any],
) -> None:
    """`scripts/mc_run.sh --down` must succeed even if the stack is
    already down. Pinned because the orchestration script is the
    smoke test's only teardown path.
    """
    import subprocess

    script: Path = compose_up_minecraft_stack["script"]
    repo_root: Path = compose_up_minecraft_stack["repo_root"]
    # Running --down once on the live stack is the canonical
    # teardown; running it again must still exit 0.
    for _ in range(2):
        completed = subprocess.run(
            ["bash", str(script), "--down"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            pytest.fail(
                f"mc_run.sh --down exited {completed.returncode}; "
                f"stderr:\n{completed.stderr[-800:]}"
            )
