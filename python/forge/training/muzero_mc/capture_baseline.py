"""Baseline-capture support for the v0.5 first-real-run flow.

Drives N episodes against a running (or freshly-brought-up) Minecraft
self-play stack and writes a single JSON snapshot per variant. The
output file is the source-of-truth a sibling plotter
(``scripts/mc_plot_baseline.py``) consumes to render the
trained-vs-random comparison report.

Lives under ``forge.training.muzero_mc`` so it inherits the package's
``EXIT_*`` codes, ``--log-level``, mypy + ruff gates, and import-time
isolation from ``tests/python/integration/`` (the existing E2E
``_helpers.py`` is *test-only* and can't be safely imported from
operator-facing tools).

No hard-coded values — every numeric/path default flows through this
module's constants (or the runner's own config).
"""

from __future__ import annotations

__all__ = [
    "BaselineRecord",
    "CaptureConfig",
    "DEFAULT_DOCKER_LOGS_TAIL",
    "DEFAULT_METRICS_URL",
    "DEFAULT_RUNNER_CONTAINER",
    "DEFAULT_TIMEOUT_SECS",
    "DEFAULT_TRAJECTORY_GLOB_PATTERN",
    "VARIANT_RANDOM",
    "VARIANT_TRAINED",
    "capture_baseline",
    "load_episode_records",
    "resolve_trajectory_dir",
]

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# --- Shared defaults (single source of truth) -----------------------

VARIANT_RANDOM: Final[str] = "random"
VARIANT_TRAINED: Final[str] = "trained"
ALL_VARIANTS: Final[tuple[str, ...]] = (VARIANT_RANDOM, VARIANT_TRAINED)

DEFAULT_METRICS_URL: Final[str] = "http://127.0.0.1:9090/metrics"
DEFAULT_RUNNER_CONTAINER: Final[str] = "forge-mc-runner"
DEFAULT_TIMEOUT_SECS: Final[int] = 3600
DEFAULT_TRAJECTORY_GLOB_PATTERN: Final[str] = "ep-*.json*"
DEFAULT_DOCKER_LOGS_TAIL: Final[int] = 50
DEFAULT_POLL_INTERVAL_SECS: Final[float] = 5.0

# Gauges the snapshot pulls out of the Prometheus scrape for the
# summary header. Kept in one tuple so the JSON output stays in sync
# with the documented schema.
SUMMARY_GAUGES: Final[tuple[str, ...]] = (
    "forge_mc_model_version",
)
SUMMARY_COUNTERS: Final[tuple[str, ...]] = (
    "forge_mc_episode_total",
    "forge_mc_steps_total",
    "forge_mc_protocol_errors_total",
)


# --- Data shapes ---------------------------------------------------


@dataclass
class CaptureConfig:
    """User-facing knobs for a single ``capture-baseline`` invocation."""

    variant: str
    episodes: int
    out_path: Path
    trajectory_dir: Path
    metrics_url: str = DEFAULT_METRICS_URL
    runner_container: str = DEFAULT_RUNNER_CONTAINER
    timeout_secs: int = DEFAULT_TIMEOUT_SECS
    poll_interval_secs: float = DEFAULT_POLL_INTERVAL_SECS
    trajectory_glob_pattern: str = DEFAULT_TRAJECTORY_GLOB_PATTERN
    docker_logs_tail: int = DEFAULT_DOCKER_LOGS_TAIL

    def __post_init__(self) -> None:
        if self.variant not in ALL_VARIANTS:
            msg = f"variant must be one of {ALL_VARIANTS}, got {self.variant!r}"
            raise ValueError(msg)
        if self.episodes <= 0:
            msg = f"episodes must be >= 1, got {self.episodes}"
            raise ValueError(msg)
        if self.timeout_secs <= 0:
            msg = f"timeout_secs must be >= 1, got {self.timeout_secs}"
            raise ValueError(msg)
        if self.poll_interval_secs <= 0:
            msg = f"poll_interval_secs must be > 0, got {self.poll_interval_secs}"
            raise ValueError(msg)
        self.out_path = Path(self.out_path).resolve()
        self.trajectory_dir = Path(self.trajectory_dir).resolve()


@dataclass
class BaselineRecord:
    """Single per-episode entry in the snapshot JSON."""

    episode_id: str
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    obs_dim: int
    action_dim: int
    schema_id: str


def resolve_trajectory_dir(variant: str, root: Path | None = None) -> Path:
    """Per-variant trajectory directory — namespaces the two runs so the
    trainer's :func:`_trim_replay_buffer` cannot evict baseline files mid-
    capture (peer-review gap #5).
    """
    if variant not in ALL_VARIANTS:
        msg = f"variant must be one of {ALL_VARIANTS}, got {variant!r}"
        raise ValueError(msg)
    base = Path(root) if root is not None else Path("trajectories")
    return Path(f"{base}.{variant}").resolve()


# --- Trajectory parsing -------------------------------------------


def load_episode_records(
    trajectory_dir: Path,
    *,
    glob_pattern: str = DEFAULT_TRAJECTORY_GLOB_PATTERN,
) -> list[BaselineRecord]:
    """Read every ``ep-*.json[.gz]`` file under ``trajectory_dir`` and
    project it to a :class:`BaselineRecord`.

    Tolerates either uncompressed ``.json`` or gzipped ``.json.gz``
    trajectories — both formats are emitted by the runner's
    ``TrajectoryWriter`` depending on ``trajectory_compression``.

    Sorted by ``episode_id`` so the JSON output is reproducible
    regardless of filesystem ordering.
    """
    paths = sorted(trajectory_dir.glob(glob_pattern))
    records: list[BaselineRecord] = []
    for path in paths:
        try:
            blob = _read_trajectory_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to parse trajectory %s: %s", path, exc)
            continue
        record = _trajectory_to_record(blob)
        if record is not None:
            records.append(record)
    return records


def _read_trajectory_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    if not isinstance(data, dict):
        msg = f"{path} did not deserialise to a JSON object"
        raise json.JSONDecodeError(msg, doc=str(path), pos=0)
    return data


def _trajectory_to_record(blob: dict[str, Any]) -> BaselineRecord | None:
    """Project a `TrajectoryV2` JSON payload into a `BaselineRecord`.

    Returns ``None`` (and emits a WARN) when the trajectory is missing
    one of the fields the snapshot schema requires. This keeps a
    single corrupt file from poisoning the whole capture.
    """
    episode_id = blob.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        logger.warning("trajectory missing string episode_id: %r", blob)
        return None
    steps_block = blob.get("steps")
    if not isinstance(steps_block, list):
        logger.warning("trajectory %s has no `steps` list", episode_id)
        return None
    total_reward = 0.0
    last_terminated = False
    last_truncated = False
    for entry in steps_block:
        if not isinstance(entry, dict):
            continue
        reward = entry.get("reward")
        if isinstance(reward, (int, float)):
            total_reward += float(reward)
        if entry.get("terminated"):
            last_terminated = True
        if entry.get("truncated"):
            last_truncated = True
    return BaselineRecord(
        episode_id=episode_id,
        total_reward=total_reward,
        steps=len(steps_block),
        terminated=last_terminated,
        truncated=last_truncated,
        obs_dim=int(blob.get("obs_dim", 0)),
        action_dim=int(blob.get("action_count", 0)),
        schema_id=str(blob.get("schema_id", "")),
    )


# --- Capture orchestration ----------------------------------------


@dataclass
class _CaptureProgress:
    """Internal bookkeeping for ``capture_baseline``."""

    episodes_target: int
    started_at: float
    manifest_versions_seen: set[float] = field(default_factory=set)
    last_episode_count: int = 0


def capture_baseline(
    cfg: CaptureConfig,
    *,
    metrics_fetcher: Any = None,
    trajectory_loader: Any = None,
    docker_log_reader: Any = None,
    sleeper: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """Drive a single baseline-capture session.

    Polls the runner's Prometheus endpoint until
    ``forge_mc_episode_total >= cfg.episodes`` (or
    ``cfg.timeout_secs`` elapses). On success, reads every emitted
    trajectory under ``cfg.trajectory_dir`` and writes the
    snapshot JSON to ``cfg.out_path``.

    All external surfaces (HTTP scrape, trajectory file read, docker
    logs, sleep, clock) are injectable — defaults wire the
    stdlib/forge canonical implementations, but unit tests can pass
    in-process stubs to drive the orchestration without a live stack.

    Returns the snapshot dict that was written to disk.
    """
    metrics_fetcher = metrics_fetcher or _default_metrics_fetcher
    trajectory_loader = trajectory_loader or (
        lambda: load_episode_records(
            cfg.trajectory_dir, glob_pattern=cfg.trajectory_glob_pattern
        )
    )
    docker_log_reader = docker_log_reader or (
        lambda: _default_docker_log_reader(cfg.runner_container, cfg.docker_logs_tail)
    )
    sleeper = sleeper or time.sleep
    clock = clock or time.monotonic

    started_iso = _utc_now_isoformat()
    progress = _CaptureProgress(
        episodes_target=cfg.episodes,
        started_at=clock(),
    )

    logger.info(
        "capture-baseline start: variant=%s episodes=%d trajectory_dir=%s out=%s",
        cfg.variant,
        cfg.episodes,
        cfg.trajectory_dir,
        cfg.out_path,
    )

    last_scrape: str = ""
    while True:
        elapsed = clock() - progress.started_at
        if elapsed > cfg.timeout_secs:
            logger.error(
                "capture-baseline timed out after %.0fs (target %d episodes, observed %d)",
                elapsed,
                cfg.episodes,
                progress.last_episode_count,
            )
            tail = docker_log_reader()
            if tail:
                logger.error("runner container logs (tail %d):\n%s", cfg.docker_logs_tail, tail)
            msg = (
                f"capture-baseline timed out after {elapsed:.0f}s; "
                f"observed {progress.last_episode_count}/{cfg.episodes} episodes"
            )
            raise TimeoutError(msg)

        try:
            last_scrape = metrics_fetcher(cfg.metrics_url)
        except Exception as exc:  # noqa: BLE001 — surface AND keep polling
            logger.debug("metrics scrape transient error: %s", exc)
            sleeper(cfg.poll_interval_secs)
            continue

        observed = int(_scrape_counter_local(last_scrape, "forge_mc_episode_total"))
        if observed > progress.last_episode_count:
            progress.last_episode_count = observed
            logger.info(
                "capture-baseline progress: %d/%d episodes",
                observed,
                cfg.episodes,
            )

        version = _scrape_gauge_local(last_scrape, "forge_mc_model_version")
        if version is not None:
            progress.manifest_versions_seen.add(version)

        if observed >= cfg.episodes:
            break
        sleeper(cfg.poll_interval_secs)

    ended_iso = _utc_now_isoformat()
    per_episode_records = trajectory_loader()
    logger.info(
        "capture-baseline collected %d trajectories (target %d)",
        len(per_episode_records),
        cfg.episodes,
    )

    snapshot = {
        "variant": cfg.variant,
        "episodes_target": cfg.episodes,
        "episodes_observed": progress.last_episode_count,
        "trajectory_dir": str(cfg.trajectory_dir),
        "started_at": started_iso,
        "ended_at": ended_iso,
        "manifest_versions_seen": sorted(progress.manifest_versions_seen),
        "summary_counters": _summary_counters(last_scrape),
        "summary_gauges": _summary_gauges(last_scrape),
        "prometheus_snapshot": last_scrape,
        "per_episode": [asdict(rec) for rec in per_episode_records],
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_path.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
    logger.info("capture-baseline wrote %s", cfg.out_path)
    return snapshot


def _summary_counters(scrape_text: str) -> dict[str, float]:
    return {name: _scrape_counter_local(scrape_text, name) for name in SUMMARY_COUNTERS}


def _summary_gauges(scrape_text: str) -> dict[str, float | None]:
    return {name: _scrape_gauge_local(scrape_text, name) for name in SUMMARY_GAUGES}


def _default_metrics_fetcher(url: str) -> str:
    # Local import keeps ``capture_baseline.py``'s top-level imports
    # cheap — the metrics module pulls in ``urllib.request`` which we
    # don't want to evaluate during ``argparse --help``.
    from forge.utils.metrics import fetch_prometheus_metrics

    return fetch_prometheus_metrics(url)


def _default_docker_log_reader(container: str, tail: int) -> str:
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return "<docker CLI not available>"
    try:
        completed = subprocess.run(
            ["docker", "logs", f"--tail={tail}", container],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<docker logs failed: {exc}>"
    return completed.stdout + completed.stderr


def _scrape_counter_local(scrape_text: str, name: str) -> float:
    from forge.utils.metrics import scrape_counter

    return scrape_counter(scrape_text, name)


def _scrape_gauge_local(scrape_text: str, name: str) -> float | None:
    from forge.utils.metrics import scrape_gauge

    return scrape_gauge(scrape_text, name)


def _utc_now_isoformat() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def collect_per_episode_payload(records: Iterable[BaselineRecord]) -> list[dict[str, Any]]:
    """Public helper: dataclass list → list of plain dicts. Used by the
    plot generator to project the snapshot without re-importing the
    dataclass.
    """
    return [asdict(rec) for rec in records]
