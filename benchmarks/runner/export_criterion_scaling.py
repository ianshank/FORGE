#!/usr/bin/env python3
"""Export Criterion ``multi_agent_scaling`` results to a committed JSON report.

Criterion writes per-benchmark estimates under ``target/criterion/``. This
helper walks that tree, extracts the square/hex scaling groups, and writes a
stable JSON artefact that can be committed under
``benchmarks/baselines/<profile>/multi_agent_scaling.json``.

Two throughput numbers are recorded per row so they cannot be confused:

* ``env_steps_per_sec`` -- whole-world ``step()`` calls per second
  (``1e9 / mean_ns``). This is the quantity comparable to the Python/PyO3
  headline.
* ``agent_steps_per_sec`` -- agent-normalised throughput
  (``num_agents * env_steps_per_sec``), matching Criterion's
  ``Throughput::Elements(num_agents)`` report.

Invocation
----------

    python3 benchmarks/runner/export_criterion_scaling.py \\
        --criterion-dir target/criterion \\
        --out benchmarks/baselines/cloud_agent/multi_agent_scaling.json \\
        --profile cloud_agent

Exit codes
----------

* ``0`` -- report written.
* ``2`` -- input tree missing, malformed, or contained no scaling rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("export_criterion_scaling")

#: Exit code when the Criterion tree is missing, malformed, or empty.
EXIT_INPUT_ERROR: int = 2

#: Criterion group names produced by ``crates/forge-bench/benches/multi_agent_scaling.rs``.
SCALING_GROUPS: tuple[str, ...] = (
    "multi_agent_scaling_square",
    "multi_agent_scaling_hex",
)

#: Default world side length, matching ``DEFAULT_WORLD_SIDE`` in the bench.
DEFAULT_WORLD_SIDE: int = 128

#: Default RNG seed, matching ``BENCH_SEED`` in ``forge-bench::env``.
DEFAULT_SEED: int = 42

#: Subdirectory Criterion uses for the just-completed run.
NEW_DIR_NAME: str = "new"

#: Estimates filename Criterion writes inside each run directory.
ESTIMATES_FILENAME: str = "estimates.json"

#: Top-level keys of a committed scaling report. Tests lock this set so a
#: new field cannot appear (or an old one vanish) without updating the gate.
REPORT_FIELDS: frozenset[str] = frozenset(
    {
        "producer",
        "profile",
        "git_sha",
        "world_side",
        "seed",
        "agent_counts",
        "hardware",
        "variants",
    }
)

#: Per-variant keys. ``env_steps_per_sec`` is the Python-headline analogue;
#: ``agent_steps_per_sec`` is Criterion's ``Elements(num_agents)`` figure.
VARIANT_FIELDS: frozenset[str] = frozenset(
    {
        "group",
        "group_id",
        "num_agents",
        "mean_ns",
        "median_ns",
        "env_steps_per_sec",
        "agent_steps_per_sec",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_sha(repo_root: Path) -> str:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha.strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                return value.strip()
    return platform.processor() or "unknown"


def detect_hardware() -> dict[str, str]:
    """Return a portable hardware label for the measurement host."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "cpu": _cpu_model(),
        "python": platform.python_version(),
    }


def _point_estimate(payload: dict[str, Any], key: str) -> float:
    block = payload.get(key)
    if not isinstance(block, dict) or "point_estimate" not in block:
        raise ValueError(f"estimates.json missing {key}.point_estimate")
    try:
        return float(block["point_estimate"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"estimates.json {key}.point_estimate is not numeric") from exc


def parse_estimates(path: Path) -> tuple[float, float]:
    """Return ``(mean_ns, median_ns)`` from a Criterion estimates.json."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root is not a JSON object")
    return _point_estimate(payload, "mean"), _point_estimate(payload, "median")


def _group_label(group_id: str) -> str:
    if group_id.endswith("_square"):
        return "square"
    if group_id.endswith("_hex"):
        return "hex"
    return group_id


def _num_agents_from_path(estimates: Path) -> int:
    """Parse the agent count from ``.../num_agents/<N>/new/estimates.json``."""
    # estimates.parent is ``new``; estimates.parent.parent is the parameter dir.
    param_dir = estimates.parent.parent
    try:
        value = int(param_dir.name)
    except ValueError as exc:
        raise ValueError(
            f"cannot parse num_agents from path {estimates}: "
            f"expected integer directory name, got {param_dir.name!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"num_agents must be positive, got {value} from {estimates}")
    return value


def collect_rows(criterion_dir: Path) -> list[dict[str, Any]]:
    """Walk ``criterion_dir`` and return one row per scaling benchmark."""
    if not criterion_dir.is_dir():
        raise ValueError(f"Criterion directory does not exist: {criterion_dir}")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for group_id in SCALING_GROUPS:
        group_dir = criterion_dir / group_id
        if not group_dir.is_dir():
            continue
        for estimates in sorted(group_dir.glob(f"**/{NEW_DIR_NAME}/{ESTIMATES_FILENAME}")):
            try:
                num_agents = _num_agents_from_path(estimates)
                mean_ns, median_ns = parse_estimates(estimates)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if mean_ns <= 0.0:
                errors.append(f"{estimates}: mean_ns must be positive, got {mean_ns}")
                continue
            env_steps_per_sec = 1e9 / mean_ns
            rows.append(
                {
                    "group": _group_label(group_id),
                    "group_id": group_id,
                    "num_agents": num_agents,
                    "mean_ns": mean_ns,
                    "median_ns": median_ns,
                    "env_steps_per_sec": env_steps_per_sec,
                    "agent_steps_per_sec": num_agents * env_steps_per_sec,
                }
            )

    if errors and not rows:
        raise ValueError("; ".join(errors))
    for message in errors:
        LOGGER.warning("%s", message)

    rows.sort(key=lambda row: (str(row["group"]), int(row["num_agents"])))
    return rows


def build_report(
    *,
    criterion_dir: Path,
    profile: str,
    repo_root: Path,
    world_side: int,
    seed: int,
) -> dict[str, Any]:
    rows = collect_rows(criterion_dir)
    if not rows:
        raise ValueError(
            f"no {SCALING_GROUPS} rows under {criterion_dir}. "
            "Remedy: run `cargo bench -p forge-bench --bench multi_agent_scaling` "
            "then re-run this exporter."
        )
    agent_counts = sorted({int(row["num_agents"]) for row in rows})
    return {
        "producer": "multi_agent_scaling",
        "profile": profile,
        "git_sha": _git_sha(repo_root),
        "world_side": world_side,
        "seed": seed,
        "agent_counts": agent_counts,
        "hardware": detect_hardware(),
        "variants": rows,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path("target/criterion"),
        help="Criterion output directory (default: target/criterion).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON path.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Hardware profile label stored in the report (e.g. cloud_agent, reference_a).",
    )
    parser.add_argument(
        "--world-side",
        type=int,
        default=int(os.environ.get("FORGE_BENCH_WORLD", DEFAULT_WORLD_SIDE)),
        help="World side length recorded in the report (default: FORGE_BENCH_WORLD or 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("FORGE_BENCH_SEED", DEFAULT_SEED)),
        help="RNG seed recorded in the report (default: FORGE_BENCH_SEED or 42).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help="Repository root used to resolve git SHA (default: derived from this file).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    try:
        report = build_report(
            criterion_dir=args.criterion_dir,
            profile=args.profile,
            repo_root=args.repo_root,
            world_side=args.world_side,
            seed=args.seed,
        )
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return EXIT_INPUT_ERROR

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("wrote %d variant(s) to %s", len(report["variants"]), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
