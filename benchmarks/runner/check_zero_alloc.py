"""Post-processor for the allocation audit JSON produced by the
``allocation_audit`` binary in ``forge-bench``.

Reads the JSON report, asserts the zero-allocation invariant per variant,
and exits non-zero if any variant violates the contract. Designed to be
driven from CI so allocation regressions block merges.

Invocation
----------

    python benchmarks/runner/check_zero_alloc.py \
        --input /tmp/alloc_audit.json \
        [--allow Move_Up,PickUp] \
        [--max-bytes 0] \
        [--json /tmp/alloc_audit_summary.json]

Exit codes
----------

* ``0`` -- every non-allowlisted variant is zero-allocation.
* ``1`` -- one or more variants violated the invariant.
* ``2`` -- input file missing, malformed, or unreadable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("check_zero_alloc")

#: Default ceiling for ``total_bytes`` per measured region. Zero means "must
#: not allocate at all", which matches the documented "zero allocation on
#: hot path" contract. Overridable from the CLI so reviewers can relax the
#: threshold when investigating a regression.
DEFAULT_MAX_BYTES: int = 0

#: Exit code when one or more variants violate the contract.
EXIT_VIOLATION: int = 1

#: Exit code when the report file is missing or malformed.
EXIT_INPUT_ERROR: int = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the JSON report emitted by allocation_audit.",
    )
    parser.add_argument(
        "--allow",
        type=str,
        default="",
        help=(
            "Comma-separated variant labels that are expected to allocate "
            "(e.g. Communicate_0 if the comms path legitimately copies). "
            "Empty by default."
        ),
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=(
            "Maximum total_bytes allowed per variant (default: "
            f"{DEFAULT_MAX_BYTES}). Raise this to investigate regressions "
            "without silently accepting them."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable pass/fail summary.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        LOGGER.error("input file does not exist: %s", path)
        sys.exit(EXIT_INPUT_ERROR)
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded: dict[str, Any] = json.load(handle)
            return loaded
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("failed to read %s: %s", path, exc)
        sys.exit(EXIT_INPUT_ERROR)


def _build_allow_set(raw: str) -> frozenset[str]:
    return frozenset(tok.strip() for tok in raw.split(",") if tok.strip())


def _evaluate(
    variants: list[dict[str, Any]],
    allow: frozenset[str],
    max_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    violations: list[dict[str, Any]] = []
    allowed_hits: list[dict[str, Any]] = []

    for row in variants:
        name = str(row.get("variant", "<unknown>"))
        total_bytes = int(row.get("total_bytes", 0))
        total_blocks = int(row.get("total_blocks", 0))
        iters = int(row.get("iters", 0))
        is_violation = total_bytes > max_bytes or total_blocks > 0
        payload: dict[str, Any] = {
            "variant": name,
            "iters": iters,
            "total_blocks": total_blocks,
            "total_bytes": total_bytes,
            "max_bytes": int(row.get("max_bytes", 0)),
        }
        if not is_violation:
            LOGGER.debug("clean: %s", payload)
            continue
        if name in allow:
            LOGGER.info("allowlisted allocation: %s", payload)
            allowed_hits.append(payload)
        else:
            LOGGER.error("allocation violation: %s", payload)
            violations.append(payload)

    return violations, allowed_hits


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    report = _load_report(args.input)
    variants = report.get("variants", [])
    if not isinstance(variants, list) or not variants:
        LOGGER.error("report %s has no variants", args.input)
        return EXIT_INPUT_ERROR

    allow = _build_allow_set(args.allow)
    violations, allowed_hits = _evaluate(variants, allow, args.max_bytes)

    summary: dict[str, Any] = {
        "input": str(args.input),
        "max_bytes": args.max_bytes,
        "allow": sorted(allow),
        "violations": violations,
        "allowlisted": allowed_hits,
        "clean_count": len(variants) - len(violations) - len(allowed_hits),
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("wrote summary to %s", args.json)

    if violations:
        LOGGER.error("zero-allocation contract violated by %d variant(s)", len(violations))
        return EXIT_VIOLATION
    LOGGER.info("all %d variants satisfy the zero-allocation contract", len(variants))
    return 0


if __name__ == "__main__":
    sys.exit(main())
