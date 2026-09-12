"""Guard that published Python throughput claims do not exceed committed evidence.

The README / CHARTER headline is a floor ("N+ steps/second from Python").
That floor must not exceed ``steps_per_sec`` in the committed
``benchmarks/baselines/cloud_agent/pyo3_step.json`` report produced by
``tests/python/test_step_throughput.py``.

Failure messages state the remedy: lower the claim, or regenerate the
PyO3 report on a labeled profile and update the docs in the same change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYO3_REPORT = REPO_ROOT / "benchmarks" / "baselines" / "cloud_agent" / "pyo3_step.json"

#: Documents that publish a Python steps/second floor.
CLAIM_FILES: tuple[str, ...] = (
    "README.md",
    "BENCHMARKS.md",
    "docs/CHARTER.md",
    "docs/architecture.md",
    "docs/cloud_edge_proposal.md",
    "web/space/README.md",
)

#: ``130,000+ steps/second from Python`` or ``130K+ steps/sec from Python``.
_PROSE_CLAIM = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<k>K)?\+\s*steps(?:/second|/sec)\s+from Python",
    re.IGNORECASE,
)
#: Performance table row: ``Steps/second (from Python) | 130,000+``.
_TABLE_CLAIM = re.compile(
    r"Steps/second \(from Python\)\s*\|\s*(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<k>K)?\+",
    re.IGNORECASE,
)


def _token_to_steps(num: str, k: str | None) -> int:
    value = int(num.replace(",", ""))
    if k:
        return value * 1000
    return value


def parse_python_steps_claims(text: str) -> list[int]:
    """Return every published Python steps/second floor in ``text``."""
    claims = [_token_to_steps(m.group("num"), m.group("k")) for m in _PROSE_CLAIM.finditer(text)]
    claims.extend(_token_to_steps(m.group("num"), m.group("k")) for m in _TABLE_CLAIM.finditer(text))
    return claims


def load_measured_steps_per_sec(path: Path) -> float:
    if not path.is_file():
        raise AssertionError(
            f"PyO3 throughput report missing at {path.relative_to(REPO_ROOT)}. "
            "Remedy: run "
            "`FORGE_RUN_STEP_THROUGHPUT=1 "
            "FORGE_STEP_THROUGHPUT_OUT=benchmarks/baselines/cloud_agent/pyo3_step.json "
            "pytest tests/python/test_step_throughput.py -s --no-cov` "
            "after `maturin develop`, then commit the report."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"PyO3 throughput report {path.relative_to(REPO_ROOT)} is malformed: {exc}. "
            "Remedy: regenerate the report with test_step_throughput.py."
        ) from exc
    measured = payload.get("steps_per_sec")
    try:
        value = float(measured)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"PyO3 throughput report {path.relative_to(REPO_ROOT)} has non-numeric "
            f"steps_per_sec={measured!r}. Remedy: regenerate the report."
        ) from exc
    if value <= 0.0:
        raise AssertionError(
            f"PyO3 throughput report {path.relative_to(REPO_ROOT)} has "
            f"steps_per_sec={value}. Remedy: regenerate the report."
        )
    return value


def test_parse_python_steps_claims_prose_and_table() -> None:
    text = (
        "running at 80,000+ steps/second from Python.\n"
        "130K+ steps/sec from Python, <8 us/step.\n"
        "| Steps/second (from Python) | 75,000+ |\n"
    )
    assert parse_python_steps_claims(text) == [80_000, 130_000, 75_000]


def test_parse_python_steps_claims_ignores_unrelated_numbers() -> None:
    text = "300x compression over full trajectories. 10K-tick episode."
    assert parse_python_steps_claims(text) == []


def test_published_python_throughput_claims_do_not_exceed_evidence() -> None:
    measured = load_measured_steps_per_sec(PYO3_REPORT)
    offenders: list[str] = []
    for rel in CLAIM_FILES:
        path = REPO_ROOT / rel
        assert path.is_file(), f"claim file missing: {rel}"
        offenders.extend(
            f"{rel} claims {claim:,}+ steps/second from Python, but "
            f"{PYO3_REPORT.relative_to(REPO_ROOT)} measured {measured:,.0f}"
            for claim in parse_python_steps_claims(path.read_text(encoding="utf-8"))
            if claim > measured
        )
    assert not offenders, (
        "Published Python throughput floor exceeds committed evidence:\n- "
        + "\n- ".join(offenders)
        + f"\nRemedy: lower the claim to a floor at or below {measured:,.0f}, "
        "or regenerate pyo3_step.json on the labeled profile and update the "
        "docs in the same change."
    )
