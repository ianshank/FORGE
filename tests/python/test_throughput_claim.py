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

#: ``SPS @ 8 | 40,000+`` or ``SPS @ 8 (ForgeAsyncVecEnv) | 40,000+``.
_SPS_AT_N = re.compile(
    r"SPS @ (?P<n>\d+)(?:\s*\([^)]*\))?\s*\|\s*(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<k>K)?\+",
    re.IGNORECASE,
)
#: Prose ``SPS @ 8: 40,000+``.
_SPS_AT_N_PROSE = re.compile(
    r"SPS @ (?P<n>\d+)\s*[:=]\s*(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<k>K)?\+",
    re.IGNORECASE,
)
#: ``replay fidelity 100%`` / table ``replay fidelity | 100%``.
_FIDELITY = re.compile(
    r"replay fidelity[^0-9%]{0,40}(?P<num>\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

VECENV_REPORT = REPO_ROOT / "benchmarks" / "baselines" / "cloud_agent" / "vecenv_step.json"
CLAIM_FILES: tuple[str, ...] = (
    "README.md",
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


def parse_sps_at_n_claims(text: str) -> list[tuple[int, int]]:
    """Return ``(n_envs, claimed_sps)`` floors published as ``SPS @ N``."""
    found: list[tuple[int, int]] = []
    for pattern in (_SPS_AT_N, _SPS_AT_N_PROSE):
        found.extend(
            (int(match.group("n")), _token_to_steps(match.group("num"), match.group("k")))
            for match in pattern.finditer(text)
        )
    return found


def parse_replay_fidelity_claims(text: str) -> list[float]:
    """Return published replay-fidelity percentages."""
    return [float(match.group("num")) for match in _FIDELITY.finditer(text)]


def load_vecenv_sps_by_n(path: Path) -> dict[int, float]:
    if not path.is_file():
        raise AssertionError(
            f"VecEnv SPS@N report missing at {path.relative_to(REPO_ROOT)}. "
            "Remedy: run "
            "`FORGE_RUN_VECENV_THROUGHPUT=1 "
            "FORGE_VECENV_THROUGHPUT_OUT=benchmarks/baselines/cloud_agent/vecenv_step.json "
            "pytest tests/python/test_vecenv_throughput.py -s -o addopts=` "
            "after `maturin develop`, then commit the report."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"VecEnv report {path.relative_to(REPO_ROOT)} is malformed: {exc}."
        ) from exc
    by_n = payload.get("by_n")
    if not isinstance(by_n, dict) or not by_n:
        raise AssertionError(
            f"VecEnv report {path.relative_to(REPO_ROOT)} has empty by_n. "
            "Do not publish SPS @ N until ForgeAsyncVecEnv is measured."
        )
    measured: dict[int, float] = {}
    for key, row in by_n.items():
        try:
            n_envs = int(key)
            value = float(row["steps_per_sec"])
        except (TypeError, ValueError, KeyError) as exc:
            raise AssertionError(
                f"VecEnv report {path.relative_to(REPO_ROOT)} has a bad row "
                f"for n_envs={key!r}: {exc}."
            ) from exc
        if value <= 0.0:
            raise AssertionError(
                f"VecEnv report n={n_envs} has steps_per_sec={value}."
            )
        measured[n_envs] = value
    return measured


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


def test_parse_sps_at_n_and_fidelity() -> None:
    text = (
        "| SPS @ 8 (ForgeAsyncVecEnv) | 40,000+ |\n"
        "SPS @ 1: 15,000+\n"
        "CompactReplay replay fidelity 100% on the v2 corpus.\n"
    )
    assert parse_sps_at_n_claims(text) == [(8, 40_000), (1, 15_000)]
    assert parse_replay_fidelity_claims(text) == [100.0]


def test_published_vecenv_sps_at_n_do_not_exceed_evidence() -> None:
    measured = load_vecenv_sps_by_n(VECENV_REPORT)
    offenders: list[str] = []
    for rel in CLAIM_FILES:
        path = REPO_ROOT / rel
        text = path.read_text(encoding="utf-8")
        for n_envs, claim in parse_sps_at_n_claims(text):
            if n_envs not in measured:
                offenders.append(
                    f"{rel} claims SPS @ {n_envs} but "
                    f"{VECENV_REPORT.relative_to(REPO_ROOT)} has no row for that N"
                )
                continue
            if claim > measured[n_envs]:
                offenders.append(
                    f"{rel} claims SPS @ {n_envs} = {claim:,}+ but measured "
                    f"{measured[n_envs]:,.0f}"
                )
    assert not offenders, (
        "Published SPS @ N exceeds committed ForgeAsyncVecEnv evidence:\n- "
        + "\n- ".join(offenders)
        + "\nRemedy: lower the floor, or regenerate vecenv_step.json."
    )


def test_published_replay_fidelity_is_at_most_100_percent() -> None:
    offenders: list[str] = []
    for rel in CLAIM_FILES:
        path = REPO_ROOT / rel
        offenders.extend(
            f"{rel} claims replay fidelity {value}% (> 100)"
            for value in parse_replay_fidelity_claims(path.read_text(encoding="utf-8"))
            if value > 100.0
        )
    assert not offenders, "Replay fidelity percent must be <= 100:\n- " + "\n- ".join(
        offenders
    )

