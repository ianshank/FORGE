#!/usr/bin/env python3
"""Fail when requires-python advertises minors CI does not smoke."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

_REQUIRES_RE = re.compile(r'^requires-python\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
_PY_VER_RE = re.compile(r'python-version:\s*["\']?(\d+\.\d+)["\']?', re.MULTILINE)


def parse_requires_floor(text: str) -> tuple[int, int]:
    match = _REQUIRES_RE.search(text)
    if not match:
        raise ValueError("pyproject.toml has no requires-python")
    spec = match.group(1).strip()
    m = re.search(r">=\s*(\d+)\.(\d+)", spec)
    if not m:
        raise ValueError(f"unsupported requires-python form: {spec!r}")
    return int(m.group(1)), int(m.group(2))


def parse_ci_minors(text: str) -> set[tuple[int, int]]:
    minors = {
        (int(a), int(b))
        for a, b in (m.group(1).split(".") for m in _PY_VER_RE.finditer(text))
    }
    if not minors:
        raise ValueError("no python-version pins found in ci.yml")
    return minors


def evaluate(
    requires_floor: tuple[int, int], ci_minors: set[tuple[int, int]]
) -> list[str]:
    problems: list[str] = []
    lowest_ci = min(ci_minors)
    if requires_floor < lowest_ci:
        problems.append(
            f"requires-python floor {requires_floor[0]}.{requires_floor[1]} is below "
            f"lowest CI pin {lowest_ci[0]}.{lowest_ci[1]} (advertises untested minors)"
        )
    return problems


def main() -> int:
    requires = parse_requires_floor(PYPROJECT.read_text(encoding="utf-8"))
    ci_minors = parse_ci_minors(CI_WORKFLOW.read_text(encoding="utf-8"))
    problems = evaluate(requires, ci_minors)
    if problems:
        print("Python support-matrix honesty check failed:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print(
        f"OK: requires-python >={requires[0]}.{requires[1]} ; CI pins "
        f"{sorted(f'{a}.{b}' for a, b in ci_minors)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
