#!/usr/bin/env python3
"""PreToolUse reminder: refresh CompactReplay goldens when staging
fingerprint sources.

`configs/scenarios/*.toml` and `ForgeConfig` defaults
(`crates/forge-types/src/config.rs`, `constants.rs`) fold into
`CompactReplay.config_hash`. Editing them without regenerating
`tests/golden/replays/` plus a `docs/results/replay_flip_log.md` row
fails `cargo test -p forge-replay --test golden_replay`.

Advisory only: always exits 0 (fail open). Not a substitute for the
golden test. Contract matches the other `.claude/hooks/guard_*.py`
PreToolUse Bash hooks (JSON on stdin).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

_GIT_COMMIT_RE = re.compile(r"\bgit\s+commit\b")

#: Repo-relative files whose Default/serde shape is in the CompactReplay hash.
PINNED_CONFIG_SOURCE_PATHS: frozenset[str] = frozenset(
    {
        "crates/forge-types/src/config.rs",
        "crates/forge-types/src/constants.rs",
    }
)

_SCENARIO_PREFIX = "configs/scenarios/"
_GOLDEN_PREFIX = "tests/golden/replays/"
_FLIP_LOG = "docs/results/replay_flip_log.md"

_REMINDER = (
    "REMINDER: staged scenario TOML or ForgeConfig fields change the "
    "CompactReplay v2 fingerprint.\n\n"
    "Staged fingerprint sources:\n{files}\n\n"
    "Refresh the golden corpus in the same commit:\n"
    "  UPDATE_GOLDEN_REPLAYS=1 cargo test -p forge-replay --test golden_replay\n"
    "  then add a row to docs/results/replay_flip_log.md\n\n"
    "Do not UPDATE_GOLDEN_REPLAYS in the same cargo test invocation that "
    "also snapshot-reads the golden (tests run in parallel).\n"
    "This reminder does not block the commit."
)


def _staged_paths(cwd: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", cwd, "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def matching_fingerprint_sources(staged: list[str]) -> list[str]:
    """Return staged paths that change CompactReplay config_hash inputs."""
    return [
        path
        for path in staged
        if path in PINNED_CONFIG_SOURCE_PATHS
        or (path.startswith(_SCENARIO_PREFIX) and path.endswith(".toml"))
    ]


def golden_refresh_is_staged(staged: list[str]) -> bool:
    """True when a golden replay file and the flip log are both staged."""
    has_golden = any(path.startswith(_GOLDEN_PREFIX) for path in staged)
    has_flip = _FLIP_LOG in staged
    return has_golden and has_flip


def evaluate(payload: dict) -> tuple[int, str]:
    """Return (exit_code, message). Always 0 — advisory reminder only."""
    command = (payload.get("tool_input") or {}).get("command") or ""
    if payload.get("tool_name") != "Bash" or not _GIT_COMMIT_RE.search(command):
        return 0, ""

    cwd = payload.get("cwd") or "."
    staged = _staged_paths(cwd)
    matched = matching_fingerprint_sources(staged)
    if not matched:
        return 0, ""
    if golden_refresh_is_staged(staged):
        return 0, ""

    files = "\n".join(f"  - {p}" for p in matched)
    return 0, _REMINDER.format(files=files)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        _code, message = evaluate(payload)
    except Exception:
        return 0

    if message:
        print(message, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
