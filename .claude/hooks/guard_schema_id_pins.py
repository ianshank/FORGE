#!/usr/bin/env python3
"""PreToolUse reminder: bump Rust/JS/Python schema_id pins when staging
Minecraft contract TOML.

`configs/minecraft/{action_map,rewards,milestone_rewards,crafting_rewards,
block_embeddings}.toml` are hashed into xlang pins. Editing the TOML
without updating the three language pins fails CI, but only after the
commit. This hook prints a reminder on `git commit` when any of those
paths are staged.

Advisory only: always exits 0 (fail open). Not a substitute for the
xlang tests. Contract matches the other `.claude/hooks/guard_*.py`
PreToolUse Bash hooks (JSON on stdin).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

_GIT_COMMIT_RE = re.compile(r"\bgit\s+commit\b")

#: Repo-relative paths whose content change must bump paired xlang pins.
PINNED_MC_CONFIG_PATHS: frozenset[str] = frozenset(
    {
        "configs/minecraft/action_map.toml",
        "configs/minecraft/rewards.toml",
        "configs/minecraft/milestone_rewards.toml",
        "configs/minecraft/crafting_rewards.toml",
        "configs/minecraft/block_embeddings.toml",
    }
)

_REMINDER = (
    "REMINDER: staged Minecraft contract TOML must keep Rust / JS / Python "
    "xlang pins in lockstep.\n\n"
    "Staged:\n{files}\n\n"
    "Update all three together:\n"
    "  - crates/forge-env-mc/src/{{action_map,reward_config,block_embeddings}}.rs "
    "(xlang_*_pinned_to_known_good / xlang_shipped_rewards_schema_id_folds_nested_files)\n"
    "  - mc-bot/test/{{schema_id,reward_config,block_embeddings}}.test.ts\n"
    "  - tests/python/training/test_muzero_mc_schema_id.py\n\n"
    "Nested reward file *contents* fold into schema_id; path-string rename "
    "without a content change does not bump. block_embeddings.toml is a "
    "separate obs-layout pin, not today's two-input schema_id.\n"
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


def matching_pinned_paths(staged: list[str]) -> list[str]:
    """Return staged paths that are Minecraft xlang-pin sources."""
    return [path for path in staged if path in PINNED_MC_CONFIG_PATHS]


def evaluate(payload: dict) -> tuple[int, str]:
    """Return (exit_code, message). Always 0 — advisory reminder only."""
    command = (payload.get("tool_input") or {}).get("command") or ""
    if payload.get("tool_name") != "Bash" or not _GIT_COMMIT_RE.search(command):
        return 0, ""

    cwd = payload.get("cwd") or "."
    matched = matching_pinned_paths(_staged_paths(cwd))
    if not matched:
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
