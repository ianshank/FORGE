#!/usr/bin/env python3
"""PreToolUse guard: scan staged changes for secrets before a `git commit`
actually runs, rather than finding out from CI (or not at all) after the
push.

`.github/workflows/security.yml`'s own `gitleaks` job is explicitly
report-only: it scans git *history* after a push, and its own step
comment says findings "are REPORTED... but never fail the build". Nothing
in this repo scans *before* a commit is made, so a secret can land in
local history -- and possibly get pushed -- well before anyone notices.
This hook closes that gap for any `Bash` command containing `git commit`:
it runs `gitleaks protect --staged`, which scans exactly the staged diff
(not full history -- fast, and scoped to what this commit would actually
introduce), and blocks the commit if it finds something.

Contract (Claude Code PreToolUse hooks): reads the hook payload as JSON on
stdin; exit 0 allows, exit 2 blocks and feeds the stderr message back to
the model as the reason. Every failure mode (unreadable payload, gitleaks
not installed, not a git repo, a gitleaks internal error unrelated to an
actual finding, any other internal error) fails OPEN (exit 0) -- this
guard must never become the reason a legitimate commit can't happen, and
it is intentionally *not* a substitute for the CI job (which still runs
against full history as a backstop) -- only an earlier, cheaper checkpoint.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys

_GIT_COMMIT_RE = re.compile(r"\bgit\s+commit\b")

# gitleaks' own default exit code for "leaks found" (its --exit-code flag
# would let this be something else, but this hook doesn't set that flag,
# so the default applies). Any OTHER non-zero code is a gitleaks-internal
# problem (bad config, scan error) unrelated to the user's actual staged
# content -- fail open on that rather than block a commit for a scanner
# bug.
_GITLEAKS_LEAKS_FOUND_EXIT_CODE = 1


def _is_git_repo(cwd: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def evaluate(payload: dict) -> tuple[int, str]:
    """Return (exit_code, message). message is empty when allowing."""
    command = (payload.get("tool_input") or {}).get("command") or ""
    if payload.get("tool_name") != "Bash" or not _GIT_COMMIT_RE.search(command):
        return 0, ""

    gitleaks = shutil.which("gitleaks")
    if gitleaks is None:
        return 0, ""

    cwd = payload.get("cwd") or "."
    if not _is_git_repo(cwd):
        return 0, ""

    try:
        result = subprocess.run(
            [gitleaks, "protect", "--staged", "--redact", "--no-banner"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0, ""

    if result.returncode != _GITLEAKS_LEAKS_FOUND_EXIT_CODE:
        return 0, ""

    findings = (result.stdout.strip() or result.stderr.strip())[:4000]
    message = (
        "BLOCKED: gitleaks found a likely secret in your staged changes:\n\n"
        f"{findings}\n\n"
        "Unstage and remove the secret before committing (`git restore "
        "--staged <file>`, edit it out, re-add). If this is a false "
        "positive, add a `.gitleaksignore` entry or a `gitleaks:allow` "
        "comment on that line."
    )
    return 2, message


def main() -> int:
    # Single broad try/except around the whole body: every failure mode
    # here must fail open, never block a legitimate commit because the
    # guard itself broke (or because gitleaks isn't installed -- CI's own
    # gitleaks job is the backstop, not this hook).
    try:
        payload = json.load(sys.stdin)
        code, message = evaluate(payload)
    except Exception:
        return 0

    if code != 0 and message:
        print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
