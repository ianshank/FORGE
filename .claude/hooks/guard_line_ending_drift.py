#!/usr/bin/env python3
"""PreToolUse guard: block a `git commit` that silently flips a tracked
file's line-ending convention across most of the file, rather than finding
out from an unreviewably huge diff after the fact.

This is not hypothetical: while preparing PR #134 on this repo, a Python
`open(p).read()` / `open(p, 'w').write(s)` edit round-trip silently
flattened CHANGELOG.md from CRLF to LF -- Python's default universal-
newlines translation strips `\r` on read and never restores it on write.
CHANGELOG.md was the one file in the repo carrying CRLF (every other .md
is LF), so a ~40-line content edit landed as a ~2000-line whole-file
rewrite. Nothing caught it before the commit; it was only noticed
afterward via an abnormally large `git diff --stat`.

Detection: for each already-tracked file about to be committed (a
brand-new file's line-ending choice isn't "drift" -- there's nothing to
drift from), compare the fraction of CRLF-terminated lines in the
committed blob (HEAD) against the fraction in the staged blob (the git
index). A change of more than half the file's lines is the signature of a
whole-file EOL flip -- no legitimate content edit does this; it is always
a tooling side effect.

Files with an explicit `.gitattributes` line-ending declaration for their
path (`-text`, or an `eol=` attribute) are exempt: that declaration IS the
record of a deliberate decision (see CHANGELOG.md's own entry, added by
the commit that fixed this exact incident) -- this guard has nothing
further to check there.

Contract (Claude Code PreToolUse hooks): reads the hook payload as JSON on
stdin; exit 0 allows, exit 2 blocks and feeds the stderr message back to
the model as the reason. Every failure mode (unreadable payload, not a
git repo, any internal error) fails OPEN (exit 0) -- this guard must
never become the reason a legitimate commit can't happen, and it is
intentionally advisory-strength, not a substitute for actually reviewing
the diff.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

_GIT_COMMIT_RE = re.compile(r"\bgit\s+commit\b")

# A file whose CRLF-line fraction moves by more than this between HEAD and
# the staged version is treated as a whole-file EOL flip. 0.5 (half the
# file's lines) comfortably separates "a normal edit touched a few lines
# whose endings happened to differ" from "something rewrote the whole file."
_FLIP_THRESHOLD = 0.5

# Skip files with fewer lines than this on either side: on a small file, a
# handful of genuinely-edited lines can swing the ratio past the threshold
# without any whole-file rewrite having happened.
_MIN_LINES_TO_CHECK = 20


def _crlf_fraction(content: bytes) -> float:
    total = content.count(b"\n")
    if total == 0:
        return 0.0
    return content.count(b"\r\n") / total


def _staged_modified_files(cwd: str) -> list[str]:
    # --diff-filter=M: only files that already existed at HEAD and are
    # being modified. A newly-added file has no prior convention to drift
    # from; a deleted file has no staged content to inspect.
    result = subprocess.run(
        ["git", "-C", cwd, "diff", "--cached", "--name-only", "--diff-filter=M"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _is_attribute_exempt(cwd: str, path: str) -> bool:
    """True if .gitattributes declares this path binary or a specific eol=."""
    result = subprocess.run(
        ["git", "-C", cwd, "check-attr", "text", "eol", "--", path],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        # Format: "<path>: <attr>: <value>"
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        attr, value = parts[1].strip(), parts[2].strip()
        if attr == "text" and value == "unset":
            return True
        if attr == "eol" and value not in ("unspecified", "unset"):
            return True
    return False


def _blob_at(cwd: str, ref: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", cwd, "show", f"{ref}:{path}"],
        capture_output=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def evaluate(payload: dict) -> tuple[int, str]:
    """Return (exit_code, message). message is empty when allowing."""
    command = (payload.get("tool_input") or {}).get("command") or ""
    if payload.get("tool_name") != "Bash" or not _GIT_COMMIT_RE.search(command):
        return 0, ""

    cwd = payload.get("cwd") or "."
    flipped: list[str] = []

    for path in _staged_modified_files(cwd):
        if _is_attribute_exempt(cwd, path):
            continue

        head_blob = _blob_at(cwd, "HEAD", path)
        # Empty ref, not ":": _blob_at's own template is f"{ref}:{path}", so
        # ref=":" would build "::path" (git error, silently caught as "not
        # found" by the returncode check) instead of the intended ":path"
        # -- git's syntax for "this path's stage-0 (normal) index content."
        staged_blob = _blob_at(cwd, "", path)
        if head_blob is None or staged_blob is None:
            continue

        head_lines = head_blob.count(b"\n")
        staged_lines = staged_blob.count(b"\n")
        if min(head_lines, staged_lines) < _MIN_LINES_TO_CHECK:
            continue

        delta = abs(_crlf_fraction(head_blob) - _crlf_fraction(staged_blob))
        if delta > _FLIP_THRESHOLD:
            flipped.append(path)

    if not flipped:
        return 0, ""

    files = "\n".join(f"  - {p}" for p in flipped)
    message = (
        "BLOCKED: this commit flips the line-ending convention across most "
        "of the following already-tracked file(s):\n\n"
        f"{files}\n\n"
        "This is almost always a tooling side effect, not an intentional "
        "edit -- e.g. a Python `open(p).read()` / `open(p, 'w').write(s)` "
        "round-trip silently strips CRLF to LF on read and never restores "
        "it on write. Prefer `sed` or the Edit tool over a Python text-mode "
        "read/write for a file whose line-ending convention you haven't "
        "verified.\n\n"
        "If this file's line endings should genuinely change, do it "
        "deliberately: check `file <path>` before and after, and add a "
        "`.gitattributes` entry (see CHANGELOG.md's own entry for the "
        "pattern) recording the file's convention -- that exempts it from "
        "this check going forward."
    )
    return 2, message


def main() -> int:
    # Single broad try/except around the whole body: every failure mode
    # here must fail open, never block a legitimate commit because the
    # guard itself broke.
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
