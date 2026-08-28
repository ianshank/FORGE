#!/usr/bin/env python3
"""PreToolUse guard: block Bash deletions whose glob/pattern would sweep up
a git-tracked file, not just the generated artifacts the author intended.

Motivated by a real incident during a FORGE tech-debt pass: a cleanup
command

    find /home/user/FORGE -maxdepth 1 -name ".coverage*" -delete

was meant to remove coverage.py's generated `.coverage.<host>.<pid>.<rand>`
temp files, but the glob also matched the tracked `.coveragerc` config and
deleted it. A stop-hook's "uncommitted changes" check caught it that time
(`git status` showed `D .coveragerc`, restored via `git checkout --`), but
nothing would have caught it had the very next command been a commit.

This hook cross-checks delete patterns against `git ls-files` ground truth
instead of a hand-maintained allow/deny list: a pattern that only matches
gitignored build artifacts (target/, node_modules/, __pycache__/, ...)
naturally matches zero tracked files and is allowed silently; a pattern
that also matches something tracked is blocked with the specific match(es)
named, so the fix is obvious.

Contract (Claude Code PreToolUse hooks): reads the hook payload as JSON on
stdin; exit 0 allows, exit 2 blocks and feeds the stderr message back to
the model as the reason. Every failure mode below (unreadable payload,
git unavailable, not a repo, unparseable command) fails OPEN (exit 0) --
this guard must never become the reason a legitimate command can't run.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
import subprocess
import sys

# Command shapes with a "hidden" blast radius: the files actually touched
# aren't spelled out literally in the command text, so a glob broader than
# intended can silently catch something it shouldn't. A bare `rm literal`
# (no wildcard, no recursion) is deliberately NOT matched here -- its blast
# radius is exactly the one named file, already visible in the command text
# and the tool-call permission prompt.
_FIND_DELETE_RE = re.compile(r"\bfind\b.*(-delete\b|-exec\s+rm\b)")
_RM_RE = re.compile(r"(^|[;&|]\s*)rm(\.exe)?\s")


def _tracked_files(cwd: str) -> list[str] | None:
    """Return repo-relative tracked paths under cwd, or None if not a repo."""
    try:
        check = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if check.returncode != 0 or check.stdout.strip() != "true":
            return None
        listing = subprocess.run(
            ["git", "-C", cwd, "ls-files"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if listing.returncode != 0:
            return None
        return [line for line in listing.stdout.splitlines() if line]
    except (OSError, subprocess.SubprocessError):
        return None


def _find_name_patterns(command: str) -> list[str]:
    r"""Extract every `-name PATTERN` argument (quoted or bare).

    The bare-argument branch stops at whitespace *and* at `;`/`&`/`|`: a
    real shell treats those as command separators regardless of adjacent
    whitespace (`-name .cover*;rm -rf /` is a `-name .cover*` argument
    followed by a separate `rm -rf /` command, not one `.cover*;rm`
    argument) -- a plain `\S+` would swallow the separator and everything
    after it into the "pattern", corrupting the match and letting a
    tracked-file hit slip through unblocked.
    """
    matches = re.findall(r"-name\s+(?:'([^']*)'|\"([^\"]*)\"|([^\s;&|]+))", command)
    return [next(g for g in groups if g) for groups in matches if any(groups)]


def _rm_targets(command: str) -> list[str]:
    """Best-effort extraction of rm arguments worth checking.

    Kept intentionally narrow: only arguments containing a shell glob
    character, or any argument at all once a recursive flag has been seen
    (an `rm -rf some_dir/` has unbounded blast radius even without a glob
    character in the name).
    """
    try:
        # punctuation_chars=True makes `;`, `&&`, `||`, `|` their own
        # tokens even with no surrounding whitespace (`rm -f .cover*;git
        # status` -> [..., ".cover*", ";", "git", "status"], not
        # [..., ".cover*;git", "status"]). Plain shlex.split() only splits
        # on whitespace, so a glob glued to a following command without a
        # space would be missed entirely -- a real false-negative, not a
        # hypothetical one.
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []

    targets: list[str] = []
    in_rm = False
    saw_recursive = False
    for tok in tokens:
        if tok in ("rm", "rm.exe"):
            in_rm = True
            saw_recursive = False
            continue
        if not in_rm:
            continue
        if tok in (";", "&&", "||", "|"):
            in_rm = False
            continue
        if tok.startswith("-"):
            if tok == "--recursive" or ("r" in tok.lstrip("-") and not tok.startswith("--")):
                saw_recursive = True
            continue
        if any(ch in tok for ch in "*?[") or saw_recursive:
            targets.append(tok)
    return targets


def _matches_tracked(pattern: str, tracked: list[str]) -> list[str]:
    hits = []
    for f in tracked:
        base = f.rsplit("/", 1)[-1]
        if fnmatch.fnmatch(base, pattern) or fnmatch.fnmatch(f, pattern):
            hits.append(f)
    return hits


def evaluate(payload: dict) -> tuple[int, str]:
    """Return (exit_code, message). message is empty when allowing."""
    if payload.get("tool_name") != "Bash":
        return 0, ""

    command = (payload.get("tool_input") or {}).get("command") or ""
    if not command:
        return 0, ""

    is_find_delete = bool(_FIND_DELETE_RE.search(command))
    is_rm = bool(_RM_RE.search(command))
    if not (is_find_delete or is_rm):
        return 0, ""

    cwd = payload.get("cwd") or "."
    tracked = _tracked_files(cwd)
    if tracked is None:
        return 0, ""

    patterns: list[str] = []
    if is_find_delete:
        patterns.extend(_find_name_patterns(command))
    if is_rm:
        patterns.extend(_rm_targets(command))

    all_hits: dict[str, list[str]] = {}
    for pat in patterns:
        hits = _matches_tracked(pat, tracked)
        if hits:
            all_hits[pat] = hits

    if not all_hits:
        return 0, ""

    lines = [
        "BLOCKED: this delete pattern matches git-tracked file(s), not just "
        "generated/ignored ones:",
    ]
    for pat, hits in all_hits.items():
        shown = ", ".join(hits[:5])
        more = f" (+{len(hits) - 5} more)" if len(hits) > 5 else ""
        lines.append(f"  pattern {pat!r} matches: {shown}{more}")
    lines.append(
        "If this is intentional, scope the command to exclude the tracked "
        "path(s) above, or delete them explicitly by name so the intent is "
        "visible in the command itself rather than relying on a wildcard "
        "that also happens to catch them."
    )
    return 2, "\n".join(lines)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # can't read the payload -- never block on a guard bug

    try:
        code, message = evaluate(payload)
    except Exception:
        return 0

    if code != 0 and message:
        print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
