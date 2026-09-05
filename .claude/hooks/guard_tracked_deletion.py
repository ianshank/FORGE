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

Detection deliberately favors recall over precision: whether `rm`/`find
-delete` is "really" being invoked (vs. merely appearing as a token) is
NOT what gates blocking -- an over-liberal detection only costs one extra,
cheap `git ls-files` cross-reference for a command that turns out to need
no scrutiny (still allowed silently, zero externally visible difference).
What gates blocking is always a REAL match against tracked files (or, for
a `find ... -delete` with no usable name/path filter at all, an
unverifiable and therefore unbounded blast radius). An adversarial review
of an earlier version of this hook confirmed real, reproducible bypasses
of a *precision*-favoring `rm` detector (`find . -name X | xargs rm -rf`,
`sudo rm -rf`, `VAR=x rm -rf`, `(rm -rf ...)`, `{ rm -rf ...; }`, a single
leading space, a newline instead of `;`/`&&`) -- see `_command_tokens`.

Contract (Claude Code PreToolUse hooks): reads the hook payload as JSON on
stdin; exit 0 allows, exit 2 blocks and feeds the stderr message back to
the model as the reason. Every failure mode (unreadable payload, git
unavailable, not a repo, unparseable command, any other internal error)
fails OPEN (exit 0) -- this guard must never become the reason a
legitimate command can't run.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
import subprocess
import sys

_RM_TOKENS = frozenset({"rm", "rm.exe"})
_SEPARATOR_TOKENS = frozenset({";", "&&", "||", "|"})

def _selector_re(flag: str) -> re.Pattern[str]:
    return re.compile(rf"{flag}\s+(?:'([^']*)'|\"([^\"]*)\"|([^\s;&|]+))")


# Each (regex, case_insensitive) pair. `-name`/`-iname` and
# `-path`/`-ipath`/`-wholename`/`-iwholename` never overlap as substrings
# of each other (an `-i...` flag's `-` is always followed by `i`, never by
# the case-sensitive flag's own first letter), so matching all four
# independently can't double-count one occurrence under two flags.
_SELECTOR_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (_selector_re("-name"), False),
    (_selector_re("-iname"), True),
    (_selector_re("-(?:path|wholename)"), False),
    (_selector_re("-i(?:path|wholename)"), True),
)
# `-regex`/`-iregex` selectors exist but use find's own regex dialect, not
# a shell glob -- not attempted here; their presence is tracked separately
# so a `find ... -regex ... -delete` with no ALSO-present -name/-path
# selector still counts as "no usable filter" (unbounded blast radius)
# rather than silently falling through as if no selector had been
# requested at all.
_REGEX_SELECTOR_RE = re.compile(r"-i?regex\b")
_FIND_DELETE_ACTION_RE = re.compile(r"-delete\b|-exec\s+rm\b")


def _find_piped_to_rm(tokens: list[str]) -> bool:
    """True if a `find` invocation's output is piped into something that
    invokes `rm` within the same pipeline segment -- e.g. `find . -name X
    | xargs rm -rf`. The actual delete target in that shape comes from
    find's OWN `-name`/`-iname`/`-path`/`-ipath` selector, not from any
    literal argument after `rm` (xargs supplies it at runtime, so there's
    nothing for `_rm_targets` to extract), which is why this is detected
    separately and, once true, routes through the same find-selector
    extraction as `-delete`/`-exec rm` rather than `_rm_targets`.
    """
    saw_find = False
    saw_pipe_after_find = False
    for tok in tokens:
        if tok in _SEPARATOR_TOKENS - {"|"}:
            saw_find = False
            saw_pipe_after_find = False
            continue
        if tok == "find":
            saw_find = True
            continue
        if tok == "|" and saw_find:
            saw_pipe_after_find = True
            continue
        if saw_pipe_after_find and tok in _RM_TOKENS:
            return True
    return False


def _lex(text: str) -> list[str] | None:
    """shlex `text` into tokens, or None if it doesn't parse.

    `punctuation_chars=True` makes `;`, `&`, `&&`, `|`, `||`, and `(`/`)`
    their own tokens even with no surrounding whitespace (`rm -f
    .cover*;git status` -> [..., ".cover*", ";", "git", "status"], not
    [..., ".cover*;git", "status"]) -- plain shlex.split() only splits on
    whitespace, so a glob glued to a following command without a space
    would be missed entirely.
    """
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return None


def _continues_next_line(raw_line: str, tokens: list[str]) -> bool:
    """Does `raw_line` continue into the next one rather than ending a command?

    Two cases, and getting either wrong is a false *negative* -- the
    dangerous direction -- because a spurious separator would end `rm`'s
    argument list early and let the real targets through unexamined:

    * a trailing backslash is an explicit line continuation;
    * a line whose last token is `&&`, `||`, `|` or `;` is mid-list, and
      already carries its own separator, so adding another is pointless.
    """
    return raw_line.rstrip().endswith("\\") or (
        bool(tokens) and tokens[-1] in _SEPARATOR_TOKENS
    )


def _command_tokens(command: str) -> list[str] | None:
    """Tokenize the whole command, or None if it doesn't parse.

    Newlines are turned into explicit `;` separator tokens, because shlex
    does the opposite of what is wanted here: a newline is ordinary
    whitespace to it, so it is *consumed* rather than emitted. This
    function's docstring used to claim a newline was "tokenized the same as
    if it were `;`-separated"; it was not, and the consequence was a
    false positive that fired in practice.

        rm -rf build_output/
        grep -n pattern .gitignore

    lexes to [rm, -rf, build_output/, grep, -n, pattern, .gitignore], with
    nothing to tell `_rm_targets` that `rm`'s arguments ended at the line
    break -- so with `-r` seen, `.gitignore` is collected as a recursive
    delete target and a tracked file is reported. Any multi-line command
    that mentions a tracked path after an `rm -r` on an earlier line hits
    this, which is common enough to train people to work around the guard.

    Lines are lexed individually and rejoined with `;`, except where the
    previous line continues (see `_continues_next_line`). A line that fails
    to lex on its own -- a quoted string spanning lines is the realistic
    case -- makes this fall back to lexing the whole command at once, which
    is exactly the previous behaviour, so nothing that used to be caught
    stops being caught.
    """
    lines = command.splitlines()
    if len(lines) <= 1:
        return _lex(command)

    tokens: list[str] = []
    previous_raw = ""
    for raw_line in lines:
        line_tokens = _lex(raw_line)
        if line_tokens is None:
            # A line that doesn't parse alone (e.g. an open quote continuing
            # onto the next line). Fall back to whole-command lexing rather
            # than guessing; over-collecting targets is the safe direction.
            return _lex(command)
        # Whether to separate is decided by the line just *ended*, not the
        # one about to start -- the trailing backslash and the dangling
        # `&&` both live at the end of the previous line.
        if tokens and line_tokens and not _continues_next_line(previous_raw, tokens):
            tokens.append(";")
        tokens.extend(line_tokens)
        previous_raw = raw_line
    return tokens


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


def _find_selectors(command: str) -> tuple[list[tuple[str, bool]], bool]:
    """Extract (pattern, case_insensitive) pairs from -name/-iname/-path/
    -ipath/-wholename/-iwholename selectors, plus whether an unevaluated
    -regex/-iregex selector is present.
    """
    selectors: list[tuple[str, bool]] = []
    for regex, case_insensitive in _SELECTOR_PATTERNS:
        for match in regex.finditer(command):
            pattern = next(g for g in match.groups() if g)
            selectors.append((pattern, case_insensitive))
    has_unevaluated_regex = bool(_REGEX_SELECTOR_RE.search(command))
    return selectors, has_unevaluated_regex


def _rm_targets(tokens: list[str]) -> list[tuple[str, bool]]:
    """Best-effort extraction of (target, is_recursive) pairs worth
    checking from an already-tokenized command.

    Kept intentionally narrow: only arguments containing a shell glob
    character, or any argument at all once a recursive flag has been seen
    (an `rm -rf some_dir` has unbounded blast radius even without a glob
    character in the name -- and even without a trailing `/`, which `rm`
    doesn't require to recurse into a directory). `is_recursive` tells the
    caller whether -r/-rf was active for this specific target, so it can
    also be matched as a directory prefix, not just a literal file glob.
    """
    targets: list[tuple[str, bool]] = []
    in_rm = False
    saw_recursive = False
    for tok in tokens:
        if tok in _RM_TOKENS:
            in_rm = True
            saw_recursive = False
            continue
        if not in_rm:
            continue
        if tok in _SEPARATOR_TOKENS or tok in ("(", ")", "{", "}"):
            in_rm = False
            continue
        if tok.startswith("-"):
            if tok == "--recursive" or ("r" in tok.lstrip("-") and not tok.startswith("--")):
                saw_recursive = True
            continue
        if any(ch in tok for ch in "*?[") or saw_recursive:
            targets.append((tok, saw_recursive))
    return targets


def _matches_tracked(
    pattern: str,
    tracked: list[str],
    *,
    case_insensitive: bool = False,
    directory_prefix: bool = False,
) -> list[str]:
    """Match `pattern` against tracked repo-relative paths.

    Tries the pattern's own basename against each tracked file's basename
    (handles a bare glob, a `./`-relative pattern, and an absolute-looking
    pattern alike, without needing to resolve an absolute path against
    cwd -- conservative, matching this guard's existing basename-fallback
    design, and it only ever widens what counts as a match, never
    narrows it), and a leading-`./`-stripped pattern against the tracked
    file's full relative path (handles a directory-scoped pattern like
    `demo_ui/.cover*`).

    `directory_prefix=True` (set for an `rm -r`/`-rf` target) additionally
    treats `pattern` as a directory to recurse into: a tracked file whose
    path starts with `pattern/` -- or equals `pattern` outright -- counts
    as a hit even with no glob character and no trailing slash on
    `pattern` (`rm -rf src` deletes `src/main.rs` exactly as `rm -rf src/`
    would; a plain fnmatch of the literal string "src" or "src/" against
    "src/main.rs" never matches, which is the gap this branch closes).
    """

    def norm(s: str) -> str:
        return s.lower() if case_insensitive else s

    pattern_basename = pattern.rsplit("/", 1)[-1]
    stripped = pattern
    while stripped.startswith("./"):
        stripped = stripped[2:]
    dir_prefix = stripped.rstrip("/")

    hits = []
    for f in tracked:
        base = f.rsplit("/", 1)[-1]
        nf = norm(f)
        if (
            fnmatch.fnmatch(norm(base), norm(pattern_basename))
            or fnmatch.fnmatch(nf, norm(stripped))
            or (
                directory_prefix
                and dir_prefix
                and (nf == norm(dir_prefix) or nf.startswith(norm(dir_prefix) + "/"))
            )
        ):
            hits.append(f)
    return hits


def _collect_candidates(
    command: str, tokens: list[str], *, is_find_delete: bool, is_rm: bool
) -> tuple[list[tuple[str, bool, bool]], bool, bool]:
    """Return ((pattern, case_insensitive, directory_prefix) triples to
    cross-check, whether a find-delete's blast radius couldn't be bounded
    at all, and whether that's specifically because only an unevaluated
    -regex/-iregex selector was present).
    """
    candidates: list[tuple[str, bool, bool]] = []
    unbounded_find_delete = False
    unevaluated_regex_only = False

    if is_find_delete:
        selectors, has_unevaluated_regex = _find_selectors(command)
        candidates.extend((pat, case_insensitive, False) for pat, case_insensitive in selectors)
        if not selectors:
            # No -name/-iname/-path/-ipath filter this guard can verify --
            # either genuinely no filter (`find . -type f -delete` deletes
            # every regular file) or only an unevaluated -regex/-iregex.
            # Either way the blast radius can't be confirmed safe, so this
            # blocks outright rather than silently passing "nothing to
            # check" through.
            unbounded_find_delete = True
            unevaluated_regex_only = has_unevaluated_regex

    if is_rm:
        candidates.extend((pat, False, is_recursive) for pat, is_recursive in _rm_targets(tokens))

    return candidates, unbounded_find_delete, unevaluated_regex_only


def _build_block_message(
    all_hits: dict[str, list[str]], *, unbounded_find_delete: bool, unevaluated_regex_only: bool
) -> str:
    lines = []
    if all_hits:
        lines.append(
            "BLOCKED: this delete pattern matches git-tracked file(s), not "
            "just generated/ignored ones:"
        )
        for pat, hits in all_hits.items():
            shown = ", ".join(hits[:5])
            more = f" (+{len(hits) - 5} more)" if len(hits) > 5 else ""
            lines.append(f"  pattern {pat!r} matches: {shown}{more}")
        lines.append(
            "If this is intentional, scope the command to exclude the "
            "tracked path(s) above, or delete them explicitly by name so "
            "the intent is visible in the command itself rather than "
            "relying on a wildcard that also happens to catch them."
        )
        return "\n".join(lines)

    assert unbounded_find_delete  # only caller of this branch guarantees it
    reason = (
        "only a -regex/-iregex filter, which this guard doesn't evaluate"
        if unevaluated_regex_only
        else "no -name/-iname/-path/-ipath filter at all"
    )
    lines.append(
        f"BLOCKED: this `find ... -delete`/`-exec rm` has {reason} -- "
        "its blast radius can't be confirmed safe against this repo's "
        "tracked files."
    )
    lines.append(
        "Scope it with an explicit -name/-path filter, or delete the "
        "specific file(s) by name instead."
    )
    return "\n".join(lines)


def evaluate(payload: dict) -> tuple[int, str]:
    """Return (exit_code, message). message is empty when allowing."""
    command = (payload.get("tool_input") or {}).get("command") or ""
    if payload.get("tool_name") != "Bash" or not command:
        return 0, ""

    tokens = _command_tokens(command)
    if tokens is None:
        return 0, ""

    is_rm = any(tok in _RM_TOKENS for tok in tokens)
    is_find_delete = "find" in tokens and (
        bool(_FIND_DELETE_ACTION_RE.search(command)) or _find_piped_to_rm(tokens)
    )
    if not (is_rm or is_find_delete):
        return 0, ""

    cwd = payload.get("cwd") or "."
    tracked = _tracked_files(cwd)
    if tracked is None:
        return 0, ""

    candidates, unbounded_find_delete, unevaluated_regex_only = _collect_candidates(
        command, tokens, is_find_delete=is_find_delete, is_rm=is_rm
    )

    all_hits: dict[str, list[str]] = {}
    for pattern, case_insensitive, directory_prefix in candidates:
        hits = _matches_tracked(
            pattern, tracked, case_insensitive=case_insensitive, directory_prefix=directory_prefix
        )
        if hits:
            all_hits[pattern] = hits

    if not all_hits and not unbounded_find_delete:
        return 0, ""

    message = _build_block_message(
        all_hits, unbounded_find_delete=unbounded_find_delete, unevaluated_regex_only=unevaluated_regex_only
    )
    return 2, message


def main() -> int:
    # Single broad try/except around the whole body, not just json.load:
    # every failure mode here -- an unreadable stdin, a malformed payload,
    # a bug in evaluate() -- must fail open, never block a legitimate
    # command because the guard itself broke.
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
