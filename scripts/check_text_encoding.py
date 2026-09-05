"""Text-encoding drift guard for tracked source files.

Two failure modes have each bitten this repository, both of them silent at
review time because git renders the affected file as ``Bin`` and shows no
diff at all:

1. **A raw NUL byte in a text file.** Commit ``0303612`` had to remove one
   from ``dashboard/e2e/aqa/networkGuard.ts``; the same thing happened again
   in ``mc-bot/test/security.test.ts``, where a test payload for NUL-byte
   injection was written as a literal ``\\x00`` instead of the ``\\u0000``
   escape. Both files still compiled and their tests still passed, so no
   existing gate noticed. The cost is not correctness but reviewability: a
   binary blob cannot be diffed, cannot carry line-level review comments,
   and hides every subsequent change to that file.

2. **Line-ending flattening.** Commit ``d51ba6d`` had to restore CRLF in
   ``CHANGELOG.md`` after a text-mode edit rewrote ~2000 lines. The repo
   pins that file ``-text`` in ``.gitattributes`` precisely to stop this,
   but nothing verified the pin held.

``.claude/hooks/guard_line_ending_drift.py`` covers (2) for Claude Code
sessions only -- it is a ``PreToolUse`` hook, so a human contributor using
plain ``git commit`` gets nothing, and neither hook covers (1) at all.
This script is the contributor-agnostic backstop: it runs in CI (the
``python-lint`` job) and via ``make text-check``, so it applies to every
commit regardless of who or what authored it.

Exit codes
----------
``0`` clean, ``1`` drift detected, ``2`` the check itself could not run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

EXIT_OK: Final[int] = 0
EXIT_DRIFT: Final[int] = 1
EXIT_ERROR: Final[int] = 2

# Extensions whose content is legitimately binary. Everything tracked that is
# NOT one of these is treated as text and must contain no NUL bytes.
BINARY_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".webp",
        ".bmp",
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".xz",
        ".zst",
        ".bz2",
        ".wasm",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
        ".bin",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".mp4",
        ".webm",
        ".mp3",
        ".wav",
        ".ogg",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".a",
        ".o",
        ".rlib",
    }
)

# Files that are known and intended to carry CRLF. This is a *snapshot*, not
# a policy endorsement -- most of these predate the convention and are simply
# not worth a whole-file rewrite (which would bury their real history under a
# 100%-changed diff). The point of pinning the set is that it cannot grow or
# shrink by accident: a new CRLF file, or one of these silently flattened to
# LF, fails the check and forces a deliberate decision.
#
# CHANGELOG.md is the only entry that is also declared `-text` in
# .gitattributes, because it is the one actively edited by tooling.
#
# To change this set intentionally: make the edit, run
# `python3 scripts/check_text_encoding.py --print-inventory`, and paste the
# result here in the same commit.
EXPECTED_CRLF: Final[frozenset[str]] = frozenset(
    {
        ".gitignore",
        "CHANGELOG.md",
        "configs/minecraft/env.toml",
        "configs/minecraft/reset.toml",
        "crates/forge-bench/Cargo.toml",
        "crates/forge-civ/Cargo.toml",
        "crates/forge-civ/src/grid_topology.rs",
        "crates/forge-cloud/src/backend.rs",
        "crates/forge-env-forge/tests/forge_env_parity.rs",
        "crates/forge-env-mc/src/error.rs",
        "crates/forge-env-mc/src/protocol.rs",
        "crates/forge-mangomas/src/transfer/export.rs",
        "docs/plans/minecraft_rl_integration_plan_v2.md",
        "mc-bot/package.json",
        "tests/python/test_sb3_integration.py",
    }
)


def tracked_files(repo_root: Path) -> list[str]:
    """Return every path git tracks, as repo-relative POSIX strings.

    Raises:
        RuntimeError: if ``git ls-files`` cannot be run.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        raise RuntimeError(f"could not enumerate tracked files: {exc}") from exc
    return [chunk.decode("utf-8") for chunk in completed.stdout.split(b"\0") if chunk]


def is_text_candidate(path: str) -> bool:
    """True when ``path`` should be held to the text-encoding rules."""
    return Path(path).suffix.lower() not in BINARY_SUFFIXES


def scan(repo_root: Path) -> tuple[list[str], set[str]]:
    """Scan the tracked tree.

    Returns:
        ``(nul_offenders, crlf_paths)`` -- text files containing a NUL byte,
        and every tracked file containing at least one CRLF pair.
    """
    nul_offenders: list[str] = []
    crlf_paths: set[str] = set()

    for rel in tracked_files(repo_root):
        blob = repo_root / rel
        try:
            data = blob.read_bytes()
        except (OSError, IsADirectoryError):
            # Submodule entry or a path removed from the working tree; the
            # index still lists it, but there is nothing here to inspect.
            continue

        if b"\r\n" in data:
            crlf_paths.add(rel)
        if is_text_candidate(rel) and b"\x00" in data:
            nul_offenders.append(rel)

    return nul_offenders, crlf_paths


def describe_nul(repo_root: Path, rel: str) -> str:
    """Render a one-line locator for the first NUL byte in ``rel``."""
    data = (repo_root / rel).read_bytes()
    index = data.index(b"\x00")
    line = data[:index].count(b"\n") + 1
    start = data.rfind(b"\n", 0, index) + 1
    end = data.find(b"\n", index)
    excerpt = data[start : end if end != -1 else len(data)]
    return f"{rel}:{line}: {excerpt!r}"


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring for exit codes."""
    args = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parent.parent

    try:
        nul_offenders, crlf_paths = scan(repo_root)
    except RuntimeError as exc:
        print(f"check_text_encoding: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if "--print-inventory" in args:
        for path in sorted(crlf_paths):
            print(f'        "{path}",')
        return EXIT_OK

    failed = False

    if nul_offenders:
        failed = True
        print(
            "NUL byte in text-tracked file(s). git renders these as binary, so "
            "they cannot be diffed or reviewed line-by-line.\n"
            "If the byte is a deliberate test payload, write it as an escape "
            "(\\u0000 in TS/JS, \\0 in Rust, \\x00 in Python) instead of a "
            "literal:",
            file=sys.stderr,
        )
        for rel in sorted(nul_offenders):
            print(f"  {describe_nul(repo_root, rel)}", file=sys.stderr)

    unexpected = sorted(crlf_paths - EXPECTED_CRLF)
    if unexpected:
        failed = True
        print(
            "\nFile(s) newly carrying CRLF. This repo is LF by convention "
            "(.editorconfig); if the change is intended, add the path to "
            "EXPECTED_CRLF in this script in the same commit:",
            file=sys.stderr,
        )
        for rel in unexpected:
            print(f"  {rel}", file=sys.stderr)

    flattened = sorted(EXPECTED_CRLF - crlf_paths)
    if flattened:
        failed = True
        print(
            "\nFile(s) expected to carry CRLF have been flattened to LF. This "
            "is the d51ba6d incident: a text-mode edit rewriting every line. "
            "Restore the line endings, or drop the path from EXPECTED_CRLF if "
            "the flattening was intended:",
            file=sys.stderr,
        )
        for rel in flattened:
            print(f"  {rel}", file=sys.stderr)

    if failed:
        return EXIT_DRIFT

    print(
        f"check_text_encoding: OK "
        f"({len(crlf_paths)} CRLF file(s) as expected, no NUL bytes in text files)"
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
