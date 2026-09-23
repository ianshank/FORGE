"""Tests for ``scripts/check_text_encoding.py``.

Every test runs against a throwaway git repository created under
``tmp_path`` -- the module's ``scan()`` takes ``repo_root`` as a parameter,
so no monkeypatching is needed and no real repository file is ever touched,
even transiently.

The guard exists because two encoding faults have each shipped here and
neither was visible in review (git renders the affected file as ``Bin``):
a raw NUL byte making a ``.ts`` file binary (commit ``0303612``, and again
in ``mc-bot/test/security.test.ts``), and ``CHANGELOG.md`` being flattened
from CRLF to LF by a text-mode edit (commit ``d51ba6d``). Both failure
modes are asserted here, in both directions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import check_text_encoding as cte
import pytest


def _init_repo(root: Path, files: dict[str, bytes]) -> Path:
    """Create a git repo at ``root`` containing ``files`` and stage them."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for rel, payload in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_scan_reports_clean_tree(tmp_path: Path) -> None:
    """A tree of plain LF text files yields no offenders and no CRLF paths."""
    repo = _init_repo(
        tmp_path / "clean",
        {"src/main.rs": b"fn main() {}\n", "README.md": b"# hi\n"},
    )

    nul_offenders, crlf_paths = cte.scan(repo)

    assert nul_offenders == []
    assert crlf_paths == set()


def test_scan_flags_nul_byte_in_text_file(tmp_path: Path) -> None:
    """A literal NUL in a .ts file is reported -- the 0303612 recurrence."""
    repo = _init_repo(
        tmp_path / "nul_test",
        {"test/security.test.ts": b"const payloads = [\n  '@s\x00',\n];\n"},
    )

    nul_offenders, _ = cte.scan(repo)

    assert nul_offenders == ["test/security.test.ts"]


def test_scan_ignores_nul_in_genuinely_binary_file(tmp_path: Path) -> None:
    """A NUL inside a .png is normal content, not drift."""
    repo = _init_repo(tmp_path / "bin", {"docs/chart.png": b"\x89PNG\x00\x00data"})

    nul_offenders, _ = cte.scan(repo)

    assert nul_offenders == []


def test_scan_collects_crlf_paths(tmp_path: Path) -> None:
    """Only the CRLF file is collected; its LF sibling is not."""
    repo = _init_repo(
        tmp_path / "crlf",
        {"CHANGELOG.md": b"# Changelog\r\n\r\nentry\r\n", "plain.md": b"# plain\n"},
    )

    _, crlf_paths = cte.scan(repo)

    assert crlf_paths == {"CHANGELOG.md"}


def test_describe_nul_locates_line_and_shows_context(tmp_path: Path) -> None:
    """The locator names the 1-based line and echoes the offending line."""
    repo = _init_repo(
        tmp_path / "locate",
        {"a.ts": b"line one\nline two\nbad '@s\x00' here\n"},
    )

    described = cte.describe_nul(repo, "a.ts")

    assert described.startswith("a.ts:3:")
    assert "@s" in described


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/main.rs", True),
        ("test/x.test.ts", True),
        ("CHANGELOG.md", True),
        ("Makefile", True),
        ("docs/chart.PNG", False),
        ("model/net.onnx", False),
        ("pkg/forge.wasm", False),
    ],
)
def test_is_text_candidate(path: str, expected: bool) -> None:
    """Binary suffixes are exempt from the NUL rule; everything else is not."""
    assert cte.is_text_candidate(path) is expected


@pytest.mark.skipif(sys.platform == "win32", reason="Git core.autocrlf causes false failures on Windows.")
def test_expected_crlf_matches_the_real_repository() -> None:
    """The pinned EXPECTED_CRLF snapshot still describes the actual tree.

    This is the test that makes the snapshot load-bearing: if someone adds a
    CRLF file, or flattens one of the pinned ones, this fails and forces the
    constant to be updated deliberately rather than drifting.
    """
    repo_root = Path(cte.__file__).resolve().parent.parent

    _, crlf_paths = cte.scan(repo_root)

    assert crlf_paths == set(cte.EXPECTED_CRLF)


def test_repository_has_no_nul_bytes_in_text_files() -> None:
    """End-to-end: the real tree is clean, so the CI gate is green today."""
    repo_root = Path(cte.__file__).resolve().parent.parent

    nul_offenders, _ = cte.scan(repo_root)

    assert nul_offenders == []
