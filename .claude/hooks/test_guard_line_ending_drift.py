#!/usr/bin/env python3
"""Deterministic self-tests for guard_line_ending_drift.py.

Stdlib-only (unittest + tempfile), no external tool dependency (unlike
guard_staged_secrets.py, this guard shells out only to `git` -- always
present in this hook's own execution environment): `python3
.claude/hooks/test_guard_line_ending_drift.py -v`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_HOOK_PATH = Path(__file__).resolve().parent / "guard_line_ending_drift.py"
_spec = importlib.util.spec_from_file_location("guard_line_ending_drift", _HOOK_PATH)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

# 25 lines clears _MIN_LINES_TO_CHECK (20) with room to spare.
_LINE_COUNT = 25


def _run_git(cwd: str, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _crlf_content(lines: int = _LINE_COUNT) -> bytes:
    return b"".join(f"line {i}\r\n".encode() for i in range(lines))


def _lf_content(lines: int = _LINE_COUNT) -> bytes:
    return b"".join(f"line {i}\n".encode() for i in range(lines))


class GitRepo:
    """A throwaway git repo, empty except for an initial commit."""

    def __enter__(self) -> GitRepo:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = self._tmp.name
        _run_git(self.path, "init", "-q")
        _run_git(self.path, "config", "user.email", "test@example.com")
        _run_git(self.path, "config", "user.name", "Test")
        (Path(self.path) / "README.md").write_text("# scratch repo\n")
        _run_git(self.path, "add", "README.md")
        _run_git(self.path, "commit", "-q", "-m", "init")
        return self

    def commit_file(self, name: str, content: bytes) -> None:
        """Write, add, and commit -- establishes this content as HEAD."""
        (Path(self.path) / name).write_bytes(content)
        _run_git(self.path, "add", name)
        _run_git(self.path, "commit", "-q", "-m", f"add {name}")

    def stage(self, name: str, content: bytes) -> None:
        """Write and add without committing -- this becomes the staged blob."""
        (Path(self.path) / name).write_bytes(content)
        _run_git(self.path, "add", name)

    def __exit__(self, *exc: object) -> None:
        self._tmp.cleanup()


def _commit_payload(repo: GitRepo) -> dict:
    return {"tool_name": "Bash", "cwd": repo.path, "tool_input": {"command": 'git commit -m "x"'}}


class EvaluateTests(unittest.TestCase):
    def test_ignores_non_bash_tool_calls(self) -> None:
        payload = {
            "tool_name": "Edit",
            "cwd": "/nonexistent",
            "tool_input": {"file_path": "/x", "old_string": "a", "new_string": "b"},
        }
        code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_ignores_bash_commands_without_git_commit(self) -> None:
        with GitRepo() as repo:
            payload = {"tool_name": "Bash", "cwd": repo.path, "tool_input": {"command": "git status"}}
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_fails_open_outside_a_git_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, message = guard.evaluate(
                {"tool_name": "Bash", "cwd": tmp, "tool_input": {"command": 'git commit -m "x"'}}
            )
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_blocks_a_wholesale_crlf_to_lf_flip(self) -> None:
        """The exact incident this guard exists for: a tracked CRLF file
        gets silently flattened to LF by a content-preserving edit."""
        with GitRepo() as repo:
            repo.commit_file("CHANGES.md", _crlf_content())
            repo.stage("CHANGES.md", _lf_content())
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED", message)
        self.assertIn("CHANGES.md", message)

    def test_blocks_a_wholesale_lf_to_crlf_flip(self) -> None:
        """The reverse direction is equally a tooling accident, not just
        CRLF->LF -- e.g. an editor with the wrong EOL setting."""
        with GitRepo() as repo:
            repo.commit_file("NOTES.md", _lf_content())
            repo.stage("NOTES.md", _crlf_content())
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED", message)

    def test_allows_a_normal_content_edit_with_stable_line_endings(self) -> None:
        with GitRepo() as repo:
            repo.commit_file("CHANGES.md", _lf_content())
            edited = _lf_content() + b"one more line\n"
            repo.stage("CHANGES.md", edited)
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_allows_a_brand_new_file_regardless_of_its_line_endings(self) -> None:
        """A new file has no prior convention to drift from -- --diff-filter=M
        excludes it entirely."""
        with GitRepo() as repo:
            repo.stage("NEW.md", _crlf_content())
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_allows_a_small_file_even_with_a_full_flip(self) -> None:
        """Below _MIN_LINES_TO_CHECK, a full flip is indistinguishable from
        a few genuinely-edited lines -- deliberately not flagged."""
        with GitRepo() as repo:
            repo.commit_file("tiny.md", b"a\r\nb\r\n")
            repo.stage("tiny.md", b"a\nb\n")
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_exempts_a_file_declared_dash_text_in_gitattributes(self) -> None:
        """The CHANGELOG.md fix commit's own pattern: once a file's
        convention is recorded explicitly, this guard defers to that
        decision instead of re-flagging every future edit."""
        with GitRepo() as repo:
            repo.commit_file("CHANGES.md", _crlf_content())
            (Path(repo.path) / ".gitattributes").write_text("CHANGES.md -text\n")
            _run_git(repo.path, "add", ".gitattributes")
            repo.stage("CHANGES.md", _lf_content())
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_exempts_a_file_declared_eol_in_gitattributes(self) -> None:
        with GitRepo() as repo:
            repo.commit_file("CHANGES.md", _crlf_content())
            (Path(repo.path) / ".gitattributes").write_text("CHANGES.md text eol=lf\n")
            _run_git(repo.path, "add", ".gitattributes")
            repo.stage("CHANGES.md", _lf_content())
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_main_fails_open_on_malformed_stdin_json(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_HOOK_PATH)],
            input="not valid json{{{",
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_main_blocks_end_to_end_via_subprocess(self) -> None:
        with GitRepo() as repo:
            repo.commit_file("CHANGES.md", _crlf_content())
            repo.stage("CHANGES.md", _lf_content())
            payload = json.dumps(_commit_payload(repo))
            result = subprocess.run(
                [sys.executable, str(_HOOK_PATH)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("BLOCKED", result.stderr)


if __name__ == "__main__":
    unittest.main()
