#!/usr/bin/env python3
"""Deterministic self-tests for guard_tracked_deletion.py.

Stdlib-only (unittest + tempfile), no pytest dependency, so it can run
standalone in any environment that has git + python3: `python3
.claude/hooks/test_guard_tracked_deletion.py -v`. Exercises `evaluate()`
directly against real temporary git repos rather than mocking git, so a
change in git's actual output format would be caught here too.

Run in CI as a step in the `python-lint` job (fast, zero project deps).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_HOOK_PATH = Path(__file__).resolve().parent / "guard_tracked_deletion.py"
_spec = importlib.util.spec_from_file_location("guard_tracked_deletion", _HOOK_PATH)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _run_git(cwd: str, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


class TrackedRepo:
    """A throwaway git repo with a couple of tracked + ignored files."""

    def __enter__(self) -> TrackedRepo:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = self._tmp.name
        _run_git(self.path, "init", "-q")
        _run_git(self.path, "config", "user.email", "test@example.com")
        _run_git(self.path, "config", "user.name", "Test")

        (Path(self.path) / ".coveragerc").write_text("[run]\nsource = forge\n")
        (Path(self.path) / "README.md").write_text("# scratch repo\n")
        Path(self.path, "target").mkdir()
        (Path(self.path) / "target" / "debug.bin").write_text("binary\n")
        (Path(self.path) / ".gitignore").write_text("target/\n*.pyc\n")

        _run_git(self.path, "add", ".coveragerc", "README.md", ".gitignore")
        _run_git(self.path, "commit", "-q", "-m", "init")
        return self

    def __exit__(self, *exc: object) -> None:
        self._tmp.cleanup()


class EvaluateTests(unittest.TestCase):
    def test_blocks_the_actual_coveragerc_incident(self) -> None:
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": 'find . -maxdepth 1 -name ".coverage*" -delete'},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 2)
        self.assertIn(".coveragerc", message)

    def test_allows_delete_scoped_to_ignored_directory(self) -> None:
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": "rm -rf target/"},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_allows_find_delete_matching_only_ignored_files(self) -> None:
        with TrackedRepo() as repo:
            Path(repo.path, "cache.pyc").write_text("x")
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": 'find . -name "*.pyc" -delete'},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_blocks_rm_glob_matching_tracked_file(self) -> None:
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": "rm -f .cover*"},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 2)
        self.assertIn(".coveragerc", message)

    def test_blocks_rm_glob_glued_to_next_command_with_no_space(self) -> None:
        # Regression test for a PR review finding (GitHub Copilot, PR #117):
        # shlex.split() only splits on whitespace, so `rm -f .cover*;git
        # status` used to glue ".cover*;git" into a single token, which
        # never matched a tracked file and let the delete through unblocked.
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": "rm -f .cover*;git status"},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 2)
        self.assertIn(".coveragerc", message)

    def test_blocks_rm_glob_glued_to_double_ampersand_with_no_space(self) -> None:
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": "rm -f .cover*&&echo done"},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 2)
        self.assertIn(".coveragerc", message)

    def test_blocks_find_delete_pattern_glued_to_next_command_with_no_space(self) -> None:
        # Same bug class as above, in the sibling find -name extraction
        # path: a bare (unquoted) pattern regex that doesn't stop at shell
        # metacharacters would swallow ";rm -rf /" into the "pattern" too,
        # silently failing to match the tracked file it should have caught.
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": 'find . -name .cover*;echo done -delete'},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 2)
        self.assertIn(".coveragerc", message)

    def test_allows_rm_glob_scoped_to_ignored_dir_with_no_space_chain(self) -> None:
        # False-positive guard: the same no-space-separator parsing must
        # not start blocking safe compound commands it previously allowed.
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": "rm -rf target/&&echo done"},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_allows_literal_single_file_rm_without_recursion_or_glob(self) -> None:
        # No wildcard, no -r: blast radius is exactly the named file and
        # already visible in the command text -- out of scope for this guard.
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": "rm README.md"},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_ignores_non_bash_tool_calls(self) -> None:
        payload = {
            "tool_name": "Edit",
            "cwd": "/nonexistent",
            "tool_input": {"file_path": "/x", "old_string": "a", "new_string": "b"},
        }
        code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_ignores_non_destructive_bash_commands(self) -> None:
        with TrackedRepo() as repo:
            payload = {
                "tool_name": "Bash",
                "cwd": repo.path,
                "tool_input": {"command": "git status && cat README.md"},
            }
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_fails_open_outside_a_git_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "tool_name": "Bash",
                "cwd": tmp,
                "tool_input": {"command": 'find . -name "*" -delete'},
            }
            code, message = guard.evaluate(payload)
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
        with TrackedRepo() as repo:
            payload = json.dumps(
                {
                    "tool_name": "Bash",
                    "cwd": repo.path,
                    "tool_input": {"command": 'find . -maxdepth 1 -name ".coverage*" -delete'},
                }
            )
            result = subprocess.run(
                [sys.executable, str(_HOOK_PATH)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn(".coveragerc", result.stderr)


if __name__ == "__main__":
    unittest.main()
