#!/usr/bin/env python3
"""Deterministic self-tests for guard_staged_secrets.py.

Stdlib-only (unittest + tempfile), no pytest dependency: `python3
.claude/hooks/test_guard_staged_secrets.py -v`. Tests that don't need the
real `gitleaks` binary always run; the two that do (confirming an actual
secret is detected, and that clean staged content passes) skip themselves
when `gitleaks` isn't on PATH, matching this repo's existing advisory
framing for gitleaks -- not every environment running these tests has it
installed, and this guard must degrade gracefully (fail open) exactly
like it would in production when the binary is missing.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HOOK_PATH = Path(__file__).resolve().parent / "guard_staged_secrets.py"
_spec = importlib.util.spec_from_file_location("guard_staged_secrets", _HOOK_PATH)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

_HAS_GITLEAKS = shutil.which("gitleaks") is not None

# A well-known example secret (e.g. AWS's own published EXAMPLE access key
# ID, or a plausible-looking GitHub PAT) is deliberately NOT used here:
# gitleaks' default ruleset is tuned against real-world false-positive
# noise and, confirmed empirically, does not flag either -- almost
# certainly precisely because they're widely copy-pasted placeholders in
# docs/tutorials, not because the detection is broken. Using a
# repo-local custom rule (via a `.gitleaks.toml` gitleaks reads
# automatically from the target path, same precedence gitleaks documents
# for its own `--config` resolution) sidesteps needing to reverse-engineer
# the exact shape of a built-in rule, and keeps this test's pass/fail
# behavior independent of gitleaks' bundled ruleset ever changing.
_FIXTURE_SECRET_REGEX = "TEST-FIXTURE-SECRET-[A-Za-z0-9]{20}"
_FAKE_SECRET = "TEST-FIXTURE-SECRET-AbCdEfGhIj1234567890"
_GITLEAKS_TOML = f"""\
title = "test fixture config"

[[rules]]
id = "test-fixture-secret"
description = "Test fixture secret for guard_staged_secrets tests"
regex = '''{_FIXTURE_SECRET_REGEX}'''
"""


def _run_git(cwd: str, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


class GitRepo:
    """A throwaway git repo, empty except for an initial commit. Carries a
    custom `.gitleaks.toml` (see _FIXTURE_SECRET_REGEX above) so tests
    don't depend on the exact shape of gitleaks' own bundled rules.
    """

    def __enter__(self) -> GitRepo:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = self._tmp.name
        _run_git(self.path, "init", "-q")
        _run_git(self.path, "config", "user.email", "test@example.com")
        _run_git(self.path, "config", "user.name", "Test")
        (Path(self.path) / "README.md").write_text("# scratch repo\n")
        (Path(self.path) / ".gitleaks.toml").write_text(_GITLEAKS_TOML)
        _run_git(self.path, "add", "README.md", ".gitleaks.toml")
        _run_git(self.path, "commit", "-q", "-m", "init")
        return self

    def stage(self, name: str, content: str) -> None:
        (Path(self.path) / name).write_text(content)
        _run_git(self.path, "add", name)

    def __exit__(self, *exc: object) -> None:
        self._tmp.cleanup()


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
            payload = {"tool_name": "Bash", "cwd": tmp, "tool_input": {"command": 'git commit -m "x"'}}
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_fails_open_when_gitleaks_not_on_path(self) -> None:
        with GitRepo() as repo:
            repo.stage("secret.txt", f"api_key = {_FAKE_SECRET}\n")
            payload = {"tool_name": "Bash", "cwd": repo.path, "tool_input": {"command": 'git commit -m "x"'}}
            with patch.object(guard.shutil, "which", return_value=None):
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

    @unittest.skipUnless(_HAS_GITLEAKS, "gitleaks not installed")
    def test_blocks_a_staged_secret(self) -> None:
        with GitRepo() as repo:
            repo.stage("secret.txt", f"api_key = {_FAKE_SECRET}\n")
            payload = {"tool_name": "Bash", "cwd": repo.path, "tool_input": {"command": 'git commit -m "x"'}}
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED", message)

    @unittest.skipUnless(_HAS_GITLEAKS, "gitleaks not installed")
    def test_allows_clean_staged_content(self) -> None:
        with GitRepo() as repo:
            repo.stage("notes.txt", "just some notes, nothing sensitive\n")
            payload = {"tool_name": "Bash", "cwd": repo.path, "tool_input": {"command": 'git commit -m "x"'}}
            code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    @unittest.skipUnless(_HAS_GITLEAKS, "gitleaks not installed")
    def test_main_blocks_end_to_end_via_subprocess(self) -> None:
        with GitRepo() as repo:
            repo.stage("secret.txt", f"api_key = {_FAKE_SECRET}\n")
            payload = json.dumps(
                {"tool_name": "Bash", "cwd": repo.path, "tool_input": {"command": 'git commit -m "x"'}}
            )
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
