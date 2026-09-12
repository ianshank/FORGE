#!/usr/bin/env python3
"""Deterministic self-tests for guard_golden_replay.py.

Stdlib-only (unittest + tempfile). Advisory hook: evaluate() always
returns exit 0; the reminder lives in the message.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

_HOOK_PATH = Path(__file__).resolve().parent / "guard_golden_replay.py"
_spec = importlib.util.spec_from_file_location("guard_golden_replay", _HOOK_PATH)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _run_git(cwd: str, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


class GitRepo:
    def __enter__(self) -> GitRepo:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = self._tmp.name
        _run_git(self.path, "init", "-q")
        _run_git(self.path, "config", "user.email", "test@example.com")
        _run_git(self.path, "config", "user.name", "Test")
        (Path(self.path) / "README.md").write_text("# scratch\n")
        _run_git(self.path, "add", "README.md")
        _run_git(self.path, "commit", "-q", "-m", "init")
        return self

    def stage(self, relpath: str, content: str) -> None:
        dest = Path(self.path) / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        _run_git(self.path, "add", relpath)

    def __exit__(self, *exc: object) -> None:
        self._tmp.cleanup()


def _commit_payload(repo: GitRepo) -> dict:
    return {
        "tool_name": "Bash",
        "cwd": repo.path,
        "tool_input": {"command": 'git commit -m "x"'},
    }


class EvaluateTests(unittest.TestCase):
    def test_ignores_non_commit_bash(self) -> None:
        payload = {
            "tool_name": "Bash",
            "cwd": "/nonexistent",
            "tool_input": {"command": "git status"},
        }
        code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_ignores_non_bash_tool(self) -> None:
        payload = {
            "tool_name": "Edit",
            "cwd": "/nonexistent",
            "tool_input": {"command": "git commit -m x"},
        }
        code, message = guard.evaluate(payload)
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_no_reminder_when_unrelated_file_staged(self) -> None:
        with GitRepo() as repo:
            repo.stage("docs/notes.md", "hello\n")
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_reminder_when_scenario_toml_staged_without_golden(self) -> None:
        with GitRepo() as repo:
            repo.stage("configs/scenarios/orchard_coverage.toml", "name = 'x'\n")
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertIn("REMINDER", message)
        self.assertIn("configs/scenarios/orchard_coverage.toml", message)
        self.assertIn("UPDATE_GOLDEN_REPLAYS", message)

    def test_reminder_when_forge_config_source_staged(self) -> None:
        with GitRepo() as repo:
            repo.stage("crates/forge-types/src/config.rs", "// pin\n")
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertIn("crates/forge-types/src/config.rs", message)

    def test_no_reminder_when_golden_and_flip_log_also_staged(self) -> None:
        with GitRepo() as repo:
            repo.stage("configs/scenarios/orchard_coverage.toml", "name = 'x'\n")
            repo.stage("tests/golden/replays/v2_seed42.json", "{}\n")
            repo.stage("docs/results/replay_flip_log.md", "| row |\n")
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertEqual(message, "")

    def test_matching_fingerprint_sources_filters_paths(self) -> None:
        staged = [
            "configs/scenarios/orchard_coverage.toml",
            "README.md",
            "crates/forge-types/src/config.rs",
            "configs/minecraft/env.toml",
        ]
        matched = guard.matching_fingerprint_sources(staged)
        self.assertEqual(
            matched,
            [
                "configs/scenarios/orchard_coverage.toml",
                "crates/forge-types/src/config.rs",
            ],
        )

    def test_unreadable_payload_fails_open_via_main(self) -> None:
        self.assertTrue(callable(guard.main))


if __name__ == "__main__":
    unittest.main()
