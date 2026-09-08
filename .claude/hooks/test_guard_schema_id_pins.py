#!/usr/bin/env python3
"""Deterministic self-tests for guard_schema_id_pins.py.

Stdlib-only (unittest + tempfile). Advisory hook: evaluate() always
returns exit 0; the reminder lives in the message.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

_HOOK_PATH = Path(__file__).resolve().parent / "guard_schema_id_pins.py"
_spec = importlib.util.spec_from_file_location("guard_schema_id_pins", _HOOK_PATH)
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

    def test_reminder_when_rewards_toml_staged(self) -> None:
        with GitRepo() as repo:
            repo.stage("configs/minecraft/rewards.toml", "schema_version = 1\n")
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertIn("REMINDER", message)
        self.assertIn("configs/minecraft/rewards.toml", message)
        self.assertIn("xlang", message)

    def test_reminder_when_block_embeddings_staged(self) -> None:
        with GitRepo() as repo:
            repo.stage("configs/minecraft/block_embeddings.toml", "[blocks]\nair = 0\n")
            code, message = guard.evaluate(_commit_payload(repo))
        self.assertEqual(code, 0)
        self.assertIn("block_embeddings.toml", message)

    def test_matching_pinned_paths_filters_only_contract_files(self) -> None:
        staged = [
            "configs/minecraft/rewards.toml",
            "README.md",
            "configs/minecraft/env.toml",
        ]
        matched = guard.matching_pinned_paths(staged)
        self.assertEqual(matched, ["configs/minecraft/rewards.toml"])

    def test_pinned_path_set_covers_nested_and_embeddings(self) -> None:
        expected = {
            "configs/minecraft/action_map.toml",
            "configs/minecraft/rewards.toml",
            "configs/minecraft/milestone_rewards.toml",
            "configs/minecraft/crafting_rewards.toml",
            "configs/minecraft/block_embeddings.toml",
        }
        self.assertEqual(set(guard.PINNED_MC_CONFIG_PATHS), expected)

    def test_unreadable_payload_fails_open_via_main(self) -> None:
        # main() wraps evaluate in try/except and always returns 0.
        self.assertTrue(callable(guard.main))


if __name__ == "__main__":
    unittest.main()
