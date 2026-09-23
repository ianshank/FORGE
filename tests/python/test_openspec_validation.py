"""OpenSpec change package and specification validation gate.

Validates that:
1. The OpenSpec change package adheres to formatting standards (no em dashes, no emojis).
2. All 6 capability specs and core change documents exist.
3. Every capability spec has required sections, scenarios, and explicit falsifier scenarios.
4. The OpenSpec CLI strict validation gate executes cleanly via `openspec validate --all --strict`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGE_DIR = REPO_ROOT / "openspec" / "changes" / "archive" / "v0.6.0-release-baseline-hardening"

REQUIRED_SPECS = (
    "python-distribution",
    "rl-api-conformance",
    "evaluation-evidence",
    "model-artifact-provenance",
    "drone-robustness",
    "developer-adoption",
)

REQUIRED_CORE_FILES = (
    "proposal.md",
    "design.md",
    "tasks.md",
)

EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]")


def test_openspec_core_files_exist() -> None:
    """All core change documents must be present in the change package."""
    assert CHANGE_DIR.is_dir(), f"Change dir missing at {CHANGE_DIR}"
    for filename in REQUIRED_CORE_FILES:
        filepath = CHANGE_DIR / filename
        assert filepath.is_file(), f"Missing required file: {filepath.relative_to(REPO_ROOT)}"


def test_openspec_capability_specs_exist() -> None:
    """All 6 capability specs must exist under specs/<name>/spec.md."""
    for spec_name in REQUIRED_SPECS:
        spec_path = CHANGE_DIR / "specs" / spec_name / "spec.md"
        assert spec_path.is_file(), f"Missing capability spec: {spec_path.relative_to(REPO_ROOT)}"


def test_openspec_files_have_no_em_dashes_or_emojis() -> None:
    """OpenSpec files must not contain em dashes or emojis."""
    for path in CHANGE_DIR.rglob("*.md"):
        content = path.read_text(encoding="utf-8")
        assert "\u2014" not in content, (
            f"{path.relative_to(REPO_ROOT)} contains em dash (\u2014); use standard hyphens or commas instead"
        )
        assert not EMOJI_PATTERN.search(content), (
            f"{path.relative_to(REPO_ROOT)} contains emoji characters; keep specification strictly text-based"
        )


def test_openspec_capability_specs_have_scenarios_and_falsifiers() -> None:
    """Each capability spec must define scenarios and explicit falsifiers."""
    for spec_name in REQUIRED_SPECS:
        spec_path = CHANGE_DIR / "specs" / spec_name / "spec.md"
        content = spec_path.read_text(encoding="utf-8")
        assert "## Purpose" in content, f"{spec_name} missing '## Purpose' section"
        assert "Requirement:" in content, f"{spec_name} missing 'Requirement:' blocks"
        assert "#### Scenario:" in content, f"{spec_name} missing '#### Scenario:' blocks"
        assert "Falsifier:" in content, f"{spec_name} missing explicit 'Falsifier:' scenario"


def test_openspec_cli_validate_strict() -> None:
    """Run openspec validate --all --strict via the CLI or wrapper."""
    import sys
    openspec_path = shutil.which("openspec")
    cmd = (
        [openspec_path, "validate", "--all", "--strict"]
        if openspec_path
        else [shutil.which("npx") or "npx", "--yes", "@fission-ai/openspec", "validate", "--all", "--strict"]
    )
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"`{' '.join(cmd)}` failed with exit code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
