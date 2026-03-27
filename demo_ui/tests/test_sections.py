"""test_sections.py — Functional tests for all 8 FORGE demo sections.

These tests call forge_runner.run_section() directly (in --quick mode)
and verify that each section produces meaningful output, contains expected
keywords, and does not error out.

NOTE: These tests require `forge_env` to be installed:
    maturin build --release -m crates/forge-python/Cargo.toml
    pip install target/wheels/*.whl
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Skip all tests if forge_env is not installed
forge_env_available = True
try:
    import forge_env  # noqa: F401
except ImportError:
    forge_env_available = False

pytestmark = pytest.mark.skipif(
    not forge_env_available,
    reason="forge_env not installed — build & install the Rust extension first",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def collect_lines(gen: AsyncGenerator[str, None]) -> list[str]:
    """Drain an async generator into a list of lines."""
    return [line async for line in gen]


# ---------------------------------------------------------------------------
# Section-level expected keywords
# ---------------------------------------------------------------------------

SECTION_KEYWORDS: dict[str, list[str]] = {
    "worldgen": ["Seed", "World Generation", "Ground"],
    "navigation": ["pos=", "stamina", "Navigation"],
    "gathering": ["inventory", "Wood", "Fiber"],
    "crafting": ["recipes", "Plank", "Torch"],
    "multiagent": ["agent_0", "agent_1", "Multi-Agent"],
    "daynight": ["Dawn", "Day", "Dusk", "Night"],
    "determinism": ["hash=", "identical", "Deterministic"],
    "performance": ["Steps/second", "us/step"],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("section", list(SECTION_KEYWORDS.keys()))
async def test_section_produces_output(section: str) -> None:
    """Every section produces at least one non-empty output line."""
    from demo_ui.backend.forge_runner import run_section

    lines = await collect_lines(run_section(section, seed=42, quick=True))
    non_empty = [line for line in lines if line.strip()]
    assert len(non_empty) > 0, f"Section '{section}' produced no output"


@pytest.mark.anyio
@pytest.mark.parametrize("section,keywords", list(SECTION_KEYWORDS.items()))
async def test_section_contains_keywords(section: str, keywords: list[str]) -> None:
    """Each section's output contains its expected keywords."""
    from demo_ui.backend.forge_runner import run_section

    lines = await collect_lines(run_section(section, seed=42, quick=True))
    combined = " ".join(lines)
    for kw in keywords:
        assert kw in combined, (
            f"Section '{section}': expected keyword '{kw}' not found in output.\n"
            f"First 500 chars:\n{combined[:500]}"
        )


@pytest.mark.anyio
async def test_section_does_not_crash_with_different_seed() -> None:
    """worldgen section runs cleanly with a non-default seed."""
    from demo_ui.backend.forge_runner import run_section

    lines = await collect_lines(run_section("worldgen", seed=1337, quick=True))
    non_empty = [line for line in lines if line.strip()]
    assert len(non_empty) > 0


@pytest.mark.anyio
async def test_unknown_section_returns_error() -> None:
    """Passing an unknown section key yields an ERROR line and terminates."""
    from demo_ui.backend.forge_runner import run_section

    lines = await collect_lines(run_section("not_a_section", seed=42, quick=True))
    assert any("ERROR" in line for line in lines)


@pytest.mark.anyio
async def test_run_all_yields_section_markers() -> None:
    """run_all() yields __SECTION_START__ and __SECTION_END__ markers."""
    from demo_ui.backend.forge_runner import run_all

    lines = await collect_lines(run_all(seed=42, quick=True))
    starts = [line for line in lines if "__SECTION_START__" in line]
    ends = [line for line in lines if "__SECTION_END__" in line]
    assert len(starts) == 8, f"Expected 8 SECTION_START markers, got {len(starts)}"
    assert len(ends) == 8, f"Expected 8 SECTION_END markers, got {len(ends)}"
