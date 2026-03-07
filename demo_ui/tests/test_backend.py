"""test_backend.py — Unit and integration tests for the FORGE demo backend.

Coverage:
  - parse_results_md()        (unit)
  - GET /api/sections          (integration)
  - GET /api/results           (integration)
  - GET /health                (integration)
  - POST /api/run/{section}    (integration — SSE stream)
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_results_md(tmp_path: Path) -> Path:
    """Write a minimal demo_results.md into a temp dir."""
    content = textwrap.dedent("""\
        # FORGE Demo Results

        **Date:** 2026-02-26
        **Seed:** 42
        **Platform:** Windows 11 (x86_64), Rust 1.93.1, Python 3.11.9
        **Result:** 8/8 sections passed in 0.1s

        ## Summary

        | Section            | Status |
        |--------------------|--------|
        | World Generation   | PASS   |
        | Navigation         | PASS   |
        | Resource Gathering | PASS   |
        | Crafting           | PASS   |
        | Multi-Agent        | PASS   |
        | Day/Night Cycle    | PASS   |
        | Determinism        | PASS   |
        | Performance        | PASS   |


        ## 8. Performance Benchmark

        | Metric         | Value          |
        |----------------|----------------|
        | Steps/second   | 136,419        |
        | us/step        | 7.33           |
    """)
    p = tmp_path / "demo_results.md"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
async def client() -> AsyncIterator[AsyncClient]:
    """AsyncClient over the FastAPI ASGI app."""
    from demo_ui.backend.main import app  # noqa: PLC0415

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Unit tests — parse_results_md
# ---------------------------------------------------------------------------


def test_parse_results_md_structure(sample_results_md: Path) -> None:
    """parse_results_md returns the expected top-level keys."""
    from demo_ui.backend.forge_runner import parse_results_md  # noqa: PLC0415

    result = parse_results_md(sample_results_md)
    assert "date" in result
    assert "seed" in result
    assert "platform" in result
    assert "result" in result
    assert "sections" in result
    assert "performance" in result


def test_parse_results_md_values(sample_results_md: Path) -> None:
    """parse_results_md extracts correct field values."""
    from demo_ui.backend.forge_runner import parse_results_md  # noqa: PLC0415

    result = parse_results_md(sample_results_md)
    assert result["date"] == "2026-02-26"
    assert result["seed"] == 42
    assert "Windows" in result["platform"]
    assert "8/8" in result["result"]


def test_parse_results_md_sections(sample_results_md: Path) -> None:
    """parse_results_md extracts all 8 section statuses."""
    from demo_ui.backend.forge_runner import parse_results_md  # noqa: PLC0415

    result = parse_results_md(sample_results_md)
    sections = result["sections"]
    assert len(sections) == 8
    for sec in sections:
        assert sec["status"] == "PASS"


def test_parse_results_md_performance(sample_results_md: Path) -> None:
    """parse_results_md extracts performance metrics."""
    from demo_ui.backend.forge_runner import parse_results_md  # noqa: PLC0415

    result = parse_results_md(sample_results_md)
    perf = result["performance"]
    assert perf["steps_per_second"] == "136,419"
    assert perf["us_per_step"] == "7.33"


def test_parse_results_md_missing_file(tmp_path: Path) -> None:
    """parse_results_md returns error dict when file missing."""
    from demo_ui.backend.forge_runner import parse_results_md  # noqa: PLC0415

    result = parse_results_md(tmp_path / "nonexistent.md")
    assert "error" in result
    assert result["sections"] == []


# ---------------------------------------------------------------------------
# Integration tests — API
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health_endpoint(client: AsyncClient) -> None:
    """GET /health returns 200 and ok status."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_sections_endpoint_count(client: AsyncClient) -> None:
    """GET /api/sections returns exactly 8 sections."""
    resp = await client.get("/api/sections")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 8


@pytest.mark.anyio
async def test_sections_endpoint_keys(client: AsyncClient) -> None:
    """GET /api/sections contains all expected section keys."""
    from demo_ui.backend.forge_runner import SECTIONS  # noqa: PLC0415

    resp = await client.get("/api/sections")
    keys = {s["key"] for s in resp.json()}
    assert keys == set(SECTIONS.keys())


@pytest.mark.anyio
async def test_sections_endpoint_schema(client: AsyncClient) -> None:
    """Each section object has key, name, and index fields."""
    resp = await client.get("/api/sections")
    for sec in resp.json():
        assert "key" in sec
        assert "name" in sec
        assert "index" in sec


@pytest.mark.anyio
async def test_results_endpoint(client: AsyncClient) -> None:
    """GET /api/results returns a dict with expected keys."""
    resp = await client.get("/api/results")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    # Either real data or an error dict — both are acceptable
    assert "sections" in data or "error" in data


@pytest.mark.anyio
async def test_run_unknown_section(client: AsyncClient) -> None:
    """POST /api/run/<bad> returns 404."""
    resp = await client.post("/api/run/not_a_section", json={"seed": 42, "quick": True})
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_run_section_returns_stream(client: AsyncClient) -> None:
    """POST /api/run/worldgen returns text/event-stream content type."""
    resp = await client.post("/api/run/worldgen", json={"seed": 42, "quick": True})
    assert resp.status_code == 200
    ct = resp.headers.get("content-type", "")
    assert "text/event-stream" in ct


@pytest.mark.anyio
async def test_run_all_returns_stream(client: AsyncClient) -> None:
    """POST /api/run-all returns text/event-stream content type."""
    resp = await client.post("/api/run-all", json={"seed": 42, "quick": True})
    assert resp.status_code == 200
    ct = resp.headers.get("content-type", "")
    assert "text/event-stream" in ct


# ---------------------------------------------------------------------------
# Sanity tests — SECTIONS constant
# ---------------------------------------------------------------------------


def test_sections_constant_is_dict() -> None:
    from demo_ui.backend.forge_runner import SECTIONS  # noqa: PLC0415

    assert isinstance(SECTIONS, dict)
    assert len(SECTIONS) == 8


def test_sections_constant_has_expected_keys() -> None:
    from demo_ui.backend.forge_runner import SECTIONS  # noqa: PLC0415

    expected = {
        "worldgen",
        "navigation",
        "gathering",
        "crafting",
        "multiagent",
        "daynight",
        "determinism",
        "performance",
    }
    assert set(SECTIONS.keys()) == expected
