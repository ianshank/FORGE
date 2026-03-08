"""Playwright E2E browser tests for the FORGE demo UI.

These tests require the ``playwright`` package and Chromium browser::

    pip install playwright
    playwright install chromium

Run with::

    pytest demo_ui/tests/test_e2e_browser.py -v

The tests start the FastAPI server automatically and verify that the
browser-based demo UI renders correctly and responds to user interaction.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Skip the entire module if playwright is not installed.
pw = pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, sync_playwright  # noqa: E402

SERVER_PORT = 18765
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"


@pytest.fixture(scope="module")
def server() -> Generator[subprocess.Popen[bytes], None, None]:
    """Start the FastAPI demo server for the duration of the test module."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "demo_ui.backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(SERVER_PORT),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for the server to be ready
    import httpx

    for _ in range(30):
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail("Demo server did not start within 15 seconds")

    yield proc

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="module")
def browser_page(server: subprocess.Popen[bytes]) -> Generator[Page, None, None]:
    """Launch a headless Chromium browser and navigate to the demo UI."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(SERVER_URL, wait_until="networkidle")
        yield page
        browser.close()


class TestPageLoad:
    """Verify the demo UI loads and renders basic elements."""

    def test_title_present(self, browser_page: Page) -> None:
        """The page should contain a title or header with 'FORGE'."""
        content = browser_page.content()
        assert "FORGE" in content.upper()

    def test_section_nav_rendered(self, browser_page: Page) -> None:
        """The sidebar should have section navigation buttons."""
        # Look for buttons or links that represent demo sections
        buttons = browser_page.query_selector_all("button, [role='button'], .section-btn")
        assert len(buttons) > 0, "Expected section navigation buttons"

    def test_terminal_area_exists(self, browser_page: Page) -> None:
        """A terminal or output area should be present on the page."""
        terminal = browser_page.query_selector(
            "#terminal, .terminal, [data-testid='terminal'], pre"
        )
        assert terminal is not None, "Expected a terminal/output area"

    def test_no_console_errors(self, browser_page: Page) -> None:
        """The page should load without JavaScript console errors."""
        errors: list[str] = []
        browser_page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        # Reload to capture any errors
        browser_page.reload(wait_until="networkidle")
        assert len(errors) == 0, f"Console errors detected: {errors}"


class TestHealthEndpoint:
    """Verify the health endpoint works via the browser fetch."""

    def test_health_returns_ok(self, browser_page: Page) -> None:
        """Fetching /health from the browser should return a valid response."""
        result = browser_page.evaluate(
            """async () => {
                const r = await fetch('/health');
                return { status: r.status, body: await r.json() };
            }"""
        )
        assert result["status"] == 200
        assert result["body"]["status"] == "ok"


class TestSectionsAPI:
    """Verify the sections API is accessible from the browser."""

    def test_sections_endpoint(self, browser_page: Page) -> None:
        """GET /api/sections should return section metadata."""
        result = browser_page.evaluate(
            """async () => {
                const r = await fetch('/api/sections');
                return { status: r.status, body: await r.json() };
            }"""
        )
        assert result["status"] == 200
        sections = result["body"]
        assert len(sections) == 8, f"Expected 8 sections, got {len(sections)}"
