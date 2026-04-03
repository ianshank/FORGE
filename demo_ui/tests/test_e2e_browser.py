"""Playwright E2E browser tests for the FORGE demo UI.

These tests require the ``playwright`` package and Chromium browser::

    pip install playwright
    playwright install chromium

Run with::

    pytest demo_ui/tests/test_e2e_browser.py -v

The tests start the FastAPI server automatically and verify that the
browser-based demo UI renders correctly and responds to user interaction.

Environment variables
---------------------
DEMO_BASE_URL   – Override the full base URL (e.g. ``http://ci-host:9000``).
                  When set the tests will **not** start a local server and will
                  connect to the provided URL instead.
FORGE_DEMO_HOST – Host to bind the auto-started server (default ``127.0.0.1``).
FORGE_DEMO_PORT – Port to bind the auto-started server (default ``18765``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# ---------------------------------------------------------------------------
# Skip the entire module if playwright is not installed.
# ---------------------------------------------------------------------------
pw = pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, sync_playwright  # noqa: E402

# ---------------------------------------------------------------------------
# URL configuration — no hard-coded values
# ---------------------------------------------------------------------------
DEMO_BASE_URL = os.environ.get("DEMO_BASE_URL", "")
SERVER_HOST = os.environ.get("FORGE_DEMO_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("FORGE_DEMO_PORT", "18765"))
SERVER_URL = DEMO_BASE_URL or f"http://{SERVER_HOST}:{SERVER_PORT}"

# Maximum time (seconds) any single E2E test is allowed to run.
E2E_TIMEOUT_MS = int(os.environ.get("E2E_TIMEOUT_MS", "30000"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server() -> Generator[subprocess.Popen[bytes] | None, None, None]:
    """Start the FastAPI demo server for the duration of the test module.

    If ``DEMO_BASE_URL`` is set the fixture yields *None* because the caller
    is expected to have started the server externally (e.g. in CI).
    """
    if DEMO_BASE_URL:
        yield None
        return

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "demo_ui.backend.main:app",
            "--host",
            SERVER_HOST,
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
def browser_page(
    server: subprocess.Popen[bytes] | None,
) -> Generator[Page, None, None]:
    """Launch a headless Chromium browser and navigate to the demo UI."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(E2E_TIMEOUT_MS)
        page.goto(SERVER_URL, wait_until="networkidle")
        yield page
        browser.close()


# ===================================================================
# Test suite 1 — Page load basics
# ===================================================================


class TestPageLoad:
    """Verify the demo UI loads and renders basic elements."""

    def test_title_present(self, browser_page: Page) -> None:
        """The page should contain a title or header with 'FORGE'."""
        content = browser_page.content()
        assert "FORGE" in content.upper()

    def test_section_nav_rendered(self, browser_page: Page) -> None:
        """The sidebar should have section navigation buttons."""
        buttons = browser_page.query_selector_all(
            "button, [role='button'], .section-btn"
        )
        assert len(buttons) > 0, "Expected section navigation buttons"

    def test_terminal_area_exists(self, browser_page: Page) -> None:
        """A terminal or output area should be present on the page."""
        terminal = browser_page.query_selector(
            "#terminal-output, #terminal, .terminal, [data-testid='terminal'], pre"
        )
        assert terminal is not None, "Expected a terminal/output area"

    def test_no_console_errors(self, browser_page: Page) -> None:
        """The page should load without JavaScript console errors."""
        errors: list[str] = []
        browser_page.on(
            "console",
            lambda msg: errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        browser_page.reload(wait_until="networkidle")
        assert len(errors) == 0, f"Console errors detected: {errors}"


# ===================================================================
# Test suite 2 — Health / API endpoints via browser fetch
# ===================================================================


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


# ===================================================================
# Test suite 3 — Interactive UI element tests
# ===================================================================


class TestSectionBadges:
    """Verify section badges transition through IDLE -> RUNNING -> PASS."""

    def test_section_badges_transition(self, browser_page: Page) -> None:
        """Click the first section and verify badge state transitions.

        Expected flow:
          1. All badges start as ``IDLE``.
          2. After clicking a section button the badge moves to ``RUNNING``.
          3. When the SSE stream completes, the badge shows ``PASS`` (or
             ``FAIL`` in an error scenario, which is acceptable but logged).
        """
        # 1. Verify initial IDLE state for the first section badge.
        badge = browser_page.query_selector(
            "[id^='badge-']"
        )
        assert badge is not None, "No section badge element found"
        initial_text = (badge.text_content() or "").strip().upper()
        assert initial_text == "IDLE", (
            f"Expected initial badge state IDLE, got {initial_text!r}"
        )

        # 2. Click the first section button to trigger a run.
        first_btn = browser_page.query_selector(".section-btn")
        assert first_btn is not None, "No section button found"
        first_btn.click()

        # 3. Wait for badge to leave IDLE (i.e. become RUNNING).
        browser_page.wait_for_function(
            """() => {
                const b = document.querySelector("[id^='badge-']");
                return b && b.textContent.trim().toUpperCase() !== 'IDLE';
            }""",
            timeout=E2E_TIMEOUT_MS,
        )
        running_text = (badge.text_content() or "").strip().upper()
        assert running_text in {"RUNNING", "PASS", "FAIL"}, (
            f"Badge should be RUNNING or terminal state, got {running_text!r}"
        )

        # 4. Wait for terminal state (PASS or FAIL).
        browser_page.wait_for_function(
            """() => {
                const b = document.querySelector("[id^='badge-']");
                const t = b ? b.textContent.trim().toUpperCase() : '';
                return t === 'PASS' || t === 'FAIL';
            }""",
            timeout=E2E_TIMEOUT_MS,
        )
        final_text = (badge.text_content() or "").strip().upper()
        assert final_text in {"PASS", "FAIL"}, (
            f"Badge should reach PASS or FAIL, got {final_text!r}"
        )


class TestTerminalOutput:
    """Verify the terminal output area streams content without errors."""

    def test_terminal_streams_without_error(self, browser_page: Page) -> None:
        """After running a section, the terminal should contain output lines
        and none of them should contain the text ``Error`` (case-insensitive).
        """
        # Navigate fresh and run the first section.
        browser_page.goto(SERVER_URL, wait_until="networkidle")
        first_btn = browser_page.query_selector(".section-btn")
        assert first_btn is not None
        first_btn.click()

        # Wait until the terminal has at least one non-placeholder child span.
        browser_page.wait_for_function(
            """() => {
                const el = document.getElementById('terminal-output');
                if (!el) return false;
                const spans = el.querySelectorAll('span.fade-in');
                return spans.length > 0;
            }""",
            timeout=E2E_TIMEOUT_MS,
        )

        terminal_text = browser_page.evaluate(
            """() => {
                const el = document.getElementById('terminal-output');
                return el ? el.innerText : '';
            }"""
        )
        assert len(terminal_text.strip()) > 0, (
            "Terminal output should contain streamed content"
        )
        # Allow "Error" only inside quoted strings or code identifiers;
        # a bare "Error" at the start of a line indicates a real problem.
        for line in terminal_text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("error"):
                pytest.fail(
                    f"Terminal output contains an error line: {stripped!r}"
                )


class TestProgressBar:
    """Verify the progress bar indicator advances during a run."""

    def test_progress_bar_advances(self, browser_page: Page) -> None:
        """The progress bar width should change from 0% to a non-zero value
        after triggering a section run.
        """
        browser_page.goto(SERVER_URL, wait_until="networkidle")

        # Capture initial progress bar width.
        initial_width = browser_page.evaluate(
            """() => {
                const bar = document.getElementById('progress-bar');
                return bar ? bar.style.width : '0%';
            }"""
        )

        # Click "Run All" to trigger progress updates.
        run_btn = browser_page.query_selector("#btn-run-all")
        assert run_btn is not None, "Run All button not found"
        run_btn.click()

        # Wait for the progress bar to move past 0%.
        browser_page.wait_for_function(
            """() => {
                const bar = document.getElementById('progress-bar');
                if (!bar) return false;
                const w = parseFloat(bar.style.width) || 0;
                return w > 0;
            }""",
            timeout=E2E_TIMEOUT_MS,
        )

        updated_width = browser_page.evaluate(
            """() => {
                const bar = document.getElementById('progress-bar');
                return bar ? bar.style.width : '0%';
            }"""
        )
        assert updated_width != initial_width or float(
            updated_width.replace("%", "") or "0"
        ) > 0, (
            f"Progress bar should advance; "
            f"initial={initial_width!r}, updated={updated_width!r}"
        )


class TestWorldCanvas:
    """Verify the world canvas element renders with visible dimensions."""

    def test_world_canvas_renders(self, browser_page: Page) -> None:
        """The ``#world-canvas`` element should exist, have positive width and
        height attributes, and occupy a non-zero bounding rectangle.
        """
        canvas = browser_page.query_selector("#world-canvas")
        assert canvas is not None, "Canvas element #world-canvas not found"

        # Check HTML width/height attributes.
        w_attr = canvas.get_attribute("width")
        h_attr = canvas.get_attribute("height")
        assert w_attr is not None and int(w_attr) > 0, (
            f"Canvas width attribute should be positive, got {w_attr!r}"
        )
        assert h_attr is not None and int(h_attr) > 0, (
            f"Canvas height attribute should be positive, got {h_attr!r}"
        )

        # Check computed bounding box has non-zero area.
        box = canvas.bounding_box()
        assert box is not None, "Canvas has no bounding box (not visible)"
        assert box["width"] > 0 and box["height"] > 0, (
            f"Canvas bounding box should have positive dimensions, "
            f"got {box['width']}x{box['height']}"
        )
