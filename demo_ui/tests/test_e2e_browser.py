"""test_e2e_browser.py — End-to-end browser tests for the FORGE demo UI.

These tests use Playwright to spin up the FastAPI backend and interact
with the JS frontend in a real browser engine (headless or headed).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import httpx
import pytest
from playwright.async_api import Page, expect

# ---------------------------------------------------------------------------
# Test Setup & Fixtures
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.asyncio

# We use an auto-assigned port for the test server to avoid conflicts
TEST_PORT = 8799
TEST_URL = f"http://127.0.0.1:{TEST_PORT}"


@pytest.fixture(scope="module")
def server() -> Generator[None, None, None]:
    """Start the uvicorn server in a background process for the duration of the tests."""
    repo_root = str(Path(__file__).parent.parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root};{env.get('PYTHONPATH', '')}"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "demo_ui.backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(TEST_PORT),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the server to be ready by checking health endpoint
    retries = 30
    while retries > 0:
        try:
            resp = httpx.get(f"{TEST_URL}/health", timeout=2.0)
            if resp.status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(0.5)
        retries -= 1

    if retries == 0:
        proc.terminate()
        raise RuntimeError(f"Test server at {TEST_URL} failed to start in time.")

    yield

    # Teardown: stop the server
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_ui_initial_load(page: Page, server: None) -> None:
    """Verify the UI renders its initial state correctly."""
    await page.goto(TEST_URL)

    # Check header
    await expect(page.locator("h1")).to_contain_text("FORGE")

    # Check baseline status chips are populated
    await expect(page.locator(".stat-chip", has_text="Platform: ")).to_be_visible()

    # Check terminal prompt
    await expect(page.locator("#terminal-output")).to_contain_text("forge$ ready")

    # Check that 8 section buttons exist
    buttons = page.locator(".section-btn")
    await expect(buttons).to_have_count(8)

    # Verify all badges are in IDLE state
    badges = page.locator(".section-btn .badge")
    for i in range(8):
        await expect(badges.nth(i)).to_have_class(re.compile(r"\bidle\b"))


async def test_run_single_section(page: Page, server: None) -> None:
    """Verify running a single section streams output to terminal and updates badge."""
    await page.goto(TEST_URL)

    # Click the "worldgen" section button
    worldgen_btn = page.locator(".section-btn", has_text="World Generation")
    await worldgen_btn.click()

    # Badge should transition to PASS
    badge = worldgen_btn.locator(".badge")
    await expect(badge).to_have_class(re.compile(r"\bpass\b"), timeout=10000)

    # Terminal should contain worldgen keywords
    terminal = page.locator("#terminal-output")
    await expect(terminal).to_contain_text("World Generation")
    await expect(terminal).to_contain_text("[PASS] World Generation")


async def test_run_all_sections(page: Page, server: None) -> None:
    """Verify the 'Run All' flow sequentially passes all sections."""
    await page.goto(TEST_URL)

    run_all_btn = page.locator("#run-all-btn")
    await run_all_btn.click()

    # Progress text should eventually reach 8/8
    progress_text = page.locator("#progress-text")
    await expect(progress_text).to_have_text("8/8", timeout=20000)

    # Check that all 8 section badges are green PASS
    badges = page.locator(".section-btn .badge")
    for i in range(8):
        await expect(badges.nth(i)).to_have_class(re.compile(r"\bpass\b"))

    # Check terminal ends cleanly
    terminal = page.locator("#terminal-output")
    await expect(terminal).to_contain_text("8/8 sections passed in")

    # Check the progress bar is at 100% (or equivalent style)
    progress_bar = page.locator("#progress-bar-fill")
    await expect(progress_bar).to_have_attribute("style", re.compile(r"width:\s*100%"))


async def test_world_canvas_visibility(page: Page, server: None) -> None:
    """Verify the world canvas is rendered and visible."""
    await page.goto(TEST_URL)
    canvas = page.locator("#world-canvas")
    await expect(canvas).to_be_visible()

    # Check that it has non-zero size
    box = await canvas.bounding_box()
    assert box is not None
    assert box["width"] > 0
    assert box["height"] > 0
