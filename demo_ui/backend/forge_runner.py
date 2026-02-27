"""forge_runner.py — Async subprocess wrapper for forge_demo.py.

Provides:
  - ``run_section``: async generator streaming stdout lines from forge_demo.py
  - ``run_all``: async generator that runs all sections sequentially
  - ``parse_results_md``: parse demo_results.md into a structured dict
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

__all__ = ["SECTIONS", "parse_results_md", "run_all", "run_section"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FORGE_ROOT = Path(__file__).parent.parent.parent  # FORGE/
DEMO_SCRIPT = FORGE_ROOT / "examples" / "forge_demo.py"
RESULTS_MD = FORGE_ROOT / "demo_results.md"

SECTIONS: dict[str, str] = {
    "worldgen": "World Generation",
    "navigation": "Navigation",
    "gathering": "Resource Gathering",
    "crafting": "Crafting",
    "multiagent": "Multi-Agent",
    "daynight": "Day/Night Cycle",
    "determinism": "Determinism",
    "performance": "Performance",
}

# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


async def run_section(
    section: str,
    seed: int = 42,
    quick: bool = True,
    timeout: float = 120.0,
) -> AsyncGenerator[str, None]:
    """Async generator that yields stdout lines from forge_demo.py.

    Parameters
    ----------
    section:
        Demo section key (must be in ``SECTIONS``).
    seed:
        Random seed forwarded to the demo script.
    quick:
        If True, pass ``--quick`` flag to the demo script.
    timeout:
        Maximum seconds to wait for the subprocess to complete.  After
        this the process is killed and an ERROR line is yielded.
    """
    if section not in SECTIONS:
        yield f"ERROR: Unknown section '{section}'. Valid: {', '.join(SECTIONS)}\n"
        return

    cmd = [
        sys.executable,
        str(DEMO_SCRIPT),
        "--section",
        section,
        "--seed",
        str(seed),
        "--no-color",
    ]
    if quick:
        cmd.append("--quick")

    logger.debug("Starting section '%s' (seed=%d, quick=%s, timeout=%.1fs)", section, seed, quick, timeout)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(FORGE_ROOT),
        )
    except Exception as exc:
        logger.error("Failed to start process for section '%s': %s", section, exc)
        yield f"ERROR: Failed to start process: {exc}\n"
        return

    assert proc.stdout is not None
    try:
        # Stream lines with a loose deadline: read all stdout via communicate()
        # gated by asyncio.wait_for, which is available from Python 3.9+.
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except TimeoutError:
        logger.warning("Section '%s' timed out after %.1fs — killing process", section, timeout)
        proc.kill()
        await proc.communicate()  # drain to avoid resource leak
        yield f"\nERROR: Section timed out after {timeout:.0f}s\n"
        return

    for raw_line in stdout_bytes.splitlines(keepends=True):
        yield raw_line.decode("utf-8", errors="replace")

    # proc.communicate() already waited; returncode is now set.
    rc = proc.returncode
    logger.debug("Section '%s' finished with exit code %d", section, rc)
    if rc != 0:
        yield f"\nProcess exited with code {rc}\n"
    else:
        yield "\n__DONE__\n"


async def run_all(
    seed: int = 42,
    quick: bool = True,
) -> AsyncGenerator[str, None]:
    """Async generator that runs all 8 sections sequentially."""
    for section_key in SECTIONS:
        yield f"\n__SECTION_START__ {section_key}\n"
        async for line in run_section(section_key, seed=seed, quick=quick):
            yield line
        yield f"\n__SECTION_END__ {section_key}\n"


# ---------------------------------------------------------------------------
# Results parser
# ---------------------------------------------------------------------------


def parse_results_md(path: Path | None = None) -> dict[str, Any]:
    """Parse demo_results.md into a structured dict."""
    md_path = path or RESULTS_MD
    if not md_path.exists():
        return {"error": "demo_results.md not found", "sections": []}

    text = md_path.read_text(encoding="utf-8")

    # Extract top-level metadata
    date_match = re.search(r"\*\*Date:\*\*\s*(.+)", text)
    seed_match = re.search(r"\*\*Seed:\*\*\s*(\d+)", text)
    platform_match = re.search(r"\*\*Platform:\*\*\s*(.+)", text)
    result_match = re.search(r"\*\*Result:\*\*\s*(.+)", text)

    # Extract section table rows from the summary table
    # Pattern allows letters, digits, spaces, slashes, and hyphens (e.g. "Multi-Agent", "Day/Night Cycle")
    section_rows = re.findall(
        r"\|\s*([\w/\- ]+\w)\s*\|\s*(PASS|FAIL)\s*\|",
        text,
    )

    sections = [{"name": name.strip(), "status": status} for name, status in section_rows]

    # Extract performance metrics
    perf_match = re.search(
        r"\|\s*Steps/second\s*\|\s*([\d,]+)\s*\|",
        text,
    )
    steps_sec = perf_match.group(1) if perf_match else "N/A"

    us_match = re.search(r"\|\s*us/step\s*\|\s*([\d.]+)\s*\|", text)
    us_step = us_match.group(1) if us_match else "N/A"

    return {
        "date": date_match.group(1).strip() if date_match else "N/A",
        "seed": int(seed_match.group(1)) if seed_match else 42,
        "platform": platform_match.group(1).strip() if platform_match else "N/A",
        "result": result_match.group(1).strip() if result_match else "N/A",
        "sections": sections,
        "performance": {
            "steps_per_second": steps_sec,
            "us_per_step": us_step,
        },
    }
