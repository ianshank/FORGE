"""main.py — FastAPI backend for the FORGE demo UI.

Endpoints:
  GET  /              → serves frontend index.html
  GET  /static/{path} → serves static CSS/JS
  GET  /api/sections  → list of demo sections + metadata
  POST /api/run/{section} → SSE stream of forge_demo.py output
  POST /api/run-all   → SSE stream running all 8 sections sequentially
  GET  /api/results   → parsed demo_results.md as JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .forge_runner import SECTIONS, parse_results_md, run_all, run_section

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="FORGE Demo UI", version="1.0.0")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Mount static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    seed: int = 42
    quick: bool = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the frontend SPA."""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index))


@app.get("/api/sections")
async def get_sections() -> list[dict[str, Any]]:
    """Return metadata for all 8 demo sections."""
    return [
        {"key": key, "name": name, "index": idx}
        for idx, (key, name) in enumerate(SECTIONS.items(), start=1)
    ]


@app.get("/api/results")
async def get_results() -> dict[str, Any]:
    """Return parsed demo_results.md as JSON."""
    return parse_results_md()


@app.post("/api/run/{section}")
async def run_demo_section(section: str, req: RunRequest) -> StreamingResponse:
    """Stream forge_demo.py output for a specific section via SSE."""
    if section not in SECTIONS:
        raise HTTPException(status_code=404, detail=f"Unknown section: {section}")

    async def event_stream() -> AsyncIterator[str]:
        async for line in run_section(section, seed=req.seed, quick=req.quick):
            # SSE format: "data: <payload>\n\n"
            payload = line.rstrip("\n").rstrip("\r")
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: __STREAM_END__\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/run-all")
async def run_all_sections(req: RunRequest) -> StreamingResponse:
    """Stream all 8 sections sequentially via SSE."""

    async def event_stream() -> AsyncIterator[str]:
        async for line in run_all(seed=req.seed, quick=req.quick):
            payload = line.rstrip("\n").rstrip("\r")
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: __STREAM_END__\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}
