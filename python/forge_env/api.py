"""forge_env REST API — async session-pool environment server.

Exposes a stateless JSON HTTP API so that any language/tool can drive a FORGE
environment over the network.  Each "session" is an independent environment
instance stored in an in-memory TTL cache.

Endpoints
---------
POST /envs                   Create a new environment session
GET  /envs/{session_id}      Inspect session metadata
POST /envs/{session_id}/reset    Reset episode
POST /envs/{session_id}/step     Step the environment
GET  /envs/{session_id}/spaces   Observation + action space info
DELETE /envs/{session_id}    Destroy session
GET  /health                 Liveness probe

Run::

    uvicorn forge_env.api:app --reload --port 8765
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    import gymnasium

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FORGE env import
# ---------------------------------------------------------------------------

try:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
    from forge_env.wrappers import FlattenObservationWrapper, TimeLimit

    _ENV_AVAILABLE = True
except ImportError:
    _ENV_AVAILABLE = False
    logger.warning("forge_env native module not available — env calls will 404.")

# ---------------------------------------------------------------------------
# Settings (env-var driven, no hardcoded values)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


SESSION_TTL_SECONDS: int = _env_int("FORGE_SESSION_TTL", 3600)
MAX_SESSIONS: int = _env_int("FORGE_MAX_SESSIONS", 64)
CORS_ORIGINS: list[str] = os.environ.get("FORGE_CORS_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """Holds a single live environment session."""

    session_id: str
    env: gymnasium.Env  # type: ignore[type-arg]
    config: dict[str, Any]
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    step_count: int = 0


_sessions: dict[str, Session] = {}
_lock = asyncio.Lock()


async def _get_session(session_id: str) -> Session:
    async with _lock:
        sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    sess.last_used = time.monotonic()
    return sess


async def _evict_expired() -> None:
    """Remove sessions that have exceeded the TTL."""
    now = time.monotonic()
    async with _lock:
        expired = [sid for sid, s in _sessions.items() if now - s.last_used > SESSION_TTL_SECONDS]
        for sid in expired:
            with contextlib.suppress(Exception):
                _sessions[sid].env.close()
            del _sessions[sid]
    if expired:
        logger.info("Evicted %d expired sessions: %s", len(expired), expired)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CreateEnvRequest(BaseModel):
    world_width: int = Field(default=32, ge=4, le=256)
    world_height: int = Field(default=32, ge=4, le=256)
    num_agents: int = Field(default=1, ge=1, le=8)
    seed: int = Field(default=42, ge=0)
    max_steps: int = Field(default=500, ge=1, le=10_000)
    flatten_obs: bool = True

    @model_validator(mode="after")
    def check_grid(self) -> CreateEnvRequest:
        if self.world_width * self.world_height > 256 * 256:
            raise ValueError("Grid too large (max 256x256).")
        return self


class ResetRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, description="Discrete action index.")


class SessionResponse(BaseModel):
    session_id: str
    created_at: float
    step_count: int
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FORGE Environment API",
    version="0.2.0",
    description="REST API for driving FORGE RL environments over HTTP.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background TTL janitor
# ---------------------------------------------------------------------------


_background_tasks: set[asyncio.Task[None]] = set()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: start janitor on startup, cancel on shutdown."""
    async def _janitor_loop() -> None:
        while True:
            await asyncio.sleep(60)
            await _evict_expired()

    task: asyncio.Task[None] = asyncio.create_task(_janitor_loop())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _generic_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error for %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict[str, Any]:
    """Liveness probe — always returns 200 OK."""
    return {
        "status": "ok",
        "env_available": _ENV_AVAILABLE,
        "active_sessions": len(_sessions),
    }


@app.post("/envs", response_model=SessionResponse, status_code=201, tags=["sessions"])
async def create_env(req: CreateEnvRequest) -> SessionResponse:
    """Create a new environment session."""
    if not _ENV_AVAILABLE:
        raise HTTPException(status_code=503, detail="forge_env native module not available.")

    await _evict_expired()

    config: dict[str, Any] = {
        "world": {"width": req.world_width, "height": req.world_height},
        "agents": {"num_agents": req.num_agents},
    }
    env_any: Any = ForgeGymnasiumEnv(config=config)
    env_any = TimeLimit(env_any, max_steps=req.max_steps)
    if req.flatten_obs:
        env_any = FlattenObservationWrapper(env_any)

    session_id = str(uuid.uuid4())
    sess = Session(session_id=session_id, env=env_any, config=config)

    async with _lock:
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(
                status_code=429,
                detail=f"Session limit ({MAX_SESSIONS}) reached. Delete idle sessions first.",
            )
        _sessions[session_id] = sess

    logger.info("Created session %s (grid=%dx%d agents=%d)", session_id, req.world_width, req.world_height, req.num_agents)
    return SessionResponse(
        session_id=session_id,
        created_at=sess.created_at,
        step_count=0,
        config=config,
    )


@app.get("/envs/{session_id}", response_model=SessionResponse, tags=["sessions"])
async def get_env(session_id: str) -> SessionResponse:
    """Retrieve metadata for an existing session."""
    sess = await _get_session(session_id)
    return SessionResponse(
        session_id=sess.session_id,
        created_at=sess.created_at,
        step_count=sess.step_count,
        config=sess.config,
    )


@app.post("/envs/{session_id}/reset", tags=["environment"])
async def reset_env(session_id: str, req: ResetRequest | None = None) -> dict[str, Any]:
    """Reset the episode and return the initial observation."""
    if req is None:
        req = ResetRequest()
    sess = await _get_session(session_id)
    obs, info = sess.env.reset(seed=req.seed)
    sess.step_count = 0
    return {"observation": _to_list(obs), "info": _safe_info(info)}


@app.post("/envs/{session_id}/step", tags=["environment"])
async def step_env(session_id: str, req: StepRequest) -> dict[str, Any]:
    """Advance the environment by one step."""
    sess = await _get_session(session_id)
    try:
        obs, reward, terminated, truncated, info = sess.env.step(req.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Step error: {exc}") from exc

    sess.step_count += 1
    return {
        "observation": _to_list(obs),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": _safe_info(info),
    }


@app.get("/envs/{session_id}/spaces", tags=["environment"])
async def get_spaces(session_id: str) -> dict[str, Any]:
    """Return observation and action space descriptions."""
    sess = await _get_session(session_id)
    obs_space = sess.env.observation_space
    act_space = sess.env.action_space
    return {
        "observation_space": {
            "shape": list(obs_space.shape),
            "dtype": str(obs_space.dtype),
            "low": _to_list(obs_space.low) if hasattr(obs_space, "low") else None,
            "high": _to_list(obs_space.high) if hasattr(obs_space, "high") else None,
        },
        "action_space": {
            "n": int(act_space.n),
            "type": "Discrete",
        },
    }


@app.delete("/envs/{session_id}", status_code=204, tags=["sessions"])
async def delete_env(session_id: str) -> None:
    """Destroy a session and release its resources."""
    async with _lock:
        sess = _sessions.pop(session_id, None)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    with contextlib.suppress(Exception):
        sess.env.close()
    logger.info("Deleted session %s", session_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to Python lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_list(x) for x in obj]
    return obj


def _safe_info(info: Any) -> dict[str, Any]:
    """Sanitise info dict for JSON serialisation."""
    if not isinstance(info, dict):
        return {}
    return {k: _to_list(v) for k, v in info.items() if isinstance(k, str)}
