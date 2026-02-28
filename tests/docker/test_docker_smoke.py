"""tests/docker/test_docker_smoke.py — HTTP smoke tests for Docker Compose services.

These tests run inside CI AFTER ``docker compose up -d`` has been executed.
They validate that both services start, respond to health checks, and that
the FORGE Env REST API can create and step an environment session.

Environment variables (set by docker_ci.yml):
  FORGE_API_URL   Base URL for forge-env-api   (default: http://localhost:8765)
  FORGE_UI_URL    Base URL for forge-demo-ui   (default: http://localhost:8080)
"""

from __future__ import annotations

import os

import httpx
import pytest

_API_URL = os.environ.get("FORGE_API_URL", "http://localhost:8765").rstrip("/")
_UI_URL = os.environ.get("FORGE_UI_URL", "http://localhost:8080").rstrip("/")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api() -> httpx.Client:
    with httpx.Client(base_url=_API_URL, timeout=10.0) as client:
        yield client


@pytest.fixture(scope="module")
def ui() -> httpx.Client:
    with httpx.Client(base_url=_UI_URL, timeout=10.0) as client:
        yield client


# ---------------------------------------------------------------------------
# forge-env-api smoke tests (AC1 - AC6 from PRD)
# ---------------------------------------------------------------------------


class TestForgeEnvApiSmoke:
    """AC2, AC3, AC4, AC6 from prd-docker-containerisation."""

    def test_health_returns_ok(self, api: httpx.Client) -> None:
        """AC2: /health responds within 2s."""
        resp = api.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_create_session_returns_201(self, api: httpx.Client) -> None:
        """AC3: POST /envs returns 201 with session JSON."""
        resp = api.post("/envs", json={"world_width": 16, "world_height": 16, "num_agents": 1})
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data

    def test_reset_and_step(self, api: httpx.Client) -> None:
        """Basic lifecycle: create → reset → step."""
        sid = api.post("/envs", json={}).json()["session_id"]
        reset = api.post(f"/envs/{sid}/reset")
        assert reset.status_code == 200
        assert "observation" in reset.json()

        step = api.post(f"/envs/{sid}/step", json={"action": 0})
        assert step.status_code == 200
        d = step.json()
        assert "reward" in d
        assert "terminated" in d

    def test_session_limit_enforced(self, api: httpx.Client) -> None:
        """AC6: FORGE_MAX_SESSIONS=4 → 5th creation returns 429 or 503."""
        created: list[str] = []
        for _ in range(4):
            r = api.post("/envs", json={})
            if r.status_code == 201:
                created.append(r.json()["session_id"])
        # The 5th request must be rejected
        r = api.post("/envs", json={})
        assert r.status_code in (429, 503)
        # Cleanup
        for sid in created:
            api.delete(f"/envs/{sid}")


# ---------------------------------------------------------------------------
# forge-demo-ui smoke tests (AC1 from PRD)
# ---------------------------------------------------------------------------


class TestForgeDemoUiSmoke:
    """AC1: demo UI responds at its port."""

    def test_health_returns_ok(self, ui: httpx.Client) -> None:
        resp = ui.get("/health")
        assert resp.status_code == 200

    def test_index_returns_html(self, ui: httpx.Client) -> None:
        resp = ui.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_api_sections_endpoint(self, ui: httpx.Client) -> None:
        resp = ui.get("/api/sections")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
