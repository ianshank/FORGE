"""test_api.py — Unit + integration tests for forge_env REST API.

Tests use FastAPI's TestClient and patch forge_env.api.ForgeGymnasiumEnv
so the tests run without a built Rust extension.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# ---------------------------------------------------------------------------
# Make forge_env importable from the repo's python/ directory
# ---------------------------------------------------------------------------
_PYTHON_DIR = Path(__file__).parent.parent.parent / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from forge_env.api import _sessions, app  # noqa: E402

# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient

    _CLIENT_AVAILABLE = True
except ImportError:
    _CLIENT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mock env factory
# ---------------------------------------------------------------------------

def _make_mock_env() -> MagicMock:
    """Return a mock env that looks like a wrapped ForgeGymnasiumEnv."""
    env = MagicMock()
    obs = np.zeros(64, dtype=np.float32)
    env.observation_space.shape = (64,)
    env.observation_space.dtype = np.float32
    env.observation_space.low = np.full(64, -1.0)
    env.observation_space.high = np.full(64, 1.0)
    env.action_space.n = 10
    env.reset.return_value = (obs, {})
    env.step.return_value = (obs, 1.0, False, False, {})
    env.close.return_value = None
    return env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_env() -> Generator[None, None, None]:
    """Patch ForgeGymnasiumEnv, TimeLimit, FlattenObservationWrapper in api module."""
    _sessions.clear()
    mock_cls = MagicMock(side_effect=lambda **_kw: _make_mock_env())
    with (
        patch("forge_env.api.ForgeGymnasiumEnv", mock_cls),
        patch("forge_env.api.TimeLimit", side_effect=lambda env, **_kw: env),
        patch("forge_env.api.FlattenObservationWrapper", side_effect=lambda env: env),
        patch("forge_env.api._ENV_AVAILABLE", True),
    ):
        yield
    _sessions.clear()


@pytest.fixture()
def client() -> TestClient:
    if not _CLIENT_AVAILABLE:
        pytest.skip("fastapi[testclient] not installed")
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_reports_active_sessions(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json()["active_sessions"] == 0


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_create_env_returns_201(self, client: TestClient) -> None:
        resp = client.post("/envs", json={"world_width": 16, "world_height": 16, "num_agents": 1})
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["step_count"] == 0

    def test_get_env_returns_session(self, client: TestClient) -> None:
        sid = client.post("/envs", json={}).json()["session_id"]
        resp = client.get(f"/envs/{sid}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid

    def test_get_nonexistent_session_404(self, client: TestClient) -> None:
        resp = client.get("/envs/does-not-exist")
        assert resp.status_code == 404

    def test_delete_env(self, client: TestClient) -> None:
        sid = client.post("/envs", json={}).json()["session_id"]
        resp = client.delete(f"/envs/{sid}")
        assert resp.status_code == 204
        assert client.get(f"/envs/{sid}").status_code == 404

    def test_delete_nonexistent_404(self, client: TestClient) -> None:
        resp = client.delete("/envs/ghost")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Reset & Step
# ---------------------------------------------------------------------------


class TestResetAndStep:
    def test_reset_returns_observation(self, client: TestClient) -> None:
        sid = client.post("/envs", json={}).json()["session_id"]
        resp = client.post(f"/envs/{sid}/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert isinstance(data["observation"], list)

    def test_reset_with_seed(self, client: TestClient) -> None:
        sid = client.post("/envs", json={}).json()["session_id"]
        resp = client.post(f"/envs/{sid}/reset", json={"seed": 99})
        assert resp.status_code == 200

    def test_step_returns_full_payload(self, client: TestClient) -> None:
        sid = client.post("/envs", json={}).json()["session_id"]
        client.post(f"/envs/{sid}/reset")
        resp = client.post(f"/envs/{sid}/step", json={"action": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "terminated" in data
        assert "truncated" in data

    def test_step_increments_step_count(self, client: TestClient) -> None:
        sid = client.post("/envs", json={}).json()["session_id"]
        client.post(f"/envs/{sid}/reset")
        client.post(f"/envs/{sid}/step", json={"action": 0})
        client.post(f"/envs/{sid}/step", json={"action": 1})
        data = client.get(f"/envs/{sid}").json()
        assert data["step_count"] == 2

    def test_step_unknown_session_404(self, client: TestClient) -> None:
        resp = client.post("/envs/ghost/step", json={"action": 0})
        assert resp.status_code == 404

    def test_reset_unknown_session_404(self, client: TestClient) -> None:
        resp = client.post("/envs/ghost/reset")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


class TestSpaces:
    def test_spaces_returns_obs_and_act(self, client: TestClient) -> None:
        sid = client.post("/envs", json={}).json()["session_id"]
        resp = client.get(f"/envs/{sid}/spaces")
        assert resp.status_code == 200
        data = resp.json()
        assert "observation_space" in data
        assert "action_space" in data
        assert data["action_space"]["n"] == 10
        assert data["observation_space"]["shape"] == [64]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_world_size_rejected(self, client: TestClient) -> None:
        resp = client.post("/envs", json={"world_width": -1, "world_height": 32})
        assert resp.status_code == 422

    def test_zero_agents_rejected(self, client: TestClient) -> None:
        resp = client.post("/envs", json={"num_agents": 0})
        assert resp.status_code == 422

    def test_max_steps_zero_rejected(self, client: TestClient) -> None:
        resp = client.post("/envs", json={"max_steps": 0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Service unavailable
# ---------------------------------------------------------------------------


class TestUnavailable:
    def test_create_env_503_when_native_unavailable(self, client: TestClient) -> None:
        with patch("forge_env.api._ENV_AVAILABLE", False):
            resp = client.post("/envs", json={})
        assert resp.status_code == 503
