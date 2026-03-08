"""Tests for forge.utils.dashboard_client module."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from forge.utils.dashboard_client import DashboardClient, _to_camel_case

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _to_camel_case tests
# ---------------------------------------------------------------------------


class TestToCamelCase:
    """Tests for the _to_camel_case helper function."""

    def test_to_camel_case_basic(self) -> None:
        """Snake_case keys are converted to camelCase."""
        result = _to_camel_case({"mean_reward": 3.14, "loss_policy": 0.01})
        assert result == {"meanReward": 3.14, "lossPolicy": 0.01}

    def test_to_camel_case_single_word(self) -> None:
        """Single-word keys remain unchanged."""
        result = _to_camel_case({"episode": 10})
        assert result == {"episode": 10}

    def test_to_camel_case_empty_dict(self) -> None:
        """Empty dict returns empty dict."""
        assert _to_camel_case({}) == {}

    def test_to_camel_case_multiple_underscores(self) -> None:
        """Keys with multiple underscores are converted correctly."""
        result = _to_camel_case({"steps_per_second": 42})
        assert result == {"stepsPerSecond": 42}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_httpx_client() -> MagicMock:
    """Return a mock httpx.Client instance."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    client.post.return_value = response
    return client


@pytest.fixture()
def dashboard(mock_httpx_client: MagicMock) -> DashboardClient:
    """Return a DashboardClient with a pre-injected mock session."""
    dc = DashboardClient("http://localhost:8080")
    dc._session = mock_httpx_client
    return dc


# ---------------------------------------------------------------------------
# DashboardClient tests
# ---------------------------------------------------------------------------


class TestDashboardClient:
    """Tests for the DashboardClient class."""

    def test_post_training_metrics_success(
        self, dashboard: DashboardClient, mock_httpx_client: MagicMock
    ) -> None:
        """Successful POST returns True and sends camelCase payload."""
        result = dashboard.post_training_metrics(mean_reward=3.14, episode=10)

        assert result is True
        mock_httpx_client.post.assert_called_once_with(
            "http://localhost:8080/api/training-metrics",
            json={"meanReward": 3.14, "episode": 10},
        )

    def test_post_training_metrics_connection_error(
        self, dashboard: DashboardClient, mock_httpx_client: MagicMock
    ) -> None:
        """ConnectError is swallowed and returns False."""
        mock_httpx_client.post.side_effect = OSError("refused")

        result = dashboard.post_training_metrics(episode=1)

        assert result is False

    def test_post_decision_traces_success(
        self, dashboard: DashboardClient, mock_httpx_client: MagicMock
    ) -> None:
        """Traces are camelCase-converted and sent as a list."""
        traces = [
            {"agent_id": 0, "tick": 5, "intent_label": "hold"},
            {"agent_id": 1, "tick": 6, "intent_label": "move"},
        ]
        result = dashboard.post_decision_traces(traces)

        assert result is True
        expected_payload = [
            {"agentId": 0, "tick": 5, "intentLabel": "hold"},
            {"agentId": 1, "tick": 6, "intentLabel": "move"},
        ]
        mock_httpx_client.post.assert_called_once_with(
            "http://localhost:8080/api/decision-traces",
            json=expected_payload,
        )

    def test_post_decision_traces_empty_list(
        self, dashboard: DashboardClient, mock_httpx_client: MagicMock
    ) -> None:
        """Empty trace list is sent without error."""
        result = dashboard.post_decision_traces([])

        assert result is True
        mock_httpx_client.post.assert_called_once_with(
            "http://localhost:8080/api/decision-traces",
            json=[],
        )

    def test_close_with_session(
        self, dashboard: DashboardClient, mock_httpx_client: MagicMock
    ) -> None:
        """Closing with an active session calls close on the client."""
        dashboard.close()

        mock_httpx_client.close.assert_called_once()
        assert dashboard._session is None

    def test_close_without_session(self) -> None:
        """Closing without a session does not raise."""
        dc = DashboardClient("http://localhost:8080")
        assert dc._session is None
        dc.close()  # should not raise
        assert dc._session is None

    def test_lazy_session_creation(self) -> None:
        """Session is not created at construction time."""
        dc = DashboardClient("http://localhost:8080")
        assert dc._session is None

    @patch("forge.utils.dashboard_client.httpx", create=True)
    def test_lazy_session_created_on_first_request(
        self, mock_httpx_mod: MagicMock
    ) -> None:
        """Session is created lazily on first _post call."""
        mock_client_instance = MagicMock()
        response = MagicMock()
        response.status_code = 200
        mock_client_instance.post.return_value = response
        mock_httpx_mod.Client.return_value = mock_client_instance

        dc = DashboardClient("http://localhost:8080")
        assert dc._session is None

        with patch.dict("sys.modules", {"httpx": mock_httpx_mod}):
            dc.post_training_metrics(episode=1)

        assert dc._session is not None

    def test_post_http_error(
        self, dashboard: DashboardClient, mock_httpx_client: MagicMock
    ) -> None:
        """Non-200 status code returns False."""
        response = MagicMock()
        response.status_code = 500
        mock_httpx_client.post.return_value = response

        result = dashboard.post_training_metrics(episode=1)

        assert result is False
