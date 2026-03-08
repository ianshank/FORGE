"""HTTP client for pushing training metrics and traces to the forge-server dashboard.

Usage::

    client = DashboardClient("http://localhost:8080")
    client.post_training_metrics(episode=10, mean_reward=3.14, loss_policy=0.01)
    client.post_decision_traces([{"agentId": 0, "tick": 5, "intentLabel": "hold"}])
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 5.0


class DashboardClient:
    """Posts training metrics and decision traces to the forge-server REST API.

    All network errors are logged and swallowed so the training loop is
    never blocked by dashboard connectivity issues.
    """

    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Any = None
        logger.info("DashboardClient targeting %s", self.base_url)

    def _get_session(self) -> Any:
        """Lazily create an httpx.Client (avoids import at module level)."""
        if self._session is None:
            import httpx  # noqa: PLC0415

            self._session = httpx.Client(timeout=self.timeout)
        return self._session

    def post_training_metrics(self, **kwargs: Any) -> bool:
        """POST training metrics to /api/training-metrics.

        Keyword arguments are serialized as camelCase JSON matching the
        Rust ``TrainingMetrics`` struct.

        Returns:
            True if the server accepted the payload, False otherwise.
        """
        payload = _to_camel_case(kwargs)
        return self._post("/api/training-metrics", payload)

    def post_decision_traces(self, traces: list[dict[str, Any]]) -> bool:
        """POST a batch of decision trace entries to /api/decision-traces.

        Args:
            traces: List of dicts with keys matching ``DecisionTraceEntry``.

        Returns:
            True if the server accepted the payload, False otherwise.
        """
        payload = [_to_camel_case(t) for t in traces]
        return self._post("/api/decision-traces", payload)

    def _post(self, path: str, payload: Any) -> bool:
        """Send a POST request and return whether it succeeded."""
        url = f"{self.base_url}{path}"
        try:
            session = self._get_session()
            resp = session.post(url, json=payload)
            if resp.status_code == 200:
                return True
            logger.warning("Dashboard POST %s returned %d", path, resp.status_code)
        except Exception:
            logger.debug("Dashboard POST %s failed (server may be offline)", path, exc_info=True)
        return False

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None


def _to_camel_case(d: dict[str, Any]) -> dict[str, Any]:
    """Convert snake_case keys to camelCase for Rust serde compatibility."""
    result = {}
    for key, value in d.items():
        parts = key.split("_")
        camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
        result[camel] = value
    return result
