"""MLflow configuration plumbing for FORGE.

Provides :class:`MlflowSettings`, a single source of truth for every knob
the :class:`forge.training.loggers.MLflowLogger` needs.  All values flow
from one of three places, in increasing precedence:

1. Hard-coded **defaults** that are intentionally null / empty so we never
   bake a server URI or experiment name into the codebase.
2. **Environment variables** following the upstream MLflow contract
   (``MLFLOW_TRACKING_URI``, ``MLFLOW_REGISTRY_URI``,
   ``MLFLOW_EXPERIMENT_NAME``, ``MLFLOW_RUN_NAME``,
   ``MLFLOW_ARTIFACT_LOCATION``, ``MLFLOW_HTTP_REQUEST_TIMEOUT``,
   ``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING``,
   ``MLFLOW_TRACKING_USERNAME``, ``MLFLOW_TRACKING_PASSWORD``,
   ``MLFLOW_TRACKING_TOKEN``).  Plus FORGE-specific tag-bag overrides
   via ``FORGE_MLFLOW_TAGS`` (``KEY=VALUE,KEY2=VALUE2`` form).
3. **Explicit overrides** passed to :meth:`MlflowSettings.merge`, used by
   CLI flags and programmatic callers.

The dataclass is deliberately backwards compatible: callers that already
pass ``experiment_name=...`` / ``run_name=...`` / ``tracking_uri=...``
directly into :class:`MLflowLogger` keep working unchanged.

Logging uses :mod:`logging` for structured emission so debug traces can
be enabled by raising the logger level to ``DEBUG``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public env-var names — declared once so callers (and tests) never reach
# into ``os.environ`` with magic strings.
# ---------------------------------------------------------------------------

ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_REGISTRY_URI = "MLFLOW_REGISTRY_URI"
ENV_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"
ENV_RUN_NAME = "MLFLOW_RUN_NAME"
ENV_ARTIFACT_LOCATION = "MLFLOW_ARTIFACT_LOCATION"
ENV_HTTP_TIMEOUT = "MLFLOW_HTTP_REQUEST_TIMEOUT"
ENV_SYSTEM_METRICS = "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"
ENV_TRACKING_USERNAME = "MLFLOW_TRACKING_USERNAME"
ENV_TRACKING_PASSWORD = "MLFLOW_TRACKING_PASSWORD"  # env-var name, not a credential
ENV_TRACKING_TOKEN = "MLFLOW_TRACKING_TOKEN"  # env-var name, not a credential
ENV_FORGE_TAGS = "FORGE_MLFLOW_TAGS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

__all__ = [
    "ENV_ARTIFACT_LOCATION",
    "ENV_EXPERIMENT_NAME",
    "ENV_FORGE_TAGS",
    "ENV_HTTP_TIMEOUT",
    "ENV_REGISTRY_URI",
    "ENV_RUN_NAME",
    "ENV_SYSTEM_METRICS",
    "ENV_TRACKING_PASSWORD",
    "ENV_TRACKING_TOKEN",
    "ENV_TRACKING_URI",
    "ENV_TRACKING_USERNAME",
    "MlflowSettings",
    "parse_tag_string",
]


def _coerce_bool(raw: str | None, *, default: bool = False) -> bool:
    """Parse a flag-style environment variable into a bool.

    Recognises ``1/true/yes/on`` and ``0/false/no/off`` (case-insensitive).
    Unrecognised values fall back to ``default`` with a debug log entry
    so misconfiguration is visible without aborting startup.
    """
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    logger.debug("MLflow env flag could not be parsed as bool: %r — using default=%s", raw, default)
    return default


def _coerce_float(raw: str | None) -> float | None:
    """Parse an optional float from an env var; ``None`` on absent/invalid."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("MLflow env float could not be parsed: %r — ignoring", raw)
        return None


def parse_tag_string(raw: str | None) -> dict[str, str]:
    """Parse ``KEY=VALUE,KEY2=VALUE2`` style tag strings into a dict.

    Empty/None input → empty dict.  Whitespace around keys/values is
    trimmed.  Pairs without ``=`` are skipped with a warning so a typo in
    the CLI never silently drops tagging entirely.
    """
    if not raw:
        return {}
    tags: dict[str, str] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            logger.warning("Skipping malformed MLflow tag (missing '='): %r", chunk)
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            logger.warning("Skipping MLflow tag with empty key: %r", chunk)
            continue
        tags[key] = value
    return tags


@dataclass
class MlflowSettings:
    """Resolved MLflow configuration for a FORGE training run.

    All fields are optional and default to ``None`` / empty so that an
    instance constructed with no arguments behaves like a "use MLflow's
    own defaults" sentinel — convenient for tests and for callers that
    only want to override one field.
    """

    tracking_uri: str | None = None
    registry_uri: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    artifact_location: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    log_system_metrics: bool = False
    http_request_timeout: float | None = None
    nested: bool = False

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MlflowSettings:
        """Build settings from process environment.

        Args:
            env: Optional mapping to read from.  Defaults to ``os.environ``
                so tests can inject a controlled mapping without touching
                global state.
        """
        env = env if env is not None else os.environ
        settings = cls(
            tracking_uri=env.get(ENV_TRACKING_URI) or None,
            registry_uri=env.get(ENV_REGISTRY_URI) or None,
            experiment_name=env.get(ENV_EXPERIMENT_NAME) or None,
            run_name=env.get(ENV_RUN_NAME) or None,
            artifact_location=env.get(ENV_ARTIFACT_LOCATION) or None,
            tags=parse_tag_string(env.get(ENV_FORGE_TAGS)),
            log_system_metrics=_coerce_bool(env.get(ENV_SYSTEM_METRICS)),
            http_request_timeout=_coerce_float(env.get(ENV_HTTP_TIMEOUT)),
        )
        logger.debug("MlflowSettings.from_env -> %s", settings.describe())
        return settings

    def merge(self, **overrides: Any) -> MlflowSettings:
        """Return a new :class:`MlflowSettings` with non-``None`` overrides applied.

        Tag dicts are *merged* (override values win on collision) rather
        than replaced wholesale so callers can layer CLI tags on top of
        env-supplied tags.
        """
        clean: dict[str, Any] = {}
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "tags":
                merged_tags = dict(self.tags)
                merged_tags.update(value)
                clean["tags"] = merged_tags
            else:
                clean[key] = value
        return replace(self, **clean)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply_to(self, mlflow_module: Any) -> None:
        """Set process-level MLflow URIs on the supplied ``mlflow`` module.

        Splitting this out of :class:`MLflowLogger` keeps the logger
        focused on per-run state while still allowing callers (e.g.
        autologgers) to configure MLflow without spinning up a run.
        """
        if self.tracking_uri:
            mlflow_module.set_tracking_uri(self.tracking_uri)
            logger.info("MLflow tracking_uri set to %s", self.tracking_uri)
        if self.registry_uri:
            mlflow_module.set_registry_uri(self.registry_uri)
            logger.info("MLflow registry_uri set to %s", self.registry_uri)
        if self.http_request_timeout is not None:
            # The upstream client reads MLFLOW_HTTP_REQUEST_TIMEOUT lazily;
            # write it back so the next REST call picks the override up
            # regardless of import order.
            os.environ[ENV_HTTP_TIMEOUT] = str(self.http_request_timeout)
            logger.debug("MLflow http_request_timeout = %s", self.http_request_timeout)

    def describe(self) -> dict[str, Any]:
        """Return a dict suitable for structured log lines / params.

        Credentials are intentionally excluded — they live in env vars
        owned by the MLflow client, not in this dataclass.
        """
        return {
            "tracking_uri": self.tracking_uri,
            "registry_uri": self.registry_uri,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "artifact_location": self.artifact_location,
            "tags": dict(self.tags),
            "log_system_metrics": self.log_system_metrics,
            "http_request_timeout": self.http_request_timeout,
            "nested": self.nested,
        }
