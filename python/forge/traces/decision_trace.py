"""Decision trace dataclass for structured agent decision logging."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from forge.utils import dataclass_from_dict

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_VERSION = 1


@dataclass
class DecisionTrace:
    """Record of a single agent decision for analysis and training."""

    tick: int = 0
    agent_id: str = ""
    action: int = 0
    confidence: float = 0.0
    search_depth: int = 0
    ucb1_score: float = 0.0
    intent_label: str = ""
    preconditions: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    schema_version: int = DEFAULT_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionTrace:
        """Create a DecisionTrace from a dictionary.

        Handles backwards compatibility by migrating older schema versions
        and ignoring unknown fields from newer versions.
        """
        version = data.get("schema_version", 1)
        data = cls._migrate(data, version)
        return dataclass_from_dict(cls, data)

    @staticmethod
    def _migrate(data: dict[str, Any], from_version: int) -> dict[str, Any]:
        """Migrate trace data from older schema versions.

        Ensures backwards compatibility when loading traces saved with
        earlier schema versions.
        """
        if from_version < 1:
            data.setdefault("schema_version", DEFAULT_SCHEMA_VERSION)
            data.setdefault("intent_label", "")
            data.setdefault("expected_outcome", "")
        # Future: if from_version < 2: migrate v1 -> v2
        return data

    def to_dict(self) -> dict[str, Any]:
        """Convert this trace to a dictionary."""
        return asdict(self)
