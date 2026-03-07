"""Decision trace dataclass for structured agent decision logging."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

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
        """Create a DecisionTrace from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert this trace to a dictionary."""
        return asdict(self)
