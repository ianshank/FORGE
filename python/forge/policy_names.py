"""Stable collection/policy identifiers shared by CLI, collector, and agents.

Keep string literals here so callers never inline policy names. Adding a
policy is a one-line change plus a collector branch.
"""

from __future__ import annotations

from typing import Final

POLICY_RANDOM: Final[str] = "random"
POLICY_MCTS: Final[str] = "mcts"
POLICY_SKILL: Final[str] = "skill"
POLICY_LLM: Final[str] = "llm"

COLLECTION_POLICY_CHOICES: Final[tuple[str, ...]] = (
    POLICY_RANDOM,
    POLICY_MCTS,
    POLICY_SKILL,
    POLICY_LLM,
)
DEFAULT_COLLECTION_POLICY: Final[str] = POLICY_RANDOM
