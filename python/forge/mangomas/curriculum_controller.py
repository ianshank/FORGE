"""Platform curriculum controller for MangoMAS integration.

Manages car/drone curriculum with adaptive difficulty progression
through 5 tiers of increasing complexity.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.mangomas.config import (
    DEFAULT_CAR_TIERS,
    DEFAULT_DRONE_TIERS,
    CurriculumConfig,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases — canonical source is config module
CAR_TIERS = DEFAULT_CAR_TIERS
DRONE_TIERS = DEFAULT_DRONE_TIERS


@dataclass
class TierStatus:
    """Status of a curriculum tier."""

    tier: int
    name: str
    success_rate: float
    episodes_completed: int
    is_unlocked: bool


class PlatformCurriculumController:
    """Manages car/drone curriculum with adaptive difficulty.

    Tracks success rates per tier, promotes/demotes tiers based on
    rolling window performance, and samples scenarios accordingly.
    """

    def __init__(
        self,
        platform: str | None = None,
        config: CurriculumConfig | None = None,
        tiers: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self.platform = platform or self.config.platform
        if tiers:
            self._tiers = tiers
        else:
            self._tiers = self.config.resolved_tiers(platform=self.platform)
        self._current_tier = 1
        self._max_unlocked_tier = 1
        self._rng = np.random.default_rng(self.config.seed)

        # Per-tier outcome tracking
        self._tier_outcomes: dict[int, deque[bool]] = {
            t["tier"]: deque(maxlen=self.config.window_size)
            for t in self._tiers
        }
        self._total_episodes = 0

        logger.info(
            "PlatformCurriculumController: platform=%s, %d tiers",
            self.platform,
            len(self._tiers),
        )

    @property
    def current_tier(self) -> int:
        """Current active tier."""
        return self._current_tier

    @property
    def max_unlocked_tier(self) -> int:
        """Highest tier that has been unlocked."""
        return self._max_unlocked_tier

    def tier_info(self, tier: int) -> dict[str, Any]:
        """Get information about a specific tier."""
        for t in self._tiers:
            if t["tier"] == tier:
                return t
        raise ValueError(f"Unknown tier: {tier}")

    def success_rate(self, tier: int | None = None) -> float:
        """Get the success rate for a tier (default: current)."""
        tier = tier or self._current_tier
        outcomes = self._tier_outcomes.get(tier, deque())
        if len(outcomes) == 0:
            return 0.0
        return sum(outcomes) / len(outcomes)

    def record_outcome(self, success: bool, metrics: dict[str, Any] | None = None) -> None:
        """Record the outcome of an episode on the current tier."""
        self._tier_outcomes[self._current_tier].append(success)
        self._total_episodes += 1

        # Check for promotion after warmup
        if self._total_episodes >= self.config.warmup_episodes:
            self._maybe_adjust_tier()

        if metrics:
            logger.debug(
                "Tier %d outcome: success=%s, metrics=%s",
                self._current_tier,
                success,
                metrics,
            )

    def _maybe_adjust_tier(self) -> None:
        """Check if we should promote or demote the current tier."""
        rate = self.success_rate()
        tier_def = self.tier_info(self._current_tier)
        threshold = tier_def["success_threshold"]

        # Promote: success rate exceeds threshold
        if rate >= threshold and self._current_tier < len(self._tiers):
            self._current_tier += 1
            self._max_unlocked_tier = max(self._max_unlocked_tier, self._current_tier)
            logger.info(
                "Promoted to tier %d (rate=%.2f >= %.2f)",
                self._current_tier,
                rate,
                threshold,
            )

        # Demote: success rate is too low (with deadband)
        elif rate < threshold - self.config.adjustment_rate and self._current_tier > 1:
            self._current_tier -= 1
            logger.info(
                "Demoted to tier %d (rate=%.2f < %.2f)",
                self._current_tier,
                rate,
                threshold - self.config.adjustment_rate,
            )

    def sample_scenario(self) -> dict[str, Any]:
        """Sample a scenario config for the current tier."""
        tier_def = self.tier_info(self._current_tier)
        return {
            "tier": self._current_tier,
            "name": tier_def["name"],
            "forge_scenario": tier_def["forge_scenario"],
            "success_threshold": tier_def["success_threshold"],
        }

    def status(self) -> list[TierStatus]:
        """Get status of all tiers."""
        return [
            TierStatus(
                tier=t["tier"],
                name=t["name"],
                success_rate=self.success_rate(t["tier"]),
                episodes_completed=len(self._tier_outcomes[t["tier"]]),
                is_unlocked=t["tier"] <= self._max_unlocked_tier,
            )
            for t in self._tiers
        ]

    def export_state(self, path: str | Path) -> None:
        """Export curriculum state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "platform": self.platform,
            "current_tier": self._current_tier,
            "max_unlocked_tier": self._max_unlocked_tier,
            "total_episodes": self._total_episodes,
            "tiers": [
                {
                    "tier": t["tier"],
                    "name": t["name"],
                    "success_rate": self.success_rate(t["tier"]),
                    "episodes": len(self._tier_outcomes[t["tier"]]),
                }
                for t in self._tiers
            ],
        }

        with path.open("w") as f:
            json.dump(state, f, indent=2)
        logger.info("Curriculum state exported to %s", path)
