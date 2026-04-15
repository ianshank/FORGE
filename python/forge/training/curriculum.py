"""Task-tier curriculum for progressive training difficulty.

Implements staged introduction of harder task tiers as training progresses,
adapted from AlphaGalerkin's BoardSizeCurriculum for FORGE's tier system.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """A single stage in the curriculum schedule.

    Attributes:
        start_step: Training step at which this stage activates.
        task_tiers: Which task tiers are available during this stage.
        tier_weights: Sampling weights for each tier (must match length of
            *task_tiers*).  Weights are normalised automatically.
    """

    start_step: int
    task_tiers: list[int]
    tier_weights: list[float]

    def __post_init__(self) -> None:
        """Validate and normalise weights."""
        if len(self.tier_weights) != len(self.task_tiers):
            msg = (
                f"tier_weights length ({len(self.tier_weights)}) must match "
                f"task_tiers length ({len(self.task_tiers)})"
            )
            raise ValueError(msg)
        total = sum(self.tier_weights)
        if total <= 0:
            msg = f"tier_weights must sum to a positive value, got {total}"
            raise ValueError(msg)
        self.tier_weights = [w / total for w in self.tier_weights]


def _default_stages() -> list[CurriculumStage]:
    """Return the default three-stage curriculum."""
    return [
        CurriculumStage(
            start_step=0,
            task_tiers=[0, 1],
            tier_weights=[0.7, 0.3],
        ),
        CurriculumStage(
            start_step=1000,
            task_tiers=[0, 1, 2],
            tier_weights=[0.3, 0.4, 0.3],
        ),
        CurriculumStage(
            start_step=5000,
            task_tiers=[0, 1, 2, 3],
            tier_weights=[0.1, 0.3, 0.3, 0.3],
        ),
    ]


@dataclass
class TaskTierCurriculumConfig:
    """Configuration for the task-tier curriculum.

    Attributes:
        stages: Ordered list of curriculum stages.
        seed: Random seed for reproducible tier sampling.
    """

    stages: list[CurriculumStage] = field(default_factory=_default_stages)
    seed: int = 42


class TaskTierCurriculum:
    """Sample task tiers according to a staged curriculum.

    Stages are ordered by ``start_step``.  At each training step the most
    recently activated stage determines the available tiers and their
    sampling weights.

    Example::

        curriculum = TaskTierCurriculum()
        for step in range(10_000):
            tier = curriculum.sample_tier(step)
            task = make_task(tier)
    """

    def __init__(self, config: TaskTierCurriculumConfig | None = None) -> None:
        """Initialize curriculum.

        Args:
            config: Curriculum configuration.  Uses defaults when *None*.
        """
        self.config = config or TaskTierCurriculumConfig()
        self._stages = sorted(self.config.stages, key=lambda s: s.start_step)
        if not self._stages:
            msg = "At least one curriculum stage is required"
            raise ValueError(msg)
        self._rng = random.Random(self.config.seed)
        logger.info(
            "TaskTierCurriculum initialised with %d stages",
            len(self._stages),
        )

    def current_stage(self, step: int) -> CurriculumStage:
        """Return the active stage for the given training step.

        Args:
            step: Current training step.

        Returns:
            The most recently activated ``CurriculumStage``.
        """
        active = self._stages[0]
        for stage in self._stages:
            if step >= stage.start_step:
                active = stage
            else:
                break
        return active

    def sample_tier(self, current_step: int) -> int:
        """Sample a task tier for the given training step.

        Args:
            current_step: Current training step.

        Returns:
            A task tier integer drawn from the current stage's distribution.
        """
        stage = self.current_stage(current_step)
        chosen = self._rng.choices(
            stage.task_tiers,
            weights=stage.tier_weights,
            k=1,
        )
        return chosen[0]
