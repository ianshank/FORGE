"""FORGE integration module: cross-layer training orchestration."""

from __future__ import annotations

from forge.integration.integrated_trainer import (
    IntegratedTrainer,
    IntegrationTrainerConfig,
)
from forge.integration.meta_learner import MetaLearner, MetaLearnerConfig

__all__ = [
    "IntegratedTrainer",
    "IntegrationTrainerConfig",
    "MetaLearner",
    "MetaLearnerConfig",
]
