"""FORGE training module.

Provides experiment-tracking loggers, checkpointing, rollout buffers,
stability monitors, and curriculum scheduling for FORGE agent training.
"""

from __future__ import annotations

from forge.training.curriculum import (
    TaskTierCurriculum,
    TaskTierCurriculumConfig,
)
from forge.training.loggers import (
    CompositeLogger,
    ForgeLogger,
    MLflowLogger,
    MlflowSettings,
    TensorBoardLogger,
    WandbLogger,
    make_logger,
)
from forge.training.stability import (
    EarlyStopping,
    EarlyStoppingConfig,
    PlateauDetector,
    PlateauDetectorConfig,
)

# Public surface of `forge.training`. Listed explicitly so ruff F401 recognises
# the re-exports as intentional and downstream `from forge.training import *`
# remains stable.
__all__ = [
    "CompositeLogger",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ForgeLogger",
    "MLflowLogger",
    "MlflowSettings",
    "PlateauDetector",
    "PlateauDetectorConfig",
    "TaskTierCurriculum",
    "TaskTierCurriculumConfig",
    "TensorBoardLogger",
    "WandbLogger",
    "make_logger",
]
