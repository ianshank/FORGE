"""FORGE-MangoMAS integration: adapters, sweep, pre-training, and curriculum.

Provides bidirectional bridges between FORGE's discrete simulation and
MangoMAS's continuous agent system for decision-level knowledge transfer.

Modules:
    config: Configuration dataclasses for all integration components.
    adapters: Action/observation space mapping (continuous ↔ discrete).
    sweep_runner: MCTS hyperparameter grid search orchestration.
    bdi_trainer: BDI intention classifier pre-training.
    constitutional_trainer: Constraint-aware policy pre-training.
    rssm_pretrainer: RSSM world model component pre-training.
    curiosity_optimizer: Curiosity channel weight meta-learning.
    curriculum_controller: Platform-specific adaptive curriculum.
    export: Weight and configuration bundle export pipeline.
    batch: High-throughput episode collection interface.
"""

from __future__ import annotations

from forge.mangomas.adapters import ActionSpaceAdapter, ObservationAdapter
from forge.mangomas.batch import BatchCollector, BatchResult, EpisodeData
from forge.mangomas.bc_trainer import BCDataset, BCTrainer, BCTrainerConfig, BCTrainResult
from forge.mangomas.bdi_trainer import BDIDataset, BDIPreTrainer, BDITrainResult
from forge.mangomas.config import (
    ActionAdapterConfig,
    BatchCollectorConfig,
    BDITrainerConfig,
    ConstitutionalTrainerConfig,
    CuriosityOptimizerConfig,
    CurriculumConfig,
    MangoMASBridgeConfig,
    ObservationAdapterConfig,
    PipelineConfig,
    PipelineExecutionConfig,
    PipelineLoggingConfig,
    PipelinePathsConfig,
    RSSMPreTrainConfig,
    SweepConfig,
    TeacherConfig,
    TransferConfig,
)
from forge.mangomas.constitutional_trainer import (
    ConstitutionalDataset,
    ConstitutionalPreTrainer,
    ConstitutionalTrainResult,
)
from forge.mangomas.curiosity_optimizer import CuriosityWeightOptimizer, CuriosityWeights
from forge.mangomas.curriculum_controller import PlatformCurriculumController, TierStatus
from forge.mangomas.export import ExportManifest, WeightExporter
from forge.mangomas.rssm_pretrainer import RSSMPreTrainer, RSSMTrainResult, SequenceDataset
from forge.mangomas.sweep_runner import (
    ComparisonReport,
    MCTSSweepRunner,
    SweepReport,
    SweepResult,
)
from forge.mangomas.teacher_trace import (
    TeacherDecisionTrace,
    TeacherTraceReader,
    TeacherTraceWriter,
)

__all__ = [
    "ActionAdapterConfig",
    "ActionSpaceAdapter",
    "BCDataset",
    "BCTrainResult",
    "BCTrainer",
    "BCTrainerConfig",
    "BDIDataset",
    "BDIPreTrainer",
    "BDITrainResult",
    "BDITrainerConfig",
    "BatchCollector",
    "BatchCollectorConfig",
    "BatchResult",
    "ComparisonReport",
    "ConstitutionalDataset",
    "ConstitutionalPreTrainer",
    "ConstitutionalTrainResult",
    "ConstitutionalTrainerConfig",
    "CuriosityOptimizerConfig",
    "CuriosityWeightOptimizer",
    "CuriosityWeights",
    "CurriculumConfig",
    "EpisodeData",
    "ExportManifest",
    "MCTSSweepRunner",
    "MangoMASBridgeConfig",
    "ObservationAdapter",
    "ObservationAdapterConfig",
    "PipelineConfig",
    "PipelineExecutionConfig",
    "PipelineLoggingConfig",
    "PipelinePathsConfig",
    "PlatformCurriculumController",
    "RSSMPreTrainConfig",
    "RSSMPreTrainer",
    "RSSMTrainResult",
    "SequenceDataset",
    "SweepConfig",
    "SweepReport",
    "SweepResult",
    "TeacherConfig",
    "TeacherDecisionTrace",
    "TeacherTraceReader",
    "TeacherTraceWriter",
    "TierStatus",
    "TransferConfig",
    "WeightExporter",
]
