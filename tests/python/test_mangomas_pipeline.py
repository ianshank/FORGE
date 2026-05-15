"""Tests for the stage-based MangoMAS drone training pipeline."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
from forge.mangomas.config import (
    MangoMASBridgeConfig,
    PipelineConfig,
    PipelineExecutionConfig,
    PipelinePathsConfig,
)
from forge.mangomas.pipeline import CollectedTrainingData, MangoMASDroneTrainingPipeline


def _sample_collected_data(include_raw: bool = True) -> CollectedTrainingData:
    observations = [
        np.arange(132, dtype=np.float32).reshape(6, 22) / 100.0,
        np.arange(132, 264, dtype=np.float32).reshape(6, 22) / 100.0,
    ]
    action_names = [
        ["Move", "Ascend", "Hover", "Communicate", "Scan"],
        ["Move", "Hover", "Move", "DropPayload", "Noop"],
    ]
    action_ids = [
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        np.array([5, 4, 3, 2, 1], dtype=np.int64),
    ]
    rewards = [
        np.array([1.0, 0.5, 0.25, 0.0, 1.5], dtype=np.float32),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    ]
    dones = [
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    ]
    raw_observations = []
    if include_raw:
        raw_observations = [
            [
                {
                    "battery": 0.8,
                    "altitude": 0.3,
                    "stamina_inverse": 0.2,
                    "boundary_distance": 0.5,
                    "threat_proximity": 0.9,
                }
                for _ in range(5)
            ],
            [
                {
                    "battery": 0.6,
                    "altitude": 0.4,
                    "stamina_inverse": 0.3,
                    "boundary_distance": 0.4,
                    "threat_proximity": 0.8,
                }
                for _ in range(5)
            ],
        ]
    return CollectedTrainingData(
        observations=observations,
        action_names=action_names,
        action_ids=action_ids,
        rewards=rewards,
        dones=dones,
        raw_observations=raw_observations,
    )


def _pipeline_config(tmp_path: Path, run_name: str) -> MangoMASBridgeConfig:
    config = MangoMASBridgeConfig()
    config.pipeline = PipelineConfig(
        paths=PipelinePathsConfig(output_root=str(tmp_path), run_name=run_name),
        execution=PipelineExecutionConfig(resume=True, fail_fast=True),
    )
    config.rssm_pretrain.num_epochs = 2
    config.rssm_pretrain.batch_size = 2
    config.bdi_trainer.num_epochs = 2
    config.bdi_trainer.batch_size = 4
    config.constitutional_trainer.num_epochs = 2
    config.constitutional_trainer.batch_size = 4
    return config


def test_pipeline_runs_all_available_stages(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path, "full-run")
    pipeline = MangoMASDroneTrainingPipeline(config)

    def evaluate_curiosity(weights: dict[str, float]) -> float:
        return weights.get("social", 0.0) * 2.0 + weights.get("epistemic", 0.0)

    def evaluate_sweep(params: dict[str, object], episodes: int) -> tuple[float, float, float]:
        reward = float(params["c_puct"]) * 10.0 - float(params["num_simulations"]) * 0.01
        return reward, 0.5, float(episodes * 100)

    result = pipeline.run(
        _sample_collected_data(include_raw=True),
        base_seed=123,
        curiosity_evaluate_fn=evaluate_curiosity,
        sweep_evaluate_fn=evaluate_sweep,
        curriculum_outcomes=[True, True, False, True, True],
    )

    assert result.manifest_path.exists()
    assert result.export_manifest_path is not None
    assert result.export_manifest_path.exists()
    assert result.stage_by_name("bdi") is not None
    assert result.stage_by_name("constitutional") is not None
    assert result.stage_by_name("rssm") is not None
    assert result.stage_by_name("curiosity") is not None
    assert result.stage_by_name("sweep") is not None
    assert result.stage_by_name("curriculum") is not None
    assert result.stage_by_name("export") is not None

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["platform"] == "drone"
    assert manifest["resolved_seed"] == 123
    # BC stage was added in PR #42; it runs (and emits a `skipped` note) even
    # when no teacher data is present, so the manifest now records 8 stages.
    assert len(manifest["stages"]) == 8

    export_manifest = json.loads(result.export_manifest_path.read_text(encoding="utf-8"))
    assert set(export_manifest["components"]) == {
        "bdi",
        "constitutional",
        "rssm",
        "mcts",
        "curiosity",
        "curriculum",
    }


def test_pipeline_skips_optional_stages_without_inputs(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path, "partial-run")
    pipeline = MangoMASDroneTrainingPipeline(config)

    result = pipeline.run(_sample_collected_data(include_raw=False), base_seed=5)

    assert result.stage_by_name("bdi") is not None
    assert result.stage_by_name("rssm") is not None
    assert result.stage_by_name("constitutional").status == "skipped"
    assert result.stage_by_name("curiosity").status == "skipped"
    assert result.stage_by_name("sweep").status == "skipped"
    assert result.stage_by_name("curriculum").status == "skipped"
    assert result.stage_by_name("export") is not None

    export_manifest = json.loads(result.export_manifest_path.read_text(encoding="utf-8"))
    assert set(export_manifest["components"]) == {"bdi", "rssm"}
