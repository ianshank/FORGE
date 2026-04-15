"""Stage-based MangoMAS drone training pipeline foundations."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from forge.mangomas.bdi_trainer import BDIPreTrainer
from forge.mangomas.constitutional_trainer import ConstitutionalPreTrainer
from forge.mangomas.curiosity_optimizer import CuriosityWeightOptimizer
from forge.mangomas.curriculum_controller import PlatformCurriculumController
from forge.mangomas.export import WeightExporter
from forge.mangomas.rssm_pretrainer import RSSMPreTrainer
from forge.mangomas.sweep_runner import MCTSSweepRunner
from forge.utils.logging_config import setup_logging
from forge.utils.seed import derive_seed

if TYPE_CHECKING:
    from forge.mangomas.config import MangoMASBridgeConfig

logger = logging.getLogger(__name__)

CuriosityEvaluateFn = Callable[[dict[str, float]], float]
SweepEvaluateFn = Callable[[dict[str, Any], int], tuple[float, float, float]]


@dataclass
class CollectedTrainingData:
    """Episode data required by the MangoMAS training stages."""

    observations: list[NDArray[np.float32]]
    action_names: list[list[str]]
    action_ids: list[NDArray[np.int64]]
    rewards: list[NDArray[np.float32]]
    dones: list[NDArray[np.float32]]
    raw_observations: list[list[dict[str, float]]] = field(default_factory=list)

    def validate(self) -> None:
        episode_count = len(self.observations)
        if episode_count == 0:
            raise ValueError("CollectedTrainingData must contain at least one episode")

        lengths = [
            len(self.action_names),
            len(self.action_ids),
            len(self.rewards),
            len(self.dones),
        ]
        if any(length != episode_count for length in lengths):
            raise ValueError("All CollectedTrainingData fields must have the same episode count")
        if self.raw_observations and len(self.raw_observations) != episode_count:
            raise ValueError("raw_observations must match the episode count when provided")

    @property
    def num_episodes(self) -> int:
        return len(self.observations)

    @property
    def state_dim(self) -> int:
        if not self.observations or self.observations[0].ndim != 2:
            return 0
        return int(self.observations[0].shape[1])

    def per_episode_step_counts(self) -> list[int]:
        counts: list[int] = []
        for index, obs in enumerate(self.observations):
            obs_steps = max(int(obs.shape[0]) - 1, 0)
            count = min(
                obs_steps,
                len(self.action_names[index]),
                int(self.action_ids[index].shape[0]),
                int(self.rewards[index].shape[0]),
                int(self.dones[index].shape[0]),
            )
            if self.raw_observations:
                count = min(count, len(self.raw_observations[index]))
            counts.append(count)
        return counts

    def step_observations(self) -> list[NDArray[np.float32]]:
        return [
            episode[:count].astype(np.float32, copy=False)
            for episode, count in zip(self.observations, self.per_episode_step_counts())
        ]

    def step_action_names(self) -> list[list[str]]:
        return [
            episode[:count]
            for episode, count in zip(self.action_names, self.per_episode_step_counts())
        ]

    def step_rewards(self) -> list[NDArray[np.float32]]:
        return [
            episode[:count].astype(np.float32, copy=False)
            for episode, count in zip(self.rewards, self.per_episode_step_counts())
        ]

    def step_raw_observations(self) -> list[list[dict[str, float]]]:
        if not self.raw_observations:
            return []
        return [
            episode[:count]
            for episode, count in zip(self.raw_observations, self.per_episode_step_counts())
        ]

    def flattened_step_observations(self) -> NDArray[np.float32]:
        step_observations = self.step_observations()
        if not step_observations:
            return np.zeros((0, self.state_dim), dtype=np.float32)
        if all(int(episode.shape[0]) == 0 for episode in step_observations):
            return np.zeros((0, self.state_dim), dtype=np.float32)
        return cast(
            NDArray[np.float32],
            np.concatenate(step_observations, axis=0).astype(np.float32, copy=False),
        )

    def flattened_action_ids(self) -> NDArray[np.int64]:
        arrays = [
            episode[:count].astype(np.int64, copy=False)
            for episode, count in zip(self.action_ids, self.per_episode_step_counts())
        ]
        if not arrays or all(int(array.shape[0]) == 0 for array in arrays):
            return np.zeros((0,), dtype=np.int64)
        return cast(
            NDArray[np.int64],
            np.concatenate(arrays, axis=0).astype(np.int64, copy=False),
        )

    def flattened_rewards(self) -> NDArray[np.float32]:
        arrays = self.step_rewards()
        if not arrays or all(int(array.shape[0]) == 0 for array in arrays):
            return np.zeros((0,), dtype=np.float32)
        return cast(
            NDArray[np.float32],
            np.concatenate(arrays, axis=0).astype(np.float32, copy=False),
        )

    def flattened_raw_observations(self) -> list[dict[str, float]]:
        flattened: list[dict[str, float]] = []
        for episode in self.step_raw_observations():
            flattened.extend(episode)
        return flattened


@dataclass
class PipelineStageResult:
    """Status and outputs for a single pipeline stage."""

    name: str
    status: str
    duration_secs: float
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class PipelineRunResult:
    """Aggregate result for a pipeline run."""

    run_dir: Path
    export_dir: Path
    manifest_path: Path
    export_manifest_path: Path | None
    stage_results: list[PipelineStageResult]
    resolved_seed: int

    def stage_by_name(self, name: str) -> PipelineStageResult | None:
        for stage in self.stage_results:
            if stage.name == name:
                return stage
        return None


class MangoMASDroneTrainingPipeline:
    """Reusable stage-based orchestration for MangoMAS drone training."""

    def __init__(
        self,
        config: MangoMASBridgeConfig,
        base_output_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.base_output_dir = (
            Path(base_output_dir)
            if base_output_dir is not None
            else Path(config.pipeline.paths.output_root)
        )

    def configure_logging(self, run_dir: Path) -> None:
        """Configure root logging for a pipeline run."""
        log_file = run_dir / self.config.pipeline.paths.log_file_name
        setup_logging(
            level=self.config.pipeline.logging.level,
            json_format=self.config.pipeline.logging.json_format,
            log_file=str(log_file),
        )

    def resolve_run_dir(self, base_seed: int, run_name: str | None = None) -> Path:
        """Resolve the filesystem directory used for a pipeline-aligned run."""
        return self._resolve_run_dir(base_seed, run_name)

    def run(
        self,
        collected_data: CollectedTrainingData,
        *,
        base_seed: int | None = None,
        sweep_evaluate_fn: SweepEvaluateFn | None = None,
        curiosity_evaluate_fn: CuriosityEvaluateFn | None = None,
        curriculum_outcomes: list[bool] | None = None,
        run_name: str | None = None,
        configure_logging: bool = False,
    ) -> PipelineRunResult:
        """Run the available MangoMAS stages and export a manifest."""
        collected_data.validate()
        resolved_seed = base_seed if base_seed is not None else self.config.batch_collector.seed
        run_dir = self.resolve_run_dir(resolved_seed, run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        if configure_logging:
            self.configure_logging(run_dir)

        export_dir = run_dir / self.config.pipeline.paths.export_dir_name
        export_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting MangoMAS drone pipeline: run_dir=%s seed=%d", run_dir, resolved_seed)

        stage_results: list[PipelineStageResult] = []
        stage_artifacts: dict[str, Path] = {}
        stop_after_stage = self.config.pipeline.execution.stop_after_stage

        stage_result, artifact = self._run_bdi_stage(collected_data, run_dir, resolved_seed)
        stage_results.append(stage_result)
        if artifact is not None:
            stage_artifacts["bdi"] = artifact
        if self._should_stop(stage_result.name, stop_after_stage):
            return self._finalize_run(run_dir, export_dir, resolved_seed, stage_results, stage_artifacts)

        stage_result, artifact = self._run_constitutional_stage(
            collected_data,
            run_dir,
            resolved_seed,
        )
        stage_results.append(stage_result)
        if artifact is not None:
            stage_artifacts["constitutional"] = artifact
        if self._should_stop(stage_result.name, stop_after_stage):
            return self._finalize_run(run_dir, export_dir, resolved_seed, stage_results, stage_artifacts)

        stage_result, artifact = self._run_rssm_stage(collected_data, run_dir, resolved_seed)
        stage_results.append(stage_result)
        if artifact is not None:
            stage_artifacts["rssm"] = artifact
        if self._should_stop(stage_result.name, stop_after_stage):
            return self._finalize_run(run_dir, export_dir, resolved_seed, stage_results, stage_artifacts)

        stage_result, artifact = self._run_curiosity_stage(
            run_dir,
            resolved_seed,
            curiosity_evaluate_fn,
        )
        stage_results.append(stage_result)
        if artifact is not None:
            stage_artifacts["curiosity"] = artifact
        if self._should_stop(stage_result.name, stop_after_stage):
            return self._finalize_run(run_dir, export_dir, resolved_seed, stage_results, stage_artifacts)

        stage_result, artifact = self._run_sweep_stage(run_dir, resolved_seed, sweep_evaluate_fn)
        stage_results.append(stage_result)
        if artifact is not None:
            stage_artifacts["sweep"] = artifact
        if self._should_stop(stage_result.name, stop_after_stage):
            return self._finalize_run(run_dir, export_dir, resolved_seed, stage_results, stage_artifacts)

        stage_result, artifact = self._run_curriculum_stage(
            run_dir,
            resolved_seed,
            curriculum_outcomes,
        )
        stage_results.append(stage_result)
        if artifact is not None:
            stage_artifacts["curriculum"] = artifact
        if self._should_stop(stage_result.name, stop_after_stage):
            return self._finalize_run(run_dir, export_dir, resolved_seed, stage_results, stage_artifacts)

        return self._finalize_run(run_dir, export_dir, resolved_seed, stage_results, stage_artifacts)

    def _should_stop(self, stage_name: str, stop_after_stage: str) -> bool:
        return bool(stop_after_stage) and stage_name == stop_after_stage

    def _resolve_run_dir(self, base_seed: int, run_name: str | None) -> Path:
        resolved_name = (
            run_name
            or self.config.pipeline.paths.run_name
            or f"{self.config.platform}-seed-{base_seed}"
        )
        run_dir = self.base_output_dir / resolved_name
        if run_dir.exists() and not self.config.pipeline.execution.resume:
            msg = f"Run directory already exists and resume is disabled: {run_dir}"
            raise FileExistsError(msg)
        return run_dir

    def _stage_dir(self, run_dir: Path, stage_name: str) -> Path:
        stage_dir = run_dir / "stages" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def _run_bdi_stage(
        self,
        collected_data: CollectedTrainingData,
        run_dir: Path,
        base_seed: int,
    ) -> tuple[PipelineStageResult, Path | None]:
        started = time.perf_counter()
        stage_dir = self._stage_dir(run_dir, "bdi")
        trainer_config = replace(
            self.config.bdi_trainer,
            seed=derive_seed(base_seed, "bdi"),
        )
        trainer = BDIPreTrainer(
            config=trainer_config,
            overrides=self.config.transfer.bdi_mapping_overrides,
        )
        dataset = trainer.build_dataset(
            collected_data.step_observations(),
            collected_data.step_action_names(),
            collected_data.step_rewards(),
        )
        result = trainer.train(dataset)
        weights_path = stage_dir / "bdi_weights.npz"
        trainer.export_weights(weights_path)
        duration = time.perf_counter() - started
        return (
            PipelineStageResult(
                name="bdi",
                status="completed",
                duration_secs=duration,
                metrics={
                    "num_samples": dataset.num_samples,
                    "final_loss": result.final_loss,
                    "final_accuracy": result.final_accuracy,
                    "epochs_run": result.epochs_run,
                    "seed": trainer_config.seed,
                },
                outputs={"weights": str(weights_path)},
            ),
            weights_path,
        )

    def _run_constitutional_stage(
        self,
        collected_data: CollectedTrainingData,
        run_dir: Path,
        base_seed: int,
    ) -> tuple[PipelineStageResult, Path | None]:
        started = time.perf_counter()
        if not collected_data.raw_observations:
            return (
                PipelineStageResult(
                    name="constitutional",
                    status="skipped",
                    duration_secs=time.perf_counter() - started,
                    notes="raw_observations are required for the constitutional stage",
                ),
                None,
            )

        stage_dir = self._stage_dir(run_dir, "constitutional")
        trainer_config = replace(
            self.config.constitutional_trainer,
            seed=derive_seed(base_seed, "constitutional"),
        )
        trainer = ConstitutionalPreTrainer(config=trainer_config)
        dataset = trainer.build_dataset(
            collected_data.flattened_step_observations(),
            collected_data.flattened_action_ids(),
            collected_data.flattened_rewards(),
            collected_data.flattened_raw_observations(),
        )
        result = trainer.train(dataset)
        weights_path = stage_dir / "constitutional_weights.npz"
        trainer.export_weights(weights_path)
        duration = time.perf_counter() - started
        return (
            PipelineStageResult(
                name="constitutional",
                status="completed",
                duration_secs=duration,
                metrics={
                    "num_samples": dataset.num_samples,
                    "final_loss": result.final_loss,
                    "violation_rate": result.final_violation_rate,
                    "epochs_run": result.epochs_run,
                    "seed": trainer_config.seed,
                },
                outputs={"weights": str(weights_path)},
            ),
            weights_path,
        )

    def _run_rssm_stage(
        self,
        collected_data: CollectedTrainingData,
        run_dir: Path,
        base_seed: int,
    ) -> tuple[PipelineStageResult, Path | None]:
        started = time.perf_counter()
        stage_dir = self._stage_dir(run_dir, "rssm")
        trainer_config = replace(
            self.config.rssm_pretrain,
            seed=derive_seed(base_seed, "rssm"),
        )
        trainer = RSSMPreTrainer(config=trainer_config)
        dataset = trainer.build_sequences(
            collected_data.observations,
            collected_data.action_ids,
            collected_data.rewards,
            collected_data.dones,
        )
        result = trainer.train(dataset)
        weights_path = stage_dir / "rssm_weights.npz"
        trainer.export_all(weights_path)
        duration = time.perf_counter() - started
        return (
            PipelineStageResult(
                name="rssm",
                status="completed",
                duration_secs=duration,
                metrics={
                    "num_sequences": dataset.num_sequences,
                    "sequence_length": dataset.sequence_length,
                    "total_loss": result.total_loss,
                    "epochs_run": result.epochs_run,
                    "seed": trainer_config.seed,
                },
                outputs={"weights": str(weights_path)},
            ),
            weights_path,
        )

    def _run_curiosity_stage(
        self,
        run_dir: Path,
        base_seed: int,
        evaluate_fn: CuriosityEvaluateFn | None,
    ) -> tuple[PipelineStageResult, Path | None]:
        started = time.perf_counter()
        if evaluate_fn is None:
            return (
                PipelineStageResult(
                    name="curiosity",
                    status="skipped",
                    duration_secs=time.perf_counter() - started,
                    notes="No curiosity evaluate function was provided",
                ),
                None,
            )

        stage_dir = self._stage_dir(run_dir, "curiosity")
        optimizer_config = replace(
            self.config.curiosity_optimizer,
            seed=derive_seed(base_seed, "curiosity"),
        )
        optimizer = CuriosityWeightOptimizer(config=optimizer_config)
        result = optimizer.optimize(evaluate_fn)
        weights_path = stage_dir / "curiosity_weights.json"
        self._write_json(weights_path, result.weights)
        duration = time.perf_counter() - started
        return (
            PipelineStageResult(
                name="curiosity",
                status="completed",
                duration_secs=duration,
                metrics={
                    "fitness": result.fitness,
                    "iterations": result.iterations,
                    "seed": optimizer_config.seed,
                },
                outputs={"weights": str(weights_path)},
            ),
            weights_path,
        )

    def _run_sweep_stage(
        self,
        run_dir: Path,
        base_seed: int,
        evaluate_fn: SweepEvaluateFn | None,
    ) -> tuple[PipelineStageResult, Path | None]:
        started = time.perf_counter()
        if evaluate_fn is None:
            return (
                PipelineStageResult(
                    name="sweep",
                    status="skipped",
                    duration_secs=time.perf_counter() - started,
                    notes="No sweep evaluate function was provided",
                ),
                None,
            )

        stage_dir = self._stage_dir(run_dir, "sweep")
        sweep_config = replace(self.config.sweep, seed=derive_seed(base_seed, "sweep"))
        runner = MCTSSweepRunner(config=sweep_config)
        report = runner.run_sweep(evaluate_fn)
        config_path = stage_dir / "optimal_mcts_config.json"
        runner.export_optimal_config(report, config_path)
        duration = time.perf_counter() - started
        best_reward = report.best.mean_reward if report.best is not None else 0.0
        return (
            PipelineStageResult(
                name="sweep",
                status="completed",
                duration_secs=duration,
                metrics={
                    "configs_tested": len(report.results),
                    "best_mean_reward": best_reward,
                    "seed": sweep_config.seed,
                },
                outputs={"config": str(config_path)},
            ),
            config_path,
        )

    def _run_curriculum_stage(
        self,
        run_dir: Path,
        base_seed: int,
        curriculum_outcomes: list[bool] | None,
    ) -> tuple[PipelineStageResult, Path | None]:
        started = time.perf_counter()
        if curriculum_outcomes is None:
            return (
                PipelineStageResult(
                    name="curriculum",
                    status="skipped",
                    duration_secs=time.perf_counter() - started,
                    notes="No curriculum outcomes were provided",
                ),
                None,
            )

        stage_dir = self._stage_dir(run_dir, "curriculum")
        curriculum_config = replace(
            self.config.curriculum,
            seed=derive_seed(base_seed, "curriculum"),
        )
        controller = PlatformCurriculumController(
            platform=self.config.platform,
            config=curriculum_config,
            tiers=curriculum_config.tiers or None,
        )
        for outcome in curriculum_outcomes:
            controller.record_outcome(outcome)
        state_path = stage_dir / "curriculum_state.json"
        controller.export_state(state_path)
        duration = time.perf_counter() - started
        return (
            PipelineStageResult(
                name="curriculum",
                status="completed",
                duration_secs=duration,
                metrics={
                    "episodes": len(curriculum_outcomes),
                    "current_tier": controller.current_tier,
                    "max_unlocked_tier": controller.max_unlocked_tier,
                    "success_rate": controller.success_rate(),
                    "seed": curriculum_config.seed,
                },
                outputs={"state": str(state_path)},
            ),
            state_path,
        )

    def _finalize_run(
        self,
        run_dir: Path,
        export_dir: Path,
        resolved_seed: int,
        stage_results: list[PipelineStageResult],
        stage_artifacts: dict[str, Path],
    ) -> PipelineRunResult:
        export_manifest_path = self._run_export_stage(export_dir, stage_results, stage_artifacts)
        manifest_path = run_dir / self.config.pipeline.paths.manifest_name
        manifest_payload = {
            "platform": self.config.platform,
            "resolved_seed": resolved_seed,
            "run_dir": str(run_dir),
            "export_dir": str(export_dir),
            "config": asdict(self.config),
            "stages": [asdict(stage) for stage in stage_results],
            "export_manifest": str(export_manifest_path) if export_manifest_path else None,
        }
        self._write_json(manifest_path, manifest_payload)
        logger.info("MangoMAS drone pipeline finished: manifest=%s", manifest_path)
        return PipelineRunResult(
            run_dir=run_dir,
            export_dir=export_dir,
            manifest_path=manifest_path,
            export_manifest_path=export_manifest_path,
            stage_results=stage_results,
            resolved_seed=resolved_seed,
        )

    def _run_export_stage(
        self,
        export_dir: Path,
        stage_results: list[PipelineStageResult],
        stage_artifacts: dict[str, Path],
    ) -> Path | None:
        exporter = WeightExporter(export_dir, platform=self.config.platform)
        exported_any = False

        bdi_path = stage_artifacts.get("bdi")
        if bdi_path is not None:
            exporter.export_bdi_weights(self._load_npz_weights(bdi_path))
            exported_any = True

        constitutional_path = stage_artifacts.get("constitutional")
        if constitutional_path is not None:
            exporter.export_constitutional_weights(self._load_npz_weights(constitutional_path))
            exported_any = True

        rssm_path = stage_artifacts.get("rssm")
        if rssm_path is not None:
            exporter.export_rssm_weights(self._load_npz_weights(rssm_path))
            exported_any = True

        sweep_path = stage_artifacts.get("sweep")
        if sweep_path is not None:
            exporter.export_mcts_config(self._load_json(sweep_path))
            exported_any = True

        curiosity_path = stage_artifacts.get("curiosity")
        if curiosity_path is not None:
            exporter.export_curiosity_weights(self._load_json(curiosity_path))
            exported_any = True

        curriculum_path = stage_artifacts.get("curriculum")
        if curriculum_path is not None:
            exporter.export_curriculum_state(self._load_json(curriculum_path))
            exported_any = True

        if not exported_any:
            return None

        manifest_path = exporter.finalize() / "manifest.json"
        stage_results.append(
            PipelineStageResult(
                name="export",
                status="completed",
                duration_secs=0.0,
                metrics={"exported_components": len(stage_artifacts)},
                outputs={"manifest": str(manifest_path)},
            )
        )
        return manifest_path

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        if not isinstance(data, dict):
            msg = f"Expected a JSON object in {path}"
            raise ValueError(msg)
        return data

    def _load_npz_weights(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
