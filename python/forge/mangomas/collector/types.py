"""Types and data structures for MangoMAS FORGE scenario collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from forge.mangomas.pipeline import CollectedTrainingData


@dataclass(frozen=True)
class ResolvedForgeScenario:
    """Executable FORGE scenario metadata resolved from a path or scenario id."""

    scenario_id: str
    name: str
    source_path: Path
    difficulty_tier: int
    min_agents: int
    max_agents: int
    forge_config: dict[str, Any]


@dataclass(frozen=True)
class ScenarioRolloutSummary:
    """Collection summary for a single scenario."""

    scenario_id: str
    episodes_collected: int
    mean_reward: float
    success_rate: float
    source_path: Path


@dataclass(frozen=True)
class ScenarioCollectionResult:
    """Collected MangoMAS training data and scenario-level summaries."""

    training_data: CollectedTrainingData
    curriculum_outcomes: list[bool]
    scenario_summaries: list[ScenarioRolloutSummary]
    resolved_scenarios: list[ResolvedForgeScenario]

    def total_episodes(self) -> int:
        return self.training_data.num_episodes

    def total_steps(self) -> int:
        return int(sum(self.training_data.per_episode_step_counts()))

    def total_reward(self) -> float:
        return float(sum(float(np.sum(rewards)) for rewards in self.training_data.step_rewards()))

    def success_rate(self) -> float:
        if not self.curriculum_outcomes:
            return 0.0
        return float(np.mean(self.curriculum_outcomes))

    def mean_episode_reward(self) -> float:
        if self.total_episodes() == 0:
            return 0.0
        return self.total_reward() / float(self.total_episodes())

    def to_report_dict(
        self,
        *,
        mode: str,
        platform: str,
        policy_name: str,
        base_seed: int,
        scenario_refs: Sequence[str | Path],
        config_paths: Sequence[str | Path],
        run_name: str,
    ) -> dict[str, Any]:
        resolved_by_id = {scenario.scenario_id: scenario for scenario in self.resolved_scenarios}
        scenarios: list[dict[str, Any]] = []
        for summary in self.scenario_summaries:
            resolved = resolved_by_id.get(summary.scenario_id)
            scenarios.append(
                {
                    "scenario_id": summary.scenario_id,
                    "name": resolved.name if resolved is not None else summary.scenario_id,
                    "difficulty_tier": resolved.difficulty_tier if resolved is not None else 0,
                    "min_agents": resolved.min_agents if resolved is not None else 0,
                    "max_agents": resolved.max_agents if resolved is not None else 0,
                    "episodes_collected": summary.episodes_collected,
                    "mean_reward": summary.mean_reward,
                    "success_rate": summary.success_rate,
                    "source_path": str(summary.source_path),
                }
            )

        return {
            "version": "1.0",
            "mode": mode,
            "platform": platform,
            "policy_name": policy_name,
            "resolved_seed": base_seed,
            "run_name": run_name,
            "config_paths": [str(path) for path in config_paths],
            "scenario_refs": [str(ref) for ref in scenario_refs],
            "totals": {
                "episodes": self.total_episodes(),
                "steps": self.total_steps(),
                "total_reward": self.total_reward(),
                "mean_episode_reward": self.mean_episode_reward(),
                "success_rate": self.success_rate(),
                "num_scenarios": len(self.scenario_summaries),
            },
            "scenarios": scenarios,
        }


@dataclass(frozen=True)
class _EpisodeRollout:
    observations: np.ndarray
    action_names: list[str]
    action_ids: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    raw_observations: list[dict[str, float]]
    total_reward: float
    success: bool
    teacher_intentions: list[int] | None = None
    teacher_rationales: list[str] | None = None
    teacher_subgoals: list[list[str]] | None = None
    teacher_value_hats: list[float] | None = None
    teacher_constraint_critiques: list[dict[str, bool]] | None = None
    teacher_top_k_probs: list[list[dict[str, Any]]] | None = None
    teacher_prompt_tokens: list[int] | None = None
    teacher_completion_tokens: list[int] | None = None
    teacher_latency_ms: list[float] | None = None
    teacher_providers: list[str] | None = None
    action_space_size: int = 0
