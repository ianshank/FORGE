"""FORGE scenario collection for MangoMAS training pipelines."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from forge.mangomas.adapters import ObservationAdapter
from forge.mangomas.collector.action_decoder import (
    decode_action_name,
    skill_category_for_action_id,
    skill_category_for_action_name,
)
from forge.mangomas.collector.async_rollout import _acollect_training_data_from_scenarios
from forge.mangomas.collector.scenario import (
    DEFAULT_EVAL_REGISTRY_DIR,
    DEFAULT_SCENARIO_DIR,
    resolve_forge_scenarios,
)
from forge.mangomas.collector.sync_rollout import (
    _allocate_episode_counts,
    _build_llm_agent,
    _collect_episode_rollout,
    _collect_scenario_rollouts,
    _create_policy_agent,
    _default_env_factory,
    _prepare_env_config,
)
from forge.mangomas.collector.types import (
    ResolvedForgeScenario,
    ScenarioCollectionResult,
    ScenarioRolloutSummary,
    _EpisodeRollout,
)
from forge.mangomas.collector.writer import write_collection_report
from forge.mangomas.pipeline import CollectedTrainingData
from forge.policy_names import DEFAULT_COLLECTION_POLICY, POLICY_LLM
from forge.utils.seed import derive_seed

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from forge.mangomas.config import MangoMASBridgeConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_EVAL_REGISTRY_DIR",
    "DEFAULT_SCENARIO_DIR",
    "ResolvedForgeScenario",
    "ScenarioCollectionResult",
    "ScenarioRolloutSummary",
    "_EpisodeRollout",
    "_build_llm_agent",
    "_collect_episode_rollout",
    "_create_policy_agent",
    "collect_training_data_from_scenarios",
    "decode_action_name",
    "resolve_forge_scenarios",
    "skill_category_for_action_id",
    "skill_category_for_action_name",
    "write_collection_report",
]


def collect_training_data_from_scenarios(
    *,
    base_forge_config: Mapping[str, Any],
    mangomas_config: MangoMASBridgeConfig,
    scenario_refs: Sequence[str | Path],
    total_episodes: int,
    base_seed: int,
    policy_name: str = DEFAULT_COLLECTION_POLICY,
    search_dirs: Sequence[str | Path] | None = None,
    env_factory: Callable[[dict[str, Any]], Any] | None = None,
    teacher_config: Any = None,
    provider_factory: Callable[[Any], Any] | None = None,
) -> ScenarioCollectionResult:
    """Collect MangoMAS training data from resolved FORGE scenario rollouts.

    When ``policy_name == POLICY_LLM`` a ``teacher_config`` must be supplied
    (typically ``mangomas_config.teacher``). For ``teacher_config.concurrency
    > 1`` the call dispatches into an asyncio path that runs episodes
    concurrently while still writing teacher trace shards in
    ``episode_index`` order — see :func:`_acollect_scenario_rollouts_concurrent`.
    """
    if policy_name == POLICY_LLM and teacher_config is None:
        teacher_config = mangomas_config.teacher

    if (
        policy_name == POLICY_LLM
        and teacher_config is not None
        and int(getattr(teacher_config, "concurrency", 1)) > 1
    ):
        import asyncio

        return asyncio.run(
            _acollect_training_data_from_scenarios(
                base_forge_config=base_forge_config,
                mangomas_config=mangomas_config,
                scenario_refs=scenario_refs,
                total_episodes=total_episodes,
                base_seed=base_seed,
                policy_name=policy_name,
                search_dirs=search_dirs,
                env_factory=env_factory,
                teacher_config=teacher_config,
                provider_factory=provider_factory,
            )
        )

    scenarios = resolve_forge_scenarios(scenario_refs, search_dirs=search_dirs)
    episode_counts = _allocate_episode_counts(total_episodes, len(scenarios))
    observation_adapter = ObservationAdapter(
        config=mangomas_config.observation_adapter,
        platform=mangomas_config.platform,
    )
    build_env = env_factory or _default_env_factory

    observations: list[np.ndarray] = []
    action_names: list[list[str]] = []
    action_ids: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    raw_observations: list[list[dict[str, float]]] = []
    teacher_intentions: list[list[int]] = []
    teacher_rationales: list[list[str]] = []
    teacher_subgoals: list[list[list[str]]] = []
    teacher_value_hats: list[list[float]] = []
    teacher_constraint_critiques: list[list[dict[str, bool]]] = []
    teacher_top_k_probs: list[list[list[dict[str, Any]]]] = []
    curriculum_outcomes: list[bool] = []
    scenario_summaries: list[ScenarioRolloutSummary] = []
    action_space_sizes: list[int] = []

    for scenario, scenario_episodes in zip(scenarios, episode_counts):
        if scenario.min_agents > 1:
            logger.warning(
                "Scenario %s requests %d agents; Python MangoMAS collection uses a single "
                "controllable agent for compatibility with ForgeGymnasiumEnv",
                scenario.scenario_id,
                scenario.min_agents,
            )

        scenario_seed = derive_seed(base_seed, f"scenario:{scenario.scenario_id}")
        env_config = _prepare_env_config(
            base_forge_config, scenario, mangomas_config, scenario_seed
        )
        env = build_env(env_config)

        try:
            scenario_rollouts, scenario_rewards, scenario_successes = _collect_scenario_rollouts(
                env=env,
                env_config=env_config,
                scenario=scenario,
                scenario_episodes=scenario_episodes,
                base_seed=base_seed,
                observation_adapter=observation_adapter,
                policy_name=policy_name,
                mangomas_config=mangomas_config,
                teacher_config=teacher_config,
                provider_factory=provider_factory,
            )
            for rollout in scenario_rollouts:
                observations.append(rollout.observations)
                action_names.append(rollout.action_names)
                action_ids.append(rollout.action_ids)
                rewards.append(rollout.rewards)
                dones.append(rollout.dones)
                raw_observations.append(rollout.raw_observations)
                curriculum_outcomes.append(rollout.success)
                if rollout.teacher_intentions is not None:
                    teacher_intentions.append(rollout.teacher_intentions)
                    teacher_rationales.append(rollout.teacher_rationales or [])
                    teacher_subgoals.append(rollout.teacher_subgoals or [])
                    teacher_value_hats.append(rollout.teacher_value_hats or [])
                    teacher_constraint_critiques.append(rollout.teacher_constraint_critiques or [])
                    teacher_top_k_probs.append(rollout.teacher_top_k_probs or [])
                if rollout.action_space_size > 0:
                    action_space_sizes.append(int(rollout.action_space_size))

        finally:
            env.close()

        mean_reward = float(np.mean(scenario_rewards)) if scenario_rewards else 0.0
        success_rate = float(np.mean(scenario_successes)) if scenario_successes else 0.0
        scenario_summaries.append(
            ScenarioRolloutSummary(
                scenario_id=scenario.scenario_id,
                episodes_collected=scenario_episodes,
                mean_reward=mean_reward,
                success_rate=success_rate,
                source_path=scenario.source_path,
            )
        )
        logger.info(
            "Collected MangoMAS data from %s: episodes=%d mean_reward=%.3f success_rate=%.2f",
            scenario.scenario_id,
            scenario_episodes,
            mean_reward,
            success_rate,
        )

    return ScenarioCollectionResult(
        training_data=CollectedTrainingData(
            observations=observations,
            action_names=action_names,
            action_ids=action_ids,
            rewards=rewards,
            dones=dones,
            raw_observations=raw_observations,
            teacher_intentions=teacher_intentions,
            teacher_rationales=teacher_rationales,
            teacher_subgoals=teacher_subgoals,
            teacher_value_hats=teacher_value_hats,
            teacher_constraint_critiques=teacher_constraint_critiques,
            teacher_top_k_probs=teacher_top_k_probs,
            action_space_sizes=action_space_sizes,
        ),
        curriculum_outcomes=curriculum_outcomes,
        scenario_summaries=scenario_summaries,
        resolved_scenarios=scenarios,
    )
