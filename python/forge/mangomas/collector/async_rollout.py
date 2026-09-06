"""Asynchronous rollout collection and concurrent scenario execution for MangoMAS."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from forge.mangomas.adapters import ObservationAdapter
from forge.mangomas.collector.action_decoder import decode_action_name
from forge.mangomas.collector.scenario import resolve_forge_scenarios
from forge.mangomas.collector.sync_rollout import (
    _allocate_episode_counts,
    _augment_observation,
    _build_llm_agent,
    _default_env_factory,
    _default_provider_factory,
    _determine_episode_success,
    _extract_raw_observation,
    _prepare_env_config,
    _resolve_max_steps,
)
from forge.mangomas.collector.types import (
    ResolvedForgeScenario,
    ScenarioCollectionResult,
    ScenarioRolloutSummary,
    _EpisodeRollout,
)
from forge.mangomas.collector.writer import (
    _flush_rollout_to_writer,
    _open_trace_writer_if_enabled,
)
from forge.mangomas.pipeline import CollectedTrainingData
from forge.utils.observation import flatten_obs
from forge.utils.seed import derive_seed

if TYPE_CHECKING:
    from pathlib import Path

    from forge.mangomas.config import MangoMASBridgeConfig

logger = logging.getLogger(__name__)


async def _acollect_episode_rollout(
    *,
    env: Any,
    env_config: Mapping[str, Any],
    episode_seed: int,
    max_steps: int,
    observation_adapter: ObservationAdapter,
    policy_agent: Any,
    platform: str,
    comm_vocab_size: int,
    drone_enabled: bool,
    agri_enabled: bool,
    hex_enabled: bool,
    scenario_id: str,
    episode_index: int,
    teacher_config: Any,
) -> _EpisodeRollout:
    """Async version of :func:`_collect_episode_rollout`.

    The agent's ``aact`` is awaited so a true async provider (e.g.
    ``AsyncOpenAI`` behind ``LMStudioProvider.acomplete``) does not block
    the event loop. No trace writer is passed in — async records are
    collected from the returned rollout and written by the orchestrator
    in episode-index order to keep on-disk bytes deterministic.
    """
    obs, _info = env.reset(seed=episode_seed)
    enriched_obs = _augment_observation(obs, env_config, platform)

    episode_observations = [observation_adapter.adapt(enriched_obs)]
    episode_raw_observations: list[dict[str, float]] = []
    episode_action_names: list[str] = []
    episode_action_ids: list[int] = []
    episode_rewards: list[float] = []
    episode_dones: list[float] = []
    total_reward = 0.0
    final_info: Mapping[str, Any] | dict[str, Any] = {}

    teacher_intentions: list[int] = []
    teacher_rationales: list[str] = []
    teacher_subgoals: list[list[str]] = []
    teacher_value_hats: list[float] = []
    teacher_constraint_critiques: list[dict[str, bool]] = []
    teacher_top_k_probs: list[list[dict[str, Any]]] = []
    teacher_prompt_tokens: list[int] = []
    teacher_completion_tokens: list[int] = []
    teacher_latency_ms: list[float] = []
    teacher_providers: list[str] = []

    for _step in range(max_steps):
        episode_raw_observations.append(_extract_raw_observation(enriched_obs))
        flat_obs = flatten_obs(enriched_obs)
        action_id, trace_info = await policy_agent.aact(flat_obs)
        discrete_action = int(action_id)
        next_obs, reward, terminated, truncated, info = env.step(discrete_action)
        next_enriched_obs = _augment_observation(next_obs, env_config, platform)

        episode_action_ids.append(discrete_action)
        episode_action_names.append(
            decode_action_name(
                discrete_action,
                comm_vocab_size,
                drone_enabled,
                agri_enabled=agri_enabled,
                hex_enabled=hex_enabled,
            )
        )
        episode_rewards.append(float(reward))
        done = bool(terminated or truncated)
        episode_dones.append(1.0 if done else 0.0)
        total_reward += float(reward)
        episode_observations.append(observation_adapter.adapt(next_enriched_obs))

        if isinstance(trace_info, dict):
            intention = trace_info.get("intention")
            teacher_intentions.append(int(intention) if intention is not None else -1)
            teacher_rationales.append(str(trace_info.get("rationale") or ""))
            teacher_subgoals.append(list(trace_info.get("subgoals") or []))
            value_hat = trace_info.get("value_hat")
            teacher_value_hats.append(float(value_hat) if value_hat is not None else 0.0)
            teacher_constraint_critiques.append(dict(trace_info.get("constraint_critique") or {}))
            teacher_top_k_probs.append(list(trace_info.get("top_k_probs") or []))
            teacher_prompt_tokens.append(int(trace_info.get("prompt_tokens") or 0))
            teacher_completion_tokens.append(int(trace_info.get("completion_tokens") or 0))
            teacher_latency_ms.append(float(trace_info.get("latency_ms") or 0.0))
            teacher_providers.append(str(trace_info.get("provider") or ""))

        enriched_obs = next_enriched_obs
        final_info = info if isinstance(info, Mapping) else {}

        if done:
            break

    return _EpisodeRollout(
        observations=np.asarray(episode_observations, dtype=np.float32),
        action_names=episode_action_names,
        action_ids=np.asarray(episode_action_ids, dtype=np.int64),
        rewards=np.asarray(episode_rewards, dtype=np.float32),
        dones=np.asarray(episode_dones, dtype=np.float32),
        raw_observations=episode_raw_observations,
        total_reward=total_reward,
        success=_determine_episode_success(total_reward, final_info),
        teacher_intentions=teacher_intentions,
        teacher_rationales=teacher_rationales,
        teacher_subgoals=teacher_subgoals,
        teacher_value_hats=teacher_value_hats,
        teacher_constraint_critiques=teacher_constraint_critiques,
        teacher_top_k_probs=teacher_top_k_probs,
        teacher_prompt_tokens=teacher_prompt_tokens,
        teacher_completion_tokens=teacher_completion_tokens,
        teacher_latency_ms=teacher_latency_ms,
        teacher_providers=teacher_providers,
        action_space_size=int(env.action_space.n),
    )


async def _acollect_scenario_rollouts_concurrent(
    *,
    scenario: ResolvedForgeScenario,
    scenario_episodes: int,
    base_forge_config: Mapping[str, Any],
    base_seed: int,
    observation_adapter: ObservationAdapter,
    mangomas_config: MangoMASBridgeConfig,
    teacher_config: Any,
    provider_factory: Callable[[Any], Any] | None,
    env_factory: Callable[[dict[str, Any]], Any],
    concurrency: int,
) -> tuple[list[_EpisodeRollout], list[float], list[bool]]:
    """Run ``scenario_episodes`` episodes concurrently for one scenario.

    Each coroutine owns its own env + agent instance so per-episode RNG
    is fully independent. The provider is shared across episodes because
    a single ``AsyncOpenAI`` client is safe to call from many coroutines
    concurrently. After :func:`asyncio.gather` returns, results are
    sorted by ``episode_index`` and written to teacher trace shards in
    that deterministic order.
    """
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    factory = provider_factory or _default_provider_factory
    shared_provider = factory(teacher_config)

    async def _run_episode(
        episode_index: int,
    ) -> tuple[int, _EpisodeRollout]:
        async with sem:
            scenario_seed = derive_seed(base_seed, f"scenario:{scenario.scenario_id}")
            env_config = _prepare_env_config(
                base_forge_config, scenario, mangomas_config, scenario_seed
            )
            env = env_factory(env_config)
            action_space_size = int(env.action_space.n)
            agent_seed = derive_seed(
                base_seed,
                f"policy:{scenario.scenario_id}:{episode_index}",
            )
            agent = _build_llm_agent(
                teacher_config,
                action_space_size,
                agent_seed,
                provider_factory=lambda _cfg: shared_provider,
            )
            comm_vocab_size = int(env_config.get("agents", {}).get("comm_vocab_size", 0))
            drone_enabled = bool(env_config.get("drone", {}).get("enabled", False))
            agri_enabled = bool(env_config.get("agri", {}).get("enabled", False)) and drone_enabled
            hex_enabled = str(env_config.get("world", {}).get("grid_type", "")).lower() == "hex"
            max_steps = _resolve_max_steps(env_config, mangomas_config)
            try:
                rollout = await _acollect_episode_rollout(
                    env=env,
                    env_config=env_config,
                    episode_seed=derive_seed(
                        base_seed,
                        f"{scenario.scenario_id}:{episode_index}",
                    ),
                    max_steps=max_steps,
                    observation_adapter=observation_adapter,
                    policy_agent=agent,
                    platform=mangomas_config.platform,
                    comm_vocab_size=comm_vocab_size,
                    drone_enabled=drone_enabled,
                    agri_enabled=agri_enabled,
                    hex_enabled=hex_enabled,
                    scenario_id=scenario.scenario_id,
                    episode_index=episode_index,
                    teacher_config=teacher_config,
                )
            finally:
                env.close()
            return (episode_index, rollout)

    results = await asyncio.gather(*(_run_episode(i) for i in range(scenario_episodes)))
    # Deterministic on-disk order: write shards in episode_index order.
    results_sorted = sorted(results, key=lambda pair: pair[0])
    rollouts: list[_EpisodeRollout] = []
    rewards: list[float] = []
    successes: list[bool] = []
    for episode_index, rollout in results_sorted:
        writer = _open_trace_writer_if_enabled(teacher_config, scenario.scenario_id, episode_index)
        try:
            if writer is not None:
                _flush_rollout_to_writer(
                    rollout,
                    scenario_id=scenario.scenario_id,
                    episode_index=episode_index,
                    teacher_config=teacher_config,
                    writer=writer,
                )
        finally:
            if writer is not None:
                writer.close()
        rollouts.append(rollout)
        rewards.append(rollout.total_reward)
        successes.append(rollout.success)
    return rollouts, rewards, successes


async def _acollect_training_data_from_scenarios(
    *,
    base_forge_config: Mapping[str, Any],
    mangomas_config: MangoMASBridgeConfig,
    scenario_refs: Sequence[str | Path],
    total_episodes: int,
    base_seed: int,
    policy_name: str,
    search_dirs: Sequence[str | Path] | None,
    env_factory: Callable[[dict[str, Any]], Any] | None,
    teacher_config: Any,
    provider_factory: Callable[[Any], Any] | None,
) -> ScenarioCollectionResult:
    """Async entry point used when ``teacher_config.concurrency > 1``.

    Runs each scenario sequentially (to bound total in-flight envs) but
    parallelises episodes within a scenario under a Semaphore. Episode
    traces are flushed to disk in ``episode_index`` order so the on-disk
    bytes match the serial path's bytes for the same ``base_seed``.
    """
    _ = policy_name  # only "llm" is dispatched here today
    scenarios = resolve_forge_scenarios(scenario_refs, search_dirs=search_dirs)
    episode_counts = _allocate_episode_counts(total_episodes, len(scenarios))
    observation_adapter = ObservationAdapter(
        config=mangomas_config.observation_adapter,
        platform=mangomas_config.platform,
    )
    build_env = env_factory or _default_env_factory
    concurrency = int(teacher_config.concurrency)

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
        (
            scenario_rollouts,
            scenario_rewards,
            scenario_successes,
        ) = await _acollect_scenario_rollouts_concurrent(
            scenario=scenario,
            scenario_episodes=scenario_episodes,
            base_forge_config=base_forge_config,
            base_seed=base_seed,
            observation_adapter=observation_adapter,
            mangomas_config=mangomas_config,
            teacher_config=teacher_config,
            provider_factory=provider_factory,
            env_factory=build_env,
            concurrency=concurrency,
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
            "concurrent collection scenario=%s episodes=%d concurrency=%d mean_reward=%.3f success_rate=%.2f",
            scenario.scenario_id,
            scenario_episodes,
            concurrency,
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
