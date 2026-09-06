"""Synchronous rollout collection and observation enrichment for MangoMAS."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from forge.mangomas.collector.action_decoder import decode_action_name
from forge.mangomas.collector.scenario import _copy_mapping, _deep_merge
from forge.mangomas.collector.types import ResolvedForgeScenario, _EpisodeRollout
from forge.mangomas.collector.writer import _open_trace_writer_if_enabled
from forge.utils.observation import flatten_obs
from forge.utils.seed import derive_seed

if TYPE_CHECKING:
    from forge.mangomas.adapters import ObservationAdapter
    from forge.mangomas.config import MangoMASBridgeConfig

logger = logging.getLogger(__name__)


def _compute_boundary_distance(position: Any, world_width: int, world_height: int) -> float:
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        return 0.0
    pos_x = float(position[0])
    pos_y = float(position[1])
    max_distance = max(min(world_width, world_height) / 2.0, 1.0)
    edge_distance = min(
        pos_x, pos_y, max(world_width - 1 - pos_x, 0.0), max(world_height - 1 - pos_y, 0.0)
    )
    return float(np.clip(edge_distance / max_distance, 0.0, 1.0))


def _compute_threat_proximity(obs: Mapping[str, Any]) -> float:
    grid = np.asarray(obs.get("grid_view", []), dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] < 2:
        return 0.0
    return float(np.clip(np.mean(grid[:, :, 1]), 0.0, 1.0))


def _augment_observation(
    obs: Mapping[str, Any],
    env_config: Mapping[str, Any],
    platform: str,
) -> dict[str, Any]:
    enriched = {str(key): value for key, value in obs.items()}
    world = _copy_mapping(
        env_config.get("world") if isinstance(env_config.get("world"), Mapping) else None
    )
    drone = _copy_mapping(
        env_config.get("drone") if isinstance(env_config.get("drone"), Mapping) else None
    )
    world_width = int(world.get("width", 1))
    world_height = int(world.get("height", 1))
    altitude = float(enriched.get("altitude", 0.0))
    max_altitude = float(drone.get("max_altitude", 1) or 1)
    if altitude > 1.0 and max_altitude > 0.0:
        altitude /= max_altitude
    stamina = float(enriched.get("stamina", 1.0))
    battery = float(enriched.get("battery", stamina))
    enriched["altitude"] = float(np.clip(altitude, 0.0, 1.0))
    enriched["battery"] = float(np.clip(battery, 0.0, 1.0))
    enriched.setdefault("morphology", 2.0 if platform == "drone" else 0.0)
    enriched.setdefault("heading", 0.0)
    enriched["stamina_inverse"] = float(np.clip(1.0 - stamina, 0.0, 1.0))
    enriched["boundary_distance"] = _compute_boundary_distance(
        enriched.get("position"),
        world_width,
        world_height,
    )
    enriched["threat_proximity"] = _compute_threat_proximity(enriched)
    return enriched


def _extract_raw_observation(obs: Mapping[str, Any]) -> dict[str, float]:
    return {
        "battery": float(obs.get("battery", 1.0)),
        "altitude": float(obs.get("altitude", 0.0)),
        "stamina_inverse": float(obs.get("stamina_inverse", 0.0)),
        "boundary_distance": float(obs.get("boundary_distance", 0.0)),
        "threat_proximity": float(obs.get("threat_proximity", 0.0)),
    }


def _allocate_episode_counts(total_episodes: int, num_scenarios: int) -> list[int]:
    if total_episodes < num_scenarios:
        msg = (
            "Collection episodes must be >= number of scenarios so each selected scenario "
            "receives at least one rollout"
        )
        raise ValueError(msg)
    base = total_episodes // num_scenarios
    remainder = total_episodes % num_scenarios
    return [base + (1 if index < remainder else 0) for index in range(num_scenarios)]


def _default_env_factory(config: dict[str, Any]) -> Any:
    from forge_env.gymnasium_env import ForgeGymnasiumEnv

    return ForgeGymnasiumEnv(config=config)


def _create_policy_agent(
    policy_name: str,
    action_space_size: int,
    seed: int,
    *,
    teacher_config: Any = None,
    provider_factory: Callable[[Any], Any] | None = None,
) -> Any:
    from forge.agents.base_agent import AgentConfig

    if policy_name == "random":
        from forge.agents.random_agent import RandomAgent

        return RandomAgent(
            config=AgentConfig(name="mangomas-random"),
            action_space_size=action_space_size,
            seed=seed,
        )

    if policy_name == "mcts":
        from forge.agents.mcts_agent import MCTSAgent, MCTSConfig

        return MCTSAgent(
            config=MCTSConfig(name="mangomas-mcts"),
            action_space_size=action_space_size,
            seed=seed,
        )

    if policy_name == "llm":
        if teacher_config is None:
            msg = "policy='llm' requires a TeacherConfig"
            raise ValueError(msg)
        return _build_llm_agent(
            teacher_config,
            action_space_size,
            seed,
            provider_factory=provider_factory,
        )

    msg = f"Unsupported MangoMAS collection policy: {policy_name}"
    raise ValueError(msg)


def _default_provider_factory(teacher_config: Any) -> Any:
    """Build a CognitiveProvider from a ``TeacherConfig``."""
    from forge.cognitive.providers import create_provider

    kwargs: dict[str, Any] = {}
    if teacher_config.api_key:
        kwargs["api_key"] = teacher_config.api_key
    if teacher_config.base_url:
        kwargs["base_url"] = teacher_config.base_url
    if teacher_config.model:
        kwargs["model"] = teacher_config.model
    if teacher_config.timeout_secs:
        kwargs["timeout_secs"] = teacher_config.timeout_secs
    kwargs["max_retries"] = teacher_config.max_retries
    kwargs["retry_backoff_secs"] = teacher_config.retry_backoff_secs
    return create_provider(teacher_config.provider, **kwargs)


def _build_llm_agent(
    teacher_config: Any,
    action_space_size: int,
    seed: int,
    *,
    provider_factory: Callable[[Any], Any] | None = None,
) -> Any:
    """Instantiate a structured LLMAgent driven by ``teacher_config``."""
    from forge.cognitive.llm_agent import LLMAgent, StructuredLLMAgentConfig

    factory = provider_factory or _default_provider_factory
    provider = factory(teacher_config)
    response_schema_path = (
        teacher_config.response_schema_path if teacher_config.response_format_enabled else ""
    )
    structured_config = StructuredLLMAgentConfig(
        name="mangomas-llm",
        provider_name=teacher_config.provider,
        model=teacher_config.model,
        temperature=teacher_config.temperature,
        max_tokens=teacher_config.max_tokens,
        system_prompt=teacher_config.system_prompt,
        prompt_template_path=teacher_config.prompt_template_path,
        response_schema_path=response_schema_path,
        few_shot_examples_path=teacher_config.few_shot_examples_path,
        include_legal_actions=teacher_config.include_legal_actions,
        log_payloads=teacher_config.log_payloads,
        validate_action=teacher_config.validate_action,
        seed=teacher_config.seed if teacher_config.seed != 0 else None,
        top_p=teacher_config.top_p,
        timeout_secs=teacher_config.timeout_secs,
        payload_preview_chars=teacher_config.payload_preview_chars,
        legal_actions=tuple(range(action_space_size)),
    )
    logger.info(
        "_build_llm_agent provider=%s model=%s action_space_size=%d",
        teacher_config.provider,
        teacher_config.model,
        action_space_size,
    )
    _ = seed  # reserved for future seed-derived adapters; provider seeds via teacher_config.seed
    return LLMAgent(structured_config, provider=provider)


def _prepare_env_config(
    base_forge_config: Mapping[str, Any],
    scenario: ResolvedForgeScenario,
    mangomas_config: MangoMASBridgeConfig,
    seed: int,
) -> dict[str, Any]:
    config = _deep_merge(_copy_mapping(base_forge_config), scenario.forge_config)
    world = config.setdefault("world", {})
    agents = config.setdefault("agents", {})
    task = config.setdefault("task", {})
    drone = config.setdefault("drone", {})

    world["seed"] = seed
    agents["num_agents"] = 1
    task.setdefault("enabled", True)

    if mangomas_config.platform == "drone":
        drone["enabled"] = True
        drone["num_aerial"] = 1
        drone["num_ground_vehicles"] = 0
    else:
        drone.setdefault("enabled", False)

    return config


def _resolve_max_steps(env_config: Mapping[str, Any], mangomas_config: MangoMASBridgeConfig) -> int:
    configured_max_steps = int(mangomas_config.batch_collector.max_steps)
    scenario_max_steps = int(env_config.get("task", {}).get("max_episode_length", 0))
    if scenario_max_steps > 0:
        return min(configured_max_steps, scenario_max_steps)
    return configured_max_steps


def _determine_episode_success(total_reward: float, info: Mapping[str, Any] | Any) -> bool:
    success = bool(total_reward > 0.0)
    tasks_completed = info.get("tasks_completed") if isinstance(info, Mapping) else None
    if isinstance(tasks_completed, list):
        success = success or any(bool(task_list) for task_list in tasks_completed)
    return success


def _collect_episode_rollout(
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
    scenario_id: str = "",
    episode_index: int = 0,
    teacher_config: Any = None,
    trace_writer: Any = None,
) -> _EpisodeRollout:
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

    capture_teacher = teacher_config is not None
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
        action_id, trace_info = policy_agent.act(flat_obs)
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

        if capture_teacher and isinstance(trace_info, dict):
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

            if trace_writer is not None:
                from forge.mangomas.teacher_trace import TeacherDecisionTrace

                trace_writer.log(
                    TeacherDecisionTrace(
                        scenario_id=scenario_id,
                        episode_index=episode_index,
                        step_index=_step,
                        observation=_extract_raw_observation(enriched_obs),
                        legal_actions=list(range(int(env.action_space.n))),
                        action_id=discrete_action,
                        intention=teacher_intentions[-1] if teacher_intentions[-1] >= 0 else None,
                        subgoals=teacher_subgoals[-1],
                        rationale=teacher_rationales[-1],
                        value_hat=teacher_value_hats[-1],
                        constraint_critique=teacher_constraint_critiques[-1],
                        top_k_probs=teacher_top_k_probs[-1],
                        provider=str(trace_info.get("provider") or ""),
                        model=str(teacher_config.model),
                        prompt_tokens=int(trace_info.get("prompt_tokens") or 0),
                        completion_tokens=int(trace_info.get("completion_tokens") or 0),
                        latency_ms=float(trace_info.get("latency_ms") or 0.0),
                        schema_version=teacher_config.trace_schema_version,
                    )
                )

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
        teacher_intentions=teacher_intentions if capture_teacher else None,
        teacher_rationales=teacher_rationales if capture_teacher else None,
        teacher_subgoals=teacher_subgoals if capture_teacher else None,
        teacher_value_hats=teacher_value_hats if capture_teacher else None,
        teacher_constraint_critiques=teacher_constraint_critiques if capture_teacher else None,
        teacher_top_k_probs=teacher_top_k_probs if capture_teacher else None,
        teacher_prompt_tokens=teacher_prompt_tokens if capture_teacher else None,
        teacher_completion_tokens=teacher_completion_tokens if capture_teacher else None,
        teacher_latency_ms=teacher_latency_ms if capture_teacher else None,
        teacher_providers=teacher_providers if capture_teacher else None,
        action_space_size=int(env.action_space.n),
    )


def _collect_scenario_rollouts(
    *,
    env: Any,
    env_config: Mapping[str, Any],
    scenario: ResolvedForgeScenario,
    scenario_episodes: int,
    base_seed: int,
    observation_adapter: ObservationAdapter,
    policy_name: str,
    mangomas_config: MangoMASBridgeConfig,
    teacher_config: Any = None,
    provider_factory: Callable[[Any], Any] | None = None,
) -> tuple[list[_EpisodeRollout], list[float], list[bool]]:
    action_space_size = int(env.action_space.n)
    policy_agent = _create_policy_agent(
        policy_name,
        action_space_size,
        derive_seed(base_seed, f"policy:{scenario.scenario_id}"),
        teacher_config=teacher_config,
        provider_factory=provider_factory,
    )
    comm_vocab_size = int(env_config.get("agents", {}).get("comm_vocab_size", 0))
    drone_enabled = bool(env_config.get("drone", {}).get("enabled", False))
    agri_enabled = bool(env_config.get("agri", {}).get("enabled", False)) and drone_enabled
    hex_enabled = str(env_config.get("world", {}).get("grid_type", "")).lower() == "hex"
    max_steps = _resolve_max_steps(env_config, mangomas_config)

    scenario_rollouts: list[_EpisodeRollout] = []
    scenario_rewards: list[float] = []
    scenario_successes: list[bool] = []
    for episode_index in range(scenario_episodes):
        writer = _open_trace_writer_if_enabled(teacher_config, scenario.scenario_id, episode_index)
        try:
            rollout = _collect_episode_rollout(
                env=env,
                env_config=env_config,
                episode_seed=derive_seed(base_seed, f"{scenario.scenario_id}:{episode_index}"),
                max_steps=max_steps,
                observation_adapter=observation_adapter,
                policy_agent=policy_agent,
                platform=mangomas_config.platform,
                comm_vocab_size=comm_vocab_size,
                drone_enabled=drone_enabled,
                agri_enabled=agri_enabled,
                hex_enabled=hex_enabled,
                scenario_id=scenario.scenario_id,
                episode_index=episode_index,
                teacher_config=teacher_config,
                trace_writer=writer,
            )
        finally:
            if writer is not None:
                writer.close()
        scenario_rollouts.append(rollout)
        scenario_rewards.append(rollout.total_reward)
        scenario_successes.append(rollout.success)
    return scenario_rollouts, scenario_rewards, scenario_successes
