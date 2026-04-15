"""FORGE scenario collection for MangoMAS training pipelines."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from forge.mangomas.adapters import ObservationAdapter
from forge.mangomas.pipeline import CollectedTrainingData
from forge.utils.observation import flatten_obs
from forge.utils.seed import derive_seed

if TYPE_CHECKING:
    from forge.mangomas.config import MangoMASBridgeConfig

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO_DIR = Path("configs/scenarios")
DEFAULT_EVAL_REGISTRY_DIR = Path("configs/eval_registry")
_FORGE_BASE_ACTIONS = 40
_FORGE_DRONE_ACTION_COUNT = 19
_FORGE_AGRI_ACTION_COUNT = 14
_FORGE_HEX_ACTION_COUNT = 6


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


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return {str(key): _deep_copy(item) for key, item in value.items()}


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = _copy_mapping(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _deep_copy(value)
    return merged


def _normalize_identifier(value: str) -> str:
    normalized: list[str] = []
    previous_was_separator = False
    for char in value:
        if char.isascii() and char.isalnum():
            normalized.append(char.lower())
            previous_was_separator = False
        elif normalized and not previous_was_separator:
            normalized.append("_")
            previous_was_separator = True
    while normalized and normalized[-1] == "_":
        normalized.pop()
    return "".join(normalized)


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file_handle:
        loaded = tomllib.load(file_handle)
    if not isinstance(loaded, Mapping):
        msg = f"Scenario file must deserialize to a table: {path}"
        raise ValueError(msg)
    return _copy_mapping(loaded)


def _resolve_path(identifier: str | Path, search_dirs: Sequence[Path]) -> Path:
    raw_path = Path(identifier)
    if raw_path.is_file():
        return raw_path

    if raw_path.suffix == ".toml":
        candidate = Path.cwd() / raw_path
        if candidate.is_file():
            return candidate

    stem = raw_path.stem if raw_path.suffix else str(identifier)
    for directory in search_dirs:
        candidate = directory / f"{stem}.toml"
        if candidate.is_file():
            return candidate

    msg = f"Could not resolve FORGE scenario '{identifier}'"
    raise FileNotFoundError(msg)


def _resolve_eval_manifest(path: Path, data: Mapping[str, Any]) -> ResolvedForgeScenario:
    scenario = _copy_mapping(
        data.get("scenario") if isinstance(data.get("scenario"), Mapping) else None
    )
    forge_config = _copy_mapping(
        data.get("forge") if isinstance(data.get("forge"), Mapping) else None
    )
    scenario_id = str(
        scenario.get("id") or _normalize_identifier(str(scenario.get("name") or path.stem))
    )
    return ResolvedForgeScenario(
        scenario_id=scenario_id,
        name=str(scenario.get("name") or scenario_id),
        source_path=path,
        difficulty_tier=int(scenario.get("difficulty_tier", 1)),
        min_agents=int(
            scenario.get("min_agents", forge_config.get("agents", {}).get("num_agents", 1))
        ),
        max_agents=int(
            scenario.get("max_agents", forge_config.get("agents", {}).get("num_agents", 1))
        ),
        forge_config=forge_config,
    )


def _resolve_high_level_manifest(path: Path, data: Mapping[str, Any]) -> ResolvedForgeScenario:
    scenario = data.get("scenario")
    if not isinstance(scenario, Mapping):
        msg = f"High-level scenario is missing [scenario]: {path}"
        raise ValueError(msg)

    map_data = _copy_mapping(
        scenario.get("map") if isinstance(scenario.get("map"), Mapping) else None
    )
    objectives = _copy_mapping(
        scenario.get("objectives") if isinstance(scenario.get("objectives"), Mapping) else None
    )
    difficulty = _copy_mapping(
        scenario.get("difficulty") if isinstance(scenario.get("difficulty"), Mapping) else None
    )

    scenario_name = str(scenario.get("name") or path.stem)
    scenario_id = _normalize_identifier(scenario_name)
    if not scenario_id:
        msg = f"Scenario name must resolve to a non-empty id: {path}"
        raise ValueError(msg)

    forge_config: dict[str, Any] = {}
    grid_size = map_data.get("grid_size")
    if grid_size is not None:
        forge_config.setdefault("world", {})["width"] = int(grid_size)
        forge_config.setdefault("world", {})["height"] = int(grid_size)

    time_limit = objectives.get("time_limit")
    if time_limit is not None:
        forge_config.setdefault("task", {})["max_episode_length"] = int(time_limit)

    difficulty_tier = int(difficulty.get("base_tier", 1))
    forge_config.setdefault("task", {})["max_tier"] = difficulty_tier
    forge_config.setdefault("agents", {})["num_agents"] = int(scenario.get("min_agents", 1))

    return ResolvedForgeScenario(
        scenario_id=scenario_id,
        name=scenario_name,
        source_path=path,
        difficulty_tier=difficulty_tier,
        min_agents=int(scenario.get("min_agents", 1)),
        max_agents=int(scenario.get("max_agents", scenario.get("min_agents", 1))),
        forge_config=forge_config,
    )


def resolve_forge_scenarios(
    scenario_refs: Sequence[str | Path],
    search_dirs: Sequence[str | Path] | None = None,
) -> list[ResolvedForgeScenario]:
    """Resolve scenario ids or paths into executable FORGE scenario configs."""
    resolved_dirs = [
        Path(directory)
        for directory in (search_dirs or (DEFAULT_SCENARIO_DIR, DEFAULT_EVAL_REGISTRY_DIR))
    ]
    resolved: list[ResolvedForgeScenario] = []
    for scenario_ref in scenario_refs:
        path = _resolve_path(scenario_ref, resolved_dirs)
        data = _load_toml(path)
        if isinstance(data.get("forge"), Mapping):
            resolved.append(_resolve_eval_manifest(path, data))
        else:
            resolved.append(_resolve_high_level_manifest(path, data))
    return resolved


def write_collection_report(
    result: ScenarioCollectionResult,
    output_path: str | Path,
    *,
    mode: str,
    platform: str,
    policy_name: str,
    base_seed: int,
    scenario_refs: Sequence[str | Path],
    config_paths: Sequence[str | Path],
    run_name: str,
) -> Path:
    """Write a JSON report summarizing a MangoMAS collection run."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_report_dict(
        mode=mode,
        platform=platform,
        policy_name=policy_name,
        base_seed=base_seed,
        scenario_refs=scenario_refs,
        config_paths=config_paths,
        run_name=run_name,
    )
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return report_path


def _decode_non_drone_action_name(action_id: int, comm_vocab_size: int) -> str | None:
    action_label: str | None = None
    if action_id == 0:
        action_label = "Noop"
    elif 1 <= action_id <= 4:
        action_label = ("MoveUp", "MoveDown", "MoveLeft", "MoveRight")[action_id - 1]
    elif action_id == 5:
        action_label = "PickUp"
    elif 6 <= action_id <= 15:
        action_label = "Drop"
    elif 16 <= action_id <= 25:
        action_label = "Use"
    elif 26 <= action_id <= 34:
        action_label = "Craft"
    elif 35 <= action_id <= 38:
        action_label = "Push"
    elif action_id == 39:
        action_label = "Interact"
    elif _FORGE_BASE_ACTIONS <= action_id < _FORGE_BASE_ACTIONS + comm_vocab_size:
        action_label = "Communicate"
    return action_label


def _decode_drone_action_name(drone_offset: int) -> str | None:
    if drone_offset <= 4:
        return ("Ascend", "Descend", "Hover", "TakeOff", "Land")[drone_offset]
    if 5 <= drone_offset <= 8:
        return "Scan"
    if 9 <= drone_offset <= 18:
        return "DropPayload"
    return None


def _decode_agri_action_name(agri_offset: int) -> str | None:
    if 0 <= agri_offset <= 9:
        return "Spray"
    if agri_offset == 10:
        return "ScanMultispectral"
    if agri_offset == 11:
        return "ScanThermal"
    if agri_offset == 12:
        return "RelaySoilData"
    if agri_offset == 13:
        return "GenerateReport"
    return None


def _decode_hex_action_name(hex_offset: int) -> str | None:
    if 0 <= hex_offset < _FORGE_HEX_ACTION_COUNT:
        return "Move"
    return None


def decode_action_name(
    action_id: int,
    comm_vocab_size: int,
    drone_enabled: bool,
    agri_enabled: bool = False,
    hex_enabled: bool = False,
) -> str:
    """Decode a FORGE discrete action id into a stable semantic label."""
    non_drone_label = _decode_non_drone_action_name(action_id, comm_vocab_size)
    if non_drone_label is not None:
        return non_drone_label

    offset = action_id - _FORGE_BASE_ACTIONS - comm_vocab_size
    if offset < 0:
        return "UnknownAction"

    if drone_enabled:
        drone_label = _decode_drone_action_name(offset)
        if drone_label is not None:
            return drone_label
        offset -= _FORGE_DRONE_ACTION_COUNT

    if agri_enabled and drone_enabled:
        agri_label = _decode_agri_action_name(offset)
        if agri_label is not None:
            return agri_label
        offset -= _FORGE_AGRI_ACTION_COUNT

    if hex_enabled:
        hex_label = _decode_hex_action_name(offset)
        if hex_label is not None:
            return hex_label

    return "UnknownAction"


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


def _create_policy_agent(policy_name: str, action_space_size: int, seed: int) -> Any:
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

    msg = f"Unsupported MangoMAS collection policy: {policy_name}"
    raise ValueError(msg)


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

    for _step in range(max_steps):
        episode_raw_observations.append(_extract_raw_observation(enriched_obs))
        flat_obs = flatten_obs(enriched_obs)
        action_id, _trace = policy_agent.act(flat_obs)
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
) -> tuple[list[_EpisodeRollout], list[float], list[bool]]:
    action_space_size = int(env.action_space.n)
    policy_agent = _create_policy_agent(
        policy_name,
        action_space_size,
        derive_seed(base_seed, f"policy:{scenario.scenario_id}"),
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
        )
        scenario_rollouts.append(rollout)
        scenario_rewards.append(rollout.total_reward)
        scenario_successes.append(rollout.success)
    return scenario_rollouts, scenario_rewards, scenario_successes


def collect_training_data_from_scenarios(
    *,
    base_forge_config: Mapping[str, Any],
    mangomas_config: MangoMASBridgeConfig,
    scenario_refs: Sequence[str | Path],
    total_episodes: int,
    base_seed: int,
    policy_name: str = "random",
    search_dirs: Sequence[str | Path] | None = None,
    env_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> ScenarioCollectionResult:
    """Collect MangoMAS training data from resolved FORGE scenario rollouts."""
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
    curriculum_outcomes: list[bool] = []
    scenario_summaries: list[ScenarioRolloutSummary] = []

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
            )
            for rollout in scenario_rollouts:
                observations.append(rollout.observations)
                action_names.append(rollout.action_names)
                action_ids.append(rollout.action_ids)
                rewards.append(rollout.rewards)
                dones.append(rollout.dones)
                raw_observations.append(rollout.raw_observations)
                curriculum_outcomes.append(rollout.success)

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
        ),
        curriculum_outcomes=curriculum_outcomes,
        scenario_summaries=scenario_summaries,
        resolved_scenarios=scenarios,
    )
