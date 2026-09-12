"""FORGE scenario resolution and TOML parsing for MangoMAS."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py39/py310
    import tomli as tomllib

from forge.mangomas.collector.types import ResolvedForgeScenario

DEFAULT_SCENARIO_DIR = Path("configs/scenarios")
DEFAULT_EVAL_REGISTRY_DIR = Path("configs/eval_registry")


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


# Known ForgeConfig drone/agri keys. Extra high-level fields (waypoints,
# initial_disease_tiles, …) must not be copied — Rust ForgeConfig denies
# unknown fields when the dict is deserialized in PyO3.
_DRONE_KEYS: tuple[str, ...] = (
    "enabled",
    "num_aerial",
    "num_ground_vehicles",
    "max_altitude",
    "starting_battery",
    "max_battery",
    "aerial_drain_rate",
    "recharge_rate",
    "scan_range",
    "scan_cost",
    "hover_cost",
    "ascend_cost",
    "descend_cost",
    "altitude_vision_bonus",
    "vehicle_turn_radius",
    "fall_damage_per_level",
)
_AGRI_KEYS: tuple[str, ...] = (
    "enabled",
    "ndvi_scan_radius",
    "thermal_scan_radius",
    "scan_battery_cost",
    "spray_radius",
    "spray_efficacy",
    "spray_battery_cost",
    "disease_spread_rate",
    "num_soil_nodes",
    "soil_relay_range",
    "soil_reading_interval",
    "report_generation_cost",
    "report_scan_radius",
    "moisture_drain_rate",
    "cropland_density",
    "pasture_density",
)
# Mirrors crates/forge-types/src/constants.rs
_DEFAULT_SURVEY_THRESHOLD = 0.8
_DEFAULT_SPRAY_THRESHOLD = 0.8
_DEFAULT_SOIL_COLLECT_COUNT = 1
_DEFAULT_SCENARIO_NUM_AERIAL = 1
_DEFAULT_REWARD_SCALE = 1.0
_DEFAULT_MAX_EPISODE_LENGTH = 10_000


def _copy_known(src: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in src and src[key] is not None:
            out[key] = _deep_copy(src[key])
    return out


def _atom(predicate: dict[str, Any]) -> dict[str, Any]:
    return {"Atom": predicate}


def _task_definition(
    task_id: int,
    description: str,
    goal: dict[str, Any],
    tier: int,
    estimated_steps: int,
    reward: float,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "description": description,
        "goal": goal,
        "tier": tier,
        "estimated_steps": estimated_steps,
        "reward": reward,
        "dense_reward_weights": [1.0],
    }


def _map_sequence_step(
    step: str,
    objectives: Mapping[str, Any],
    soil_nodes: int,
) -> dict[str, Any] | None:
    survey = float(objectives.get("survey_threshold", _DEFAULT_SURVEY_THRESHOLD))
    spray = float(objectives.get("spray_threshold", _DEFAULT_SPRAY_THRESHOLD))
    collect = int(
        objectives.get("collect_threshold", max(soil_nodes, _DEFAULT_SOIL_COLLECT_COUNT))
    )
    if step == "survey":
        return _atom({"FieldSurveyed": survey})
    if step == "spray":
        return _atom({"AreaSprayed": spray})
    if step in {"relay_soil", "collect"}:
        return _atom({"SoilDataCollected": [0, collect]})
    if step in {"generate_report", "report"}:
        return _atom({"FieldReportGenerated": 0})
    return None


def _compile_objectives(
    objectives: Mapping[str, Any],
    tier: int,
    soil_nodes: int,
) -> list[dict[str, Any]]:
    kind = str(objectives.get("type") or "")
    reward = float(objectives.get("completion_bonus", _DEFAULT_REWARD_SCALE))
    estimated = int(objectives.get("time_limit", _DEFAULT_MAX_EPISODE_LENGTH))
    goal: dict[str, Any] | None = None
    if kind == "survey":
        threshold = float(objectives.get("survey_threshold", _DEFAULT_SURVEY_THRESHOLD))
        goal = _atom({"FieldSurveyed": threshold})
    elif kind == "collect":
        count = int(
            objectives.get("collect_threshold", max(soil_nodes, _DEFAULT_SOIL_COLLECT_COUNT))
        )
        goal = _atom({"SoilDataCollected": [0, count]})
    elif kind == "sequence":
        steps_raw = objectives.get("steps")
        steps: list[str] = list(steps_raw) if isinstance(steps_raw, list) else []
        mapped = [
            item
            for step in steps
            if isinstance(step, str)
            for item in [_map_sequence_step(step, objectives, soil_nodes)]
            if item is not None
        ]
        if mapped:
            goal = {"Sequence": mapped}
    if goal is None:
        return []
    return [_task_definition(1, f"{kind} objective", goal, tier, estimated, reward)]


def _goal_needs_agri(goal: Any) -> bool:
    if not isinstance(goal, Mapping):
        return False
    if "Atom" in goal:
        atom = goal["Atom"]
        return isinstance(atom, Mapping) and any(
            key in atom
            for key in (
                "FieldSurveyed",
                "AreaSprayed",
                "SoilDataCollected",
                "FieldReportGenerated",
                "CropHealthBelow",
                "DiseaseDetected",
                "IrrigationMapped",
            )
        )
    for key in ("And", "Or", "Sequence"):
        children = goal.get(key)
        if isinstance(children, list) and any(_goal_needs_agri(child) for child in children):
            return True
    for key in ("Before", "While", "Without"):
        inner = goal.get(key)
        if isinstance(inner, list) and inner and _goal_needs_agri(inner[0]):
            return True
        if isinstance(inner, Mapping) and _goal_needs_agri(inner):
            return True
    return False


def _compile_high_level_forge_config(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Compile a high-level ``[scenario]`` table into a Rust ForgeConfig dict."""
    map_data = _copy_mapping(
        scenario.get("map") if isinstance(scenario.get("map"), Mapping) else None
    )
    objectives = _copy_mapping(
        scenario.get("objectives") if isinstance(scenario.get("objectives"), Mapping) else None
    )
    difficulty = _copy_mapping(
        scenario.get("difficulty") if isinstance(scenario.get("difficulty"), Mapping) else None
    )
    drone_src = _copy_mapping(
        scenario.get("drone") if isinstance(scenario.get("drone"), Mapping) else None
    )
    agri_src = _copy_mapping(
        scenario.get("agri") if isinstance(scenario.get("agri"), Mapping) else None
    )

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

    drone = _copy_known(drone_src, _DRONE_KEYS)
    agri = _copy_known(agri_src, _AGRI_KEYS)
    if map_data.get("cropland_density") is not None and "cropland_density" not in agri:
        agri["cropland_density"] = float(map_data["cropland_density"])
    if map_data.get("pasture_density") is not None and "pasture_density" not in agri:
        agri["pasture_density"] = float(map_data["pasture_density"])
    if difficulty.get("disease_spread_rate") is not None:
        agri["disease_spread_rate"] = int(difficulty["disease_spread_rate"])

    soil_nodes = int(agri.get("num_soil_nodes", 0))
    scenario_tasks = _compile_objectives(objectives, difficulty_tier, soil_nodes)
    if any(_goal_needs_agri(task.get("goal")) for task in scenario_tasks):
        agri["enabled"] = True
        drone["enabled"] = True
    if drone.get("enabled") and int(drone.get("num_aerial", 0)) == 0 and int(
        drone.get("num_ground_vehicles", 0)
    ) == 0:
        drone["num_aerial"] = _DEFAULT_SCENARIO_NUM_AERIAL

    if drone:
        forge_config["drone"] = drone
    if agri:
        forge_config["agri"] = agri
    forge_config.setdefault("task", {})["scenario_tasks"] = scenario_tasks
    forge_config.setdefault("task", {})["enabled"] = True
    return forge_config


def _resolve_high_level_manifest(path: Path, data: Mapping[str, Any]) -> ResolvedForgeScenario:
    scenario = data.get("scenario")
    if not isinstance(scenario, Mapping):
        msg = f"High-level scenario is missing [scenario]: {path}"
        raise ValueError(msg)

    difficulty = _copy_mapping(
        scenario.get("difficulty") if isinstance(scenario.get("difficulty"), Mapping) else None
    )

    scenario_name = str(scenario.get("name") or path.stem)
    scenario_id = _normalize_identifier(scenario_name)
    if not scenario_id:
        msg = f"Scenario name must resolve to a non-empty id: {path}"
        raise ValueError(msg)

    difficulty_tier = int(difficulty.get("base_tier", 1))
    forge_config = _compile_high_level_forge_config(scenario)

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
