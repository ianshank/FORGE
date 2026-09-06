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
