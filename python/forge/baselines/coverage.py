"""Lawnmower and random coverage baselines for energy-aware orchard scenarios.

Action ids match ``Action::try_to_discrete_configured`` (base, comm, drone,
agri). Training remains on PyO3; this module is a graded heuristic, not a
learned policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from forge.actions import (
    FORGE_BASE_ACTIONS,
    FORGE_DEFAULT_COMM_VOCAB_SIZE,
    FORGE_DRONE_ACTION_COUNT,
)
from forge.mangomas.collector.scenario import resolve_forge_scenarios


def drone_action_ids(comm_vocab_size: int = FORGE_DEFAULT_COMM_VOCAB_SIZE) -> dict[str, int]:
    """Return discrete ids for lawnmower primitives under drone+agri layout."""
    drone_base = FORGE_BASE_ACTIONS + comm_vocab_size
    agri_base = drone_base + FORGE_DRONE_ACTION_COUNT
    return {
        "noop": 0,
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "takeoff": drone_base + 3,
        "land": drone_base + 4,
        "scan_multispectral": agri_base + 10,
    }


def boustrophedon_tiles(
    width: int,
    height: int,
    margin: int = 0,
) -> list[tuple[int, int]]:
    """Row-major snake covering the interior after ``margin``."""
    x0, y0 = margin, margin
    x1, y1 = width - margin, height - margin
    if x0 >= x1 or y0 >= y1:
        return []
    tiles: list[tuple[int, int]] = []
    even = True
    for y in range(y0, y1):
        xs = range(x0, x1) if even else range(x1 - 1, x0 - 1, -1)
        tiles.extend((x, y) for x in xs)
        even = not even
    return tiles


def manhattan_moves(
    start: tuple[int, int],
    goal: tuple[int, int],
    ids: Mapping[str, int],
) -> list[int]:
    """Cardinal steps from ``start`` to ``goal``."""
    x, y = start
    gx, gy = goal
    moves: list[int] = []
    while x < gx:
        moves.append(ids["right"])
        x += 1
    while x > gx:
        moves.append(ids["left"])
        x -= 1
    while y < gy:
        moves.append(ids["down"])
        y += 1
    while y > gy:
        moves.append(ids["up"])
        y -= 1
    return moves


def lawnmower_action_ids(
    width: int,
    height: int,
    home: tuple[int, int],
    comm_vocab_size: int = FORGE_DEFAULT_COMM_VOCAB_SIZE,
    margin: int = 0,
) -> list[int]:
    """Take off, scan every interior tile, return home, land."""
    ids = drone_action_ids(comm_vocab_size)
    actions = [ids["takeoff"]]
    cursor = home
    for tile in boustrophedon_tiles(width, height, margin):
        actions.extend(manhattan_moves(cursor, tile, ids))
        actions.append(ids["scan_multispectral"])
        cursor = tile
    actions.extend(manhattan_moves(cursor, home, ids))
    actions.append(ids["land"])
    return actions


def orchard_env_config(
    scenario_path: str = "configs/scenarios/orchard_coverage.toml",
) -> dict[str, Any]:
    """Compile the high-level orchard scenario into a ForgeConfig dict."""
    resolved = resolve_forge_scenarios([scenario_path])
    if not resolved:
        msg = f"failed to resolve orchard scenario: {scenario_path}"
        raise ValueError(msg)
    return dict(resolved[0].forge_config)


def home_from_config(forge_config: Mapping[str, Any]) -> tuple[int, int]:
    """Read spawn_home / first charger / origin from a compiled config."""
    drone: Mapping[str, Any]
    raw_drone = forge_config.get("drone") or {}
    drone = raw_drone if isinstance(raw_drone, Mapping) else {}
    spawn = drone.get("spawn_home") or {}
    if isinstance(spawn, Mapping) and "x" in spawn and "y" in spawn:
        return int(spawn["x"]), int(spawn["y"])
    tiles: Sequence[Any]
    raw_tiles = drone.get("charger_tiles") or []
    tiles = raw_tiles if isinstance(raw_tiles, Sequence) else []
    if tiles and isinstance(tiles[0], Mapping):
        tile = tiles[0]
        return int(tile["x"]), int(tile["y"])
    return 0, 0
