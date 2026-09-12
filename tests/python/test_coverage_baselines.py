"""Unit tests for orchard coverage lawnmower helpers."""

from __future__ import annotations

from pathlib import Path

from forge.baselines.coverage import (
    boustrophedon_tiles,
    home_from_config,
    lawnmower_action_ids,
    orchard_env_config,
)
from forge.mangomas.collector.scenario import resolve_forge_scenarios


def test_boustrophedon_covers_interior_once() -> None:
    tiles = boustrophedon_tiles(4, 3, margin=0)
    assert len(tiles) == 12
    assert len(set(tiles)) == 12
    assert tiles[0] == (0, 0)


def test_lawnmower_plan_is_deterministic() -> None:
    a = lawnmower_action_ids(4, 4, (0, 0), margin=1)
    b = lawnmower_action_ids(4, 4, (0, 0), margin=1)
    assert a == b
    assert a[0] != a[-1]
    ids = {a[0], a[-1]}
    assert len(ids) == 2


def test_orchard_scenario_compiles_charger_and_and_goal() -> None:
    cfg = orchard_env_config()
    assert cfg["drone"]["restrict_recharge_to_chargers"] is True
    assert home_from_config(cfg) == (0, 0)
    resolved = resolve_forge_scenarios([Path("configs/scenarios/orchard_coverage.toml")])
    assert resolved[0].scenario_id == "orchard_coverage"
