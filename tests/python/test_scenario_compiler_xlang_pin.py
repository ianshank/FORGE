"""Rust ↔ Python high-level scenario compiler pin.

Paired with ``crates/forge-types/src/scenario.rs`` tests
``xlang_orchard_coverage_pinned_to_known_good`` and
``xlang_crop_scout_pinned_to_known_good``. Drift on either side fails CI.

This is a structural pin (tasks + charger/geofence/world), not a full
``ForgeConfig`` dump: Python fills known keys while Rust overlays
``ForgeConfig::default()``.
"""

from __future__ import annotations

from pathlib import Path

from forge.actions import FORGE_DEFAULT_COMM_VOCAB_SIZE
from forge.mangomas.collector.scenario import resolve_forge_scenarios

_REPO = Path(__file__).resolve().parents[2]
_SCENARIOS = _REPO / "configs" / "scenarios"


def _atom_kind(goal: object) -> str | None:
    if not isinstance(goal, dict):
        return None
    atom = goal.get("Atom")
    if isinstance(atom, dict) and len(atom) == 1:
        return next(iter(atom))
    return None


def _atom_payload(goal: object, kind: str) -> object:
    assert isinstance(goal, dict)
    atom = goal["Atom"]
    assert isinstance(atom, dict)
    return atom[kind]


def test_python_compiler_defaults_match_rust_constants() -> None:
    """Mirrors crates/forge-types/src/constants.rs survey/battery/comm defaults."""
    assert FORGE_DEFAULT_COMM_VOCAB_SIZE == 16


def test_xlang_orchard_coverage_pinned_to_known_good() -> None:
    resolved = resolve_forge_scenarios([_SCENARIOS / "orchard_coverage.toml"])
    assert len(resolved) == 1
    scenario = resolved[0]
    assert scenario.scenario_id == "orchard_coverage"
    cfg = scenario.forge_config
    world = cfg["world"]
    drone = cfg["drone"]
    agri = cfg["agri"]
    task = cfg["task"]
    assert world["width"] == 16
    assert world["height"] == 16
    assert world["geofence_enabled"] is True
    assert world["geofence_margin"] == 1
    assert drone["enabled"] is True
    assert drone["num_aerial"] == 1
    assert drone["restrict_recharge_to_chargers"] is True
    assert drone["spawn_home"] == {"x": 0, "y": 0}
    assert drone["charger_tiles"] == [{"x": 0, "y": 0}]
    assert agri["enabled"] is True
    assert task["max_episode_length"] == 800
    goals = [item["goal"] for item in task["scenario_tasks"]]
    assert len(goals) == 1
    and_parts = goals[0]["And"]
    assert [_atom_kind(part) for part in and_parts] == [
        "FieldSurveyed",
        "BatteryAbove",
        "AgentAt",
    ]
    assert abs(float(_atom_payload(and_parts[0], "FieldSurveyed")) - 0.8) < 1e-6
    battery = _atom_payload(and_parts[1], "BatteryAbove")
    assert battery[0] == 0
    assert abs(float(battery[1]) - 0.2) < 1e-6
    agent_at = _atom_payload(and_parts[2], "AgentAt")
    assert agent_at[0] == 0
    assert agent_at[1] == {"x": 0, "y": 0}


def test_xlang_crop_scout_pinned_to_known_good() -> None:
    resolved = resolve_forge_scenarios([_SCENARIOS / "crop_scout.toml"])
    assert len(resolved) == 1
    scenario = resolved[0]
    assert scenario.scenario_id == "crop_scout"
    cfg = scenario.forge_config
    world = cfg["world"]
    assert world["width"] == 48
    assert world["height"] == 48
    assert world.get("geofence_enabled", False) is False
    assert cfg["drone"]["enabled"] is True
    assert cfg["drone"]["num_aerial"] == 1
    assert cfg["agri"]["enabled"] is True
    assert cfg["task"]["max_episode_length"] == 1500
    goal = cfg["task"]["scenario_tasks"][0]["goal"]
    assert _atom_kind(goal) == "FieldSurveyed"
    assert abs(float(_atom_payload(goal, "FieldSurveyed")) - 0.8) < 1e-6
