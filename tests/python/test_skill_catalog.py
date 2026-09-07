"""Tests for the shared hierarchical skill catalog and Python skill policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from forge.agents.base_agent import AgentConfig
from forge.agents.skills import (
    DEFAULT_SKILL_ID,
    DEFAULT_SKILLS_CONFIG_PATH,
    HierarchicalSkillPolicy,
    SkillCatalog,
    SkillSpec,
)
from forge.mangomas.collector.action_decoder import (
    ACTION_ID_NOOP,
    ACTION_ID_PICKUP,
    FORGE_AGRI_ACTION_COUNT,
    FORGE_BASE_ACTIONS,
    FORGE_DRONE_ACTION_COUNT,
    skill_category_for_action_id,
    skill_category_for_action_name,
)
from forge.mangomas.collector.sync_rollout import _create_policy_agent
from forge.policy_names import COLLECTION_POLICY_CHOICES, POLICY_SKILL

if TYPE_CHECKING:
    from pathlib import Path


def test_default_toml_catalog_loads_and_has_unique_ids() -> None:
    catalog = SkillCatalog.from_toml(DEFAULT_SKILLS_CONFIG_PATH)
    ids = [spec.id for spec in catalog.skills]
    assert catalog.default_skill == DEFAULT_SKILL_ID
    assert DEFAULT_SKILL_ID in ids
    assert len(ids) == len(set(ids))
    assert catalog.resolve_default() is not None
    resolved = catalog.resolve_default()
    assert resolved is not None
    assert catalog.effective_horizon(resolved) >= 1
    assert not catalog.enabled


def test_empty_skills_list_is_preserved() -> None:
    catalog = SkillCatalog.from_mapping({"enabled": False, "skills": []})
    assert catalog.skills == []


def test_unknown_skill_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unknown skill spec keys"):
        SkillCatalog.from_mapping(
            {
                "skills": [
                    {
                        "id": "explore",
                        "category": "explore",
                        "mystery": 1,
                    }
                ]
            }
        )


def test_missing_catalog_path_falls_back_to_builtins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "nope.toml"
    monkeypatch.setenv("FORGE_SKILLS_CONFIG_PATH", str(missing))
    catalog = SkillCatalog.from_default_path()
    assert catalog.get("explore") is not None
    assert catalog.get("idle") is not None


def test_zero_horizon_uses_catalog_default() -> None:
    catalog = SkillCatalog(default_horizon=12)
    spec = SkillSpec(id="x", category="idle", max_horizon=0)
    assert catalog.effective_horizon(spec) == 12


def test_skill_category_mapping_for_primitives() -> None:
    assert skill_category_for_action_name("Noop") == "idle"
    assert skill_category_for_action_name("MoveUp") == "navigate"
    assert skill_category_for_action_name("PickUp") == "gather"
    assert skill_category_for_action_name("Craft") == "craft"
    assert skill_category_for_action_name("Interact") == "combat"
    assert skill_category_for_action_name("TakeOff") == "aerial"
    assert (
        skill_category_for_action_id(
            ACTION_ID_NOOP, comm_vocab_size=0, drone_enabled=False
        )
        == "idle"
    )
    assert (
        skill_category_for_action_id(
            ACTION_ID_PICKUP, comm_vocab_size=0, drone_enabled=False
        )
        == "gather"
    )


def test_action_layout_constants_are_internally_consistent() -> None:
    assert FORGE_BASE_ACTIONS == 40
    assert FORGE_DRONE_ACTION_COUNT == 19
    assert FORGE_AGRI_ACTION_COUNT == 14


def test_hierarchical_skill_policy_is_deterministic() -> None:
    catalog = SkillCatalog.from_toml(DEFAULT_SKILLS_CONFIG_PATH)
    cfg = AgentConfig(name="skill-test")
    a = HierarchicalSkillPolicy(cfg, catalog, action_space_size=FORGE_BASE_ACTIONS, seed=7)
    b = HierarchicalSkillPolicy(
        AgentConfig(name="skill-test-b"),
        catalog,
        action_space_size=FORGE_BASE_ACTIONS,
        seed=7,
    )
    obs = np.zeros((4,), dtype=np.float32)
    for _ in range(24):
        act_a, trace_a = a.act(obs)
        act_b, trace_b = b.act(obs)
        assert act_a == act_b
        assert 0 <= act_a < FORGE_BASE_ACTIONS
        assert trace_a["skill_id"] == trace_b["skill_id"]


def test_hierarchical_skill_policy_empty_catalog_is_noop() -> None:
    catalog = SkillCatalog(skills=[])
    policy = HierarchicalSkillPolicy(
        AgentConfig(name="empty"), catalog, action_space_size=8, seed=1
    )
    action, trace = policy.act(np.zeros((1,), dtype=np.float32))
    assert action == ACTION_ID_NOOP
    assert trace["skill_id"] is None


def test_hierarchical_skill_policy_skips_drone_skills_when_disabled() -> None:
    catalog = SkillCatalog(
        default_skill="aerial",
        skills=[
            SkillSpec(id="aerial", category="aerial", requires_drone=True, max_horizon=4)
        ],
    )
    policy = HierarchicalSkillPolicy(
        AgentConfig(name="ground"),
        catalog,
        action_space_size=FORGE_BASE_ACTIONS,
        drone_enabled=False,
        seed=1,
    )
    action, trace = policy.act(np.zeros((1,), dtype=np.float32))
    assert action == ACTION_ID_NOOP
    assert trace["skill_id"] is None


def test_create_policy_agent_skill_wiring() -> None:
    assert POLICY_SKILL in COLLECTION_POLICY_CHOICES
    agent = _create_policy_agent(POLICY_SKILL, FORGE_BASE_ACTIONS, 3, drone_enabled=False)
    assert isinstance(agent, HierarchicalSkillPolicy)
    action, trace = agent.act(np.zeros((2,), dtype=np.float32))
    assert 0 <= action < FORGE_BASE_ACTIONS
    assert "skill_id" in trace


def test_hierarchical_skill_policy_skips_agri_when_disabled() -> None:
    catalog = SkillCatalog(
        default_skill="spray",
        skills=[SkillSpec(id="spray", category="agriculture", max_horizon=3)],
    )
    policy = HierarchicalSkillPolicy(
        AgentConfig(name="no-agri"),
        catalog,
        action_space_size=FORGE_BASE_ACTIONS + FORGE_DRONE_ACTION_COUNT + FORGE_AGRI_ACTION_COUNT,
        drone_enabled=True,
        agri_enabled=False,
        seed=42,
    )
    action, trace = policy.act(np.zeros((2,), dtype=np.float32))
    assert action == ACTION_ID_NOOP
    assert trace["skill_id"] is None


def test_hierarchical_skill_policy_clips_illegal_action() -> None:
    catalog = SkillCatalog(
        default_skill="combat",
        skills=[SkillSpec(id="combat", category="combat", max_horizon=2)],
    )
    # Restrict action space so ACTION_ID_INTERACT is out of bounds
    policy = HierarchicalSkillPolicy(
        AgentConfig(name="tiny-space"),
        catalog,
        action_space_size=2,
        seed=1,
    )
    action, trace = policy.act(np.zeros((1,), dtype=np.float32))
    assert action == ACTION_ID_NOOP
    assert trace["skill_id"] == "combat"


def test_hierarchical_skill_policy_step_count_and_state_reset() -> None:
    catalog = SkillCatalog(
        default_skill="idle",
        skills=[SkillSpec(id="idle", category="idle", max_horizon=2)],
    )
    policy = HierarchicalSkillPolicy(
        AgentConfig(name="tracker"),
        catalog,
        action_space_size=FORGE_BASE_ACTIONS,
        seed=1,
    )
    assert policy.step_count == 0
    a1, trace1 = policy.act(np.zeros((1,), dtype=np.float32))
    assert a1 == ACTION_ID_NOOP
    assert policy.step_count == 1
    assert trace1["skill_step"] == 1
    a2, trace2 = policy.act(np.zeros((1,), dtype=np.float32))
    assert a2 == ACTION_ID_NOOP
    assert policy.step_count == 2
    assert trace2["skill_step"] == 2
    # Skill completed (horizon 2 reached), next act should cycle
    _, trace3 = policy.act(np.zeros((1,), dtype=np.float32))
    assert policy.step_count == 3
    assert trace3["skill_step"] == 1

