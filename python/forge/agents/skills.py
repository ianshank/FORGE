"""Configuration-driven hierarchical skills over primitive FORGE actions.

Mirrors `forge_types::skill::SkillsConfig` and `forge_agent::skills::HierarchicalSkillAgent`
so Python trainers, collectors, and the Rust engine share one catalog
(`configs/agents/skills_default.toml`) without inlined magic numbers.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py39/py310
    import tomli as tomllib

import numpy as np

from forge.actions import (
    ACTION_ID_CRAFT_MIN,
    ACTION_ID_INTERACT,
    ACTION_ID_MOVE_MIN,
    ACTION_ID_NOOP,
    ACTION_ID_PICKUP,
    ACTION_MOVE_FAMILY_SIZE,
    AGRI_OFFSET_MULTISPECTRAL,
    DRONE_OFFSET_HOVER,
    DRONE_OFFSET_LAND,
    DRONE_OFFSET_TAKEOFF,
    FORGE_BASE_ACTIONS,
    FORGE_DRONE_ACTION_COUNT,
    skill_category_for_action_id,
)
from forge.agents.base_agent import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)

SKILLS_CONFIG_PATH_ENV: Final[str] = "FORGE_SKILLS_CONFIG_PATH"
DEFAULT_SKILLS_CONFIG_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "configs" / "agents" / "skills_default.toml"
)
DEFAULT_SKILL_ID: Final[str] = "explore"
DEFAULT_SKILL_HORIZON: Final[int] = 32
DEFAULT_SKILL_IDLE_HORIZON: Final[int] = 1
DEFAULT_SKILL_NAVIGATE_HORIZON: Final[int] = 64
DEFAULT_SKILL_GATHER_HORIZON: Final[int] = 32
DEFAULT_SKILL_EXPLORE_HORIZON: Final[int] = 16
DEFAULT_SKILL_CRAFT_HORIZON: Final[int] = 8
DEFAULT_SKILL_COMBAT_HORIZON: Final[int] = 8
DEFAULT_SKILL_AERIAL_HORIZON: Final[int] = 16
DEFAULT_SKILL_AGRI_HORIZON: Final[int] = 12
DEFAULT_SKILL_COMMUNICATE_HORIZON: Final[int] = 4
MIN_SKILL_HORIZON: Final[int] = 1
DEFAULT_SKILL_POLICY_SEED: Final[int] = 0


@dataclass
class SkillSpec:
    """One reusable skill option loaded from TOML."""

    id: str
    category: str
    enabled: bool = True
    max_horizon: int = DEFAULT_SKILL_HORIZON
    requires_drone: bool = False
    recipe_index: int = 0
    comm_token: int = 0
    target_x: int | None = None
    target_y: int | None = None


_SKILL_SPEC_FIELDS: Final[frozenset[str]] = frozenset(item.name for item in fields(SkillSpec))


def _default_skills() -> list[SkillSpec]:
    return [
        SkillSpec(id="idle", category="idle", max_horizon=DEFAULT_SKILL_IDLE_HORIZON),
        SkillSpec(id="navigate", category="navigate", max_horizon=DEFAULT_SKILL_NAVIGATE_HORIZON),
        SkillSpec(id="gather", category="gather", max_horizon=DEFAULT_SKILL_GATHER_HORIZON),
        SkillSpec(id=DEFAULT_SKILL_ID, category="explore", max_horizon=DEFAULT_SKILL_EXPLORE_HORIZON),
        SkillSpec(id="craft", category="craft", max_horizon=DEFAULT_SKILL_CRAFT_HORIZON),
        SkillSpec(id="combat", category="combat", max_horizon=DEFAULT_SKILL_COMBAT_HORIZON),
        SkillSpec(
            id="aerial",
            category="aerial",
            max_horizon=DEFAULT_SKILL_AERIAL_HORIZON,
            requires_drone=True,
        ),
        SkillSpec(
            id="agriculture",
            category="agriculture",
            max_horizon=DEFAULT_SKILL_AGRI_HORIZON,
            requires_drone=True,
        ),
        SkillSpec(
            id="communicate",
            category="communicate",
            max_horizon=DEFAULT_SKILL_COMMUNICATE_HORIZON,
        ),
    ]


def _skill_spec_from_mapping(item: dict[str, Any]) -> SkillSpec:
    unknown = set(item) - _SKILL_SPEC_FIELDS
    if unknown:
        msg = f"Unknown skill spec keys: {sorted(unknown)}"
        raise ValueError(msg)
    try:
        return SkillSpec(**item)
    except TypeError as exc:
        msg = f"Invalid skill spec {item}: {exc}"
        raise ValueError(msg) from exc


@dataclass
class SkillCatalog:
    """Python loader for the shared hierarchical skill catalog."""

    enabled: bool = False
    default_skill: str = DEFAULT_SKILL_ID
    default_horizon: int = DEFAULT_SKILL_HORIZON
    skills: list[SkillSpec] = field(default_factory=_default_skills)

    def get(self, skill_id: str) -> SkillSpec | None:
        for spec in self.skills:
            if spec.id == skill_id:
                return spec
        return None

    def enabled_skills(self) -> list[SkillSpec]:
        return [spec for spec in self.skills if spec.enabled]

    def resolve_default(self) -> SkillSpec | None:
        candidate = self.get(self.default_skill)
        if candidate is not None and candidate.enabled:
            return candidate
        enabled = self.enabled_skills()
        return enabled[0] if enabled else None

    def effective_horizon(self, spec: SkillSpec) -> int:
        if spec.max_horizon <= 0:
            return max(self.default_horizon, MIN_SKILL_HORIZON)
        return spec.max_horizon

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> SkillCatalog:
        raw_skills = payload.get("skills")
        if isinstance(raw_skills, list):
            skills = [
                _skill_spec_from_mapping(item)
                for item in raw_skills
                if isinstance(item, dict)
            ]
        else:
            skills = _default_skills()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            default_skill=str(payload.get("default_skill", DEFAULT_SKILL_ID)),
            default_horizon=int(payload.get("default_horizon", DEFAULT_SKILL_HORIZON)),
            skills=skills,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> SkillCatalog:
        config_path = Path(path)
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
        catalog = cls.from_mapping(payload)
        logger.info(
            "Loaded skill catalog from %s (%d skills, default=%s)",
            config_path,
            len(catalog.skills),
            catalog.default_skill,
        )
        return catalog

    @classmethod
    def from_default_path(cls) -> SkillCatalog:
        override = os.environ.get(SKILLS_CONFIG_PATH_ENV, "").strip()
        path = Path(override) if override else DEFAULT_SKILLS_CONFIG_PATH
        if not path.is_file():
            logger.warning("Skill catalog %s missing; using in-code defaults", path)
            return cls()
        return cls.from_toml(path)


class HierarchicalSkillPolicy(BaseAgent):
    """Selects catalog skills, then emits discrete primitive action ids."""

    def __init__(
        self,
        config: AgentConfig,
        catalog: SkillCatalog | None = None,
        *,
        action_space_size: int,
        comm_vocab_size: int = 0,
        drone_enabled: bool = False,
        agri_enabled: bool = False,
        seed: int = DEFAULT_SKILL_POLICY_SEED,
    ) -> None:
        super().__init__(config)
        self.catalog = catalog or SkillCatalog.from_default_path()
        self.action_space_size = max(int(action_space_size), 1)
        self.comm_vocab_size = max(int(comm_vocab_size), 0)
        self.drone_enabled = drone_enabled
        self.agri_enabled = agri_enabled
        self._rng = np.random.default_rng(seed)
        self._active_index: int | None = None
        self._steps_in_skill = 0
        logger.info(
            "HierarchicalSkillPolicy ready: skills=%d default=%s space=%d",
            len(self.catalog.skills),
            self.catalog.default_skill,
            self.action_space_size,
        )

    @property
    def active_skill_id(self) -> str | None:
        if self._active_index is None:
            return None
        return self.catalog.skills[self._active_index].id

    def _select_next_skill(self) -> int | None:
        n = len(self.catalog.skills)
        if n == 0:
            logger.warning("Empty skill catalog; emitting Noop")
            return None
        start = 0
        if self._active_index is not None:
            start = (self._active_index + 1) % n
        else:
            default = self.catalog.resolve_default()
            if default is not None:
                for idx, spec in enumerate(self.catalog.skills):
                    if spec.id == default.id:
                        start = idx
                        break
        for offset in range(n):
            idx = (start + offset) % n
            spec = self.catalog.skills[idx]
            if not spec.enabled:
                continue
            if spec.requires_drone and not self.drone_enabled:
                logger.debug("Skipping drone-gated skill %s", spec.id)
                continue
            if spec.category == "agriculture" and not (self.drone_enabled and self.agri_enabled):
                logger.debug("Skipping agriculture skill %s; agri disabled", spec.id)
                continue
            if spec.category == "communicate" and self.comm_vocab_size == 0:
                continue
            return idx
        return None

    def _clip(self, action_id: int) -> int:
        if action_id < 0 or action_id >= self.action_space_size:
            logger.debug("Clipping illegal action id %s to Noop", action_id)
            return ACTION_ID_NOOP
        return action_id

    def _aerial_primitive(self, remaining: int, drone_base: int) -> int:
        if not self.drone_enabled:
            return ACTION_ID_NOOP
        if remaining <= 1:
            offset = DRONE_OFFSET_LAND
        elif self._steps_in_skill > 0:
            offset = DRONE_OFFSET_HOVER
        else:
            offset = DRONE_OFFSET_TAKEOFF
        return self._clip(drone_base + offset)

    def _primitive_for(self, spec: SkillSpec, remaining: int) -> int:
        drone_base = FORGE_BASE_ACTIONS + self.comm_vocab_size
        agri_base = drone_base + (FORGE_DRONE_ACTION_COUNT if self.drone_enabled else 0)
        raw_action = ACTION_ID_NOOP
        category = spec.category

        if category in {"navigate", "explore"}:
            offset = int(self._rng.integers(0, ACTION_MOVE_FAMILY_SIZE))
            raw_action = ACTION_ID_MOVE_MIN + offset
        elif category == "gather":
            raw_action = ACTION_ID_PICKUP
        elif category == "craft":
            raw_action = ACTION_ID_CRAFT_MIN + int(spec.recipe_index)
        elif category == "combat":
            raw_action = ACTION_ID_INTERACT
        elif category == "communicate":
            raw_action = FORGE_BASE_ACTIONS + int(spec.comm_token)
        elif category == "aerial":
            return self._aerial_primitive(remaining, drone_base)
        elif category == "agriculture" and self.drone_enabled and self.agri_enabled:
            raw_action = agri_base + AGRI_OFFSET_MULTISPECTRAL

        return self._clip(raw_action)

    def act(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        del observation
        needs_new = self._active_index is None
        if not needs_new and self._active_index is not None:
            spec = self.catalog.skills[self._active_index]
            needs_new = self._steps_in_skill >= self.catalog.effective_horizon(spec)
        if needs_new:
            nxt = self._select_next_skill()
            self._active_index = nxt
            self._steps_in_skill = 0
            if nxt is not None:
                logger.info("Switching Python skill to %s", self.catalog.skills[nxt].id)
        if self._active_index is None:
            self._step_count += 1
            return ACTION_ID_NOOP, {"skill_id": None, "skill_step": 0}
        spec = self.catalog.skills[self._active_index]
        remaining = self.catalog.effective_horizon(spec) - self._steps_in_skill
        action_id = self._primitive_for(spec, remaining)
        self._steps_in_skill += 1
        self._step_count += 1
        category = skill_category_for_action_id(
            action_id,
            comm_vocab_size=self.comm_vocab_size,
            drone_enabled=self.drone_enabled,
            agri_enabled=self.agri_enabled,
        )
        logger.debug(
            "skill=%s step=%s action_id=%s category=%s",
            spec.id,
            self._steps_in_skill,
            action_id,
            category,
        )
        return action_id, {
            "skill_id": spec.id,
            "skill_category": spec.category,
            "skill_step": self._steps_in_skill,
            "decoded_category": category,
        }

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        del batch
        return {}
