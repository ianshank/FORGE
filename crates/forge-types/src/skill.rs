//! Hierarchical skill catalog over primitive [`Action`]s.
//!
//! Implements the *options* temporal-abstraction layer (Sutton, Precup &
//! Singh 1999; HRL / HMASD skill catalogs) as a configuration-driven mapping
//! from reusable skill identifiers to primitive-action families. High-level
//! policies select a [`SkillSpec`]; low-level executors emit [`Action`]s until
//! the spec's horizon (or an initiation/termination predicate) fires.
//!
//! No in-logic numeric literals: horizons, recipe indices, and identifiers
//! resolve through [`crate::constants`] or deserialized config.

use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::constants;

/// Semantic family of a temporally-extended skill option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SkillCategory {
    /// Always-legal no-op option.
    Idle,
    /// Locomotion toward a goal (cardinal or hex movement).
    Navigate,
    /// Resource pickup / drop.
    Gather,
    /// Stochastic exploration over the move family.
    Explore,
    /// Crafting and inventory-use primitives.
    Craft,
    /// Push / interact combat primitives.
    Combat,
    /// Aerial UAV primitives (takeoff, hover, land, scan, payload).
    Aerial,
    /// Agricultural scan / spray primitives.
    Agriculture,
    /// Discrete communication tokens.
    Communicate,
}

impl SkillCategory {
    /// Maps a primitive action onto its reusable skill family.
    ///
    /// Exhaustive over [`Action`] so adding a variant is a compile-time break
    /// rather than a silent `explore` fallback.
    pub fn from_action(action: &Action) -> Self {
        match action {
            Action::Noop => Self::Idle,
            Action::Move(_) | Action::MoveHex(_) => Self::Navigate,
            Action::PickUp | Action::Drop(_) => Self::Gather,
            Action::Use(_) | Action::Craft(_) => Self::Craft,
            Action::Push(_) | Action::Interact => Self::Combat,
            Action::Communicate(_) => Self::Communicate,
            Action::Ascend
            | Action::Descend
            | Action::Hover
            | Action::TakeOff
            | Action::Land
            | Action::Scan(_)
            | Action::DropPayload(_) => Self::Aerial,
            Action::Spray(_)
            | Action::ScanMultispectral
            | Action::ScanThermal
            | Action::RelaySoilData
            | Action::GenerateReport => Self::Agriculture,
        }
    }

    /// Stable lowercase identifier used in traces and TOML.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Navigate => "navigate",
            Self::Gather => "gather",
            Self::Explore => "explore",
            Self::Craft => "craft",
            Self::Combat => "combat",
            Self::Aerial => "aerial",
            Self::Agriculture => "agriculture",
            Self::Communicate => "communicate",
        }
    }
}

/// One reusable skill option: initiation is implied by `enabled` /
/// `requires_drone`; termination is the configured horizon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SkillSpec {
    /// Stable skill identifier (`idle`, `navigate`, …).
    pub id: String,
    /// Primitive-action family this option executes.
    pub category: SkillCategory,
    /// Whether a hierarchical policy may select this skill.
    pub enabled: bool,
    /// Maximum ticks before the option is forced to terminate.
    pub max_horizon: u32,
    /// When true, ground agents skip this skill (aerial initiation set).
    pub requires_drone: bool,
    /// Recipe index for [`SkillCategory::Craft`] (config, not a literal).
    pub recipe_index: u16,
    /// Communication token for [`SkillCategory::Communicate`].
    pub comm_token: u16,
    /// Optional navigation target X; `None` means world-center at runtime.
    pub target_x: Option<u16>,
    /// Optional navigation target Y; `None` means world-center at runtime.
    pub target_y: Option<u16>,
}

impl Default for SkillSpec {
    fn default() -> Self {
        Self {
            id: constants::DEFAULT_SKILL_ID.to_string(),
            category: SkillCategory::Explore,
            enabled: true,
            max_horizon: constants::DEFAULT_SKILL_HORIZON,
            requires_drone: false,
            recipe_index: constants::DEFAULT_SKILL_CRAFT_RECIPE,
            comm_token: constants::DEFAULT_SKILL_COMM_TOKEN,
            target_x: None,
            target_y: None,
        }
    }
}

fn builtin_skill_catalog() -> Vec<SkillSpec> {
    vec![
        SkillSpec {
            id: "idle".to_string(),
            category: SkillCategory::Idle,
            max_horizon: constants::DEFAULT_SKILL_IDLE_HORIZON,
            ..SkillSpec::default()
        },
        SkillSpec {
            id: "navigate".to_string(),
            category: SkillCategory::Navigate,
            max_horizon: constants::DEFAULT_SKILL_NAVIGATE_HORIZON,
            ..SkillSpec::default()
        },
        SkillSpec {
            id: "gather".to_string(),
            category: SkillCategory::Gather,
            max_horizon: constants::DEFAULT_SKILL_GATHER_HORIZON,
            ..SkillSpec::default()
        },
        SkillSpec {
            id: constants::DEFAULT_SKILL_ID.to_string(),
            category: SkillCategory::Explore,
            max_horizon: constants::DEFAULT_SKILL_EXPLORE_HORIZON,
            ..SkillSpec::default()
        },
        SkillSpec {
            id: "craft".to_string(),
            category: SkillCategory::Craft,
            max_horizon: constants::DEFAULT_SKILL_CRAFT_HORIZON,
            recipe_index: constants::DEFAULT_SKILL_CRAFT_RECIPE,
            ..SkillSpec::default()
        },
        SkillSpec {
            id: "combat".to_string(),
            category: SkillCategory::Combat,
            max_horizon: constants::DEFAULT_SKILL_COMBAT_HORIZON,
            ..SkillSpec::default()
        },
        SkillSpec {
            id: "aerial".to_string(),
            category: SkillCategory::Aerial,
            max_horizon: constants::DEFAULT_SKILL_AERIAL_HORIZON,
            requires_drone: true,
            ..SkillSpec::default()
        },
        SkillSpec {
            id: "agriculture".to_string(),
            category: SkillCategory::Agriculture,
            max_horizon: constants::DEFAULT_SKILL_AGRI_HORIZON,
            requires_drone: true,
            ..SkillSpec::default()
        },
        SkillSpec {
            id: "communicate".to_string(),
            category: SkillCategory::Communicate,
            max_horizon: constants::DEFAULT_SKILL_COMMUNICATE_HORIZON,
            comm_token: constants::DEFAULT_SKILL_COMM_TOKEN,
            ..SkillSpec::default()
        },
    ]
}

/// Configuration-driven catalog of reusable skill options.
///
/// Additive and backwards compatible: omitted `[skills]` sections deserialize
/// to `enabled = false` with the builtin catalog populated via defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SkillsConfig {
    /// Master switch. Hierarchical agents may still be constructed explicitly.
    pub enabled: bool,
    /// Skill id selected when a policy has no outstanding option.
    pub default_skill: String,
    /// Fallback horizon used by specs that set `max_horizon = 0` before validation.
    pub default_horizon: u32,
    /// Ordered catalog. Empty after explicit `skills = []`; otherwise builtins.
    #[serde(default = "builtin_skill_catalog")]
    pub skills: Vec<SkillSpec>,
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_skill: constants::DEFAULT_SKILL_ID.to_string(),
            default_horizon: constants::DEFAULT_SKILL_HORIZON,
            skills: builtin_skill_catalog(),
        }
    }
}

impl SkillsConfig {
    /// Looks up a skill by id.
    pub fn get(&self, id: &str) -> Option<&SkillSpec> {
        self.skills.iter().find(|spec| spec.id == id)
    }

    /// Enabled catalog entries in declaration order.
    pub fn enabled_skills(&self) -> impl Iterator<Item = &SkillSpec> {
        self.skills.iter().filter(|spec| spec.enabled)
    }

    /// Resolves the default skill, falling back to the first enabled entry.
    pub fn resolve_default(&self) -> Option<&SkillSpec> {
        self.get(&self.default_skill)
            .filter(|spec| spec.enabled)
            .or_else(|| self.enabled_skills().next())
    }

    /// Horizon for `spec`, substituting the catalog default when the spec is zero.
    pub fn effective_horizon(&self, spec: &SkillSpec) -> u32 {
        if spec.max_horizon == 0 {
            self.default_horizon.max(constants::MIN_SKILL_HORIZON)
        } else {
            spec.max_horizon
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Direction, HexDirection};

    #[test]
    fn from_action_is_exhaustive_over_primitive_families() {
        let cases: &[(Action, SkillCategory)] = &[
            (Action::Noop, SkillCategory::Idle),
            (Action::Move(Direction::Up), SkillCategory::Navigate),
            (Action::MoveHex(HexDirection::NE), SkillCategory::Navigate),
            (Action::PickUp, SkillCategory::Gather),
            (Action::Drop(0), SkillCategory::Gather),
            (Action::Use(0), SkillCategory::Craft),
            (Action::Craft(0), SkillCategory::Craft),
            (Action::Push(Direction::Left), SkillCategory::Combat),
            (Action::Interact, SkillCategory::Combat),
            (Action::Communicate(0), SkillCategory::Communicate),
            (Action::TakeOff, SkillCategory::Aerial),
            (Action::Hover, SkillCategory::Aerial),
            (Action::Land, SkillCategory::Aerial),
            (Action::Scan(Direction::Down), SkillCategory::Aerial),
            (Action::DropPayload(1), SkillCategory::Aerial),
            (Action::Spray(0), SkillCategory::Agriculture),
            (Action::ScanMultispectral, SkillCategory::Agriculture),
            (Action::GenerateReport, SkillCategory::Agriculture),
        ];
        for (action, expected) in cases {
            assert_eq!(
                SkillCategory::from_action(action),
                *expected,
                "action {action:?}"
            );
        }
    }

    #[test]
    fn builtin_catalog_contains_default_skill_and_unique_ids() {
        let catalog = SkillsConfig::default();
        assert!(
            !catalog.enabled,
            "skills remain opt-in for backwards compatibility"
        );
        assert_eq!(catalog.default_skill, constants::DEFAULT_SKILL_ID);
        let mut ids: Vec<&str> = catalog.skills.iter().map(|s| s.id.as_str()).collect();
        ids.sort_unstable();
        let mut dedup = ids.clone();
        dedup.dedup();
        assert_eq!(ids, dedup, "builtin skill ids must be unique");
        assert!(catalog.get(constants::DEFAULT_SKILL_ID).is_some());
        assert_eq!(
            catalog.resolve_default().map(|s| s.id.as_str()),
            Some(constants::DEFAULT_SKILL_ID)
        );
    }

    #[test]
    fn deny_unknown_fields_on_skill_spec() {
        let err = toml::from_str::<SkillSpec>(
            r#"
id = "explore"
category = "explore"
mystery = 1
"#,
        );
        assert!(err.is_err(), "unknown keys must fail closed");
    }

    #[test]
    fn omitted_skills_section_deserializes_to_defaults() {
        let parsed: SkillsConfig = toml::from_str("").unwrap();
        assert_eq!(parsed, SkillsConfig::default());
    }

    #[test]
    fn committed_toml_catalog_matches_builtin_ids() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../configs/agents/skills_default.toml"
        );
        let raw = std::fs::read_to_string(path).expect("skills_default.toml must exist");
        let loaded: SkillsConfig = toml::from_str(&raw).expect("skills_default.toml must parse");
        assert_eq!(loaded, SkillsConfig::default());
    }
}
