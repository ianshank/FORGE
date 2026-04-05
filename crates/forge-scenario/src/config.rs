//! Scenario configuration with metadata.
//!
//! A [`ScenarioConfig`] wraps a [`ForgeConfig`] with additional metadata
//! (tags, difficulty tier, author, description) for the scenario marketplace.

use std::fmt;

use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Minimum valid difficulty tier.
pub const MIN_DIFFICULTY_TIER: u8 = 1;

/// Maximum valid difficulty tier.
pub const MAX_DIFFICULTY_TIER: u8 = 6;

/// A scenario configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioConfig {
    /// Scenario metadata.
    pub scenario: ScenarioMeta,
    /// FORGE configuration overrides for this scenario.
    #[serde(default)]
    pub forge: ForgeConfig,
}

/// Metadata describing a scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioMeta {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Description of what the scenario tests.
    #[serde(default)]
    pub description: String,
    /// Searchable tags (e.g., "navigation", "crafting", "cooperation").
    #[serde(default)]
    pub tags: Vec<String>,
    /// Difficulty tier (1–6).
    #[serde(default = "default_tier")]
    pub difficulty_tier: u8,
    /// Minimum agents required.
    #[serde(default = "default_min_agents")]
    pub min_agents: u32,
    /// Maximum agents supported.
    #[serde(default = "default_max_agents")]
    pub max_agents: u32,
    /// Author of the scenario.
    #[serde(default)]
    pub author: String,
    /// Version string.
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_tier() -> u8 {
    1
}
fn default_min_agents() -> u32 {
    1
}
fn default_max_agents() -> u32 {
    1
}
fn default_version() -> String {
    "1.0".to_string()
}

impl ScenarioConfig {
    /// Parses a scenario config from a TOML string.
    #[instrument(skip_all)]
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        toml::from_str(toml_str).map_err(|e| format!("TOML parse error: {e}"))
    }

    /// Serializes to a TOML string.
    #[instrument(skip_all)]
    pub fn to_toml(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("TOML serialization error: {e}"))
    }

    /// Returns the effective ForgeConfig with scenario overrides applied.
    pub fn effective_config(&self) -> &ForgeConfig {
        &self.forge
    }

    /// Validates the scenario config, returning a list of issues.
    ///
    /// An empty list means the config is valid.
    #[instrument(skip_all)]
    pub fn validate(&self) -> Vec<String> {
        self.scenario.validate()
    }

    /// Returns true if the scenario config passes all validation checks.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

impl ScenarioMeta {
    /// Validates the scenario metadata, returning a list of issues.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.id.is_empty() {
            errors.push("scenario id must not be empty".to_string());
        }
        if self.name.is_empty() {
            errors.push("scenario name must not be empty".to_string());
        }
        if self.difficulty_tier < MIN_DIFFICULTY_TIER || self.difficulty_tier > MAX_DIFFICULTY_TIER
        {
            errors.push(format!(
                "difficulty_tier {} is outside valid range {}-{}",
                self.difficulty_tier, MIN_DIFFICULTY_TIER, MAX_DIFFICULTY_TIER
            ));
        }
        if self.min_agents > self.max_agents {
            errors.push(format!(
                "min_agents ({}) > max_agents ({})",
                self.min_agents, self.max_agents
            ));
        }
        if self.min_agents == 0 {
            errors.push("min_agents must be > 0".to_string());
        }

        errors
    }
}

impl fmt::Display for ScenarioConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (tier {}, {}-{} agents)",
            self.scenario.name,
            self.scenario.difficulty_tier,
            self.scenario.min_agents,
            self.scenario.max_agents,
        )
    }
}

impl fmt::Display for ScenarioMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} (tier {})",
            self.id, self.name, self.difficulty_tier
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_from_toml() {
        let toml = r#"
[scenario]
id = "patrol_basic"
name = "Basic Patrol"
description = "Navigate through waypoints"
tags = ["navigation", "movement"]
difficulty_tier = 2
min_agents = 1
max_agents = 4
author = "test"

[forge.world]
width = 32
height = 32
seed = 100
"#;
        let config = ScenarioConfig::from_toml(toml).unwrap();
        assert_eq!(config.scenario.id, "patrol_basic");
        assert_eq!(config.scenario.name, "Basic Patrol");
        assert_eq!(config.scenario.difficulty_tier, 2);
        assert_eq!(config.scenario.tags, vec!["navigation", "movement"]);
        assert_eq!(config.forge.world.width, 32);
        assert_eq!(config.forge.world.seed, 100);
    }

    #[test]
    fn test_scenario_defaults() {
        let toml = r#"
[scenario]
id = "minimal"
name = "Minimal Scenario"
"#;
        let config = ScenarioConfig::from_toml(toml).unwrap();
        assert_eq!(config.scenario.difficulty_tier, 1);
        assert_eq!(config.scenario.min_agents, 1);
        assert_eq!(config.scenario.max_agents, 1);
        assert_eq!(config.scenario.version, "1.0");
        assert!(config.scenario.tags.is_empty());
    }

    #[test]
    fn test_scenario_toml_roundtrip() {
        let config = ScenarioConfig {
            scenario: ScenarioMeta {
                id: "test".into(),
                name: "Test".into(),
                description: "A test scenario".into(),
                tags: vec!["test".into()],
                difficulty_tier: 3,
                min_agents: 1,
                max_agents: 2,
                author: "author".into(),
                version: "1.0".into(),
            },
            forge: ForgeConfig::default(),
        };

        let toml_str = config.to_toml().unwrap();
        let deser = ScenarioConfig::from_toml(&toml_str).unwrap();
        assert_eq!(deser.scenario.id, "test");
        assert_eq!(deser.scenario.difficulty_tier, 3);
    }

    #[test]
    fn test_invalid_toml() {
        let result = ScenarioConfig::from_toml("not valid toml {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_effective_config() {
        let toml = r#"
[scenario]
id = "test"
name = "Test"

[forge.world]
width = 64
"#;
        let config = ScenarioConfig::from_toml(toml).unwrap();
        assert_eq!(config.effective_config().world.width, 64);
    }

    #[test]
    fn test_validate_valid_scenario() {
        let toml = r#"
[scenario]
id = "test"
name = "Test"
difficulty_tier = 3
min_agents = 1
max_agents = 4
"#;
        let config = ScenarioConfig::from_toml(toml).unwrap();
        assert!(config.is_valid());
    }

    #[test]
    fn test_validate_tier_out_of_range() {
        let mut config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "test"
name = "Test"
"#,
        )
        .unwrap();
        config.scenario.difficulty_tier = 0;
        let errors = config.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("difficulty_tier"));

        config.scenario.difficulty_tier = 7;
        let errors = config.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validate_min_greater_than_max_agents() {
        let mut config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "test"
name = "Test"
"#,
        )
        .unwrap();
        config.scenario.min_agents = 5;
        config.scenario.max_agents = 2;
        let errors = config.validate();
        assert!(errors.iter().any(|e| e.contains("min_agents")));
    }

    #[test]
    fn test_validate_empty_id() {
        let mut config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = ""
name = "Test"
"#,
        )
        .unwrap();
        config.scenario.id = String::new();
        assert!(!config.is_valid());
    }

    #[test]
    fn test_validate_zero_min_agents() {
        let mut config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "test"
name = "Test"
"#,
        )
        .unwrap();
        config.scenario.min_agents = 0;
        assert!(!config.is_valid());
    }

    #[test]
    fn test_scenario_display() {
        let config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "test"
name = "Test Scenario"
difficulty_tier = 3
min_agents = 1
max_agents = 4
"#,
        )
        .unwrap();
        let display = format!("{config}");
        assert!(display.contains("Test Scenario"));
        assert!(display.contains("tier 3"));
    }

    #[test]
    fn test_scenario_meta_display() {
        let meta = ScenarioMeta {
            id: "patrol".into(),
            name: "Patrol".into(),
            difficulty_tier: 2,
            ..ScenarioMeta {
                id: String::new(),
                name: String::new(),
                description: String::new(),
                tags: vec![],
                difficulty_tier: 1,
                min_agents: 1,
                max_agents: 1,
                author: String::new(),
                version: "1.0".into(),
            }
        };
        let display = format!("{meta}");
        assert!(display.contains("patrol"));
        assert!(display.contains("Patrol"));
    }

    #[test]
    fn test_tier_range_constants() {
        assert_eq!(MIN_DIFFICULTY_TIER, 1);
        assert_eq!(MAX_DIFFICULTY_TIER, 6);
    }

    #[test]
    fn test_validate_empty_name() {
        let mut config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "test"
name = ""
"#,
        )
        .unwrap();
        config.scenario.name = String::new();
        let errors = config.validate();
        assert!(errors.iter().any(|e| e.contains("name")));
    }

    #[test]
    fn test_validate_all_valid_tiers() {
        for tier in MIN_DIFFICULTY_TIER..=MAX_DIFFICULTY_TIER {
            let mut config = ScenarioConfig::from_toml(
                r#"
[scenario]
id = "test"
name = "Test"
"#,
            )
            .unwrap();
            config.scenario.difficulty_tier = tier;
            assert!(
                config.is_valid(),
                "tier {tier} should be valid"
            );
        }
    }

    #[test]
    fn test_scenario_config_clone() {
        let config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "clone_test"
name = "Clone Test"
tags = ["a", "b"]
"#,
        )
        .unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.scenario.id, "clone_test");
        assert_eq!(cloned.scenario.tags.len(), 2);
    }

    #[test]
    fn test_scenario_debug() {
        let config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "debug_test"
name = "Debug Test"
"#,
        )
        .unwrap();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ScenarioConfig"));
    }

    #[test]
    fn test_scenario_meta_serde_roundtrip() {
        let config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "serde_test"
name = "Serde Test"
tags = ["x"]
difficulty_tier = 4
min_agents = 2
max_agents = 8
author = "forge-team"
version = "2.0"
"#,
        )
        .unwrap();
        let json = serde_json::to_string(&config).unwrap();
        let deser: ScenarioConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.scenario.id, "serde_test");
        assert_eq!(deser.scenario.difficulty_tier, 4);
        assert_eq!(deser.scenario.min_agents, 2);
        assert_eq!(deser.scenario.max_agents, 8);
        assert_eq!(deser.scenario.author, "forge-team");
        assert_eq!(deser.scenario.version, "2.0");
    }

    #[test]
    fn test_validate_multiple_errors() {
        let mut config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = ""
name = ""
"#,
        )
        .unwrap();
        config.scenario.id = String::new();
        config.scenario.name = String::new();
        config.scenario.difficulty_tier = 0;
        config.scenario.min_agents = 0;
        let errors = config.validate();
        assert!(errors.len() >= 3, "expected at least 3 errors, got {}", errors.len());
    }

    #[test]
    fn test_display_with_multi_agent() {
        let config = ScenarioConfig::from_toml(
            r#"
[scenario]
id = "multi"
name = "Multi Agent"
min_agents = 2
max_agents = 8
difficulty_tier = 5
"#,
        )
        .unwrap();
        let display = format!("{config}");
        assert!(display.contains("2-8 agents"));
        assert!(display.contains("tier 5"));
    }

    #[test]
    fn test_from_toml_missing_scenario_section() {
        let result = ScenarioConfig::from_toml("[forge]\nwidth = 32");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Difficulty tier validation accepts all tiers in valid range.
        #[test]
        fn valid_tier_passes_validation(tier in 1_u8..=6) {
            let meta = ScenarioMeta {
                id: "prop_test".into(),
                name: "PropTest".into(),
                description: String::new(),
                tags: vec![],
                difficulty_tier: tier,
                min_agents: 1,
                max_agents: 4,
                author: String::new(),
                version: "1.0".into(),
            };
            let errors = meta.validate();
            prop_assert!(!errors.iter().any(|e| e.contains("difficulty_tier")));
        }

        /// min_agents <= max_agents never produces that error.
        #[test]
        fn valid_agent_range_passes(
            min in 1_u32..=10,
            extra in 0_u32..=10,
        ) {
            let max = min + extra;
            let meta = ScenarioMeta {
                id: "test".into(),
                name: "Test".into(),
                description: String::new(),
                tags: vec![],
                difficulty_tier: 1,
                min_agents: min,
                max_agents: max,
                author: String::new(),
                version: "1.0".into(),
            };
            let errors = meta.validate();
            prop_assert!(!errors.iter().any(|e| e.contains("min_agents") && e.contains("max_agents")));
        }
    }
}
