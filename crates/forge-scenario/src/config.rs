//! Scenario configuration with metadata.
//!
//! A [`ScenarioConfig`] wraps a [`ForgeConfig`] with additional metadata
//! (tags, difficulty tier, author, description) for the scenario marketplace.

use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};

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
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        toml::from_str(toml_str).map_err(|e| format!("TOML parse error: {e}"))
    }

    /// Serializes to a TOML string.
    pub fn to_toml(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("TOML serialization error: {e}"))
    }

    /// Returns the effective ForgeConfig with scenario overrides applied.
    pub fn effective_config(&self) -> &ForgeConfig {
        &self.forge
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
}
