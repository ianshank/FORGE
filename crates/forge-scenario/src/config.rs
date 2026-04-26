//! Scenario configuration with metadata.
//!
//! A [`ScenarioConfig`] wraps a [`ForgeConfig`] with additional metadata
//! (tags, difficulty tier, author, description) for the scenario marketplace.

use std::fmt;

use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{instrument, warn};

/// Minimum valid difficulty tier.
pub const MIN_DIFFICULTY_TIER: u8 = 1;

/// Maximum valid difficulty tier.
pub const MAX_DIFFICULTY_TIER: u8 = 6;

/// Errors that can occur while parsing or serializing a [`ScenarioConfig`].
///
/// This replaces the previous `Result<_, String>` signature on
/// [`ScenarioConfig::from_toml`] / [`ScenarioConfig::to_toml`]. The new type
/// preserves the original error context (via the underlying
/// [`toml::de::Error`] / [`toml::ser::Error`]) so callers can pattern-match on
/// the failure mode instead of string-matching the message.
#[derive(Debug, Error)]
pub enum ScenarioConfigError {
    /// The TOML payload could not be parsed into a [`ScenarioConfig`].
    #[error("failed to parse scenario TOML: {0}")]
    ParseToml(#[from] toml::de::Error),

    /// A [`ScenarioConfig`] could not be serialized into a TOML string.
    #[error("failed to serialize scenario TOML: {0}")]
    SerializeToml(#[from] toml::ser::Error),
}

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
    ///
    /// On failure, returns a [`ScenarioConfigError::ParseToml`] that wraps the
    /// underlying [`toml::de::Error`] (including span/line information when
    /// the source TOML provides it).
    #[instrument(skip_all)]
    pub fn from_toml(toml_str: &str) -> Result<Self, ScenarioConfigError> {
        toml::from_str(toml_str).map_err(|e: toml::de::Error| {
            warn!(error = %e, "ScenarioConfig::from_toml: parse failure");
            ScenarioConfigError::ParseToml(e)
        })
    }

    /// Serializes to a TOML string.
    ///
    /// On failure, returns a [`ScenarioConfigError::SerializeToml`] that wraps
    /// the underlying [`toml::ser::Error`].
    #[instrument(skip_all)]
    pub fn to_toml(&self) -> Result<String, ScenarioConfigError> {
        toml::to_string_pretty(self).map_err(|e: toml::ser::Error| {
            warn!(error = %e, "ScenarioConfig::to_toml: serialization failure");
            ScenarioConfigError::SerializeToml(e)
        })
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
    #[instrument(skip(self))]
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
    fn test_scenario_from_toml_returns_parse_error_variant() {
        // Confirm the error type carries the underlying toml::de::Error and is
        // not silently downgraded to a String.
        let err =
            ScenarioConfig::from_toml("not [valid toml").expect_err("malformed TOML must error");
        assert!(
            matches!(err, ScenarioConfigError::ParseToml(_)),
            "expected ParseToml variant, got {err:?}",
        );
        // Display message must remain human-readable (registry.rs logs it).
        let msg = err.to_string();
        assert!(
            msg.contains("failed to parse scenario TOML"),
            "missing wrapper prefix in: {msg}",
        );
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
    fn test_scenario_config_all_optional_fields() {
        let toml = r#"
[scenario]
id = "full"
name = "Full Scenario"
description = "A fully specified scenario"
tags = ["combat", "navigation"]
difficulty_tier = 5
min_agents = 2
max_agents = 8
author = "tester"
version = "2.0"

[forge.world]
width = 128
height = 128
seed = 999
"#;
        let config = ScenarioConfig::from_toml(toml).unwrap();
        assert_eq!(config.scenario.id, "full");
        assert_eq!(config.scenario.description, "A fully specified scenario");
        assert_eq!(config.scenario.tags, vec!["combat", "navigation"]);
        assert_eq!(config.scenario.difficulty_tier, 5);
        assert_eq!(config.scenario.min_agents, 2);
        assert_eq!(config.scenario.max_agents, 8);
        assert_eq!(config.scenario.author, "tester");
        assert_eq!(config.scenario.version, "2.0");
        assert!(config.is_valid());
    }

    #[test]
    fn test_validate_tier_boundary_valid() {
        for tier in [MIN_DIFFICULTY_TIER, MAX_DIFFICULTY_TIER] {
            let toml = format!(
                r#"
[scenario]
id = "test"
name = "Test"
difficulty_tier = {tier}
"#
            );
            let config = ScenarioConfig::from_toml(&toml).unwrap();
            let errors = config.validate();
            assert!(
                !errors.iter().any(|e| e.contains("difficulty_tier")),
                "tier {tier} should be valid"
            );
        }
    }

    #[test]
    fn test_effective_config_preserves_defaults() {
        let toml = r#"
[scenario]
id = "minimal"
name = "Minimal"
"#;
        let config = ScenarioConfig::from_toml(toml).unwrap();
        let effective = config.effective_config();
        let default_forge = ForgeConfig::default();
        assert_eq!(effective.world.width, default_forge.world.width);
        assert_eq!(effective.world.height, default_forge.world.height);
    }

    #[test]
    fn test_scenario_to_toml_roundtrip() {
        let toml = r#"
[scenario]
id = "roundtrip"
name = "Roundtrip Test"
difficulty_tier = 3
min_agents = 1
max_agents = 4
"#;
        let config = ScenarioConfig::from_toml(toml).unwrap();
        let serialized = config.to_toml().unwrap();
        let deser = ScenarioConfig::from_toml(&serialized).unwrap();
        assert_eq!(deser.scenario.id, "roundtrip");
        assert_eq!(deser.scenario.difficulty_tier, 3);
    }
}
