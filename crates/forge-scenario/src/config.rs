//! Scenario configuration with metadata.
//!
//! A [`ScenarioConfig`] wraps a [`ForgeConfig`] with additional metadata
//! (tags, difficulty tier, author, description) for the scenario marketplace.

use std::collections::BTreeMap;
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
    /// Optional derivation diagnostics for scenarios synthesized from higher-level documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation: Option<ScenarioDerivationDiagnostics>,
}

/// Diagnostics for scenarios derived from higher-level scenario documents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScenarioDerivationDiagnostics {
    /// Source format used to build this scenario.
    pub source_format: ScenarioSourceFormat,
    /// Source fields that were translated into executable or searchable config fields.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub translated_fields: Vec<DerivedFieldMapping>,
    /// Source fields preserved only as derivation metadata because no executable mapping exists yet.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metadata_only_fields: Vec<MetadataOnlyField>,
}

/// Supported source formats for scenario loading.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioSourceFormat {
    /// A native executable scenario manifest.
    #[default]
    NativeManifest,
    /// A higher-level scenario document derived into a scenario manifest.
    HighLevelDocument,
}

/// A single source-field mapping produced during scenario derivation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DerivedFieldMapping {
    /// Source field path in the higher-level scenario document.
    pub source: String,
    /// One or more target fields that received the translated value.
    pub targets: Vec<String>,
    /// Additional notes about the translation.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

/// A field captured in diagnostics but not translated into executable config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetadataOnlyField {
    /// Source field path in the higher-level scenario document.
    pub source: String,
    /// Additional notes about why the field is metadata-only.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
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

#[derive(Debug, Clone, Deserialize)]
struct HighLevelScenarioDocument {
    scenario: HighLevelScenario,
}

#[derive(Debug, Clone, Deserialize)]
struct HighLevelScenario {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_min_agents")]
    min_agents: u32,
    #[serde(default = "default_max_agents")]
    max_agents: u32,
    #[serde(default)]
    map: HighLevelScenarioMap,
    #[serde(default)]
    objectives: HighLevelScenarioObjectives,
    #[serde(default)]
    difficulty: HighLevelScenarioDifficulty,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HighLevelScenarioMap {
    grid_size: Option<u16>,
    #[serde(default)]
    terrain_type: String,
    #[serde(flatten)]
    _extra: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HighLevelScenarioObjectives {
    #[serde(rename = "type", default)]
    objective_type: String,
    time_limit: Option<u64>,
    #[serde(flatten)]
    _extra: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HighLevelScenarioDifficulty {
    #[serde(default = "default_tier")]
    base_tier: u8,
    #[serde(default)]
    fog_of_war: bool,
    #[serde(flatten)]
    _extra: BTreeMap<String, toml::Value>,
}

impl HighLevelScenarioDocument {
    fn into_scenario_config(self) -> Result<ScenarioConfig, String> {
        let scenario_id = normalize_identifier(&self.scenario.name);
        if scenario_id.is_empty() {
            return Err("high-level scenario name must produce a non-empty scenario id".into());
        }

        let mut tags = Vec::new();
        let mut translated_fields = vec![
            derived_field(
                "scenario.name",
                &["scenario.id", "scenario.name"],
                "scenario.id is normalized from scenario.name",
            ),
            derived_field(
                "scenario.description",
                &["scenario.description"],
                "description copied directly",
            ),
            derived_field(
                "scenario.min_agents",
                &["scenario.min_agents", "forge.agents.num_agents", "scenario.tags[]"],
                "min_agents sets the evaluated agent count and agent-count tag",
            ),
            derived_field(
                "scenario.max_agents",
                &["scenario.max_agents"],
                "max_agents copied directly",
            ),
            derived_field(
                "scenario.difficulty.base_tier",
                &["scenario.difficulty_tier", "forge.task.max_tier"],
                "base tier sets scenario metadata and task generation tier ceiling",
            ),
        ];
        let mut metadata_only_fields = Vec::new();

        push_normalized_tag(&mut tags, &self.scenario.objectives.objective_type);
        if !self.scenario.objectives.objective_type.is_empty() {
            translated_fields.push(derived_field(
                "scenario.objectives.type",
                &["scenario.tags[]"],
                "objective type currently contributes searchable tags only",
            ));
        }

        push_normalized_tag(&mut tags, &self.scenario.map.terrain_type);
        if !self.scenario.map.terrain_type.is_empty() {
            translated_fields.push(derived_field(
                "scenario.map.terrain_type",
                &["scenario.tags[]"],
                "terrain type currently contributes searchable tags only",
            ));
        }

        if self.scenario.difficulty.fog_of_war {
            push_normalized_tag(&mut tags, "fog_of_war");
            translated_fields.push(derived_field(
                "scenario.difficulty.fog_of_war",
                &["scenario.tags[]"],
                "fog_of_war=true currently contributes a searchable tag only",
            ));
        } else {
            metadata_only_fields.push(metadata_only_field(
                "scenario.difficulty.fog_of_war",
                "fog_of_war=false does not currently change executable config",
            ));
        }
        push_normalized_tag(
            &mut tags,
            if self.scenario.min_agents > 1 {
                "multi_agent"
            } else {
                "single_agent"
            },
        );

        let mut forge = ForgeConfig::default();
        if let Some(grid_size) = self.scenario.map.grid_size {
            forge.world.width = grid_size;
            forge.world.height = grid_size;
            translated_fields.push(derived_field(
                "scenario.map.grid_size",
                &["forge.world.width", "forge.world.height"],
                format!("grid_size applied as a square world: {grid_size}x{grid_size}"),
            ));
        }
        if let Some(time_limit) = self.scenario.objectives.time_limit {
            forge.task.max_episode_length = time_limit;
            translated_fields.push(derived_field(
                "scenario.objectives.time_limit",
                &["forge.task.max_episode_length"],
                format!("time limit copied directly: {time_limit} ticks"),
            ));
        }
        forge.task.max_tier = self.scenario.difficulty.base_tier;
        forge.agents.num_agents = self.scenario.min_agents;

        metadata_only_fields.extend(flatten_metadata_only_fields(
            "scenario.map",
            &self.scenario.map._extra,
        ));
        metadata_only_fields.extend(flatten_metadata_only_fields(
            "scenario.objectives",
            &self.scenario.objectives._extra,
        ));
        metadata_only_fields.extend(flatten_metadata_only_fields(
            "scenario.difficulty",
            &self.scenario.difficulty._extra,
        ));

        Ok(ScenarioConfig {
            scenario: ScenarioMeta {
                id: scenario_id,
                name: self.scenario.name,
                description: self.scenario.description,
                tags,
                difficulty_tier: self.scenario.difficulty.base_tier,
                min_agents: self.scenario.min_agents,
                max_agents: self.scenario.max_agents,
                author: String::new(),
                version: default_version(),
            },
            forge,
            derivation: Some(ScenarioDerivationDiagnostics {
                source_format: ScenarioSourceFormat::HighLevelDocument,
                translated_fields,
                metadata_only_fields,
            }),
        })
    }
}

fn derived_field(
    source: impl Into<String>,
    targets: &[&str],
    detail: impl Into<String>,
) -> DerivedFieldMapping {
    DerivedFieldMapping {
        source: source.into(),
        targets: targets.iter().map(|target| (*target).to_string()).collect(),
        detail: detail.into(),
    }
}

fn metadata_only_field(source: impl Into<String>, detail: impl Into<String>) -> MetadataOnlyField {
    MetadataOnlyField {
        source: source.into(),
        detail: detail.into(),
    }
}

fn flatten_metadata_only_fields(
    prefix: &str,
    fields: &BTreeMap<String, toml::Value>,
) -> Vec<MetadataOnlyField> {
    fields
        .iter()
        .map(|(key, value)| {
            metadata_only_field(
                format!("{prefix}.{key}"),
                format!(
                    "preserved as metadata only; no executable mapping exists yet (value={})",
                    summarize_toml_value(value)
                ),
            )
        })
        .collect()
}

fn summarize_toml_value(value: &toml::Value) -> String {
    match value {
        toml::Value::String(text) => text.clone(),
        toml::Value::Integer(number) => number.to_string(),
        toml::Value::Float(number) => number.to_string(),
        toml::Value::Boolean(flag) => flag.to_string(),
        toml::Value::Datetime(datetime) => datetime.to_string(),
        toml::Value::Array(values) => format!("array(len={})", values.len()),
        toml::Value::Table(table) => format!("table(keys={})", table.len()),
    }
}

fn normalize_identifier(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut previous_was_separator = false;

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            previous_was_separator = false;
        } else if !normalized.is_empty() && !previous_was_separator {
            normalized.push('_');
            previous_was_separator = true;
        }
    }

    while normalized.ends_with('_') {
        normalized.pop();
    }

    normalized
}

fn push_normalized_tag(tags: &mut Vec<String>, value: &str) {
    let normalized = normalize_identifier(value);
    if !normalized.is_empty() && !tags.iter().any(|tag| tag == &normalized) {
        tags.push(normalized);
    }
}

impl ScenarioConfig {
    /// Parses a scenario config from a TOML string.
    ///
    /// Accepts either a native executable scenario manifest or a higher-level
    /// scenario document such as the files under `configs/scenarios`.
    #[instrument(skip_all)]
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        match toml::from_str(toml_str) {
            Ok(config) => Ok(config),
            Err(native_error) => {
                let native_error = native_error.to_string();
                Self::try_from_high_level_toml(toml_str).map_err(|high_level_error| {
                    format!(
                        "native scenario parse failed: {native_error}; high-level scenario derivation failed: {high_level_error}"
                    )
                })
            }
        }
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

    /// Returns derivation diagnostics when the scenario was synthesized from a higher-level document.
    pub fn derivation_diagnostics(&self) -> Option<&ScenarioDerivationDiagnostics> {
        self.derivation.as_ref()
    }

    fn try_from_high_level_toml(toml_str: &str) -> Result<Self, String> {
        let document: HighLevelScenarioDocument = toml::from_str(toml_str)
            .map_err(|e| format!("high-level TOML parse error: {e}"))?;
        document.into_scenario_config()
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
        assert!(config.derivation.is_none());
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
            derivation: None,
        };

        let toml_str = config.to_toml().unwrap();
        let deser = ScenarioConfig::from_toml(&toml_str).unwrap();
        assert_eq!(deser.scenario.id, "test");
        assert_eq!(deser.scenario.difficulty_tier, 3);
    }

    #[test]
    fn test_high_level_scenario_document_derivation() {
        let toml = r#"
[scenario]
name = "patrol"
description = "Agents patrol a route, visiting waypoints in order"
min_agents = 1
max_agents = 4

[scenario.map]
grid_size = 64
terrain_type = "mixed"
num_waypoints = 6

[scenario.objectives]
type = "sequence"
time_limit = 2000
completion_bonus = 5.0

[scenario.difficulty]
base_tier = 2
fog_of_war = true
"#;

        let config = ScenarioConfig::from_toml(toml).unwrap();

        assert_eq!(config.scenario.id, "patrol");
        assert_eq!(config.scenario.name, "patrol");
        assert_eq!(config.scenario.difficulty_tier, 2);
        assert_eq!(config.scenario.min_agents, 1);
        assert_eq!(config.scenario.max_agents, 4);
        assert_eq!(
            config.scenario.tags,
            vec!["sequence", "mixed", "fog_of_war", "single_agent"]
        );
        assert_eq!(config.forge.world.width, 64);
        assert_eq!(config.forge.world.height, 64);
        assert_eq!(config.forge.agents.num_agents, 1);
        assert_eq!(config.forge.task.max_episode_length, 2000);
        assert_eq!(config.forge.task.max_tier, 2);

        let derivation = config.derivation_diagnostics().unwrap();
        assert_eq!(derivation.source_format, ScenarioSourceFormat::HighLevelDocument);
        assert!(derivation.translated_fields.iter().any(|field| {
            field.source == "scenario.map.grid_size"
                && field.targets == vec!["forge.world.width", "forge.world.height"]
        }));
        assert!(derivation.metadata_only_fields.iter().any(|field| {
            field.source == "scenario.map.num_waypoints"
                && field.detail.contains("no executable mapping exists yet")
        }));
    }

    #[test]
    fn test_high_level_scenario_document_normalizes_identifier() {
        let toml = r#"
[scenario]
name = "Escort Mission Alpha"

[scenario.difficulty]
base_tier = 4
"#;

        let config = ScenarioConfig::from_toml(toml).unwrap();
        assert_eq!(config.scenario.id, "escort_mission_alpha");
        assert_eq!(config.scenario.tags, vec!["single_agent"]);
        assert!(config.derivation_diagnostics().is_some());
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
}
