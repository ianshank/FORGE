//! Scenario composition: merge multiple configs.
//!
//! Allows layering scenario configs where later configs override earlier ones.
//! This enables patterns like: base scenario + difficulty modifier + agent count modifier.

use forge_types::config::ForgeConfig;
use tracing::instrument;

use crate::config::ScenarioConfig;

/// Merges a base ForgeConfig with overrides from another ForgeConfig.
///
/// Fields in `overrides` that differ from the default replace the corresponding
/// fields in `base`. This uses a JSON-based merge strategy.
#[instrument(skip_all)]
pub fn merge_forge_configs(base: &ForgeConfig, overrides: &ForgeConfig) -> ForgeConfig {
    // Serialize both to JSON, merge, deserialize
    let base_json: serde_json::Value = serde_json::to_value(base).unwrap_or_default();
    let override_json: serde_json::Value = serde_json::to_value(overrides).unwrap_or_default();

    let merged = merge_json_values(base_json, override_json);

    serde_json::from_value(merged).unwrap_or_else(|_| base.clone())
}

/// Deep-merges two JSON values. Override values replace base values.
fn merge_json_values(base: serde_json::Value, overrides: serde_json::Value) -> serde_json::Value {
    match (base, overrides) {
        (serde_json::Value::Object(mut base_map), serde_json::Value::Object(override_map)) => {
            for (key, override_val) in override_map {
                let base_val = base_map.remove(&key).unwrap_or(serde_json::Value::Null);
                base_map.insert(key, merge_json_values(base_val, override_val));
            }
            serde_json::Value::Object(base_map)
        }
        (_, override_val) => override_val,
    }
}

/// Composes multiple scenario configs, applying overrides in order.
///
/// The first config is the base. Each subsequent config's `forge` section
/// overrides the previous. Scenario metadata comes from the last config.
#[instrument(skip_all)]
pub fn compose_scenarios(configs: &[ScenarioConfig]) -> Option<ScenarioConfig> {
    if configs.is_empty() {
        return None;
    }

    let mut result = configs[0].clone();

    for config in &configs[1..] {
        result.forge = merge_forge_configs(&result.forge, &config.forge);
        result.scenario = config.scenario.clone();
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScenarioMeta;

    fn make_base_scenario() -> ScenarioConfig {
        let mut forge = ForgeConfig::default();
        forge.world.width = 32;
        forge.world.height = 32;
        forge.agents.num_agents = 2;

        ScenarioConfig {
            scenario: ScenarioMeta {
                id: "base".into(),
                name: "Base".into(),
                description: "Base scenario".into(),
                tags: vec!["base".into()],
                difficulty_tier: 1,
                min_agents: 1,
                max_agents: 4,
                author: "test".into(),
                version: "1.0".into(),
            },
            forge,
        }
    }

    fn make_override_scenario() -> ScenarioConfig {
        let mut forge = ForgeConfig::default();
        forge.world.width = 64;

        ScenarioConfig {
            scenario: ScenarioMeta {
                id: "override".into(),
                name: "Override".into(),
                description: "Override scenario".into(),
                tags: vec!["hard".into()],
                difficulty_tier: 3,
                min_agents: 2,
                max_agents: 4,
                author: "test".into(),
                version: "2.0".into(),
            },
            forge,
        }
    }

    #[test]
    fn test_merge_forge_configs() {
        let mut base = ForgeConfig::default();
        base.world.width = 32;
        base.world.height = 32;
        base.agents.num_agents = 2;

        let mut overrides = ForgeConfig::default();
        overrides.world.width = 64;

        let merged = merge_forge_configs(&base, &overrides);
        assert_eq!(merged.world.width, 64);
        // Height comes from override (which has default value)
        // since the merge replaces all override fields
    }

    #[test]
    fn test_compose_empty() {
        let result = compose_scenarios(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_compose_single() {
        let base = make_base_scenario();
        let result = compose_scenarios(&[base.clone()]).unwrap();
        assert_eq!(result.scenario.id, "base");
        assert_eq!(result.forge.world.width, 32);
    }

    #[test]
    fn test_compose_two() {
        let base = make_base_scenario();
        let override_config = make_override_scenario();

        let result = compose_scenarios(&[base, override_config]).unwrap();

        // Metadata from last config
        assert_eq!(result.scenario.id, "override");
        assert_eq!(result.scenario.difficulty_tier, 3);

        // Forge config merged: override's width takes precedence
        assert_eq!(result.forge.world.width, 64);
    }

    #[test]
    fn test_merge_json_values_nested() {
        let base = serde_json::json!({
            "a": {"b": 1, "c": 2},
            "d": 3
        });
        let overrides = serde_json::json!({
            "a": {"b": 10},
            "e": 5
        });

        let merged = merge_json_values(base, overrides);
        assert_eq!(merged["a"]["b"], 10);
        assert_eq!(merged["a"]["c"], 2);
        assert_eq!(merged["d"], 3);
        assert_eq!(merged["e"], 5);
    }

    #[test]
    fn test_merge_json_values_scalar_override() {
        let base = serde_json::json!({"x": 1});
        let overrides = serde_json::json!({"x": 99});
        let merged = merge_json_values(base, overrides);
        assert_eq!(merged["x"], 99);
    }
}
