//! Configuration conversion from Python dicts to Rust ForgeConfig.
//!
//! Uses serde_json as an intermediate format: Python dict -> JSON string -> ForgeConfig.

use forge_types::config::ForgeConfig;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Converts a Python dictionary to a ForgeConfig by serializing through JSON.
pub fn config_from_dict(py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<ForgeConfig> {
    // Convert Python dict -> JSON string -> ForgeConfig
    let json_module = py.import_bound("json")?;
    let json_str: String = json_module.call_method1("dumps", (dict,))?.extract()?;
    let config: ForgeConfig = serde_json::from_str(&json_str).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid config: {e}"))
    })?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use forge_types::config::ForgeConfig;

    #[test]
    fn test_config_default_roundtrip_through_json() {
        let config = ForgeConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deser: ForgeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.world.width, config.world.width);
        assert_eq!(deser.world.height, config.world.height);
        assert_eq!(deser.world.seed, config.world.seed);
        assert_eq!(deser.agents.num_agents, config.agents.num_agents);
    }

    #[test]
    fn test_partial_json_uses_defaults() {
        let json = r#"{"world":{"seed":42}}"#;
        let config: ForgeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.world.seed, 42);
        assert_eq!(
            config.world.width,
            forge_types::constants::DEFAULT_WORLD_WIDTH
        );
        assert_eq!(
            config.world.height,
            forge_types::constants::DEFAULT_WORLD_HEIGHT
        );
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let result: Result<ForgeConfig, _> = serde_json::from_str("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_object_uses_all_defaults() {
        let config: ForgeConfig = serde_json::from_str("{}").unwrap();
        let default = ForgeConfig::default();
        assert_eq!(config.world.width, default.world.width);
        assert_eq!(config.agents.num_agents, default.agents.num_agents);
    }
}
