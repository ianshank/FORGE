//! Configuration validation for the FORGE simulation.
//!
//! Validates all config fields against their valid ranges before
//! constructing a [`WorldState`]. All validation logic is centralized
//! here to keep the config module focused on type definitions.

use crate::config::ForgeConfig;
use crate::error::{ConfigError, ForgeError, ForgeResult};

/// Validates a [`ForgeConfig`] against all known constraints.
///
/// Returns `Ok(())` if the config is valid, or a [`ForgeError::Config`]
/// with details about the first invalid field found.
///
/// Called automatically by `WorldState::new()`. Can also be called
/// directly to check a config before constructing a world.
pub fn validate_config(config: &ForgeConfig) -> ForgeResult<()> {
    // World dimensions within [min_dimension, max_dimension]
    if config.world.width < config.world.min_dimension
        || config.world.width > config.world.max_dimension
    {
        return Err(ForgeError::Config(ConfigError::OutOfRange {
            field: "world.width".to_string(),
            value: config.world.width.to_string(),
            min: config.world.min_dimension.to_string(),
            max: config.world.max_dimension.to_string(),
        }));
    }

    if config.world.height < config.world.min_dimension
        || config.world.height > config.world.max_dimension
    {
        return Err(ForgeError::Config(ConfigError::OutOfRange {
            field: "world.height".to_string(),
            value: config.world.height.to_string(),
            min: config.world.min_dimension.to_string(),
            max: config.world.max_dimension.to_string(),
        }));
    }

    // Resource density in [0.0, 1.0]
    if !(0.0..=1.0).contains(&config.world.resource_density) {
        return Err(ForgeError::Config(ConfigError::OutOfRange {
            field: "world.resource_density".to_string(),
            value: config.world.resource_density.to_string(),
            min: "0.0".to_string(),
            max: "1.0".to_string(),
        }));
    }

    // num_agents >= 1 when tasks are enabled (0 agents is valid for worldgen-only use)
    if config.task.enabled && config.agents.num_agents == 0 {
        return Err(ForgeError::Config(ConfigError::OutOfRange {
            field: "agents.num_agents".to_string(),
            value: "0 (tasks require at least 1 agent)".to_string(),
            min: "1".to_string(),
            max: u32::MAX.to_string(),
        }));
    }

    // reward_scale > 0.0
    if config.task.reward_scale <= 0.0 {
        return Err(ForgeError::Config(ConfigError::OutOfRange {
            field: "task.reward_scale".to_string(),
            value: config.task.reward_scale.to_string(),
            min: "0.0 (exclusive)".to_string(),
            max: "inf".to_string(),
        }));
    }

    // Note: comm_vocab_size == 0 with comm_radius > 0 is valid — it means
    // communication is disabled. No validation needed for this combination.

    // Vision radius fits within grid
    let vision_span = 2 * config.agents.default_vision_radius as u16 + 1;
    let min_side = config.world.width.min(config.world.height);
    if vision_span > min_side {
        return Err(ForgeError::Config(ConfigError::OutOfRange {
            field: "agents.default_vision_radius".to_string(),
            value: config.agents.default_vision_radius.to_string(),
            min: "0".to_string(),
            max: ((min_side - 1) / 2).to_string(),
        }));
    }

    // max_health > 0
    if config.agents.max_health <= 0 {
        return Err(ForgeError::Config(ConfigError::OutOfRange {
            field: "agents.max_health".to_string(),
            value: config.agents.max_health.to_string(),
            min: "1".to_string(),
            max: i32::MAX.to_string(),
        }));
    }

    // max_stamina > 0
    if config.agents.max_stamina <= 0 {
        return Err(ForgeError::Config(ConfigError::OutOfRange {
            field: "agents.max_stamina".to_string(),
            value: config.agents.max_stamina.to_string(),
            min: "1".to_string(),
            max: i32::MAX.to_string(),
        }));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        assert!(validate_config(&ForgeConfig::default()).is_ok());
    }

    #[test]
    fn test_invalid_width_too_small() {
        let mut config = ForgeConfig::default();
        config.world.width = 0;
        let result = validate_config(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("world.width"));
    }

    #[test]
    fn test_invalid_width_too_large() {
        let mut config = ForgeConfig::default();
        config.world.width = config.world.max_dimension + 1;
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_height() {
        let mut config = ForgeConfig::default();
        config.world.height = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_resource_density_negative() {
        let mut config = ForgeConfig::default();
        config.world.resource_density = -0.1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_resource_density_over_one() {
        let mut config = ForgeConfig::default();
        config.world.resource_density = 1.1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_zero_agents_with_tasks() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 0;
        config.task.enabled = true;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_zero_agents_without_tasks() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 0;
        config.task.enabled = false;
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_invalid_reward_scale_zero() {
        let mut config = ForgeConfig::default();
        config.task.reward_scale = 0.0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_reward_scale_negative() {
        let mut config = ForgeConfig::default();
        config.task.reward_scale = -1.0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_comm_radius_without_vocab_is_ok() {
        let mut config = ForgeConfig::default();
        config.agents.comm_radius = 5;
        config.agents.comm_vocab_size = 0;
        // comm_vocab_size == 0 means communication disabled, which is valid
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_vision_radius_too_large() {
        let mut config = ForgeConfig::default();
        config.world.min_dimension = 4;
        config.world.width = 5;
        config.world.height = 5;
        config.agents.default_vision_radius = 3; // 2*3+1 = 7 > 5
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_vision_radius_fits() {
        let mut config = ForgeConfig::default();
        config.world.min_dimension = 4;
        config.world.width = 5;
        config.world.height = 5;
        config.agents.default_vision_radius = 2; // 2*2+1 = 5 = 5, ok
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_boundary_resource_density_zero() {
        let mut config = ForgeConfig::default();
        config.world.resource_density = 0.0;
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_boundary_resource_density_one() {
        let mut config = ForgeConfig::default();
        config.world.resource_density = 1.0;
        assert!(validate_config(&config).is_ok());
    }
}
