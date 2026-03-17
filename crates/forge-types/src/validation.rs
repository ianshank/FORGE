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

    // Drone configuration validation
    if config.drone.enabled {
        if config.drone.max_altitude == 0 {
            return Err(ForgeError::Config(ConfigError::OutOfRange {
                field: "drone.max_altitude".to_string(),
                value: "0".to_string(),
                min: "1".to_string(),
                max: "255".to_string(),
            }));
        }
        let total_special = config.drone.num_aerial + config.drone.num_ground_vehicles;
        if total_special > config.agents.num_agents {
            return Err(ForgeError::Config(ConfigError::OutOfRange {
                field: "drone.num_aerial + drone.num_ground_vehicles".to_string(),
                value: total_special.to_string(),
                min: "0".to_string(),
                max: config.agents.num_agents.to_string(),
            }));
        }
        if config.drone.max_battery <= 0 {
            return Err(ForgeError::Config(ConfigError::OutOfRange {
                field: "drone.max_battery".to_string(),
                value: config.drone.max_battery.to_string(),
                min: "1".to_string(),
                max: i32::MAX.to_string(),
            }));
        }
        if config.drone.fall_damage_per_level < 0 {
            return Err(ForgeError::Config(ConfigError::OutOfRange {
                field: "drone.fall_damage_per_level".to_string(),
                value: config.drone.fall_damage_per_level.to_string(),
                min: "0".to_string(),
                max: i32::MAX.to_string(),
            }));
        }
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

    #[test]
    fn test_invalid_max_health_zero() {
        let mut config = ForgeConfig::default();
        config.agents.max_health = 0;
        let result = validate_config(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_health"));
    }

    #[test]
    fn test_invalid_max_health_negative() {
        let mut config = ForgeConfig::default();
        config.agents.max_health = -100;
        let result = validate_config(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_health"));
    }

    #[test]
    fn test_invalid_max_stamina_zero() {
        let mut config = ForgeConfig::default();
        config.agents.max_stamina = 0;
        let result = validate_config(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_stamina"));
    }

    #[test]
    fn test_invalid_max_stamina_negative() {
        let mut config = ForgeConfig::default();
        config.agents.max_stamina = -50;
        let result = validate_config(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_stamina"));
    }

    #[test]
    fn test_drone_config_disabled_passes_validation() {
        let config = ForgeConfig::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_drone_config_valid_passes() {
        let mut config = ForgeConfig::default();
        config.drone.enabled = true;
        config.drone.num_aerial = 1;
        config.agents.num_agents = 2;
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_drone_config_too_many_special_agents() {
        let mut config = ForgeConfig::default();
        config.drone.enabled = true;
        config.drone.num_aerial = 5;
        config.drone.num_ground_vehicles = 5;
        config.agents.num_agents = 3;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_drone_config_zero_max_altitude() {
        let mut config = ForgeConfig::default();
        config.drone.enabled = true;
        config.drone.max_altitude = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_drone_config_negative_fall_damage() {
        let mut config = ForgeConfig::default();
        config.drone.enabled = true;
        config.drone.fall_damage_per_level = -1;
        let result = validate_config(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("fall_damage_per_level"));
    }

    #[test]
    fn test_drone_config_zero_fall_damage_valid() {
        let mut config = ForgeConfig::default();
        config.drone.enabled = true;
        config.drone.num_aerial = 1;
        config.drone.fall_damage_per_level = 0;
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_drone_config_negative_max_battery() {
        let mut config = ForgeConfig::default();
        config.drone.enabled = true;
        config.drone.max_battery = -100;
        let result = validate_config(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_battery"));
    }

    #[test]
    fn test_drone_config_exact_agent_count() {
        let mut config = ForgeConfig::default();
        config.drone.enabled = true;
        config.drone.num_aerial = 2;
        config.drone.num_ground_vehicles = 1;
        config.agents.num_agents = 3; // exactly matches
        assert!(validate_config(&config).is_ok());
    }

    // ---- Proptest: validation invariants ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn default_config_always_valid(seed in 0u64..10000) {
                let mut config = ForgeConfig::default();
                config.world.seed = seed;
                prop_assert!(validate_config(&config).is_ok());
            }

            #[test]
            fn valid_dimensions_pass(
                width in 8u16..=256,
                height in 8u16..=256,
            ) {
                let mut config = ForgeConfig::default();
                config.world.width = width;
                config.world.height = height;
                // Adjust vision radius to fit
                let min_side = width.min(height);
                config.agents.default_vision_radius =
                    ((min_side - 1) / 2).min(5) as u8;
                prop_assert!(validate_config(&config).is_ok());
            }

            /// Zero or negative health always fails.
            #[test]
            fn invalid_health_fails(health in i32::MIN..=0) {
                let mut config = ForgeConfig::default();
                config.agents.max_health = health;
                prop_assert!(validate_config(&config).is_err());
            }

            /// Zero or negative stamina always fails.
            #[test]
            fn invalid_stamina_fails(stamina in i32::MIN..=0) {
                let mut config = ForgeConfig::default();
                config.agents.max_stamina = stamina;
                prop_assert!(validate_config(&config).is_err());
            }

            /// Resource density outside [0.0, 1.0] fails.
            #[test]
            fn invalid_resource_density(density in 1.01f32..100.0) {
                let mut config = ForgeConfig::default();
                config.world.resource_density = density;
                prop_assert!(validate_config(&config).is_err());
            }
        }
    }
}
