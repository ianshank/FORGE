//! Configuration adapter: translates MangoMAS training parameters to ForgeConfig.
//!
//! MangoMAS uses different parameter naming and ranges. This adapter
//! provides a clean translation layer so MangoMAS consumers don't need
//! to understand FORGE's internal config structure.

use forge_types::config::ForgeConfig;
use tracing::instrument;

use crate::config::{MangoMasConfig, Platform};
use crate::error::MangoMasResult;

/// Adapter for translating MangoMAS configuration to FORGE simulation config.
pub struct ConfigAdapter;

impl ConfigAdapter {
    /// Creates a FORGE config suitable for MangoMAS training.
    ///
    /// Applies platform-specific defaults (drone physics, agent morphology)
    /// and training-optimized settings (small worlds, short episodes).
    #[instrument(skip_all)]
    pub fn to_forge_config(mangomas_config: &MangoMasConfig) -> MangoMasResult<ForgeConfig> {
        let mut config = mangomas_config.batch_runner.forge_config.clone();

        // Apply platform-specific settings
        match mangomas_config.platform {
            Platform::Drone => {
                config.drone.enabled = true;
                config.drone.num_aerial = config.agents.num_agents;
            }
            Platform::Car => {
                config.drone.enabled = false;
            }
        }

        Ok(config)
    }

    /// Creates a minimal FORGE config for fast sweep evaluation.
    ///
    /// Uses small world, few agents, short episodes to maximize
    /// parameter sweep throughput.
    #[instrument(skip_all)]
    pub fn sweep_config(platform: Platform, seed: u64) -> ForgeConfig {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = seed;
        config.agents.num_agents = 1;
        config.task.max_episode_length = 500;
        config.task.enabled = true;

        match platform {
            Platform::Drone => {
                config.drone.enabled = true;
                config.drone.num_aerial = 1;
            }
            Platform::Car => {
                config.drone.enabled = false;
            }
        }

        config
    }

    /// Creates a FORGE config for surprise/novelty scenarios.
    ///
    /// Modifies world generation parameters to create controlled
    /// novelty for surprise-adaptive budget validation.
    #[instrument(skip_all)]
    pub fn novelty_config(base: &ForgeConfig, novelty_level: f32) -> ForgeConfig {
        let mut config = base.clone();
        // Increase resource scarcity and entity density for higher novelty
        config.world.resource_density = (1.0 - novelty_level).max(0.05);
        config.world.max_entities =
            ((config.world.max_entities as f32) * (1.0 + novelty_level)).min(256.0) as u16;
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drone_config() {
        let mangomas = MangoMasConfig {
            platform: Platform::Drone,
            ..MangoMasConfig::default()
        };
        let forge = ConfigAdapter::to_forge_config(&mangomas).unwrap();
        assert!(forge.drone.enabled);
    }

    #[test]
    fn test_car_config() {
        let mangomas = MangoMasConfig {
            platform: Platform::Car,
            ..MangoMasConfig::default()
        };
        let forge = ConfigAdapter::to_forge_config(&mangomas).unwrap();
        assert!(!forge.drone.enabled);
    }

    #[test]
    fn test_sweep_config_is_small() {
        let config = ConfigAdapter::sweep_config(Platform::Drone, 42);
        assert_eq!(config.world.width, 16);
        assert_eq!(config.world.height, 16);
        assert_eq!(config.agents.num_agents, 1);
        assert!(config.drone.enabled);
    }

    #[test]
    fn test_novelty_config_varies() {
        let base = ForgeConfig::default();
        let low = ConfigAdapter::novelty_config(&base, 0.0);
        let high = ConfigAdapter::novelty_config(&base, 1.0);
        assert!(low.world.resource_density > high.world.resource_density);
    }
}
