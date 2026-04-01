//! Configuration for the evaluation harness.

use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Configuration for an evaluation run.
///
/// All parameters are configurable — no hard-coded values.
/// Use [`EvalConfig::validate`] to check invariants before running.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EvalConfig {
    /// Number of episodes to run per scenario/seed combination.
    pub episodes_per_scenario: u32,
    /// Maximum steps per episode (overrides scenario-level config if set).
    pub max_steps_per_episode: u64,
    /// Base seed for reproducibility. Episode seeds are derived as `base_seed + episode_idx`.
    pub base_seed: u64,
    /// Which difficulty tiers to evaluate (empty = all 6).
    pub tiers: Vec<u8>,
    /// Number of parallel evaluation threads (0 = use rayon default).
    pub parallelism: u32,
    /// Whether to record compact replays for each episode.
    pub record_replays: bool,
    /// Whether to record full trajectories for each episode.
    pub record_trajectories: bool,
    /// Base FORGE config to use for scenarios that don't specify their own.
    pub base_forge_config: ForgeConfig,
}

impl Default for EvalConfig {
    fn default() -> Self {
        let mut forge_config = ForgeConfig::default();
        forge_config.world.width = 16;
        forge_config.world.height = 16;
        forge_config.agents.num_agents = 1;
        forge_config.task.max_episode_length = 500;

        Self {
            episodes_per_scenario: 10,
            max_steps_per_episode: 500,
            base_seed: 0,
            tiers: vec![],
            parallelism: 0,
            record_replays: false,
            record_trajectories: false,
            base_forge_config: forge_config,
        }
    }
}

impl EvalConfig {
    /// Validates the configuration, returning a list of issues.
    ///
    /// An empty list means the config is valid.
    #[instrument(skip_all)]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.episodes_per_scenario == 0 {
            errors.push("episodes_per_scenario must be > 0".to_string());
        }
        if self.max_steps_per_episode == 0 {
            errors.push("max_steps_per_episode must be > 0".to_string());
        }
        for &tier in &self.tiers {
            if tier == 0 || tier > 6 {
                errors.push(format!("tier {tier} is out of valid range 1-6"));
            }
        }
        if self.base_forge_config.world.width == 0 || self.base_forge_config.world.height == 0 {
            errors.push("world dimensions must be > 0".to_string());
        }
        if self.base_forge_config.agents.num_agents == 0 {
            errors.push("num_agents must be > 0".to_string());
        }

        errors
    }

    /// Returns true if the configuration passes all validation checks.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EvalConfig::default();
        assert_eq!(config.episodes_per_scenario, 10);
        assert_eq!(config.max_steps_per_episode, 500);
        assert_eq!(config.base_seed, 0);
        assert!(config.tiers.is_empty());
        assert!(!config.record_replays);
        assert!(!config.record_trajectories);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(EvalConfig);
    }

    #[test]
    fn test_defaults_valid_macro() {
        forge_types::assert_config_defaults_valid!(EvalConfig);
    }

    #[test]
    fn test_config_serde_roundtrip_custom_values() {
        let config = EvalConfig {
            episodes_per_scenario: 50,
            max_steps_per_episode: 1000,
            base_seed: 42,
            tiers: vec![1, 2, 3],
            parallelism: 4,
            record_replays: true,
            record_trajectories: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: EvalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.episodes_per_scenario, 50);
        assert_eq!(deser.base_seed, 42);
        assert_eq!(deser.tiers, vec![1, 2, 3]);
        assert!(deser.record_replays);
    }

    #[test]
    fn test_default_config_is_valid() {
        let config = EvalConfig::default();
        assert!(config.is_valid());
    }

    #[test]
    fn test_validate_zero_episodes() {
        let mut config = EvalConfig::default();
        config.episodes_per_scenario = 0;
        let errors = config.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("episodes_per_scenario"));
    }

    #[test]
    fn test_validate_zero_max_steps() {
        let mut config = EvalConfig::default();
        config.max_steps_per_episode = 0;
        assert!(!config.is_valid());
    }

    #[test]
    fn test_validate_invalid_tier() {
        let mut config = EvalConfig::default();
        config.tiers = vec![0, 7];
        let errors = config.validate();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_validate_valid_tiers() {
        let mut config = EvalConfig::default();
        config.tiers = vec![1, 3, 6];
        assert!(config.is_valid());
    }

    #[test]
    fn test_validate_zero_world_dimensions() {
        let mut config = EvalConfig::default();
        config.base_forge_config.world.width = 0;
        assert!(!config.is_valid());
    }

    #[test]
    fn test_validate_zero_agents() {
        let mut config = EvalConfig::default();
        config.base_forge_config.agents.num_agents = 0;
        assert!(!config.is_valid());
    }
}
