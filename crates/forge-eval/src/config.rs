//! Configuration for the evaluation harness.

use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};

/// Configuration for an evaluation run.
///
/// All parameters are configurable — no hard-coded values.
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
}
