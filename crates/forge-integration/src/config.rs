//! Configuration for the integration layer.

use serde::{Deserialize, Serialize};

use forge_cognitive::config::CognitiveConfig;
use forge_memory::config::MemoryConfig;
use forge_social::config::SocialConfig;

/// Default interval (in ticks) between memory writes.
const DEFAULT_MEMORY_WRITE_INTERVAL: u64 = 10;
/// Default weight of social rewards in total reward.
const DEFAULT_SOCIAL_REWARD_WEIGHT: f32 = 0.3;
/// Default meta-learning rate.
const DEFAULT_META_LR: f32 = 0.001;

/// Configuration for the integration orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IntegrationConfig {
    /// Whether the integration layer is enabled.
    pub enabled: bool,
    /// How often (in ticks) to write to agent memory.
    pub memory_write_interval: u64,
    /// Weight of social rewards blended into task rewards.
    pub social_reward_weight: f32,
    /// Whether meta-learning is enabled.
    pub meta_learning_enabled: bool,
    /// Meta-learning rate (controls how fast learning rules adapt).
    pub meta_lr: f32,
    /// Curriculum domains to train across.
    pub curriculum_domains: Vec<String>,
    /// Memory system configuration.
    pub memory: MemoryConfig,
    /// Social system configuration.
    pub social: SocialConfig,
    /// Cognitive system configuration.
    pub cognitive: CognitiveConfig,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            memory_write_interval: DEFAULT_MEMORY_WRITE_INTERVAL,
            social_reward_weight: DEFAULT_SOCIAL_REWARD_WEIGHT,
            meta_learning_enabled: false,
            meta_lr: DEFAULT_META_LR,
            curriculum_domains: vec![
                "navigation".to_string(),
                "crafting".to_string(),
                "social".to_string(),
                "combat".to_string(),
            ],
            memory: MemoryConfig::default(),
            social: SocialConfig::default(),
            cognitive: CognitiveConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IntegrationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.curriculum_domains.len(), 4);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = IntegrationConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deser: IntegrationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.curriculum_domains, config.curriculum_domains);
    }
}
