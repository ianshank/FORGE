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
        forge_types::assert_config_serde_roundtrip!(IntegrationConfig);
    }

    #[test]
    fn test_defaults_valid() {
        forge_types::assert_config_defaults_valid!(IntegrationConfig);
    }

    #[test]
    fn test_all_fields_accessible_with_custom_values() {
        let config = IntegrationConfig {
            enabled: true,
            memory_write_interval: 42,
            social_reward_weight: 0.75,
            meta_learning_enabled: true,
            meta_lr: 0.05,
            curriculum_domains: vec!["custom_domain".to_string()],
            memory: MemoryConfig::default(),
            social: SocialConfig::default(),
            cognitive: CognitiveConfig::default(),
        };
        assert!(config.enabled);
        assert_eq!(config.memory_write_interval, 42);
        assert_eq!(config.social_reward_weight, 0.75);
        assert!(config.meta_learning_enabled);
        assert_eq!(config.meta_lr, 0.05);
        assert_eq!(config.curriculum_domains, vec!["custom_domain".to_string()]);
        // Sub-configs are accessible
        let _mem = &config.memory;
        let _soc = &config.social;
        let _cog = &config.cognitive;
    }

    #[test]
    fn test_default_memory_write_interval() {
        let config = IntegrationConfig::default();
        assert_eq!(config.memory_write_interval, 10);
    }

    #[test]
    fn test_default_social_reward_weight() {
        let config = IntegrationConfig::default();
        assert!((config.social_reward_weight - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_meta_lr() {
        let config = IntegrationConfig::default();
        assert!((config.meta_lr - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_meta_learning_disabled() {
        let config = IntegrationConfig::default();
        assert!(!config.meta_learning_enabled);
    }

    #[test]
    fn test_curriculum_domains_default_content() {
        let config = IntegrationConfig::default();
        assert!(config
            .curriculum_domains
            .contains(&"navigation".to_string()));
        assert!(config.curriculum_domains.contains(&"crafting".to_string()));
        assert!(config.curriculum_domains.contains(&"social".to_string()));
        assert!(config.curriculum_domains.contains(&"combat".to_string()));
    }

    #[test]
    fn test_config_with_empty_curriculum_roundtrips() {
        let config = IntegrationConfig {
            curriculum_domains: vec![],
            ..IntegrationConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: IntegrationConfig = serde_json::from_str(&json).unwrap();
        assert!(deser.curriculum_domains.is_empty());
    }

    #[test]
    fn test_config_extreme_values_roundtrip() {
        let config = IntegrationConfig {
            enabled: true,
            memory_write_interval: u64::MAX,
            social_reward_weight: 1.0,
            meta_learning_enabled: true,
            meta_lr: 0.0,
            curriculum_domains: (0..100).map(|i| format!("domain_{i}")).collect(),
            ..IntegrationConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: IntegrationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.memory_write_interval, u64::MAX);
        assert_eq!(deser.curriculum_domains.len(), 100);
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let config = IntegrationConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deser: IntegrationConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(deser.memory_write_interval, config.memory_write_interval);
        assert_eq!(deser.social_reward_weight, config.social_reward_weight);
    }
}
