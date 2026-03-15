//! Configuration for the social interaction system.

use serde::{Deserialize, Serialize};

/// Default initial trust value between agents.
const DEFAULT_TRUST_INITIAL: f32 = 0.5;
/// Default trust update rate.
const DEFAULT_TRUST_UPDATE_RATE: f32 = 0.1;
/// Default cooperation reward weight.
const DEFAULT_COOPERATION_REWARD_WEIGHT: f32 = 0.3;
/// Default betrayal penalty.
const DEFAULT_BETRAYAL_PENALTY: f32 = -0.5;
/// Default alliance formation threshold.
const DEFAULT_ALLIANCE_THRESHOLD: f32 = 0.7;
/// Default maximum number of alliances per agent.
const DEFAULT_MAX_ALLIANCES: u32 = 3;

/// Configuration for the social interaction system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SocialConfig {
    /// Whether the social system is enabled.
    pub enabled: bool,
    /// Initial trust between any two agents.
    pub trust_initial: f32,
    /// Learning rate for trust updates.
    pub trust_update_rate: f32,
    /// Weight of cooperation-based rewards in total reward.
    pub cooperation_reward_weight: f32,
    /// Penalty applied when an agent betrays an alliance.
    pub betrayal_penalty: f32,
    /// Minimum trust required to form an alliance.
    pub alliance_threshold: f32,
    /// Maximum number of alliances per agent.
    pub max_alliances: u32,
}

impl Default for SocialConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            trust_initial: DEFAULT_TRUST_INITIAL,
            trust_update_rate: DEFAULT_TRUST_UPDATE_RATE,
            cooperation_reward_weight: DEFAULT_COOPERATION_REWARD_WEIGHT,
            betrayal_penalty: DEFAULT_BETRAYAL_PENALTY,
            alliance_threshold: DEFAULT_ALLIANCE_THRESHOLD,
            max_alliances: DEFAULT_MAX_ALLIANCES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SocialConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.trust_initial, DEFAULT_TRUST_INITIAL);
    }

    #[test]
    fn test_serialization_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(SocialConfig);
    }

    #[test]
    fn test_defaults_valid() {
        forge_types::assert_config_defaults_valid!(SocialConfig);
    }
}
