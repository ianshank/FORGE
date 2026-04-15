//! Configuration for replay and trajectory recording.

use serde::{Deserialize, Serialize};

use crate::compact;

/// Configuration for replay recording behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplayConfig {
    /// Whether to record compact replays (seed + actions).
    pub record_compact: bool,
    /// Whether to record full trajectories (obs + action + reward).
    pub record_trajectories: bool,
    /// Whether to include reasoning traces in trajectory steps.
    pub include_reasoning: bool,
    /// Format version for forward compatibility.
    pub format_version: u32,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            record_compact: true,
            record_trajectories: false,
            include_reasoning: false,
            format_version: compact::FORMAT_VERSION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ReplayConfig::default();
        assert!(config.record_compact);
        assert!(!config.record_trajectories);
        assert!(!config.include_reasoning);
        assert_eq!(config.format_version, 1);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(ReplayConfig);
    }

    #[test]
    fn test_defaults_valid() {
        forge_types::assert_config_defaults_valid!(ReplayConfig);
    }

    #[test]
    fn test_config_serde_roundtrip_custom_values() {
        let config = ReplayConfig {
            record_compact: false,
            record_trajectories: true,
            include_reasoning: true,
            format_version: 2,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: ReplayConfig = serde_json::from_str(&json).unwrap();
        assert!(!deser.record_compact);
        assert!(deser.record_trajectories);
        assert!(deser.include_reasoning);
        assert_eq!(deser.format_version, 2);
    }

    #[test]
    fn test_config_bincode_roundtrip() {
        let config = ReplayConfig {
            record_compact: false,
            record_trajectories: true,
            include_reasoning: true,
            format_version: 1,
        };
        let bytes = bincode::serialize(&config).unwrap();
        let deser: ReplayConfig = bincode::deserialize(&bytes).unwrap();
        assert!(!deser.record_compact);
        assert!(deser.record_trajectories);
        assert!(deser.include_reasoning);
    }

    #[test]
    fn test_config_debug_impl() {
        let config = ReplayConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ReplayConfig"));
        assert!(debug_str.contains("record_compact"));
    }

    #[test]
    fn test_config_clone() {
        let config = ReplayConfig {
            record_compact: false,
            record_trajectories: true,
            include_reasoning: true,
            format_version: 42,
        };
        let cloned = config.clone();
        assert_eq!(cloned.record_compact, config.record_compact);
        assert_eq!(cloned.record_trajectories, config.record_trajectories);
        assert_eq!(cloned.include_reasoning, config.include_reasoning);
        assert_eq!(cloned.format_version, config.format_version);
    }
}
