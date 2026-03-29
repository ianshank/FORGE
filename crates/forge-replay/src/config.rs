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
        let config = ReplayConfig {
            record_compact: false,
            record_trajectories: true,
            include_reasoning: true,
            format_version: 2,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: ReplayConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.record_compact, false);
        assert_eq!(deser.record_trajectories, true);
        assert_eq!(deser.include_reasoning, true);
        assert_eq!(deser.format_version, 2);
    }
}
