//! Integration-specific metrics for monitoring the proto-Data system.

use serde::{Deserialize, Serialize};

/// Metrics snapshot for the integration layer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    /// Total memory entries across all agents.
    pub total_memory_entries: usize,
    /// Mean trust across all agent pairs.
    pub mean_trust: f32,
    /// Mean reputation across all agents.
    pub mean_reputation: f32,
    /// Number of active alliances.
    pub active_alliances: usize,
    /// Social reward fraction of total reward.
    pub social_reward_fraction: f32,
    /// Current integration tick.
    pub tick: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_metrics() {
        let m = IntegrationMetrics::default();
        assert_eq!(m.total_memory_entries, 0);
        assert_eq!(m.mean_trust, 0.0);
    }

    #[test]
    fn test_serialization() {
        let m = IntegrationMetrics {
            total_memory_entries: 100,
            mean_trust: 0.6,
            ..IntegrationMetrics::default()
        };
        let json = serde_json::to_string(&m).unwrap();
        let deser: IntegrationMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.total_memory_entries, 100);
    }
}
