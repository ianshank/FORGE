//! Search and telemetry metrics for the edge runtime.
//!
//! These structs capture per-search performance data and telemetry
//! buffer state, enabling monitoring and tuning of edge deployment.

use serde::{Deserialize, Serialize};

/// Metrics from a single adaptive MCTS search.
///
/// Captures the simulation budget used, actual latency, and action
/// selected. Used for monitoring edge performance and tuning the
/// latency estimator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSearchMetrics {
    /// Number of MCTS simulations actually executed.
    pub simulations_used: u32,
    /// Actual wall-clock latency of the search in milliseconds.
    pub actual_latency_ms: f32,
    /// Configured latency budget in milliseconds.
    pub budget_ms: u32,
    /// Budget utilization ratio (actual_latency / budget). Values > 1.0
    /// indicate the budget was exceeded.
    pub budget_utilization: f32,
    /// Current per-simulation latency estimate in milliseconds.
    pub estimated_per_sim_ms: f32,
    /// The action ID selected by the search.
    pub action_selected: u32,
    /// Estimated value at the search root.
    pub root_value: f32,
}

/// Snapshot of telemetry collector state.
///
/// Provides a read-only view of the telemetry buffer for monitoring
/// dashboards and health checks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    /// Number of replays waiting to be flushed.
    pub pending_replays: usize,
    /// Current buffer usage in bytes.
    pub buffer_bytes: u64,
    /// Maximum buffer capacity in bytes.
    pub buffer_capacity_bytes: u64,
    /// Total replays recorded since creation.
    pub total_replays_recorded: u64,
    /// Total replays successfully flushed.
    pub total_replays_flushed: u64,
    /// Total flush operations that failed.
    pub total_flush_failures: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_search_metrics_serde_roundtrip() {
        let metrics = AdaptiveSearchMetrics {
            simulations_used: 50,
            actual_latency_ms: 42.5,
            budget_ms: 50,
            budget_utilization: 0.85,
            estimated_per_sim_ms: 0.85,
            action_selected: 3,
            root_value: 0.72,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let deser: AdaptiveSearchMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.simulations_used, 50);
        assert_eq!(deser.action_selected, 3);
        assert!((deser.budget_utilization - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_telemetry_snapshot_default() {
        let snap = TelemetrySnapshot::default();
        assert_eq!(snap.pending_replays, 0);
        assert_eq!(snap.buffer_bytes, 0);
        assert_eq!(snap.buffer_capacity_bytes, 0);
        assert_eq!(snap.total_replays_recorded, 0);
        assert_eq!(snap.total_replays_flushed, 0);
        assert_eq!(snap.total_flush_failures, 0);
    }

    #[test]
    fn test_telemetry_snapshot_serde_roundtrip() {
        let snap = TelemetrySnapshot {
            pending_replays: 5,
            buffer_bytes: 2048,
            buffer_capacity_bytes: 1_048_576,
            total_replays_recorded: 100,
            total_replays_flushed: 95,
            total_flush_failures: 2,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let deser: TelemetrySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.pending_replays, 5);
        assert_eq!(deser.buffer_bytes, 2048);
        assert_eq!(deser.total_replays_flushed, 95);
    }

    #[test]
    fn test_budget_utilization_calculation() {
        let actual_ms: f32 = 42.5;
        let budget_ms: u32 = 50;
        let utilization = actual_ms / budget_ms as f32;
        assert!((utilization - 0.85).abs() < 1e-5);
    }
}
