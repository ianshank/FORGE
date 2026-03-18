//! Server metrics collection and reporting.
//!
//! The `MetricsCollector` tracks simulation ticks.
//! Use `snapshot()` to get a point-in-time `ServerMetrics` for API responses.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// A point-in-time snapshot of server metrics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerMetrics {
    /// Total number of simulation ticks processed.
    pub simulation_ticks: u64,
    /// Current simulation steps per second.
    pub steps_per_second: f64,
    /// Number of active WebSocket connections.
    pub ws_connections: u32,
    /// Server uptime in seconds.
    pub uptime_seconds: u64,
}

/// Training metrics pushed from the Python training loop via REST.
///
/// These are forwarded to WebSocket clients so the dashboard can render
/// live training charts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingMetrics {
    /// Current training episode number.
    #[serde(default)]
    pub episode: u64,
    /// Total environment steps taken.
    #[serde(default)]
    pub total_steps: u64,
    /// Mean reward over recent episodes.
    #[serde(default)]
    pub mean_reward: f64,
    /// Win rate over recent episodes (0.0 to 1.0).
    #[serde(default)]
    pub win_rate: f64,
    /// Current curriculum difficulty level (0.0 to 1.0).
    #[serde(default)]
    pub curriculum_difficulty: f64,
    /// Environment steps per second throughput.
    #[serde(default)]
    pub steps_per_second: f64,
    /// Policy loss from the most recent update.
    #[serde(default)]
    pub loss_policy: f64,
    /// Value loss from the most recent update.
    #[serde(default)]
    pub loss_value: f64,
    /// Entropy from the most recent update.
    #[serde(default)]
    pub entropy: f64,
}

/// Decision trace entry pushed from the Python training loop.
///
/// Forwarded to dashboard clients for live decision reasoning display.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecisionTraceEntry {
    /// Which agent produced this trace.
    #[serde(default)]
    pub agent_id: u32,
    /// Simulation tick at which the decision was made.
    #[serde(default)]
    pub tick: u64,
    /// Human-readable intent label.
    #[serde(default)]
    pub intent_label: String,
    /// Agent confidence in this decision (0.0 to 1.0).
    #[serde(default)]
    pub confidence: f64,
    /// MCTS search depth that produced this decision.
    #[serde(default)]
    pub search_depth: u32,
    /// UCB1 score of the selected action.
    #[serde(default)]
    pub ucb1_score: f64,
    /// Number of alternative actions considered.
    #[serde(default)]
    pub alternatives_considered: u32,
}

/// Collects and aggregates server metrics over time.
#[derive(Debug)]
pub struct MetricsCollector {
    simulation_ticks: u64,
    last_snapshot_time: Instant,
    ticks_since_last_snapshot: u64,
    steps_per_second: f64,
}

impl MetricsCollector {
    /// Creates a new `MetricsCollector` with zeroed counters.
    #[instrument]
    pub fn new() -> Self {
        tracing::debug!("Creating new MetricsCollector");
        Self {
            simulation_ticks: 0,
            last_snapshot_time: Instant::now(),
            ticks_since_last_snapshot: 0,
            steps_per_second: 0.0,
        }
    }

    /// Records that a simulation tick has occurred.
    #[instrument(skip(self))]
    pub fn record_tick(&mut self) {
        self.simulation_ticks += 1;
        self.ticks_since_last_snapshot += 1;
        tracing::trace!(ticks = self.simulation_ticks, "Recorded simulation tick");
    }

    /// Returns a snapshot of the current server metrics.
    ///
    /// Also updates the `steps_per_second` rate based on ticks since the
    /// last snapshot call.
    #[instrument(skip(self))]
    pub fn snapshot(&mut self) -> ServerMetrics {
        tracing::trace!("Taking metrics snapshot");

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_snapshot_time).as_secs_f64();
        if elapsed > 0.0 {
            self.steps_per_second = self.ticks_since_last_snapshot as f64 / elapsed;
        }
        self.ticks_since_last_snapshot = 0;
        self.last_snapshot_time = now;

        ServerMetrics {
            simulation_ticks: self.simulation_ticks,
            steps_per_second: self.steps_per_second,
            ws_connections: 0,
            uptime_seconds: 0,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_counting() {
        let mut collector = MetricsCollector::new();
        assert_eq!(collector.snapshot().simulation_ticks, 0);

        collector.record_tick();
        collector.record_tick();
        collector.record_tick();

        let metrics = collector.snapshot();
        assert_eq!(metrics.simulation_ticks, 3);
    }

    #[test]
    fn test_metrics_serialization() {
        let metrics = ServerMetrics {
            simulation_ticks: 100,
            steps_per_second: 10.5,
            ws_connections: 3,
            uptime_seconds: 60,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("simulationTicks"));
        assert!(json.contains("stepsPerSecond"));
        assert!(json.contains("wsConnections"));
        assert!(json.contains("uptimeSeconds"));
    }

    #[test]
    fn test_training_metrics_serialization() {
        let metrics = TrainingMetrics {
            episode: 50,
            total_steps: 10000,
            mean_reward: 2.75,
            loss_policy: 0.01,
            loss_value: 0.02,
            entropy: 1.5,
            ..Default::default()
        };
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("meanReward"));
        assert!(json.contains("lossPolicy"));
        assert!(json.contains("lossValue"));
        assert!(json.contains("totalSteps"));
    }

    #[test]
    fn test_training_metrics_defaults() {
        let json = "{}";
        let metrics: TrainingMetrics = serde_json::from_str(json).unwrap();
        assert_eq!(metrics.episode, 0);
        assert_eq!(metrics.mean_reward, 0.0);
    }

    #[test]
    fn test_decision_trace_entry_serialization() {
        let entry = DecisionTraceEntry {
            agent_id: 1,
            tick: 42,
            intent_label: "flank_east".to_string(),
            confidence: 0.85,
            search_depth: 5,
            ucb1_score: 1.23,
            alternatives_considered: 4,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("intentLabel"));
        assert!(json.contains("ucb1Score"));
        assert!(json.contains("searchDepth"));
    }

    #[test]
    fn test_decision_trace_entry_defaults() {
        let json = "{}";
        let entry: DecisionTraceEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.agent_id, 0);
        assert_eq!(entry.tick, 0);
        assert!(entry.intent_label.is_empty());
        assert_eq!(entry.confidence, 0.0);
    }

    #[test]
    fn test_decision_trace_entry_roundtrip() {
        let entry = DecisionTraceEntry {
            agent_id: 3,
            tick: 100,
            intent_label: "explore".to_string(),
            confidence: 0.75,
            search_depth: 10,
            ucb1_score: 3.15,
            alternatives_considered: 7,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deser: DecisionTraceEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.agent_id, 3);
        assert_eq!(deser.tick, 100);
        assert_eq!(deser.intent_label, "explore");
        assert_eq!(deser.alternatives_considered, 7);
    }

    #[test]
    fn test_server_metrics_default() {
        let metrics = ServerMetrics::default();
        assert_eq!(metrics.simulation_ticks, 0);
        assert_eq!(metrics.steps_per_second, 0.0);
        assert_eq!(metrics.ws_connections, 0);
        assert_eq!(metrics.uptime_seconds, 0);
    }

    #[test]
    fn test_server_metrics_roundtrip() {
        let metrics = ServerMetrics {
            simulation_ticks: 500,
            steps_per_second: 25.0,
            ws_connections: 3,
            uptime_seconds: 120,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let deser: ServerMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.simulation_ticks, 500);
        assert_eq!(deser.ws_connections, 3);
    }

    #[test]
    fn test_metrics_collector_default() {
        let mut collector = MetricsCollector::default();
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.simulation_ticks, 0);
    }

    #[test]
    fn test_metrics_collector_steps_per_second() {
        let mut collector = MetricsCollector::new();
        for _ in 0..10 {
            collector.record_tick();
        }
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.simulation_ticks, 10);
        // steps_per_second should be positive after recording ticks
        assert!(snapshot.steps_per_second > 0.0);
    }

    #[test]
    fn test_metrics_collector_snapshot_resets_rate() {
        let mut collector = MetricsCollector::new();
        collector.record_tick();
        collector.record_tick();
        let _first = collector.snapshot();

        // After snapshot, ticks_since_last_snapshot is reset
        let second = collector.snapshot();
        assert_eq!(second.simulation_ticks, 2); // Total unchanged
                                                // steps_per_second should be 0 or very low since no ticks recorded
        assert!(second.steps_per_second < 1.0);
    }

    #[test]
    fn test_training_metrics_roundtrip() {
        let metrics = TrainingMetrics {
            episode: 100,
            total_steps: 50000,
            mean_reward: 3.5,
            win_rate: 0.65,
            curriculum_difficulty: 0.8,
            steps_per_second: 1000.0,
            loss_policy: 0.05,
            loss_value: 0.1,
            entropy: 2.0,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let deser: TrainingMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.episode, 100);
        assert_eq!(deser.win_rate, 0.65);
        assert_eq!(deser.entropy, 2.0);
    }
}
