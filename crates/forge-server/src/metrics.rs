//! Server metrics collection and reporting.
//!
//! The `MetricsCollector` tracks simulation ticks.
//! Use `snapshot()` to get a point-in-time `ServerMetrics` for API responses.

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

/// Collects and aggregates server metrics over time.
#[derive(Debug)]
pub struct MetricsCollector {
    simulation_ticks: u64,
    steps_per_second: f64,
}

impl MetricsCollector {
    /// Creates a new `MetricsCollector` with zeroed counters.
    #[instrument]
    pub fn new() -> Self {
        tracing::debug!("Creating new MetricsCollector");
        Self {
            simulation_ticks: 0,
            steps_per_second: 0.0,
        }
    }

    /// Records that a simulation tick has occurred.
    #[instrument(skip(self))]
    pub fn record_tick(&mut self) {
        self.simulation_ticks += 1;
        tracing::trace!(ticks = self.simulation_ticks, "Recorded simulation tick");
    }

    /// Returns a snapshot of the current server metrics.
    #[instrument(skip(self))]
    pub fn snapshot(&self) -> ServerMetrics {
        tracing::trace!("Taking metrics snapshot");
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
}
