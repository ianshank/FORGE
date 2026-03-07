//! Server metrics collection and reporting.

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// A point-in-time snapshot of server metrics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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
    ws_connections: u32,
    uptime_seconds: u64,
}

impl MetricsCollector {
    /// Creates a new `MetricsCollector` with zeroed counters.
    #[instrument]
    pub fn new() -> Self {
        tracing::debug!("Creating new MetricsCollector");
        Self {
            simulation_ticks: 0,
            steps_per_second: 0.0,
            ws_connections: 0,
            uptime_seconds: 0,
        }
    }

    /// Records that a simulation tick has occurred.
    #[instrument(skip(self))]
    pub fn record_tick(&mut self) {
        self.simulation_ticks += 1;
        tracing::trace!(ticks = self.simulation_ticks, "Recorded simulation tick");
    }

    /// Records a new WebSocket client connection.
    #[instrument(skip(self))]
    pub fn record_ws_connect(&mut self) {
        self.ws_connections += 1;
        tracing::debug!(
            connections = self.ws_connections,
            "WebSocket client connected"
        );
    }

    /// Records a WebSocket client disconnection.
    #[instrument(skip(self))]
    pub fn record_ws_disconnect(&mut self) {
        self.ws_connections = self.ws_connections.saturating_sub(1);
        tracing::debug!(
            connections = self.ws_connections,
            "WebSocket client disconnected"
        );
    }

    /// Returns a snapshot of the current server metrics.
    #[instrument(skip(self))]
    pub fn snapshot(&self) -> ServerMetrics {
        tracing::trace!("Taking metrics snapshot");
        ServerMetrics {
            simulation_ticks: self.simulation_ticks,
            steps_per_second: self.steps_per_second,
            ws_connections: self.ws_connections,
            uptime_seconds: self.uptime_seconds,
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
    fn test_connection_tracking() {
        let mut collector = MetricsCollector::new();
        assert_eq!(collector.snapshot().ws_connections, 0);

        collector.record_ws_connect();
        collector.record_ws_connect();
        assert_eq!(collector.snapshot().ws_connections, 2);

        collector.record_ws_disconnect();
        assert_eq!(collector.snapshot().ws_connections, 1);

        // Verify saturating subtraction prevents underflow.
        collector.record_ws_disconnect();
        collector.record_ws_disconnect();
        assert_eq!(collector.snapshot().ws_connections, 0);
    }
}
