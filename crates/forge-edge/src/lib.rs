#![deny(missing_docs)]
#![deny(clippy::all)]
//! Edge deployment runtime for the FORGE platform.
//!
//! Provides adaptive latency-budgeted MCTS planning, telemetry collection,
//! and a composite EdgeAgent that implements AgentInterface for direct
//! compatibility with FORGE's evaluation and training infrastructure.

pub mod adaptive_mcts;
pub mod edge_agent;
pub mod latency;
pub mod metrics;
pub mod telemetry;

pub use adaptive_mcts::AdaptiveMctsSearch;
pub use edge_agent::EdgeAgent;
pub use latency::LatencyEstimator;
pub use metrics::{AdaptiveSearchMetrics, TelemetrySnapshot};
pub use telemetry::TelemetryCollector;
