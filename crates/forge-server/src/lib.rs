//! WebSocket and REST server for FORGE simulation visualization.

pub mod api;
pub mod metrics;
pub mod state;
pub mod ws_handler;

/// Current schema version for client compatibility checks.
pub const SCHEMA_VERSION: u32 = 1;
