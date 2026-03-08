#![deny(missing_docs)]
#![deny(clippy::all)]
//! WebSocket and REST server for FORGE simulation visualization.
//!
//! This crate provides an Axum-based HTTP+WebSocket server that broadcasts
//! live simulation state to connected dashboard clients.

pub mod api;
pub mod config;
pub mod metrics;
pub mod state;
pub mod ws_handler;

/// Current schema version for client compatibility checks.
///
/// Increment this when making breaking changes to the `SimulationSnapshot`
/// or `WsMessage` JSON schema so clients can detect version mismatches.
pub const SCHEMA_VERSION: u32 = 1;
