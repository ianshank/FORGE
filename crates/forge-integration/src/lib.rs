#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-integration-layer
//!
//! > **Maturity**: `[Research]` — Research Stack Component — cognitive cross-layer orchestration.
//!
//! Cross-layer integration orchestrator for the FORGE platform.
//!
//! This crate wires together memory, cognition, and social layers into
//! a unified system that manages the full proto-Data agent lifecycle:
//!
//! - **Orchestrator** (`orchestrator`): Coordinates all subsystems per agent
//! - **Config** (`config`): Unified configuration combining all sub-configs
//! - **Metrics** (`metrics`): Integration-specific monitoring metrics
//!
//! # Architecture
//!
//! The integration layer ensures that memory, social state, and cognitive
//! processing are trained together — not bolted on separately. This follows
//! the lesson from Data's emotion chip: integration requires integrated training.

pub mod config;
/// External controller harness for ADK and DeerFlow.
pub mod controller;
/// DeerFlow sandbox super-agent harness.
pub mod deerflow;
/// Honcho memory mirror export service.
pub mod honcho;
pub mod metrics;
pub mod orchestrator;
pub mod prelude;
