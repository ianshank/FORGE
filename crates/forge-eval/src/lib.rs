#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-eval
//!
//! Agent-agnostic evaluation harness and benchmark suite for the FORGE platform.
//!
//! This crate provides:
//! - **Harness** ([`harness`]): Runs any [`AgentInterface`](forge_types::AgentInterface)
//!   against FORGE scenarios, collecting per-episode results.
//! - **Scorecard** ([`scorecard`]): Aggregated evaluation results across scenarios
//!   and difficulty tiers, suitable for leaderboards and reports.
//! - **Config** ([`config`]): Evaluation configuration with sensible defaults.
//!
//! # Architecture
//!
//! The evaluation harness uses a factory pattern (`Fn() -> Box<dyn AgentInterface>`)
//! to create fresh agent instances per episode, enabling safe parallel evaluation
//! via rayon. All constants flow through [`EvalConfig`](config::EvalConfig).

pub mod config;
pub mod harness;
pub mod scorecard;
