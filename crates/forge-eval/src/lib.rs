#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-eval
//!
//! Evaluation harness and benchmark suite for the FORGE platform.
//!
//! This crate provides:
//! - **Harness** ([`harness`]): Runs agent-agnostic and privileged agents against
//!   single scenarios or registry-backed scenario suites, collecting per-episode results.
//! - **Scorecard** ([`scorecard`]): Aggregated evaluation results across scenarios
//!   and difficulty tiers, suitable for leaderboards and reports.
//! - **Config** ([`config`]): Evaluation configuration with sensible defaults.
//!
//! # Architecture
//!
//! The primary evaluation harness uses a factory pattern
//! (`Fn() -> Box<dyn AgentInterface>`) to create fresh agent instances per episode,
//! enabling safe parallel evaluation via rayon. A privileged fast path built on the
//! legacy `forge_agent::baselines::Agent` trait is also available when clone-free
//! access to [`forge_core::WorldState`] is required. All constants flow through
//! [`EvalConfig`](config::EvalConfig).

pub mod config;
pub mod harness;
pub mod scorecard;
