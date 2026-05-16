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
//! - **Scenario suite** ([`scenario`]): Composable scenario definitions and
//!   on-disk suites loaded from TOML.
//! - **Output** ([`output`]): Persistence of replays, trajectories, and scorecards.
//! - **Config** ([`config`]): Evaluation configuration with sensible defaults.
//!
//! # Architecture
//!
//! The evaluation harness uses a factory pattern (`Fn() -> Box<dyn AgentInterface>`)
//! to create fresh agent instances per episode, enabling safe parallel evaluation
//! via rayon. All constants flow through [`EvalConfig`](config::EvalConfig).

pub mod config;
pub mod harness;
pub mod output;
pub mod scenario;
pub mod scorecard;

pub use output::{OutputConfig, ScorecardFormat};
pub use scenario::{Scenario, ScenarioError, ScenarioSuite};
