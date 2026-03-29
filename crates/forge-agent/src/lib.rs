#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-agent
//!
//! Agent framework, MCTS planning engine, and baseline agents
//! for the FORGE platform.
//!
//! This crate provides:
//! - **Forward model** (`forward_model`): Simulation interface for planning
//! - **MCTS** (`mcts`): Monte Carlo Tree Search with PUCT selection
//! - **Baselines** (`baselines`): Random, heuristic, and greedy baseline agents

pub mod adapter;
pub mod baselines;
pub mod forward_model;
pub mod mcts;
pub mod prelude;
