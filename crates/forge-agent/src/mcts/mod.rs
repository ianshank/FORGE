//! Monte Carlo Tree Search (MCTS) implementation.
//!
//! Provides a configurable MCTS planner using the PUCT (Predictor + Upper
//! Confidence bounds applied to Trees) algorithm. The search uses the
//! forward model to simulate rollouts and backpropagates value estimates.

pub mod policy;
pub mod search;
pub mod tree;
