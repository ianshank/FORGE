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
//! - **Hierarchical skills** (`skills`): Options/HRL catalog executor over primitive actions
//! - **Latent MCTS** (`latent_mcts`): MuZero-style tree search in learned latent space

pub mod adapter;
pub mod baselines;
pub mod forward_model;
pub mod latent_mcts;
pub mod mcts;
pub mod prelude;
pub mod skills;

// Crate-root convenience re-exports of the most-used types so
// downstream consumers can write `forge_agent::OnnxReloadError`
// instead of the three-segment `forge_agent::latent_mcts::
// onnx_model::OnnxReloadError`. The ONNX surface is feature-gated.
pub use crate::latent_mcts::model::{LatentForwardModel, LatentInferenceOutput, StubLatentModel};
#[cfg(feature = "onnx")]
pub use crate::latent_mcts::onnx_model::{
    validate_reload_paths, OnnxModelConfig, OnnxMuZeroModel, OnnxReloadError,
};
pub use crate::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch, LatentSearchResult};
pub use crate::latent_mcts::state::LatentState;
