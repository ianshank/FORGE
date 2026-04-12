//! Latent-space MCTS for MuZero-style planning.
//!
//! This module provides a Monte Carlo Tree Search implementation that operates
//! on learned latent states instead of full [`WorldState`] objects. It is
//! designed to work with a MuZero world model that provides:
//!
//! - **Initial inference**: observation → (latent state, policy, value)
//! - **Recurrent inference**: (latent state, action) → (next latent, reward, policy, value)
//!
//! The search algorithm is identical to standard MCTS (PUCT selection,
//! backpropagation with discounting) but carries compact latent vectors
//! through the tree instead of cloning the full simulation state.
//!
//! # Architecture
//!
//! The [`LatentForwardModel`] trait abstracts the neural network inference,
//! allowing different backends (ONNX Runtime, PyTorch via FFI, mock models).
//! The [`LatentMctsSearch`] is generic over this trait.
//!
//! # Feature flags
//!
//! - `onnx`: Enables the ONNX Runtime backend ([`OnnxMuZeroModel`]).

pub mod model;
#[cfg(feature = "onnx")]
pub mod onnx_model;
pub mod search;
pub mod state;
