#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-env-forge
//!
//! Backwards-compatible shim that exposes FORGE's `WorldState` via the
//! generic [`forge_env::Env`] trait.
//!
//! This crate is purely additive — `forge-python::ForgeEnv`,
//! `forge-agent::mcts`, and every other existing FORGE consumer of
//! `WorldState` keeps working unchanged. The new types here let Rust
//! code drive `WorldState` through the same env-agnostic interface
//! `latent_mcts` and (eventually) `forge-mc-runner` use.
//!
//! ## Single-agent assumption
//!
//! [`WorldEnv`] wraps a single agent inside the underlying multi-agent
//! `WorldState`. This mirrors the Python `ForgeEnv` Gymnasium binding,
//! which also exposes a single-agent surface. Multi-agent envs are
//! out of scope for v1 and will be added later via a `MultiAgentEnv`
//! variant that takes `Vec<Action>`.

pub mod config;
pub mod error;
pub mod flat_adapter;
pub mod world_env;

pub use config::FlatObsConfig;
pub use error::ForgeEnvError;
pub use flat_adapter::{FlatForgeEnv, ObsFlattener};
pub use world_env::WorldEnv;
