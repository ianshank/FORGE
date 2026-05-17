#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-env-mc
//!
//! Minecraft environment for FORGE, implementing [`forge_env::Env`].
//!
//! Talks over WebSocket to a Node.js `mc-bot` driving a mineflayer bot
//! against a Minecraft server. The bot computes observations and rewards
//! per the loaded `action_map.toml` + `rewards.toml`; this crate just
//! marshals messages on the wire.
//!
//! ## Zero-alloc note
//!
//! [`MinecraftEnv`]'s `reset_into` and `step_into` reuse the caller's
//! `Vec<f32>` buffer for the observation payload, satisfying the
//! buffer-filling contract. Network I/O (WebSocket read, JSON parse) still
//! allocates; this is unavoidable and excluded from the CI zero-alloc gate.
//!
//! ## Protocol
//!
//! See [`protocol`] for the wire format. Version pinned by
//! [`protocol::SCHEMA_VERSION`].

pub mod action_map;
pub mod client;
pub mod config;
pub mod error;
pub mod mc_env;
pub mod protocol;
pub mod reward_config;

pub use action_map::{ActionEntry, ActionKind, ActionMap};
pub use client::ProtocolClient;
pub use config::MinecraftEnvConfig;
pub use error::McEnvError;
pub use mc_env::{MinecraftEnv, MinecraftStepInfo};
pub use protocol::{ClientMsg, ServerMsg, SCHEMA_VERSION};
pub use reward_config::{combined_schema_id, RewardConfig};
