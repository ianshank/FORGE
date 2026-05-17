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
//! ## Zero-alloc carve-out
//!
//! [`MinecraftEnv`] does **not** implement [`forge_env::StepInto`]. Every
//! step performs WebSocket I/O (JSON parse → fresh `Vec<f32>`), which is
//! fundamentally incompatible with the FORGE zero-alloc hot-path
//! contract. CI's allocation audit excludes this crate's module path.
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
