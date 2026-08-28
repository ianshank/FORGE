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
mod hash_util;
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

/// Test-only helpers shared across this crate's unit tests. Not compiled
/// into non-test builds.
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::{Path, PathBuf};

    /// Read a repo-relative config file (e.g. `configs/minecraft/env.toml`)
    /// from the workspace root, resolved two directories up from this crate's
    /// `CARGO_MANIFEST_DIR`. Panics (with the resolved path) on failure —
    /// intended only for the "shipped default config parses" pin tests.
    pub(crate) fn read_workspace_config(rel_path: &str) -> String {
        let workspace_root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root (two levels up from CARGO_MANIFEST_DIR)")
            .to_path_buf();
        let path = workspace_root.join(rel_path);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }
}
