//! Wire protocol between the Rust client and the Node `mc-bot`.
//!
//! All messages are JSON-encoded WebSocket text frames. Binary frames
//! are reserved for a future protocol v2 and intentionally unsupported
//! in v1 (see v2 plan, §3.3.3).

use serde::{Deserialize, Serialize};

/// Pin this constant when introducing a backwards-incompatible change.
/// The bot's `Hello` reply must match.
pub const SCHEMA_VERSION: u32 = 1;

/// Messages sent from the Rust client to the Node bot.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    /// Begin a new episode. `seed` is honoured by deterministic setups
    /// (teleport-reset uses it for any randomised reset state).
    Reset {
        /// Optional seed.
        seed: Option<u64>,
    },
    /// Advance one step with the given discrete action index.
    Step {
        /// Action id from `action_map.toml`.
        action_id: u32,
    },
    /// Tear down the connection.
    Close,
}

/// Messages sent from the Node bot to the Rust client.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    /// Handshake reply with the bot's view of the schema.
    Hello {
        /// Protocol schema version.
        schema_version: u32,
        /// Discrete action count.
        action_count: u32,
        /// Flat observation dimension.
        obs_dim: usize,
        /// SHA256 of `(action_map_canonical, rewards_canonical, obs_layout)`.
        /// Client MUST refuse to start if this disagrees with its own.
        schema_id: String,
    },
    /// Per-tick observation, reward, terminal flags, and free-form info.
    Observation {
        /// Server tick at which the observation was produced.
        tick: u64,
        /// Flat observation vector.
        obs: Vec<f32>,
        /// Step reward computed by the bot's reward registry.
        reward: f32,
        /// Episode terminated naturally (e.g. goal reached, death).
        terminated: bool,
        /// Episode truncated artificially (max ticks).
        truncated: bool,
        /// Free-form diagnostic info — bot-defined.
        info: serde_json::Value,
    },
    /// Protocol-level error.
    Error {
        /// Short machine code (e.g. "INVALID_ACTION").
        code: String,
        /// Human description.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_reset_serde_roundtrip() {
        let m = ClientMsg::Reset { seed: Some(42) };
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"type\":\"reset\""), "got: {j}");
        let back: ClientMsg = serde_json::from_str(&j).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn client_step_serde_roundtrip() {
        let m = ClientMsg::Step { action_id: 7 };
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"action_id\":7"));
        let back: ClientMsg = serde_json::from_str(&j).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn server_hello_serde_roundtrip() {
        let m = ServerMsg::Hello {
            schema_version: SCHEMA_VERSION,
            action_count: 32,
            obs_dim: 960,
            schema_id: "abcd1234".to_string(),
        };
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"type\":\"hello\""));
        assert!(j.contains("\"obs_dim\":960"));
        let back: ServerMsg = serde_json::from_str(&j).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn server_observation_serde_roundtrip() {
        let m = ServerMsg::Observation {
            tick: 7,
            obs: vec![0.1, 0.2, 0.3],
            reward: 1.5,
            terminated: false,
            truncated: false,
            info: serde_json::json!({"event": "test"}),
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: ServerMsg = serde_json::from_str(&j).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn server_error_serde_roundtrip() {
        let m = ServerMsg::Error {
            code: "INVALID_ACTION".into(),
            message: "no such action_id".into(),
        };
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"type\":\"error\""));
        let back: ServerMsg = serde_json::from_str(&j).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn schema_version_is_pinned() {
        // If you change the protocol shape, bump SCHEMA_VERSION explicitly.
        assert_eq!(SCHEMA_VERSION, 1);
    }
}
