//! Wire protocol between the Rust client and the Node `mc-bot`.
//!
//! All messages are JSON-encoded WebSocket text frames. Binary frames
//! are reserved for a future protocol v2 and intentionally unsupported
//! in v1 (see v2 plan, §3.3.3).

use serde::{Deserialize, Serialize};

/// Pin this constant when introducing a backwards-incompatible change.
/// The bot's `Hello` reply must match.
pub const SCHEMA_VERSION: u32 = 1;

/// Bot is tearing down / rebuilding mineflayer. Completes the current
/// request–response pair; the MDP episode is **not** resumable.
///
/// MUST stay in sync with `mc-bot/src/protocol.ts::ERROR_CODE_RECONNECTING`.
pub const ERROR_CODE_RECONNECTING: &str = "RECONNECTING";

/// Bot already has a client attached. Completes the pair; treat as
/// transient at the runner (discard episode, continue the run).
///
/// MUST stay in sync with `mc-bot/src/protocol.ts::ERROR_CODE_BUSY`.
pub const ERROR_CODE_BUSY: &str = "BUSY";

/// True for protocol error codes that end the current episode without
/// failing the whole run. `INTERNAL`, `INVALID_ACTION`, `BAD_MESSAGE`,
/// and unknown codes are **not** transient.
#[must_use]
pub fn is_transient_error_code(code: &str) -> bool {
    code == ERROR_CODE_RECONNECTING || code == ERROR_CODE_BUSY
}

/// Frozen channel order for the block-grid observation prefix.
///
/// MUST stay in sync with the JS-side
/// `mc-bot/src/observation_grid.ts::BLOCK_FEATURE_CHANNELS` —
/// reordering on either side silently mis-trains the CNN. Coordinated
/// tests live in `xlang_block_feature_channels_pinned_to_known_good`
/// (this side) and
/// `mc-bot/test/observation_grid.test.ts::feature channel order pin`
/// (JS side); drift fails both tests simultaneously.
pub const BLOCK_FEATURE_CHANNELS: [&str; 7] = [
    "block_type_hash",
    "light_level",
    "hardness",
    "is_solid",
    "is_liquid",
    "is_dangerous",
    "biome_id_hash",
];

/// Spatial layout of the block-grid prefix in the observation vector.
///
/// When present in the bot's `Hello`, the client cross-checks each
/// dimension against `config.observation.expected_grid_shape` and
/// refuses to start on mismatch. This catches a regression where the
/// JS encoder advertises the same flat `obs_dim` but reorders the
/// grid axes (silently mis-training the CNN).
///
/// `vector_dim` is the size of the non-grid suffix in the same flat
/// observation buffer — i.e. `obs_dim == height * width * depth *
/// channels + vector_dim` must hold when grid_shape is `Some`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridShape {
    /// X-axis ego-centric tile count (2 * grid_radius + 1 on the bot).
    pub height: u32,
    /// Z-axis ego-centric tile count (2 * grid_radius + 1 on the bot).
    pub width: u32,
    /// Y-axis ego-centric tile count (2 * grid_height_radius + 1).
    pub depth: u32,
    /// Per-tile feature channel count.
    pub channels: u32,
    /// Non-grid flat-vector dim that follows the grid in `obs`.
    /// Operators may set this to `0` when the bot emits only the grid.
    #[serde(default)]
    pub vector_dim: u32,
}

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
        /// Optional spatial layout of the block-grid prefix in the
        /// observation buffer. `None` keeps the legacy flat-only
        /// contract for bots that don't emit a grid.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grid_shape: Option<GridShape>,
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
            grid_shape: None,
        };
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"type\":\"hello\""));
        assert!(j.contains("\"obs_dim\":960"));
        // grid_shape:None must be elided so legacy bots stay
        // wire-compatible with this serializer's output.
        assert!(!j.contains("grid_shape"));
        let back: ServerMsg = serde_json::from_str(&j).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn server_hello_with_grid_shape_roundtrip() {
        let m = ServerMsg::Hello {
            schema_version: SCHEMA_VERSION,
            action_count: 12,
            obs_dim: 920,
            schema_id: "abc".to_string(),
            grid_shape: Some(GridShape {
                height: 11,
                width: 11,
                depth: 11,
                channels: 7,
                vector_dim: 73,
            }),
        };
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"grid_shape\""));
        assert!(j.contains("\"channels\":7"));
        let back: ServerMsg = serde_json::from_str(&j).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn server_hello_accepts_grid_shape_without_vector_dim() {
        // Bots may omit `vector_dim` entirely when the obs is grid-only;
        // serde_default must fill it in as 0 so the wire stays
        // backwards-compatible.
        let raw = r#"{
            "type": "hello",
            "schema_version": 1,
            "action_count": 4,
            "obs_dim": 847,
            "schema_id": "abc",
            "grid_shape": {
                "height": 11, "width": 11, "depth": 11, "channels": 7
            }
        }"#;
        let back: ServerMsg = serde_json::from_str(raw).unwrap();
        match back {
            ServerMsg::Hello {
                grid_shape: Some(g),
                ..
            } => {
                assert_eq!(g.vector_dim, 0);
                assert_eq!(g.channels, 7);
            }
            other => panic!("expected Hello with grid_shape, got {other:?}"),
        }
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

    /// Pinned cross-language regression gate. This exact value MUST
    /// equal `mc-bot/src/protocol.ts::SCHEMA_VERSION`, verified by the
    /// JS-side counterpart test:
    /// `mc-bot/test/protocol.test.ts::xlang_schema_version_matches_rust`.
    ///
    /// If you bump the protocol, change BOTH constants in the same PR;
    /// otherwise both this test and the JS-side counterpart will fail.
    #[test]
    fn xlang_schema_version_pinned_to_known_good() {
        assert_eq!(
            SCHEMA_VERSION, 1,
            "protocol SCHEMA_VERSION drift — mc-bot/src/protocol.ts \
             SCHEMA_VERSION must also be bumped and its xlang test \
             updated in the same PR"
        );
    }

    /// Pinned cross-language channel-order regression gate. This exact
    /// order MUST equal
    /// `mc-bot/src/observation_grid.ts::BLOCK_FEATURE_CHANNELS`;
    /// the JS-side counterpart pins the same names via
    /// `mc-bot/test/observation_grid.test.ts`.
    ///
    /// Reordering on either side silently mis-trains the CNN — drift
    /// fails both this test and the JS-side counterpart simultaneously.
    #[test]
    fn xlang_block_feature_channels_pinned_to_known_good() {
        assert_eq!(
            BLOCK_FEATURE_CHANNELS,
            [
                "block_type_hash",
                "light_level",
                "hardness",
                "is_solid",
                "is_liquid",
                "is_dangerous",
                "biome_id_hash",
            ],
            "BLOCK_FEATURE_CHANNELS drift — coordinate with \
             mc-bot/src/observation_grid.ts + its JS-side pin test"
        );
        assert_eq!(BLOCK_FEATURE_CHANNELS.len(), 7);
    }

    #[test]
    fn is_transient_error_code_covers_reconnect_and_busy_only() {
        assert!(is_transient_error_code(ERROR_CODE_RECONNECTING));
        assert!(is_transient_error_code(ERROR_CODE_BUSY));
        assert!(!is_transient_error_code("INTERNAL"));
        assert!(!is_transient_error_code("INVALID_ACTION"));
        assert!(!is_transient_error_code("BAD_MESSAGE"));
        assert!(!is_transient_error_code("unknown"));
    }
}
