//! Minecraft env configuration. All values are config-driven; nothing
//! is hardcoded inline outside of `Default` impls (and even those use
//! named constants).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::protocol::GridShape;

/// Top-level env config. Loaded from `configs/minecraft/env.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinecraftEnvConfig {
    /// Where to find the action map TOML.
    #[serde(default = "default_action_map_path")]
    pub action_map_path: PathBuf,
    /// WebSocket connection URL (mc-bot listens here).
    #[serde(default = "default_ws_url")]
    pub ws_url: String,
    /// Receive timeout per message — drops the connection if a tick
    /// produces no observation within this window.
    #[serde(default = "default_heartbeat_ms")]
    pub heartbeat_ms: u64,
    /// Expected bot-reported schema id. When set, the handshake refuses
    /// action-map/reward-config drift before any episode starts.
    #[serde(default)]
    pub expected_schema_id: Option<String>,
    /// Episode safety knobs.
    #[serde(default)]
    pub episode: EpisodeConfig,
    /// Observation knobs.
    #[serde(default)]
    pub observation: ObservationConfig,
}

impl Default for MinecraftEnvConfig {
    fn default() -> Self {
        Self {
            action_map_path: default_action_map_path(),
            ws_url: default_ws_url(),
            heartbeat_ms: default_heartbeat_ms(),
            expected_schema_id: None,
            episode: EpisodeConfig::default(),
            observation: ObservationConfig::default(),
        }
    }
}

fn default_action_map_path() -> PathBuf {
    PathBuf::from("configs/minecraft/action_map.toml")
}

fn default_ws_url() -> String {
    "ws://127.0.0.1:8765".to_string()
}

fn default_heartbeat_ms() -> u64 {
    2_000
}

/// Episode safety / timing knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeConfig {
    /// Server-side cap on ticks before truncation.
    #[serde(default = "default_max_ticks")]
    pub max_ticks: u64,
    /// Bot repeats the most recent action for this many ticks while
    /// the planner thinks.
    #[serde(default = "default_action_repeat")]
    pub action_repeat: u32,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            max_ticks: default_max_ticks(),
            action_repeat: default_action_repeat(),
        }
    }
}

fn default_max_ticks() -> u64 {
    6_000
}
fn default_action_repeat() -> u32 {
    4
}

/// Observation knobs.
///
/// Field naming follows the JS-side `[observation]` table so a single
/// `env.toml` parses on both ends of the WebSocket — `grid_radius`,
/// `grid_height_radius`, `grid_channels`, etc. all match
/// `mc-bot/src/observation_grid.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationConfig {
    /// Ego-centric block radius in the X/Z plane. Plumbed through to
    /// the bot's `encodeBlockGrid` config — the Rust side records it
    /// for diagnostics / `expected_grid_shape` derivation.
    #[serde(default = "default_grid_radius", alias = "radius")]
    pub grid_radius: u32,
    /// Ego-centric Y-axis layer half-count. Total vertical layers =
    /// `2 * grid_height_radius + 1`.
    #[serde(default = "default_grid_height_radius")]
    pub grid_height_radius: u32,
    /// Per-tile feature-channel count. Must match the bot's
    /// `BLOCK_FEATURE_CHANNELS.length`.
    #[serde(default = "default_grid_channels")]
    pub grid_channels: u32,
    /// Whether the bot is expected to emit the block-grid prefix.
    /// When `false`, no `grid_shape` cross-check is performed.
    #[serde(default)]
    pub include_block_grid: bool,
    /// Optional zero-pad target for the flat (non-grid) suffix. The
    /// bot pads the raw flat vector up to this length so the trainer's
    /// `vector_dim` stays stable as new flat features arrive over time.
    #[serde(default)]
    pub flat_vector_dim: Option<usize>,
    /// Expected `obs_dim` — cross-checked against the bot's `Hello`.
    /// `None` means accept whatever the bot reports.
    #[serde(default)]
    pub expected_dim: Option<usize>,
    /// Expected `grid_shape` — cross-checked against the bot's `Hello`
    /// when set. Catches a regression where the bot advertises the
    /// same total `obs_dim` but reorders the grid axes.
    #[serde(default)]
    pub expected_grid_shape: Option<GridShape>,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            grid_radius: default_grid_radius(),
            grid_height_radius: default_grid_height_radius(),
            grid_channels: default_grid_channels(),
            include_block_grid: false,
            flat_vector_dim: None,
            expected_dim: None,
            expected_grid_shape: None,
        }
    }
}

fn default_grid_radius() -> u32 {
    5
}

fn default_grid_height_radius() -> u32 {
    // 0 → single Y-layer, matching MuZeroConfig.grid_flat_dim's 2D
    // (C, H, W) reshape. Set > 0 in env.toml for 3D variants
    // (requires a Conv3d branch on the Python side — deferred to
    // v0.5 Phase 2).
    0
}

fn default_grid_channels() -> u32 {
    7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_serde_roundtrip() {
        let c = MinecraftEnvConfig::default();
        let s = toml::to_string(&c).unwrap();
        let back: MinecraftEnvConfig = toml::from_str(&s).unwrap();
        assert_eq!(c.ws_url, back.ws_url);
        assert_eq!(c.heartbeat_ms, back.heartbeat_ms);
        assert_eq!(c.expected_schema_id, back.expected_schema_id);
        assert_eq!(c.episode.max_ticks, back.episode.max_ticks);
    }

    #[test]
    fn partial_toml_uses_defaults_for_omitted_fields() {
        let raw = r#"ws_url = "ws://other:1234""#;
        let c: MinecraftEnvConfig = toml::from_str(raw).unwrap();
        assert_eq!(c.ws_url, "ws://other:1234");
        // Defaults kick in for everything else.
        assert_eq!(c.heartbeat_ms, default_heartbeat_ms());
        assert_eq!(c.expected_schema_id, None);
        assert_eq!(c.episode.max_ticks, default_max_ticks());
        assert_eq!(c.observation.grid_radius, default_grid_radius());
        assert_eq!(
            c.observation.grid_height_radius,
            default_grid_height_radius()
        );
        assert_eq!(c.observation.grid_channels, default_grid_channels());
        assert!(!c.observation.include_block_grid);
        assert_eq!(c.observation.flat_vector_dim, None);
        assert_eq!(c.observation.expected_grid_shape, None);
    }

    #[test]
    fn legacy_radius_alias_still_parses() {
        // `radius = N` is the pre-v0.5 field name. The serde `alias`
        // attribute on `grid_radius` keeps old env.toml files loading
        // cleanly while new TOMLs use the explicit `grid_radius` key.
        let raw = r#"
            [observation]
            radius = 8
        "#;
        let c: MinecraftEnvConfig = toml::from_str(raw).unwrap();
        assert_eq!(c.observation.grid_radius, 8);
    }

    #[test]
    fn ships_default_env_toml_parses_with_expected_grid_shape() {
        // The shipped `configs/minecraft/env.toml` carries the v0.5
        // Phase 1 block-grid expectations. Loading it from disk and
        // checking the cross-checked fields pins the wire contract:
        // any future drift in the TOML field names or layout will
        // fail this test before it can ship.
        //
        let raw = crate::test_support::read_workspace_config("configs/minecraft/env.toml");
        let cfg: MinecraftEnvConfig = toml::from_str(&raw).expect("parse env.toml");

        assert!(cfg.observation.include_block_grid);
        assert_eq!(cfg.observation.grid_radius, 5);
        assert_eq!(cfg.observation.grid_height_radius, 0);
        assert_eq!(cfg.observation.grid_channels, 7);
        assert_eq!(cfg.observation.flat_vector_dim, Some(73));
        assert_eq!(cfg.observation.expected_dim, Some(920));
        let expected = cfg
            .observation
            .expected_grid_shape
            .expect("expected_grid_shape must be set in shipped env.toml");
        assert_eq!(expected.height, 11);
        assert_eq!(expected.width, 11);
        assert_eq!(expected.depth, 1);
        assert_eq!(expected.channels, 7);
        assert_eq!(expected.vector_dim, 73);
        // The advertised dims must add up to expected_dim — the
        // mc_env handshake derives + compares this directly.
        let derived = (expected.height as usize)
            * (expected.width as usize)
            * (expected.depth as usize)
            * (expected.channels as usize)
            + (expected.vector_dim as usize);
        assert_eq!(derived, 920);
    }

    #[test]
    fn observation_table_roundtrip_with_grid_shape() {
        let mut c = MinecraftEnvConfig::default();
        c.observation.include_block_grid = true;
        c.observation.expected_dim = Some(920);
        c.observation.flat_vector_dim = Some(73);
        c.observation.expected_grid_shape = Some(GridShape {
            height: 11,
            width: 11,
            depth: 11,
            channels: 7,
            vector_dim: 73,
        });
        let s = toml::to_string(&c).unwrap();
        let back: MinecraftEnvConfig = toml::from_str(&s).unwrap();
        assert!(back.observation.include_block_grid);
        assert_eq!(back.observation.expected_dim, Some(920));
        assert_eq!(back.observation.flat_vector_dim, Some(73));
        assert_eq!(
            back.observation.expected_grid_shape,
            c.observation.expected_grid_shape
        );
    }
}
