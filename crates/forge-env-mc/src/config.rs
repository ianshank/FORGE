//! Minecraft env configuration. All values are config-driven; nothing
//! is hardcoded inline outside of `Default` impls (and even those use
//! named constants).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationConfig {
    /// Ego-centric block radius. Bot decides exact obs construction.
    #[serde(default = "default_radius")]
    pub radius: u32,
    /// Expected `obs_dim` — cross-checked against the bot's `Hello`.
    /// `None` means accept whatever the bot reports.
    #[serde(default)]
    pub expected_dim: Option<usize>,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            radius: default_radius(),
            expected_dim: None,
        }
    }
}

fn default_radius() -> u32 {
    8
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
    }
}
