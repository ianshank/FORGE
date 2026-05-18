//! [`RunnerConfig`] — knobs that control the episode loop.
//!
//! All fields have `Default` values so a user can override only what
//! they need via TOML or `Deserialize`. No hard-coded constants leak
//! into the loop itself — every magic number lives here.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level config for `forge-mc-runner`.
///
/// The `Default` impl is suitable for local smoke runs against the
/// `mc-bot` mock server: 1 episode, short caps, in-process paths.
/// Production callers should override `episodes`, `trajectory_dir`,
/// `manifest_path`, and the planning budget.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RunnerConfig {
    /// Number of episodes to run. `0` means "run forever".
    pub episodes: u64,

    /// Hard ceiling on env steps per episode — truncates the episode
    /// even if the env hasn't reported `terminated`.
    pub max_steps_per_episode: u64,

    /// Where `TrajectoryV2` jsonl files are written. Created on first
    /// episode if missing. One file per episode (`episode_id.json`).
    pub trajectory_dir: PathBuf,

    /// Path to `model_manifest.json`. The hot-reload watcher polls this
    /// **only between episodes** for a strictly-monotonic version bump.
    pub manifest_path: PathBuf,

    /// Free-form env identifier embedded in trajectory headers.
    /// Typically `"minecraft"` or `"forge"`.
    pub env_id: String,

    /// `schema_id` the runner expects the env handshake to advertise.
    /// Cross-checked at startup; mismatch is a hard error.
    pub schema_id: String,

    /// Number of MCTS simulations per decision.
    /// `0` means "use the env's first valid action without searching" —
    /// useful for protocol smoke tests.
    pub planning_sims: u32,

    /// Hold the chosen action for this many env steps before
    /// re-planning. `1` re-plans every step.
    pub action_repeat: u32,

    /// Optional RNG seed for reproducibility. `None` means
    /// time-derived seeding per episode.
    pub base_seed: Option<u64>,

    /// TCP port for the runner's Prometheus `/metrics` endpoint.
    /// `0` disables the endpoint entirely (useful in tests).
    pub metrics_port: u16,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            episodes: 1,
            max_steps_per_episode: 1_000,
            trajectory_dir: PathBuf::from("trajectories"),
            manifest_path: PathBuf::from("models/model_manifest.json"),
            env_id: "minecraft".to_string(),
            schema_id: "unset".to_string(),
            planning_sims: 50,
            action_repeat: 1,
            base_seed: None,
            metrics_port: 9090,
        }
    }
}

impl RunnerConfig {
    /// Returns `true` iff `episodes == 0` (run forever).
    pub fn runs_forever(&self) -> bool {
        self.episodes == 0
    }

    /// Returns `true` iff metrics serving is disabled.
    pub fn metrics_disabled(&self) -> bool {
        self.metrics_port == 0
    }

    /// Validate cross-field invariants. Catches confusable misconfigs
    /// (e.g. `action_repeat == 0` would divide by zero in the runner).
    pub fn validate(&self) -> Result<(), String> {
        if self.action_repeat == 0 {
            return Err("action_repeat must be >= 1".into());
        }
        if self.env_id.is_empty() {
            return Err("env_id must be non-empty".into());
        }
        if self.schema_id.is_empty() {
            return Err("schema_id must be non-empty".into());
        }
        if self.trajectory_dir.as_os_str().is_empty() {
            return Err("trajectory_dir must be non-empty".into());
        }
        if self.manifest_path.as_os_str().is_empty() {
            return Err("manifest_path must be non-empty".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_valid() {
        let cfg = RunnerConfig::default();
        cfg.validate().unwrap();
        assert!(!cfg.runs_forever());
        assert!(!cfg.metrics_disabled());
        assert_eq!(cfg.episodes, 1);
        assert_eq!(cfg.action_repeat, 1);
    }

    #[test]
    fn runs_forever_when_episodes_zero() {
        let cfg = RunnerConfig {
            episodes: 0,
            ..RunnerConfig::default()
        };
        assert!(cfg.runs_forever());
        cfg.validate().unwrap();
    }

    #[test]
    fn metrics_port_zero_disables_metrics() {
        let cfg = RunnerConfig {
            metrics_port: 0,
            ..RunnerConfig::default()
        };
        assert!(cfg.metrics_disabled());
    }

    #[test]
    fn validate_rejects_zero_action_repeat() {
        let cfg = RunnerConfig {
            action_repeat: 0,
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("action_repeat"));
    }

    #[test]
    fn validate_rejects_empty_env_id() {
        let cfg = RunnerConfig {
            env_id: String::new(),
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("env_id"));
    }

    #[test]
    fn validate_rejects_empty_schema_id() {
        let cfg = RunnerConfig {
            schema_id: String::new(),
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("schema_id"));
    }

    #[test]
    fn validate_rejects_empty_trajectory_dir() {
        let cfg = RunnerConfig {
            trajectory_dir: PathBuf::new(),
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("trajectory_dir"));
    }

    #[test]
    fn validate_rejects_empty_manifest_path() {
        let cfg = RunnerConfig {
            manifest_path: PathBuf::new(),
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("manifest_path"));
    }

    /// `#[serde(default)]` means partial TOML reuses defaults for
    /// unspecified fields. Verifies the contract callers will rely on.
    #[test]
    fn deserialise_partial_toml_uses_defaults() {
        let toml_src = r#"
            episodes = 42
            env_id = "minecraft"
            schema_id = "abc123"
        "#;
        let cfg: RunnerConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(cfg.episodes, 42);
        assert_eq!(cfg.env_id, "minecraft");
        assert_eq!(cfg.schema_id, "abc123");
        // Untouched fields take Default values.
        assert_eq!(cfg.max_steps_per_episode, 1_000);
        assert_eq!(cfg.action_repeat, 1);
        assert_eq!(cfg.metrics_port, 9090);
    }

    #[test]
    fn json_roundtrip_preserves_all_fields() {
        let cfg = RunnerConfig {
            episodes: 7,
            max_steps_per_episode: 99,
            trajectory_dir: PathBuf::from("/tmp/t"),
            manifest_path: PathBuf::from("/tmp/m.json"),
            env_id: "forge".into(),
            schema_id: "deadbeef".into(),
            planning_sims: 25,
            action_repeat: 4,
            base_seed: Some(123),
            metrics_port: 0,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: RunnerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }
}
