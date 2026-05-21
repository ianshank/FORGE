//! [`RunnerConfig`] — knobs that control the episode loop.
//!
//! All fields have `Default` values so a user can override only what
//! they need via TOML or `Deserialize`. No hard-coded constants leak
//! into the loop itself — every magic number lives here.

use std::path::PathBuf;

use forge_replay::v2::TrajectoryGzipLevel;
use serde::{Deserialize, Serialize};

/// Knobs used by the `--dry-run` smoke mode in the runner binary.
///
/// Lifted into a struct (rather than inlined as literals at the call
/// site) so the CLI binary carries zero hard-coded values. Every
/// default below is overridable through TOML under the
/// `[dry_run]` table. The defaults match the shape of the
/// pre-extraction binary literals (`obs_dim = 8`, `action_count = 4`,
/// `latent_dim = 16`, `max_episode_len = 8`) so existing
/// `cargo run -- --dry-run` invocations behave identically.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct DryRunConfig {
    /// Observation dimensionality the stub env exposes.
    pub obs_dim: usize,
    /// Action space size the stub env exposes.
    pub action_count: u32,
    /// Latent dim of the stub model (must match the agent's
    /// `LatentMctsConfig::latent_dim` for non-stub deployments).
    pub latent_dim: usize,
    /// Hard cap on env steps per dry-run episode. The runner's own
    /// `max_steps_per_episode` is checked too; this field is the
    /// stub-env's internal `Some(max)` truncation parameter.
    pub max_episode_len: u64,
}

impl Default for DryRunConfig {
    fn default() -> Self {
        Self {
            obs_dim: 8,
            action_count: 4,
            latent_dim: 16,
            max_episode_len: 8,
        }
    }
}

/// Compression codec for trajectory files written by
/// [`crate::TrajectoryWriter`].
///
/// `None` keeps the existing `<episode_id>.json` (backwards-compat
/// default). `Gzip` writes `<episode_id>.json.gz` via
/// [`forge_replay::v2::TrajectoryV2::save_json_gz`]; the reader auto-
/// detects gzip by extension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrajectoryCompression {
    /// Plain JSON (`.json`). Default — backwards-compatible.
    #[default]
    None,
    /// Gzip-compressed JSON (`.json.gz`).
    Gzip,
}

impl TrajectoryCompression {
    /// Per-file extension this codec emits.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::None => forge_replay::v2::JSON_EXT,
            Self::Gzip => forge_replay::v2::JSON_GZ_EXT,
        }
    }
}

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

    /// Interface the metrics endpoint binds to. Defaults to
    /// ``"127.0.0.1"`` (localhost-only) for safety; compose-stack
    /// operators override to ``"0.0.0.0"`` so a sibling Prometheus
    /// container can scrape across the docker network.
    pub metrics_bind: String,

    /// Histogram bucket boundaries (in seconds) for the
    /// ``forge_mc_planning_latency_seconds`` Prometheus histogram.
    /// Defaults to the Prometheus standard latency buckets, suitable
    /// for sub-second per-decision MCTS planning calls.
    pub metrics_histogram_buckets: Vec<f64>,

    /// Compression codec for trajectory files. Defaults to `None`
    /// (plain `.json`) for backwards compatibility with existing
    /// readers. Set to `Gzip` to write `.json.gz`; the reader
    /// auto-detects by extension.
    pub trajectory_compression: TrajectoryCompression,

    /// Gzip compression level when
    /// `trajectory_compression == TrajectoryCompression::Gzip`.
    /// Defaults to `flate2::Compression::default()` (level 6).
    /// Ignored when compression is `None`.
    pub trajectory_gzip_level: TrajectoryGzipLevel,

    /// Number of multi-thread tokio worker threads to use for the
    /// runner binary's async runtime (metrics endpoint + shutdown
    /// signal). The runner loop itself runs on
    /// `tokio::task::spawn_blocking`, so a small thread count is
    /// usually enough; defaults to [`DEFAULT_TOKIO_WORKER_THREADS`].
    pub tokio_worker_threads: usize,

    /// Stub-env / stub-model knobs consumed only by `--dry-run`. The
    /// `serde(default)` means production TOMLs can omit the table
    /// entirely. See [`DryRunConfig`].
    pub dry_run: DryRunConfig,

    /// Path to the `MinecraftEnvConfig` TOML (typically
    /// `configs/minecraft/env.toml`). When set, the live runner path
    /// loads + connects through this config. `None` keeps the binary
    /// in dry-run-or-error mode (used by tests that don't touch
    /// `forge-env-mc`). May be overridden by a `--mc-config` CLI flag.
    #[serde(default)]
    pub mc_env_config_path: Option<PathBuf>,

    /// ONNX Runtime invariants the runner passes through to
    /// `into_reload_fn`. See [`OnnxRuntimeConfig`].
    #[serde(default)]
    pub onnx: OnnxRuntimeConfig,

    /// When `true`, the runner samples actions uniformly at random
    /// from `0..action_count` instead of asking the MCTS planner.
    /// Used by `scripts/mc_capture_baseline.py` to capture the
    /// untrained-baseline numbers a trained-agent comparison
    /// hinges on (v0.5 first-real-run plan, T3 + T4).
    ///
    /// Bypasses MCTS entirely — see
    /// [`crate::random_baseline`] for why uniform priors through
    /// MCTS don't produce uniform action selection.
    ///
    /// In live runs, setting this `true` also lets the runner skip
    /// the ONNX bundle load (the model never runs). The runner
    /// constructs a [`crate::random_baseline::RandomLatentModel`]
    /// purely to satisfy the type-generic bound on `Runner<E, M>`.
    #[serde(default)]
    pub random_actions: bool,
}

/// ONNX Runtime invariants the runner carries across reloads. These
/// fields DON'T change between manifest versions; they're set once at
/// runner construction and reused by every `into_reload_fn` callback.
///
/// Defaults delegate to
/// [`forge_agent::latent_mcts::onnx_model::DEFAULT_LATENT_DIM`] +
/// [`forge_agent::latent_mcts::onnx_model::DEFAULT_NUM_THREADS`] so
/// every literal lives in one place (`onnx_model.rs`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct OnnxRuntimeConfig {
    /// `action_space_size` passed to `OnnxModelConfig`. Should equal
    /// the number of entries in `configs/minecraft/action_map.toml`.
    /// `0` means "fall back to the action map's `action_count()` at
    /// connect time" — the runner sets this automatically when it
    /// loads the action map for the cross-check at handshake.
    pub action_space_size: u32,
    /// Latent dimensionality the model carries. Mirrors the
    /// `MuZeroConfig.latent_dim` used at training time.
    pub latent_dim: usize,
    /// ONNX Runtime inter-op thread count. `0` means "leave at the
    /// `ort` default" (single-threaded).
    pub num_threads: usize,
    /// Bundle directory that contains the per-role ONNX files. When
    /// `None`, defaults to `manifest_path.parent()`.
    pub bundle_dir: Option<PathBuf>,
}

impl Default for OnnxRuntimeConfig {
    fn default() -> Self {
        // Lazy const re-export so the runner crate doesn't need a
        // direct compile-time link to forge-agent when the
        // `onnx-reload` feature is off. The literals below are kept
        // in sync with `forge_agent::latent_mcts::onnx_model::DEFAULT_*`
        // by `crates/forge-mc-runner/src/config.rs::tests::onnx_runtime_defaults_match_forge_agent_consts`
        // when the `onnx-reload` feature is enabled.
        Self {
            action_space_size: 0, // 0 = auto-derive from action_map
            latent_dim: DEFAULT_LATENT_DIM_FALLBACK,
            num_threads: DEFAULT_NUM_THREADS_FALLBACK,
            bundle_dir: None,
        }
    }
}

/// Fallback for the latent dim when the `onnx-reload` feature isn't
/// compiled in. Kept byte-identical to
/// `forge_agent::latent_mcts::onnx_model::DEFAULT_LATENT_DIM` via the
/// gated test below.
pub const DEFAULT_LATENT_DIM_FALLBACK: usize = 256;

/// Fallback for the thread count. See [`DEFAULT_LATENT_DIM_FALLBACK`].
pub const DEFAULT_NUM_THREADS_FALLBACK: usize = 1;

/// Default number of multi-thread tokio worker threads for the
/// runner binary. The metrics endpoint and the SIGINT handler don't
/// need more than a couple of threads; the runner loop itself runs
/// on `tokio::task::spawn_blocking` so it doesn't consume a worker
/// slot.
pub const DEFAULT_TOKIO_WORKER_THREADS: usize = 2;

/// Prometheus standard latency buckets (in seconds), used as the
/// default for `RunnerConfig::metrics_histogram_buckets`. Each MCTS
/// planning call should land somewhere in this range under realistic
/// `planning_sims` budgets; if a deployment needs tighter resolution
/// it overrides this list via TOML.
pub const DEFAULT_METRICS_HISTOGRAM_BUCKETS_SECONDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Default localhost bind for the metrics endpoint. Pinned as a
/// `const` so the default flows through one source of truth.
pub const DEFAULT_METRICS_BIND: &str = "127.0.0.1";

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
            metrics_bind: DEFAULT_METRICS_BIND.to_string(),
            metrics_histogram_buckets: DEFAULT_METRICS_HISTOGRAM_BUCKETS_SECONDS.to_vec(),
            trajectory_compression: TrajectoryCompression::default(),
            trajectory_gzip_level: TrajectoryGzipLevel::default(),
            tokio_worker_threads: DEFAULT_TOKIO_WORKER_THREADS,
            dry_run: DryRunConfig::default(),
            mc_env_config_path: None,
            onnx: OnnxRuntimeConfig::default(),
            random_actions: false,
        }
    }
}

/// Environment variable that overrides `RunnerConfig.schema_id` at
/// runtime. Used by `scripts/mc_self_play.sh` to feed the live
/// `compute-schema-id` output into the runner WITHOUT patching the
/// static `runner.toml` file. Empty / unset env vars are treated as
/// "no override" — the TOML value wins.
pub const SCHEMA_ID_ENV_VAR: &str = "FORGE_MC_SCHEMA_ID";

impl RunnerConfig {
    /// Returns `true` iff `episodes == 0` (run forever).
    pub fn runs_forever(&self) -> bool {
        self.episodes == 0
    }

    /// Returns `true` iff metrics serving is disabled.
    pub fn metrics_disabled(&self) -> bool {
        self.metrics_port == 0
    }

    /// Apply env-var overrides in-place. Currently honours
    /// [`SCHEMA_ID_ENV_VAR`]; future env-var overrides land here.
    /// Returns `Self` for chainability after `Default::default()` or
    /// `toml::from_str`.
    ///
    /// Empty env-var values are treated as unset — the TOML value
    /// wins. This avoids surprising operators who export an empty
    /// `FORGE_MC_SCHEMA_ID=` in a shell config.
    #[must_use]
    pub fn with_env_var_overrides(mut self) -> Self {
        if let Ok(v) = std::env::var(SCHEMA_ID_ENV_VAR) {
            if !v.is_empty() {
                self.schema_id = v;
            }
        }
        self
    }

    /// Validate cross-field invariants. Catches confusable misconfigs
    /// (e.g. `action_repeat == 0` would divide by zero in the runner).
    pub fn validate(&self) -> Result<(), String> {
        if self.action_repeat == 0 {
            return Err("action_repeat must be >= 1".into());
        }
        if self.tokio_worker_threads == 0 {
            return Err("tokio_worker_threads must be >= 1".into());
        }
        if self.dry_run.obs_dim == 0 {
            return Err("dry_run.obs_dim must be >= 1".into());
        }
        if self.dry_run.action_count == 0 {
            return Err("dry_run.action_count must be >= 1".into());
        }
        if self.dry_run.latent_dim == 0 {
            return Err("dry_run.latent_dim must be >= 1".into());
        }
        if self.dry_run.max_episode_len == 0 {
            return Err("dry_run.max_episode_len must be >= 1".into());
        }
        if self.onnx.latent_dim == 0 {
            return Err("onnx.latent_dim must be >= 1".into());
        }
        // `onnx.num_threads == 0` is legal — ort interprets it as
        // "library default". `onnx.action_space_size == 0` is also
        // legal — the live runner derives it from the action map at
        // connect time.
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
        if !self.metrics_disabled() {
            if self.metrics_bind.is_empty() {
                return Err("metrics_bind must be non-empty when metrics_port != 0".into());
            }
            if self.metrics_histogram_buckets.is_empty() {
                return Err(
                    "metrics_histogram_buckets must be non-empty when metrics_port != 0".into(),
                );
            }
            // Buckets must be strictly increasing and positive for
            // Prometheus to accept them.
            let mut prev = 0.0f64;
            for (i, &b) in self.metrics_histogram_buckets.iter().enumerate() {
                if !b.is_finite() || b <= prev {
                    return Err(format!(
                        "metrics_histogram_buckets must be strictly increasing positive finite floats; \
                         index {i} ({b}) violates this (previous = {prev})"
                    ));
                }
                prev = b;
            }
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
    fn random_actions_default_is_false() {
        // Backwards-compat pin: existing TOMLs without `random_actions`
        // must continue to route planning through MCTS.
        let cfg = RunnerConfig::default();
        assert!(!cfg.random_actions);
    }

    #[test]
    fn ships_default_runner_toml_parses_with_random_actions() {
        // The v0.5 shipped runner.toml (mounted into the runner
        // container by docker/compose.minecraft.yml) MUST parse
        // cleanly + validate, and its `random_actions = true`
        // default MUST be honoured so a docker compose up runner
        // produces a baseline-capture run out of the box. Drift in
        // either the TOML's field names OR the runner's config
        // schema fails this test before it can ship.
        //
        // CARGO_MANIFEST_DIR == .../crates/forge-mc-runner; the
        // workspace root is two directories up.
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root");
        let runner_toml = workspace_root.join("configs/minecraft/runner.toml");
        let raw = std::fs::read_to_string(&runner_toml)
            .unwrap_or_else(|e| panic!("read {}: {e}", runner_toml.display()));
        let cfg: RunnerConfig = toml::from_str(&raw).expect("parse runner.toml");
        cfg.validate().expect("runner.toml must validate");
        assert!(
            cfg.random_actions,
            "shipped runner.toml MUST default to random_actions=true so the \
             v0.5 baseline-capture docker variant works out of the box"
        );
        // The container mount uses /app/configs/env.toml; runner.toml
        // must reference that absolute path so the runner can find
        // the env-config when the workspace isn't its cwd.
        assert_eq!(
            cfg.mc_env_config_path
                .as_deref()
                .map(|p| p.to_string_lossy().into_owned()),
            Some("/app/configs/env.toml".to_string()),
            "shipped runner.toml mc_env_config_path must match the docker mount target"
        );
    }

    #[test]
    fn random_actions_parses_from_partial_toml() {
        let toml_src = r#"
            env_id = "minecraft"
            schema_id = "abc"
            random_actions = true
        "#;
        let cfg: RunnerConfig = toml::from_str(toml_src).unwrap();
        assert!(cfg.random_actions);
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
            ..RunnerConfig::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: RunnerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn dry_run_default_matches_pre_extraction_literals() {
        // Pins the dry-run defaults to the literal values the
        // pre-extraction `run_dry` carried inline. If a TOML override
        // is later introduced the defaults stay backwards-compat.
        let dr = DryRunConfig::default();
        assert_eq!(dr.obs_dim, 8);
        assert_eq!(dr.action_count, 4);
        assert_eq!(dr.latent_dim, 16);
        assert_eq!(dr.max_episode_len, 8);
    }

    #[test]
    fn validate_rejects_zero_dry_run_obs_dim() {
        let cfg = RunnerConfig {
            dry_run: DryRunConfig {
                obs_dim: 0,
                ..DryRunConfig::default()
            },
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("dry_run.obs_dim"));
    }

    #[test]
    fn validate_rejects_zero_dry_run_action_count() {
        let cfg = RunnerConfig {
            dry_run: DryRunConfig {
                action_count: 0,
                ..DryRunConfig::default()
            },
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("dry_run.action_count"));
    }

    #[test]
    fn validate_rejects_zero_dry_run_latent_dim() {
        let cfg = RunnerConfig {
            dry_run: DryRunConfig {
                latent_dim: 0,
                ..DryRunConfig::default()
            },
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("dry_run.latent_dim"));
    }

    #[test]
    fn validate_rejects_zero_dry_run_max_episode_len() {
        let cfg = RunnerConfig {
            dry_run: DryRunConfig {
                max_episode_len: 0,
                ..DryRunConfig::default()
            },
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("dry_run.max_episode_len"));
    }

    #[test]
    fn validate_rejects_zero_tokio_worker_threads() {
        let cfg = RunnerConfig {
            tokio_worker_threads: 0,
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("tokio_worker_threads"));
    }

    #[test]
    fn tokio_worker_threads_default_matches_const() {
        assert_eq!(
            RunnerConfig::default().tokio_worker_threads,
            DEFAULT_TOKIO_WORKER_THREADS,
        );
    }

    /// Confirms the new `[dry_run]` TOML table parses correctly with
    /// the `#[serde(default)]` discipline — partial tables fall back
    /// to defaults for unspecified fields.
    #[test]
    fn dry_run_table_partial_toml_uses_defaults() {
        let toml_src = r#"
            env_id = "minecraft"
            schema_id = "abc"
            [dry_run]
            obs_dim = 32
        "#;
        let cfg: RunnerConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(cfg.dry_run.obs_dim, 32);
        // Other dry_run fields fall back to defaults.
        assert_eq!(cfg.dry_run.action_count, 4);
        assert_eq!(cfg.dry_run.latent_dim, 16);
    }

    #[test]
    fn mc_env_config_path_defaults_to_none() {
        // v0.4 BLOCKER addition. Existing v0.3-pre TOMLs MUST parse
        // unchanged → field is Optional with `#[serde(default)]`.
        let cfg = RunnerConfig::default();
        assert!(cfg.mc_env_config_path.is_none());
    }

    #[test]
    fn onnx_table_partial_toml_uses_defaults() {
        let toml_src = r#"
            env_id = "minecraft"
            schema_id = "abc"
            [onnx]
            action_space_size = 12
        "#;
        let cfg: RunnerConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(cfg.onnx.action_space_size, 12);
        assert_eq!(cfg.onnx.latent_dim, DEFAULT_LATENT_DIM_FALLBACK);
        assert_eq!(cfg.onnx.num_threads, DEFAULT_NUM_THREADS_FALLBACK);
        assert!(cfg.onnx.bundle_dir.is_none());
    }

    #[test]
    fn validate_rejects_zero_onnx_latent_dim() {
        let cfg = RunnerConfig {
            onnx: OnnxRuntimeConfig {
                latent_dim: 0,
                ..OnnxRuntimeConfig::default()
            },
            ..RunnerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("onnx.latent_dim"));
    }

    #[test]
    fn validate_accepts_zero_action_space_size_and_num_threads() {
        // 0 = auto-derive for action_space_size; 0 = library-default
        // for num_threads. Both are explicitly legal.
        let cfg = RunnerConfig {
            onnx: OnnxRuntimeConfig {
                action_space_size: 0,
                num_threads: 0,
                ..OnnxRuntimeConfig::default()
            },
            ..RunnerConfig::default()
        };
        cfg.validate().unwrap();
    }

    /// Pins the runner-side fallback consts byte-equal to the
    /// `forge_agent::latent_mcts::onnx_model::DEFAULT_*` consts when
    /// the `onnx-reload` feature is compiled in. The runner crate
    /// otherwise has NO direct link to forge-agent.
    #[test]
    #[cfg(feature = "onnx-reload")]
    fn onnx_runtime_defaults_match_forge_agent_consts() {
        assert_eq!(
            DEFAULT_LATENT_DIM_FALLBACK,
            forge_agent::latent_mcts::onnx_model::DEFAULT_LATENT_DIM,
            "onnx.latent_dim fallback drifted from forge-agent's source of truth",
        );
        assert_eq!(
            DEFAULT_NUM_THREADS_FALLBACK,
            forge_agent::latent_mcts::onnx_model::DEFAULT_NUM_THREADS,
            "onnx.num_threads fallback drifted from forge-agent's source of truth",
        );
    }

    /// All three env-var override scenarios consolidated into one
    /// `#[test]` so cargo's parallel test runner cannot interleave
    /// our env-var mutations across threads.
    ///
    /// Rust 2024 made `std::env::set_var` `unsafe` precisely because
    /// concurrent mutators race the process-global env table; the
    /// peer-review-flagged "single-threaded test scope" claim only
    /// holds when ALL mutators of the same env var sit in the same
    /// `#[test]` function.
    ///
    /// Sequential cases (cleared between each):
    /// 1. env set → overrides TOML.
    /// 2. env unset → TOML wins (default-state inheritance).
    /// 3. env set to empty string → treated as unset; TOML wins.
    #[test]
    fn with_env_var_overrides_covers_all_scenarios() {
        let saved = std::env::var(SCHEMA_ID_ENV_VAR).ok();

        // --- case 1: env set ---
        // SAFETY: All env mutation is contained in this single test;
        // no other test in the suite touches SCHEMA_ID_ENV_VAR, so
        // cargo's parallel runner cannot race us.
        unsafe {
            std::env::set_var(SCHEMA_ID_ENV_VAR, "sha-from-env");
        }
        let cfg = RunnerConfig {
            schema_id: "sha-from-toml".into(),
            ..RunnerConfig::default()
        }
        .with_env_var_overrides();
        assert_eq!(cfg.schema_id, "sha-from-env");

        // --- case 2: env unset ---
        unsafe {
            std::env::remove_var(SCHEMA_ID_ENV_VAR);
        }
        let cfg = RunnerConfig {
            schema_id: "sha-from-toml".into(),
            ..RunnerConfig::default()
        }
        .with_env_var_overrides();
        assert_eq!(cfg.schema_id, "sha-from-toml");

        // --- case 3: env set to empty string (treated as unset) ---
        unsafe {
            std::env::set_var(SCHEMA_ID_ENV_VAR, "");
        }
        let cfg = RunnerConfig {
            schema_id: "sha-from-toml".into(),
            ..RunnerConfig::default()
        }
        .with_env_var_overrides();
        assert_eq!(cfg.schema_id, "sha-from-toml");

        // Restore prior env so the suite stays hygienic.
        unsafe {
            if let Some(v) = saved {
                std::env::set_var(SCHEMA_ID_ENV_VAR, v);
            } else {
                std::env::remove_var(SCHEMA_ID_ENV_VAR);
            }
        }
    }
}
