//! Configuration for the evaluation harness.

use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::output::OutputConfig;

// MLflow HTTP exporter defaults.
//
// When the `http-mlflow` feature is enabled, the canonical home for these
// constants is `crate::exporters::mlflow_http`; we re-export from there so a
// single edit changes every consumer. When the feature is OFF the placeholders
// below stand in (the constants still need to exist so `EvalConfig::Default`
// compiles; their values mirror the canonical ones one-for-one — pinned by a
// test in `mlflow_http::tests` when the feature is enabled).

#[cfg(feature = "http-mlflow")]
pub use crate::exporters::mlflow_http::{
    DEFAULT_HTTP_BACKOFF_BASE_MS as DEFAULT_MLFLOW_HTTP_BACKOFF_BASE_MS,
    DEFAULT_HTTP_MAX_RETRIES as DEFAULT_MLFLOW_HTTP_MAX_RETRIES,
    DEFAULT_HTTP_TIMEOUT_MS as DEFAULT_MLFLOW_HTTP_TIMEOUT_MS,
    DEFAULT_LOG_BATCH_SIZE as DEFAULT_MLFLOW_HTTP_BATCH_SIZE,
};

/// Default HTTP request timeout for the MLflow tracking REST client (ms).
#[cfg(not(feature = "http-mlflow"))]
pub const DEFAULT_MLFLOW_HTTP_TIMEOUT_MS: u64 = 30_000;
/// Default max retries on retryable HTTP statuses (5xx, 408, 429).
#[cfg(not(feature = "http-mlflow"))]
pub const DEFAULT_MLFLOW_HTTP_MAX_RETRIES: u32 = 5;
/// Default exponential-backoff base (ms) between retries.
#[cfg(not(feature = "http-mlflow"))]
pub const DEFAULT_MLFLOW_HTTP_BACKOFF_BASE_MS: u64 = 250;
/// Default batch size for `runs/log-batch`. MLflow REST caps at 1000.
#[cfg(not(feature = "http-mlflow"))]
pub const DEFAULT_MLFLOW_HTTP_BATCH_SIZE: usize = 1_000;

/// Configuration for an evaluation run.
///
/// All parameters are configurable — no hard-coded values.
/// Use [`EvalConfig::validate`] to check invariants before running.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EvalConfig {
    /// Number of episodes to run per scenario/seed combination.
    pub episodes_per_scenario: u32,
    /// Maximum steps per episode (overrides scenario-level config if set).
    pub max_steps_per_episode: u64,
    /// Base seed for reproducibility. Episode seeds are derived as `base_seed + episode_idx`.
    pub base_seed: u64,
    /// Which difficulty tiers to evaluate (empty = all 6).
    pub tiers: Vec<u8>,
    /// Number of parallel evaluation threads (0 = use rayon default).
    pub parallelism: u32,
    /// If `true`, the harness builds a [`forge_replay::compact::CompactReplay`]
    /// for each episode. The replay is only persisted on disk when
    /// [`OutputConfig::enabled`] **and** [`OutputConfig::write_replays`] are
    /// both true; otherwise the builder is constructed and dropped (useful
    /// only when paired with `output.enabled = true`). For an observable
    /// effect, set `output.enabled = true` and `output.write_replays = true`.
    pub record_replays: bool,
    /// If `true`, the harness builds a [`forge_replay::trajectory::Trajectory`]
    /// for each episode. Same persistence rule as
    /// [`record_replays`](Self::record_replays): on-disk emission requires
    /// `output.enabled = true` **and** `output.write_trajectories = true`.
    pub record_trajectories: bool,
    /// On-disk artefact configuration. Disabled by default — when enabled,
    /// the harness persists replays and trajectories under
    /// [`OutputConfig::dir`].
    pub output: OutputConfig,
    /// Base FORGE config to use for scenarios that don't specify their own.
    pub base_forge_config: ForgeConfig,
    // --- Phase B fields (additive; default None = exporter disabled) ---
    /// Absolute path of an MLflow `mlruns/` tracking directory. When set,
    /// the harness invokes the MLflow filesystem-layout exporter after the
    /// scorecard is built, writing the run + nested per-scenario child
    /// runs that `mlflow ui --backend-store-uri <dir>` can visualize.
    /// `None` (the default) leaves Phase 1 behaviour byte-identical.
    pub mlflow_tracking_uri: Option<std::path::PathBuf>,
    /// Absolute path of the HuggingFace-export root directory. When set,
    /// the harness writes a `DatasetDict`-compatible directory under
    /// `<huggingface_export_root>/<run_id>/` that `datasets.load_from_disk`
    /// can open and `huggingface-cli upload` can push to the Hub.
    /// `None` (the default) leaves Phase 1 behaviour byte-identical.
    pub huggingface_export_root: Option<std::path::PathBuf>,
    /// Stable run identifier shared by every Phase B exporter target.
    /// `None` triggers a UUIDv4 hex run id at [`crate::RunManifest::capture`]
    /// time. Set explicitly to make multiple Phase B exporters write under
    /// the same id, or to re-export an existing scorecard idempotently.
    pub run_id: Option<String>,
    /// Experiment grouping mirrored to MLflow's experiment name and to the
    /// HF dataset card's `pretty_name`. `None` defaults to
    /// `"forge-eval-default"` in the manifest.
    pub experiment_name: Option<String>,
    // --- MLflow HTTP transport fields (added Slice 1.05; consumer wired in Slice 2) ---
    /// HTTP(S) URI of an MLflow tracking server. When set, takes precedence
    /// over [`mlflow_tracking_uri`](Self::mlflow_tracking_uri) and routes
    /// exporter calls through the REST API instead of writing to disk.
    /// `None` (the default) preserves filesystem-only behaviour.
    pub mlflow_http_tracking_uri: Option<String>,
    /// Per-request timeout (ms) for the MLflow HTTP client. Default
    /// [`DEFAULT_MLFLOW_HTTP_TIMEOUT_MS`].
    pub mlflow_http_timeout_ms: u64,
    /// Max retries on retryable HTTP statuses (5xx, 408, 429). Default
    /// [`DEFAULT_MLFLOW_HTTP_MAX_RETRIES`].
    pub mlflow_http_max_retries: u32,
    /// Exponential-backoff base (ms) between retries. Default
    /// [`DEFAULT_MLFLOW_HTTP_BACKOFF_BASE_MS`].
    pub mlflow_http_backoff_base_ms: u64,
    /// Batch size for `runs/log-batch`. MLflow REST caps at 1000; default
    /// [`DEFAULT_MLFLOW_HTTP_BATCH_SIZE`].
    pub mlflow_http_batch_size: usize,
    /// Optional bearer token for MLflow tracking server auth. When `None`,
    /// the HTTP client reads `MLFLOW_TRACKING_TOKEN` env at construction time.
    /// Never logged.
    pub mlflow_http_token: Option<String>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        let mut forge_config = ForgeConfig::default();
        forge_config.world.width = 16;
        forge_config.world.height = 16;
        forge_config.agents.num_agents = 1;
        forge_config.task.max_episode_length = 500;

        Self {
            episodes_per_scenario: 10,
            max_steps_per_episode: 500,
            base_seed: 0,
            tiers: vec![],
            parallelism: 0,
            record_replays: false,
            record_trajectories: false,
            output: OutputConfig::default(),
            base_forge_config: forge_config,
            mlflow_tracking_uri: None,
            huggingface_export_root: None,
            run_id: None,
            experiment_name: None,
            mlflow_http_tracking_uri: None,
            mlflow_http_timeout_ms: DEFAULT_MLFLOW_HTTP_TIMEOUT_MS,
            mlflow_http_max_retries: DEFAULT_MLFLOW_HTTP_MAX_RETRIES,
            mlflow_http_backoff_base_ms: DEFAULT_MLFLOW_HTTP_BACKOFF_BASE_MS,
            mlflow_http_batch_size: DEFAULT_MLFLOW_HTTP_BATCH_SIZE,
            // Read from env at construction time so `Default::default()` picks
            // up CI-provided tokens without callers having to thread them.
            mlflow_http_token: std::env::var("MLFLOW_TRACKING_TOKEN").ok(),
        }
    }
}

impl EvalConfig {
    /// Validates the configuration, returning a list of issues.
    ///
    /// An empty list means the config is valid.
    #[instrument(skip_all)]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.episodes_per_scenario == 0 {
            errors.push("episodes_per_scenario must be > 0".to_string());
        }
        if self.max_steps_per_episode == 0 {
            errors.push("max_steps_per_episode must be > 0".to_string());
        }
        for &tier in &self.tiers {
            if tier == 0 || tier > 6 {
                errors.push(format!("tier {tier} is out of valid range 1-6"));
            }
        }
        if self.base_forge_config.world.width == 0 || self.base_forge_config.world.height == 0 {
            errors.push("world dimensions must be > 0".to_string());
        }
        if self.base_forge_config.agents.num_agents == 0 {
            errors.push("num_agents must be > 0".to_string());
        }
        errors.extend(self.output.validate());

        errors
    }

    /// Returns true if the configuration passes all validation checks.
    #[instrument(skip(self))]
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EvalConfig::default();
        assert_eq!(config.episodes_per_scenario, 10);
        assert_eq!(config.max_steps_per_episode, 500);
        assert_eq!(config.base_seed, 0);
        assert!(config.tiers.is_empty());
        assert!(!config.record_replays);
        assert!(!config.record_trajectories);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(EvalConfig);
    }

    #[test]
    fn test_mlflow_http_fields_serde_roundtrip() {
        // Exercises every new HTTP field added in Slice 1.05 so a future
        // rename/remove surfaces here, not at runtime under the harness.
        let config = EvalConfig {
            mlflow_http_tracking_uri: Some("https://mlflow.example:5000".to_string()),
            mlflow_http_timeout_ms: 12_345,
            mlflow_http_max_retries: 7,
            mlflow_http_backoff_base_ms: 100,
            mlflow_http_batch_size: 250,
            mlflow_http_token: Some("REDACTED".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: EvalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.mlflow_http_tracking_uri.as_deref(), Some("https://mlflow.example:5000"));
        assert_eq!(deser.mlflow_http_timeout_ms, 12_345);
        assert_eq!(deser.mlflow_http_max_retries, 7);
        assert_eq!(deser.mlflow_http_backoff_base_ms, 100);
        assert_eq!(deser.mlflow_http_batch_size, 250);
        assert_eq!(deser.mlflow_http_token.as_deref(), Some("REDACTED"));
    }

    #[test]
    fn test_default_mlflow_http_fields_use_constants() {
        // Pins the binding between EvalConfig::default and the
        // DEFAULT_MLFLOW_HTTP_* constants so a future divergence is caught
        // at compile + test time, not in production.
        let cfg = EvalConfig::default();
        assert!(cfg.mlflow_http_tracking_uri.is_none());
        assert_eq!(cfg.mlflow_http_timeout_ms, DEFAULT_MLFLOW_HTTP_TIMEOUT_MS);
        assert_eq!(cfg.mlflow_http_max_retries, DEFAULT_MLFLOW_HTTP_MAX_RETRIES);
        assert_eq!(cfg.mlflow_http_backoff_base_ms, DEFAULT_MLFLOW_HTTP_BACKOFF_BASE_MS);
        assert_eq!(cfg.mlflow_http_batch_size, DEFAULT_MLFLOW_HTTP_BATCH_SIZE);
        assert!(cfg.mlflow_http_batch_size <= 1000, "MLflow REST caps log_batch at 1000");
    }

    #[test]
    fn test_defaults_valid_macro() {
        forge_types::assert_config_defaults_valid!(EvalConfig);
    }

    #[test]
    fn test_config_serde_roundtrip_custom_values() {
        let config = EvalConfig {
            episodes_per_scenario: 50,
            max_steps_per_episode: 1000,
            base_seed: 42,
            tiers: vec![1, 2, 3],
            parallelism: 4,
            record_replays: true,
            record_trajectories: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: EvalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.episodes_per_scenario, 50);
        assert_eq!(deser.base_seed, 42);
        assert_eq!(deser.tiers, vec![1, 2, 3]);
        assert!(deser.record_replays);
    }

    #[test]
    fn test_default_config_is_valid() {
        let config = EvalConfig::default();
        assert!(config.is_valid());
    }

    #[test]
    fn test_validate_zero_episodes() {
        let config = EvalConfig {
            episodes_per_scenario: 0,
            ..EvalConfig::default()
        };
        let errors = config.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("episodes_per_scenario"));
    }

    #[test]
    fn test_validate_zero_max_steps() {
        let config = EvalConfig {
            max_steps_per_episode: 0,
            ..EvalConfig::default()
        };
        assert!(!config.is_valid());
    }

    #[test]
    fn test_validate_invalid_tier() {
        let config = EvalConfig {
            tiers: vec![0, 7],
            ..EvalConfig::default()
        };
        let errors = config.validate();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_validate_valid_tiers() {
        let config = EvalConfig {
            tiers: vec![1, 3, 6],
            ..EvalConfig::default()
        };
        assert!(config.is_valid());
    }

    #[test]
    fn test_validate_zero_world_dimensions() {
        let mut config = EvalConfig::default();
        config.base_forge_config.world.width = 0;
        assert!(!config.is_valid());
    }

    #[test]
    fn test_validate_zero_agents() {
        let mut config = EvalConfig::default();
        config.base_forge_config.agents.num_agents = 0;
        assert!(!config.is_valid());
    }

    #[test]
    fn test_validate_empty_tiers_is_valid() {
        let config = EvalConfig {
            tiers: vec![],
            ..EvalConfig::default()
        };
        assert!(
            config.is_valid(),
            "empty tiers means 'all tiers' and is valid"
        );
    }

    #[test]
    fn test_validate_multiple_errors_accumulate() {
        let mut config = EvalConfig {
            episodes_per_scenario: 0,
            max_steps_per_episode: 0,
            tiers: vec![0],
            ..EvalConfig::default()
        };
        config.base_forge_config.world.width = 0;
        config.base_forge_config.agents.num_agents = 0;
        let errors = config.validate();
        assert!(
            errors.len() >= 4,
            "should collect multiple errors: got {}",
            errors.len()
        );
    }

    #[test]
    fn test_validate_all_valid_tiers_boundary() {
        let config = EvalConfig {
            tiers: vec![1, 2, 3, 4, 5, 6],
            ..EvalConfig::default()
        };
        assert!(config.is_valid(), "all valid tier values should pass");
    }

    #[test]
    fn test_base_seed_max_value() {
        let config = EvalConfig {
            base_seed: u64::MAX,
            ..EvalConfig::default()
        };
        assert!(config.is_valid(), "max seed should be valid");
    }

    #[test]
    fn test_parallelism_zero_is_valid() {
        let config = EvalConfig {
            parallelism: 0,
            ..EvalConfig::default()
        };
        assert!(config.is_valid(), "zero parallelism means rayon default");
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let config = EvalConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deser: EvalConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(deser.episodes_per_scenario, config.episodes_per_scenario);
        assert_eq!(deser.max_steps_per_episode, config.max_steps_per_episode);
        assert_eq!(deser.base_seed, config.base_seed);
    }
}
