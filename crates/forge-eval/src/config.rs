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
///
/// `Debug` is implemented manually (not derived) so the
/// [`Self::mlflow_http_token`] secret never lands in logs or panic
/// backtraces. `Serialize`/`Deserialize` skip the same field so a
/// config dumped to TOML/JSON doesn't write the token to disk; if
/// callers need the token after a deserialise round-trip they must
/// re-read it from env or re-set it explicitly.
#[derive(Clone, Serialize, Deserialize)]
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
    /// the harness writes a Hugging Face-friendly JSONL export under
    /// `<huggingface_export_root>/<run_id>/` that
    /// `datasets.load_dataset("json", data_files=...)` can open and
    /// `huggingface-cli upload` can push to the Hub. The export is NOT
    /// a `save_to_disk`-shaped Arrow DatasetDict — the README's
    /// `configs.data_files` declares per-split JSONL paths instead, so
    /// consumers don't need `pyarrow` to load.
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
    ///
    /// `#[serde(skip)]` — the token is NEVER serialised. A config dumped to
    /// disk (toml::to_string / serde_json::to_writer / EvalConfig::clone +
    /// log) excludes this field. Deserialising a config sets it to `None`;
    /// callers wanting auth must re-read it from env or set it explicitly
    /// after `from_str`. The custom `Debug` impl on `EvalConfig` redacts
    /// this field too — `format!("{cfg:?}")` shows `Some(<redacted>)` or
    /// `None`, never the bytes themselves.
    #[serde(skip)]
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

/// Manual `Debug` impl that redacts the bearer token. Mirrors every other
/// field via `f.debug_struct()` so log output is otherwise identical to
/// the auto-derived `Debug` we replaced.
impl std::fmt::Debug for EvalConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvalConfig")
            .field("episodes_per_scenario", &self.episodes_per_scenario)
            .field("max_steps_per_episode", &self.max_steps_per_episode)
            .field("base_seed", &self.base_seed)
            .field("tiers", &self.tiers)
            .field("parallelism", &self.parallelism)
            .field("record_replays", &self.record_replays)
            .field("record_trajectories", &self.record_trajectories)
            .field("output", &self.output)
            .field("base_forge_config", &self.base_forge_config)
            .field("mlflow_tracking_uri", &self.mlflow_tracking_uri)
            .field("huggingface_export_root", &self.huggingface_export_root)
            .field("run_id", &self.run_id)
            .field("experiment_name", &self.experiment_name)
            .field("mlflow_http_tracking_uri", &self.mlflow_http_tracking_uri)
            .field("mlflow_http_timeout_ms", &self.mlflow_http_timeout_ms)
            .field("mlflow_http_max_retries", &self.mlflow_http_max_retries)
            .field(
                "mlflow_http_backoff_base_ms",
                &self.mlflow_http_backoff_base_ms,
            )
            .field("mlflow_http_batch_size", &self.mlflow_http_batch_size)
            .field(
                "mlflow_http_token",
                &self.mlflow_http_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// MLflow REST cap for `runs/log-batch` metric arrays. Used by
/// [`EvalConfig::validate`] to reject configs that would generate
/// server-rejected requests.
pub const MLFLOW_LOG_BATCH_HARD_CAP: usize = 1_000;

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
        if self.mlflow_http_batch_size == 0 {
            errors.push("mlflow_http_batch_size must be > 0".to_string());
        }
        if self.mlflow_http_batch_size > MLFLOW_LOG_BATCH_HARD_CAP {
            errors.push(format!(
                "mlflow_http_batch_size {} exceeds MLflow REST cap of {} \
                 (server will reject runs/log-batch requests)",
                self.mlflow_http_batch_size, MLFLOW_LOG_BATCH_HARD_CAP
            ));
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
        // NOTE: `mlflow_http_token` carries `#[serde(skip)]` and is
        // intentionally NOT preserved across (de)serialisation — see the
        // dedicated `test_mlflow_http_token_is_never_serialised` test
        // below, which pins that contract.
        let config = EvalConfig {
            mlflow_http_tracking_uri: Some("https://mlflow.example:5000".to_string()),
            mlflow_http_timeout_ms: 12_345,
            mlflow_http_max_retries: 7,
            mlflow_http_backoff_base_ms: 100,
            mlflow_http_batch_size: 250,
            mlflow_http_token: None, // skip-serde means we never round-trip the token
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: EvalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deser.mlflow_http_tracking_uri.as_deref(),
            Some("https://mlflow.example:5000")
        );
        assert_eq!(deser.mlflow_http_timeout_ms, 12_345);
        assert_eq!(deser.mlflow_http_max_retries, 7);
        assert_eq!(deser.mlflow_http_backoff_base_ms, 100);
        assert_eq!(deser.mlflow_http_batch_size, 250);
    }

    /// Pin the security contract: the bearer token MUST NOT appear in the
    /// serialised representation, and MUST NOT survive a deserialise round
    /// trip. A regression here would write secrets to any TOML/JSON the
    /// config is dumped to (audit logs, config snapshots, test fixtures).
    #[test]
    fn test_mlflow_http_token_is_never_serialised() {
        let cfg = EvalConfig {
            mlflow_http_token: Some("PAT-super-secret-DO-NOT-LEAK".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialise");
        assert!(
            !json.contains("PAT-super-secret-DO-NOT-LEAK"),
            "bearer token must not appear in JSON serialisation: {json}"
        );
        assert!(
            !json.contains("mlflow_http_token"),
            "token field must be skipped entirely, not emitted with null: {json}"
        );
        let toml_str = toml::to_string(&cfg).expect("toml serialise");
        assert!(
            !toml_str.contains("PAT-super-secret-DO-NOT-LEAK"),
            "bearer token must not appear in TOML serialisation: {toml_str}"
        );

        // Deserialise back: token must be absent (None) regardless of what
        // was originally set on the source config.
        let deser: EvalConfig = serde_json::from_str(&json).expect("deserialise");
        assert!(deser.mlflow_http_token.is_none());
    }

    /// Pin the second security contract: `Debug` (and therefore every
    /// `tracing::debug!(?cfg, ...)` site) MUST redact the token.
    #[test]
    fn test_mlflow_http_token_is_redacted_in_debug_output() {
        let cfg = EvalConfig {
            mlflow_http_token: Some("PAT-super-secret-DO-NOT-LEAK".to_string()),
            ..Default::default()
        };
        let dbg = format!("{cfg:?}");
        assert!(
            !dbg.contains("PAT-super-secret-DO-NOT-LEAK"),
            "bearer token must be redacted in Debug output: {dbg}"
        );
        assert!(
            dbg.contains("<redacted>"),
            "Debug output must signal the redaction: {dbg}"
        );

        // None case: Debug should show None (not "<redacted>") so callers
        // can still distinguish unset-vs-set at log-read time.
        let unset_cfg = EvalConfig {
            mlflow_http_token: None,
            ..Default::default()
        };
        let dbg_unset = format!("{unset_cfg:?}");
        assert!(dbg_unset.contains("mlflow_http_token: None"));
    }

    /// Pin the batch_size validation contract: configs that would generate
    /// server-rejected requests (batch_size > MLflow's 1000 metric cap, or
    /// 0) must surface in `validate()` rather than fail at runtime.
    #[test]
    fn test_validate_rejects_oversized_or_zero_batch_size() {
        let mut cfg = EvalConfig {
            mlflow_http_batch_size: 0,
            ..EvalConfig::default()
        };
        let errs = cfg.validate();
        assert!(
            errs.iter()
                .any(|e| e.contains("mlflow_http_batch_size must be > 0")),
            "zero batch_size must error: {errs:?}"
        );

        cfg.mlflow_http_batch_size = MLFLOW_LOG_BATCH_HARD_CAP + 1;
        let errs = cfg.validate();
        assert!(
            errs.iter()
                .any(|e| e.contains("exceeds MLflow REST cap of 1000")),
            "oversized batch_size must error: {errs:?}"
        );

        cfg.mlflow_http_batch_size = MLFLOW_LOG_BATCH_HARD_CAP;
        let errs: Vec<String> = cfg
            .validate()
            .into_iter()
            .filter(|e| e.contains("batch_size"))
            .collect();
        assert!(
            errs.is_empty(),
            "exactly-1000 batch_size must be allowed: {errs:?}"
        );
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
        assert_eq!(
            cfg.mlflow_http_backoff_base_ms,
            DEFAULT_MLFLOW_HTTP_BACKOFF_BASE_MS
        );
        assert_eq!(cfg.mlflow_http_batch_size, DEFAULT_MLFLOW_HTTP_BATCH_SIZE);
        assert!(
            cfg.mlflow_http_batch_size <= 1000,
            "MLflow REST caps log_batch at 1000"
        );
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
