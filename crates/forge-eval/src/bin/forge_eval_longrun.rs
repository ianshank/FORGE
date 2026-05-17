//! `forge-eval-longrun` — long-running e2e evaluation CLI.
//!
//! Drives [`EvalHarness::evaluate_suite`] over a TOML-defined scenario
//! suite and routes the resulting scorecard through the `http-mlflow`
//! [`MlflowHttpSink`] (configured automatically by
//! [`crate::exporters::mlflow_http::MlflowHttpSink::from_config`] inside the
//! harness's Phase B dispatcher).
//!
//! Only available behind the `http-mlflow` feature; declared in
//! `crates/forge-eval/Cargo.toml` with `required-features = ["http-mlflow"]`
//! so `cargo build -p forge-eval` (no feature) is unaffected.
//!
//! ## Defaults
//!
//! Every flag has either an explicit `default_value`/`default_value_t` or
//! an env-var binding (`env = "FORGE_E2E_*"` / `"FORGE_MLFLOW_*"`). No
//! magic numbers live in this file — numeric defaults all come from
//! `forge_eval::exporters::mlflow_http`'s `DEFAULT_*` constants, and
//! string defaults from `DEFAULT_EXPERIMENT_NAME` /
//! `DEFAULT_EVALUATE_SCENARIO_ID`. A future tuning change to the HTTP
//! defaults propagates to the CLI without code edits.
//!
//! ## Logging
//!
//! Initialises `tracing_subscriber::fmt` with `EnvFilter::from_default_env()`,
//! so `RUST_LOG=forge_eval=debug,info ./forge-eval-longrun ...` works as
//! expected. Every stage boundary (parsing, suite load, eval start, eval
//! complete, scorecard write) emits a `tracing::info!` event with
//! structured fields.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::Parser;
use forge_eval::config::EvalConfig;
use forge_eval::exporters::mlflow_http::{
    DEFAULT_EXPERIMENT_NAME, DEFAULT_HTTP_BACKOFF_BASE_MS, DEFAULT_HTTP_MAX_RETRIES,
    DEFAULT_HTTP_TIMEOUT_MS, DEFAULT_LOG_BATCH_SIZE,
};
use forge_eval::harness::EvalHarness;
use forge_eval::scenario::ScenarioSuite;
use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::observation::Observation;
use tracing::{info, instrument};
use tracing_subscriber::EnvFilter;

/// Long-running e2e evaluation driver: loads a scenario suite, runs the
/// harness against a `NoopEvalAgent` (or any pluggable factory in future
/// versions), and streams the scorecard to an MLflow tracking server via
/// the `http-mlflow` sink.
#[derive(Debug, Parser)]
#[command(
    name = "forge-eval-longrun",
    about = "Drive EvalHarness::evaluate_suite into an MLflow HTTP tracking server.",
    long_about = "Long-running e2e evaluation driver. Loads scenarios from --suite, \
                  runs the harness against a configurable agent factory, and routes \
                  the scorecard through the http-mlflow exporter."
)]
pub struct Cli {
    /// Directory containing `*.toml` scenario files to evaluate. The
    /// harness loads every file via `ScenarioSuite::load_dir` in
    /// deterministic file-name order; subdirectories are not traversed.
    #[arg(long, env = "FORGE_E2E_SUITE")]
    pub suite: PathBuf,

    /// Number of episodes to run per scenario. Surfaced as
    /// `EvalConfig::episodes_per_scenario` (separate from the total
    /// episode count the Python orchestrator caps the run at).
    #[arg(long, env = "FORGE_E2E_EPISODES_PER_SCENARIO", default_value_t = 1)]
    pub episodes_per_scenario: u32,

    /// MLflow tracking server URI. The harness's Phase B dispatcher
    /// prefers the HTTP sink over the filesystem sink whenever this is
    /// set; absent → filesystem sink falls through if `--fs-tracking`
    /// is also configured.
    #[arg(long, env = "FORGE_MLFLOW_TRACKING_URI")]
    pub tracking_uri: String,

    /// MLflow experiment name. Defaults to the canonical
    /// `DEFAULT_EXPERIMENT_NAME` ("forge-eval-default") so runs from
    /// different invocations group together in the UI.
    #[arg(long, env = "FORGE_E2E_EXPERIMENT_NAME", default_value = DEFAULT_EXPERIMENT_NAME)]
    pub experiment_name: String,

    /// Explicit run id override. `None` triggers a UUIDv4 hex id at
    /// manifest-capture time.
    #[arg(long, env = "FORGE_E2E_RUN_ID")]
    pub run_id: Option<String>,

    /// Per-request HTTP timeout (ms). Mirrors
    /// [`DEFAULT_HTTP_TIMEOUT_MS`].
    #[arg(long, env = "FORGE_MLFLOW_HTTP_TIMEOUT_MS", default_value_t = DEFAULT_HTTP_TIMEOUT_MS)]
    pub http_timeout_ms: u64,

    /// Max HTTP retries on retryable statuses (5xx, 408, 429). Mirrors
    /// [`DEFAULT_HTTP_MAX_RETRIES`].
    #[arg(long, env = "FORGE_MLFLOW_HTTP_MAX_RETRIES", default_value_t = DEFAULT_HTTP_MAX_RETRIES)]
    pub http_max_retries: u32,

    /// Exponential backoff base (ms). Mirrors
    /// [`DEFAULT_HTTP_BACKOFF_BASE_MS`].
    #[arg(long, env = "FORGE_MLFLOW_HTTP_BACKOFF_BASE_MS", default_value_t = DEFAULT_HTTP_BACKOFF_BASE_MS)]
    pub http_backoff_base_ms: u64,

    /// `runs/log-batch` chunk size cap. MLflow REST max is 1000;
    /// mirrors [`DEFAULT_LOG_BATCH_SIZE`].
    #[arg(long, env = "FORGE_E2E_BATCH_SIZE", default_value_t = DEFAULT_LOG_BATCH_SIZE)]
    pub http_batch_size: usize,

    /// Optional HuggingFace dataset export root. When set, the harness's
    /// Phase B dispatcher additionally writes a per-run JSONL tree under
    /// `<hf_export_root>/<manifest.run_id>/` for `datasets.load_dataset`
    /// consumers. Env-bound to `FORGE_HF_EXPORT_ROOT` so CI can wire it
    /// the same way the Python orchestrator does without an extra flag.
    #[arg(long, env = "FORGE_HF_EXPORT_ROOT")]
    pub hf_export_root: Option<PathBuf>,

    /// Output directory for `scorecard.json`. The Python orchestrator
    /// (Slice 4) reads this file to verify the run completed.
    #[arg(long, env = "FORGE_E2E_OUTPUT_DIR")]
    pub output_dir: PathBuf,
}

impl Cli {
    /// Build an [`EvalConfig`] from parsed CLI flags. Pure (no I/O,
    /// no env reads) so the conversion is unit-testable.
    pub fn to_eval_config(&self) -> EvalConfig {
        EvalConfig {
            episodes_per_scenario: self.episodes_per_scenario,
            mlflow_http_tracking_uri: Some(self.tracking_uri.clone()),
            mlflow_http_timeout_ms: self.http_timeout_ms,
            mlflow_http_max_retries: self.http_max_retries,
            mlflow_http_backoff_base_ms: self.http_backoff_base_ms,
            mlflow_http_batch_size: self.http_batch_size,
            experiment_name: Some(self.experiment_name.clone()),
            run_id: self.run_id.clone(),
            huggingface_export_root: self.hf_export_root.clone(),
            ..EvalConfig::default()
        }
    }
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    info!(
        suite = %cli.suite.display(),
        episodes_per_scenario = cli.episodes_per_scenario,
        tracking_uri = %cli.tracking_uri,
        experiment_name = %cli.experiment_name,
        output_dir = %cli.output_dir.display(),
        "forge-eval-longrun: starting"
    );
    let exit = run(&cli)?;
    info!(exit_code = exit, "forge-eval-longrun: done");
    std::process::exit(exit);
}

/// Init `tracing_subscriber` with the `RUST_LOG` env filter and a `info`
/// default. Idempotent: a second call (from tests) no-ops via
/// `try_init`.
fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,forge_eval=info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

/// Run the evaluation end-to-end. Split from `main` so unit tests can
/// invoke it with a custom `Cli` without spawning a process.
#[instrument(skip(cli), fields(
    suite = %cli.suite.display(),
    tracking_uri = %cli.tracking_uri
))]
pub fn run(cli: &Cli) -> Result<i32> {
    let suite = ScenarioSuite::load_dir(&cli.suite)
        .with_context(|| format!("loading scenario suite from {}", cli.suite.display()))?;
    info!(scenarios = suite.scenarios.len(), "scenario suite loaded");

    let cfg = cli.to_eval_config();
    // Validate before constructing the harness so bad CLI/env input fails
    // fast with a structured error rather than silently degrading inside
    // the best-effort Phase B exporter dispatcher (which logs warnings on
    // exporter failure and would otherwise yield an exit-0 "success" for
    // a config the harness can't honour, e.g. an oversized batch size).
    let cfg_errors = cfg.validate();
    if !cfg_errors.is_empty() {
        anyhow::bail!(
            "invalid evaluation configuration ({} error(s)): {}",
            cfg_errors.len(),
            cfg_errors.join("; ")
        );
    }
    let harness = EvalHarness::new(cfg);
    info!("running eval suite — exporter dispatch is automatic via EvalHarness");
    let factory = noop_agent_factory();
    let scorecard = harness.evaluate_suite(&suite, &factory);
    info!(
        overall_score = scorecard.overall_score,
        total_episodes = scorecard.summary.total_episodes,
        wall_clock_seconds = scorecard.summary.wall_clock_seconds,
        "eval complete"
    );

    std::fs::create_dir_all(&cli.output_dir)
        .with_context(|| format!("creating output dir {}", cli.output_dir.display()))?;
    let scorecard_path = cli.output_dir.join("scorecard.json");
    let file = std::fs::File::create(&scorecard_path)
        .with_context(|| format!("creating {}", scorecard_path.display()))?;
    serde_json::to_writer_pretty(file, &scorecard)
        .with_context(|| format!("writing {}", scorecard_path.display()))?;
    info!(path = %scorecard_path.display(), "scorecard written");

    Ok(0)
}

// ─── NoopEvalAgent (CLI-side default factory) ──────────────────────────────
//
// Mirrors `tests::NoopEvalAgent` in harness.rs but lives here because the
// harness's version is `#[cfg(test)]`-gated. Returns `AgentResponse::from_action(0)`
// for every observation — fine for plumbing validation, NOT for producing
// useful scores. A future revision of this CLI will accept a `--policy`
// flag selecting between built-in factories (NoopAgent / RandomAgent /
// loaded-BC-weights agent).

/// Identity used by the CLI's default agent factory. Surfaces in the
/// scorecard's `agent_metadata.model_name`.
const NOOP_AGENT_NAME: &str = "ForgeEvalLongrunNoopAgent";

struct NoopEvalAgent;

impl AgentInterface for NoopEvalAgent {
    fn select_action(&mut self, _obs: &Observation, _agent_idx: usize) -> AgentResponse {
        AgentResponse::from_action(0)
    }
    fn name(&self) -> &str {
        NOOP_AGENT_NAME
    }
    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::heuristic(NOOP_AGENT_NAME)
    }
}

/// Build a Noop agent factory. Wrapped in `Arc<Mutex<>>`-free closure
/// because the trait requires `Send + Sync` and our factory closure
/// returns a fresh boxed agent per call.
fn noop_agent_factory() -> impl Fn() -> Box<dyn AgentInterface> + Send + Sync {
    || Box::new(NoopEvalAgent) as Box<dyn AgentInterface>
}

// Silence the unused `Arc`/`Mutex` imports when only the public `Cli`
// surface is consumed — keeps the future hook for richer agent factories
// without an extra import line.
#[allow(dead_code)]
fn _silence_arc_mutex() -> (Arc<()>, Mutex<()>) {
    (Arc::new(()), Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: parse a fixed argv into a `Cli` without touching the
    /// process env (clap's `try_parse_from` ignores env-binding when an
    /// arg is supplied explicitly).
    fn parse(argv: &[&str]) -> Cli {
        Cli::try_parse_from(argv).expect("Cli::try_parse_from")
    }

    /// Every required arg + every default produces a usable Cli; the
    /// defaults bind to the canonical `DEFAULT_*` constants from
    /// `mlflow_http` (no magic numbers in this file).
    #[test]
    fn cli_required_args_only_uses_canonical_defaults() {
        let cli = parse(&[
            "forge-eval-longrun",
            "--suite",
            "configs/scenarios",
            "--tracking-uri",
            "http://localhost:5000",
            "--output-dir",
            "artifacts/e2e",
        ]);
        assert_eq!(cli.suite, PathBuf::from("configs/scenarios"));
        assert_eq!(cli.tracking_uri, "http://localhost:5000");
        assert_eq!(cli.output_dir, PathBuf::from("artifacts/e2e"));
        // Every numeric / string default tracks the canonical const.
        assert_eq!(cli.experiment_name, DEFAULT_EXPERIMENT_NAME);
        assert_eq!(cli.http_timeout_ms, DEFAULT_HTTP_TIMEOUT_MS);
        assert_eq!(cli.http_max_retries, DEFAULT_HTTP_MAX_RETRIES);
        assert_eq!(cli.http_backoff_base_ms, DEFAULT_HTTP_BACKOFF_BASE_MS);
        assert_eq!(cli.http_batch_size, DEFAULT_LOG_BATCH_SIZE);
        assert_eq!(cli.episodes_per_scenario, 1);
        assert!(cli.run_id.is_none());
    }

    /// Explicit `--`-style flags override the defaults end-to-end.
    #[test]
    fn cli_explicit_flags_override_every_default() {
        let cli = parse(&[
            "forge-eval-longrun",
            "--suite",
            "s",
            "--tracking-uri",
            "http://mlflow.example:5000",
            "--output-dir",
            "out",
            "--experiment-name",
            "custom-exp",
            "--episodes-per-scenario",
            "42",
            "--run-id",
            "fixed-run-id",
            "--http-timeout-ms",
            "11000",
            "--http-max-retries",
            "9",
            "--http-backoff-base-ms",
            "333",
            "--http-batch-size",
            "250",
        ]);
        assert_eq!(cli.experiment_name, "custom-exp");
        assert_eq!(cli.episodes_per_scenario, 42);
        assert_eq!(cli.run_id.as_deref(), Some("fixed-run-id"));
        assert_eq!(cli.http_timeout_ms, 11_000);
        assert_eq!(cli.http_max_retries, 9);
        assert_eq!(cli.http_backoff_base_ms, 333);
        assert_eq!(cli.http_batch_size, 250);
    }

    /// Missing a required arg produces a clap error (not a panic). Pins
    /// the contract that `--tracking-uri` / `--suite` / `--output-dir`
    /// are mandatory.
    #[test]
    fn cli_missing_required_args_errors_cleanly() {
        for missing in ["--suite", "--tracking-uri", "--output-dir"] {
            let argv: Vec<&str> = ["forge-eval-longrun"]
                .iter()
                .chain(
                    [
                        ("--suite", "s"),
                        ("--tracking-uri", "u"),
                        ("--output-dir", "o"),
                    ]
                    .iter()
                    .filter(|(k, _)| *k != missing)
                    .flat_map(|(k, v)| [k, v]),
                )
                .copied()
                .collect();
            let err =
                Cli::try_parse_from(&argv).expect_err("must error when a required arg is absent");
            // clap returns a structured error with a `MissingRequiredArgument` kind.
            assert_eq!(
                err.kind(),
                clap::error::ErrorKind::MissingRequiredArgument,
                "missing {missing} produced {err:?}",
            );
        }
    }

    /// `to_eval_config()` populates every HTTP field from the parsed Cli
    /// and sets `mlflow_http_tracking_uri` to `Some(...)` so the harness's
    /// Phase B dispatcher routes to the HTTP sink rather than the
    /// filesystem fallback.
    #[test]
    fn cli_to_eval_config_threads_every_http_field_through() {
        let cli = parse(&[
            "forge-eval-longrun",
            "--suite",
            "s",
            "--tracking-uri",
            "http://h",
            "--output-dir",
            "o",
            "--episodes-per-scenario",
            "5",
            "--http-timeout-ms",
            "1234",
            "--http-max-retries",
            "2",
            "--http-backoff-base-ms",
            "50",
            "--http-batch-size",
            "7",
            "--experiment-name",
            "exp",
            "--run-id",
            "r",
            "--hf-export-root",
            "out/hf",
        ]);
        let cfg = cli.to_eval_config();
        assert_eq!(cfg.episodes_per_scenario, 5);
        assert_eq!(cfg.mlflow_http_tracking_uri.as_deref(), Some("http://h"));
        assert_eq!(cfg.mlflow_http_timeout_ms, 1234);
        assert_eq!(cfg.mlflow_http_max_retries, 2);
        assert_eq!(cfg.mlflow_http_backoff_base_ms, 50);
        assert_eq!(cfg.mlflow_http_batch_size, 7);
        assert_eq!(cfg.experiment_name.as_deref(), Some("exp"));
        assert_eq!(cfg.run_id.as_deref(), Some("r"));
        assert_eq!(
            cfg.huggingface_export_root.as_deref(),
            Some(std::path::Path::new("out/hf")),
            "--hf-export-root must flow into EvalConfig so the harness's HF exporter is enabled in Phase B"
        );
    }

    /// `--hf-export-root` is optional — when omitted, `EvalConfig`'s
    /// `huggingface_export_root` stays `None` and the harness skips the
    /// HF exporter without erroring out.
    #[test]
    fn cli_omits_hf_export_root_when_flag_absent() {
        let cli = parse(&[
            "forge-eval-longrun",
            "--suite",
            "s",
            "--tracking-uri",
            "http://h",
            "--output-dir",
            "o",
        ]);
        let cfg = cli.to_eval_config();
        assert!(
            cfg.huggingface_export_root.is_none(),
            "no --hf-export-root flag must leave HF disabled, got {:?}",
            cfg.huggingface_export_root,
        );
    }

    #[test]
    fn noop_agent_metadata_carries_constant_name() {
        let mut agent = NoopEvalAgent;
        assert_eq!(agent.name(), NOOP_AGENT_NAME);
        let meta = agent.metadata();
        assert_eq!(meta.model_name, NOOP_AGENT_NAME);
        // select_action returns the discrete 0 action for any observation.
        let obs = Observation::default();
        let resp = agent.select_action(&obs, 0);
        assert_eq!(resp.action_id, 0);
    }

    #[test]
    fn noop_agent_factory_yields_independent_instances() {
        let factory = noop_agent_factory();
        let a = factory();
        let b = factory();
        // Both report the same canonical name (same class), but they are
        // distinct heap allocations — factory creates a fresh agent per
        // call so episode parallelism is safe.
        assert_eq!(a.name(), b.name());
        assert_eq!(a.name(), NOOP_AGENT_NAME);
    }
}
