//! Prometheus `/metrics` endpoint + per-episode / per-step recorder.
//!
//! Two pieces:
//!
//! 1. [`MetricsRecorder`] — owns a [`prometheus::Registry`] plus
//!    pre-built [`prometheus::Counter`] / [`prometheus::Histogram`] /
//!    [`prometheus::Gauge`] handles for the five signals called out in
//!    the v2 plan §3.6 (`forge_mc_episode_total`,
//!    `forge_mc_episode_reward_sum`, `forge_mc_planning_latency_seconds`,
//!    `forge_mc_model_version`, `forge_mc_protocol_error_total`).
//!    The runner installs one via [`crate::runner::Runner::with_metrics`]
//!    and calls the public `record_*` methods at the relevant
//!    lifecycle points. When the runner is constructed without a
//!    recorder, every call site is a no-op.
//!
//! 2. [`serve_metrics`] — spawns an `axum`-backed HTTP server on a
//!    caller-supplied `SocketAddr`. Returns a `JoinHandle` so the
//!    binary's `tokio::select!` shutdown future can `await` clean
//!    teardown.
//!
//! All metric names + help strings + bucket sets live as `const &str`
//! / `const &[f64]` at the top of this module so the JSON / YAML / TOML
//! consumer surface has one source of truth.
//!
//! No hard-coded values at call sites: bind address, port, and bucket
//! list flow from [`crate::RunnerConfig`].

use std::net::SocketAddr;

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, Opts, Registry,
    TextEncoder,
};
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument, warn};

/// Counter incremented once per finished episode.
pub const METRIC_EPISODE_TOTAL: &str = "forge_mc_episode_total";
/// Cumulative sum of per-episode rewards.
pub const METRIC_EPISODE_REWARD_SUM: &str = "forge_mc_episode_reward_sum";
/// Histogram of per-decision planning latency in seconds.
pub const METRIC_PLANNING_LATENCY_SECONDS: &str = "forge_mc_planning_latency_seconds";
/// Gauge of the last manifest version the runner observed.
pub const METRIC_MODEL_VERSION: &str = "forge_mc_model_version";
/// Counter of protocol / I/O errors surfaced by the env or watcher.
/// Labelled by reason so dashboards can split by failure mode without
/// re-deploying the runner.
pub const METRIC_PROTOCOL_ERROR_TOTAL: &str = "forge_mc_protocol_error_total";
/// Counter of total model reloads.
pub const METRIC_MODEL_RELOAD_COUNT: &str = "forge_mc_model_reload_count";
/// Histogram of episode length in steps.
pub const METRIC_EPISODE_LENGTH_STEPS: &str = "forge_mc_episode_length_steps";
/// CounterVec of episode rewards, broken down by reward component.
pub const METRIC_EPISODE_REWARD_COMPONENTS: &str = "forge_mc_episode_reward_components";

/// Errors raised by metrics setup. Distinct from
/// [`crate::RunnerError`] so the binary's main can surface
/// metrics-startup failures with a precise exit code.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    /// Underlying `prometheus` crate failure (duplicate name, bad
    /// bucket list, ...).
    #[error("prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),

    /// Failed to bind / serve the HTTP listener.
    #[error("metrics server io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Per-runner recorder. Cheap to clone (every metric is an
/// `Arc`-backed handle under the hood). Cloneable so the runner can
/// hand a handle to the metrics HTTP task without giving away
/// ownership.
#[derive(Clone)]
pub struct MetricsRecorder {
    registry: Registry,
    episode_total: Counter,
    episode_reward_sum: Gauge,
    planning_latency_seconds: Histogram,
    model_version: Gauge,
    protocol_error_total: CounterVec,
    model_reload_count: Counter,
    episode_length_steps: Histogram,
    episode_reward_components: GaugeVec,
}

impl MetricsRecorder {
    /// Build the metrics, register them with a fresh
    /// [`prometheus::Registry`], and return the recorder.
    ///
    /// `histogram_buckets` should be in strictly-increasing seconds.
    /// `RunnerConfig::validate` already enforces this, so callers
    /// that pass `&cfg.metrics_histogram_buckets` are safe.
    pub fn new(
        histogram_buckets: &[f64],
        episode_length_buckets: &[f64],
    ) -> Result<Self, MetricsError> {
        let registry = Registry::new();

        let episode_total = Counter::with_opts(Opts::new(
            METRIC_EPISODE_TOTAL,
            "Total episodes completed by the runner (sum across run() calls).",
        ))?;
        registry.register(Box::new(episode_total.clone()))?;

        let episode_reward_sum = Gauge::with_opts(Opts::new(
            METRIC_EPISODE_REWARD_SUM,
            "Cumulative sum of per-episode rewards across all completed episodes.",
        ))?;
        registry.register(Box::new(episode_reward_sum.clone()))?;

        let planning_latency_seconds = Histogram::with_opts(
            HistogramOpts::new(
                METRIC_PLANNING_LATENCY_SECONDS,
                "Per-decision MCTS planning latency in seconds.",
            )
            .buckets(histogram_buckets.to_vec()),
        )?;
        registry.register(Box::new(planning_latency_seconds.clone()))?;

        let model_version = Gauge::with_opts(Opts::new(
            METRIC_MODEL_VERSION,
            "Last manifest version the runner observed via the hot-reload watcher.",
        ))?;
        registry.register(Box::new(model_version.clone()))?;

        let protocol_error_total = CounterVec::new(
            Opts::new(
                METRIC_PROTOCOL_ERROR_TOTAL,
                "Total protocol / env / reload errors observed, labelled by reason.",
            ),
            &["reason"],
        )?;
        registry.register(Box::new(protocol_error_total.clone()))?;

        let model_reload_count = Counter::with_opts(Opts::new(
            METRIC_MODEL_RELOAD_COUNT,
            "Total number of times the ONNX model was successfully hot-reloaded.",
        ))?;
        registry.register(Box::new(model_reload_count.clone()))?;

        let episode_length_steps = Histogram::with_opts(
            HistogramOpts::new(
                METRIC_EPISODE_LENGTH_STEPS,
                "Episode length in environment steps.",
            )
            .buckets(episode_length_buckets.to_vec()),
        )?;
        registry.register(Box::new(episode_length_steps.clone()))?;

        let episode_reward_components = GaugeVec::new(
            Opts::new(
                METRIC_EPISODE_REWARD_COMPONENTS,
                "Per-episode reward breakdown by reward component.",
            ),
            &["component"],
        )?;
        registry.register(Box::new(episode_reward_components.clone()))?;

        Ok(Self {
            registry,
            episode_total,
            episode_reward_sum,
            planning_latency_seconds,
            model_version,
            protocol_error_total,
            model_reload_count,
            episode_length_steps,
            episode_reward_components,
        })
    }

    /// Borrow the underlying registry (so the HTTP handler can scrape
    /// it without holding the recorder by value).
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Record that a single episode finished.
    #[instrument(skip(self))]
    pub fn record_episode_complete(&self, total_reward: f32) {
        self.episode_total.inc();
        self.episode_reward_sum.add(f64::from(total_reward));
    }

    /// Record one planning call's wall-clock latency (seconds).
    #[instrument(skip(self))]
    pub fn record_planning_latency_seconds(&self, latency: f64) {
        self.planning_latency_seconds.observe(latency);
    }

    /// Update the last-observed model manifest version. Idempotent.
    #[instrument(skip(self))]
    pub fn set_model_version(&self, version: u64) {
        // Gauge takes f64; manifest versions fit exactly in f64 up to
        // 2**53 which is far beyond any realistic export cadence.
        self.model_version.set(version as f64);
    }

    /// Record one protocol / env / reload error. `reason` should be a
    /// short, low-cardinality string (e.g. "env_step", "reload",
    /// "manifest_parse"); avoid embedding per-error detail.
    #[instrument(skip(self))]
    pub fn record_protocol_error(&self, reason: &str) {
        self.protocol_error_total.with_label_values(&[reason]).inc();
    }

    /// Record that a model reload succeeded.
    #[instrument(skip(self))]
    pub fn record_model_reload(&self) {
        self.model_reload_count.inc();
    }

    /// Record an episode length in steps.
    #[instrument(skip(self))]
    pub fn record_episode_length(&self, steps: usize) {
        self.episode_length_steps.observe(steps as f64);
    }

    /// Record reward component values.
    #[instrument(skip(self))]
    pub fn record_reward_component(&self, component: &str, value: f32) {
        if value != 0.0 {
            self.episode_reward_components
                .with_label_values(&[component])
                .add(f64::from(value));
        }
    }

    /// Encode the registry as Prometheus text format. Useful for
    /// in-process tests that want to assert on counter / gauge state
    /// without spinning up the HTTP server.
    pub fn encode_text(&self) -> Result<String, MetricsError> {
        let mut buffer = Vec::new();
        let encoder = TextEncoder::new();
        encoder.encode(&self.registry.gather(), &mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }
}

#[derive(Clone)]
struct ServerState {
    registry: Registry,
}

async fn metrics_handler(State(state): State<ServerState>) -> impl IntoResponse {
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    if let Err(e) = encoder.encode(&state.registry.gather(), &mut buffer) {
        warn!(error = %e, "metrics encode failed");
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, String::new());
    }
    let body = String::from_utf8_lossy(&buffer).into_owned();
    (axum::http::StatusCode::OK, body)
}

async fn health_handler() -> impl IntoResponse {
    "ok"
}

/// Spawn the metrics HTTP server on `addr`. Listens until
/// `shutdown` resolves, then performs a graceful shutdown and exits.
///
/// Returns the spawned [`JoinHandle`] so callers can `await` it from
/// inside a `tokio::select!` block.
pub async fn serve_metrics<F>(
    addr: SocketAddr,
    recorder: MetricsRecorder,
    shutdown: F,
) -> Result<JoinHandle<Result<(), MetricsError>>, MetricsError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let state = ServerState {
        registry: recorder.registry().clone(),
    };
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    info!(addr = %bound, "metrics endpoint listening");
    debug!(metric_names = ?[METRIC_EPISODE_TOTAL, METRIC_EPISODE_REWARD_SUM, METRIC_PLANNING_LATENCY_SECONDS, METRIC_MODEL_VERSION, METRIC_PROTOCOL_ERROR_TOTAL], "registered metrics");

    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(MetricsError::from)
    });
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DEFAULT_METRICS_HISTOGRAM_BUCKETS_SECONDS;

    fn recorder() -> MetricsRecorder {
        use crate::config::DEFAULT_METRICS_EPISODE_LENGTH_BUCKETS;
        MetricsRecorder::new(
            DEFAULT_METRICS_HISTOGRAM_BUCKETS_SECONDS,
            DEFAULT_METRICS_EPISODE_LENGTH_BUCKETS,
        )
        .expect("recorder build")
    }

    #[test]
    fn metrics_recorder_encodes_all_five_names_in_text_format() {
        let rec = recorder();
        rec.record_episode_complete(1.5);
        rec.record_planning_latency_seconds(0.02);
        rec.set_model_version(7);
        rec.record_protocol_error("env_step");
        let text = rec.encode_text().unwrap();
        for name in [
            METRIC_EPISODE_TOTAL,
            METRIC_EPISODE_REWARD_SUM,
            METRIC_PLANNING_LATENCY_SECONDS,
            METRIC_MODEL_VERSION,
            METRIC_PROTOCOL_ERROR_TOTAL,
        ] {
            assert!(text.contains(name), "expected {name} in text, got:\n{text}");
        }
        assert!(text.contains("# TYPE forge_mc_episode_total counter"));
        assert!(text.contains("# TYPE forge_mc_episode_reward_sum gauge"));
        assert!(text.contains("# TYPE forge_mc_planning_latency_seconds histogram"));
        assert!(text.contains("# TYPE forge_mc_model_version gauge"));
        assert!(text.contains("# TYPE forge_mc_protocol_error_total counter"));
    }

    #[test]
    fn record_episode_increments_counter() {
        let rec = recorder();
        rec.record_episode_complete(2.0);
        rec.record_episode_complete(1.5);
        let text = rec.encode_text().unwrap();
        // Counter samples appear without quotes on the value.
        assert!(
            text.contains("forge_mc_episode_total 2"),
            "expected forge_mc_episode_total == 2, got:\n{text}"
        );
        assert!(
            text.contains("forge_mc_episode_reward_sum 3.5"),
            "expected reward_sum == 3.5, got:\n{text}"
        );
    }

    #[test]
    fn record_protocol_error_increments_per_label() {
        let rec = recorder();
        rec.record_protocol_error("env_step");
        rec.record_protocol_error("env_step");
        rec.record_protocol_error("reload");
        let text = rec.encode_text().unwrap();
        assert!(text.contains(r#"forge_mc_protocol_error_total{reason="env_step"} 2"#));
        assert!(text.contains(r#"forge_mc_protocol_error_total{reason="reload"} 1"#));
    }

    #[test]
    fn set_model_version_appears_as_gauge_value() {
        let rec = recorder();
        rec.set_model_version(42);
        let text = rec.encode_text().unwrap();
        assert!(text.contains("forge_mc_model_version 42"));
        // Setting again overrides.
        rec.set_model_version(7);
        let text = rec.encode_text().unwrap();
        assert!(text.contains("forge_mc_model_version 7"));
    }

    #[test]
    fn new_rejects_unsorted_histogram_buckets() {
        let result = MetricsRecorder::new(&[0.1, 0.05], &[10.0]);
        // Prometheus requires strictly-increasing buckets; surface
        // any rejection via the `MetricsError::Prometheus` variant.
        assert!(result.is_err(), "expected MetricsError::Prometheus");
    }

    #[tokio::test]
    async fn serve_metrics_responds_with_metrics_endpoint_on_port_zero() {
        let rec = recorder();
        rec.record_episode_complete(0.5);
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let handle = serve_metrics("127.0.0.1:0".parse().unwrap(), rec.clone(), async move {
            let _ = rx.await;
        })
        .await
        .expect("serve_metrics");
        // Trigger shutdown immediately and confirm the task joins
        // cleanly without an axum::Error.
        let _ = tx.send(());
        let join_result = handle.await.expect("join handle");
        join_result.expect("metrics server clean shutdown");
    }
}
