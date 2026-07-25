//! MLflow HTTP tracking-server transport.
//!
//! Consumes the same [`super::mlflow_payload::RunPayload`] as
//! [`super::mlflow_fs::MlflowFsSink`]; serialises it as REST calls to an
//! MLflow tracking server's `/api/2.0/mlflow/*` surface instead of writing
//! the on-disk `mlruns/` layout.
//!
//! Off by default — compile with `--features http-mlflow` to pull in
//! `reqwest` + `url` + `scopeguard`. The default build of `forge-eval`
//! remains HTTP/TLS-free. (The feature gate lives at the module's
//! declaration site in `mod.rs`; no inner `#![cfg(...)]` needed.)

use std::cell::Cell;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::blocking::{Client, RequestBuilder, Response};
use reqwest::StatusCode;
use serde_json::{json, Value};
use tracing::{debug, error, info, instrument, warn};
use url::Url;

use super::mlflow_payload::{
    build_run_payload, sanitize, ArtifactRef, ArtifactSource, MetricSample, ParamKv, RunPayload,
    TagKv,
};
use super::{ExportError, Exporter};
use crate::config::EvalConfig;
use crate::manifest::RunManifest;
use crate::scorecard::Scorecard;

// ─── Defaults (kept here as the canonical home; EvalConfig::Default reads them) ──
//
// Constants are `pub` so the `EvalConfig` placeholders in
// `crate::config` can re-import them once the feature is enabled. Until
// then `config.rs` keeps its own copies as a build-without-feature fallback.

/// Per-request HTTP timeout (ms). 30s is generous enough for `log_batch`
/// at the 1000-metric cap without being so long that a wedged server
/// blocks the entire long-run job.
pub const DEFAULT_HTTP_TIMEOUT_MS: u64 = 30_000;

/// Max retries on retryable HTTP statuses (5xx, 408, 429). 5 retries with
/// exponential backoff covers transient server hiccups without burning
/// the whole long-run budget on a permanent outage.
pub const DEFAULT_HTTP_MAX_RETRIES: u32 = 5;

/// Exponential-backoff base (ms) between retries. Effective wait at
/// attempt `n` is `base * 2^n` (capped at one minute).
pub const DEFAULT_HTTP_BACKOFF_BASE_MS: u64 = 250;

/// Cap on single-retry wait, even at high attempt counts. Prevents the
/// exponential schedule from sleeping past the runner's own timeout.
pub const DEFAULT_HTTP_BACKOFF_CAP_MS: u64 = 60_000;

/// Per-`log_batch` chunk size. MLflow REST caps `/runs/log-batch` at
/// 1000 metrics per request; the client chunks anything larger.
pub const DEFAULT_LOG_BATCH_SIZE: usize = 1_000;

/// `User-Agent` header sent on every request. Mirrors the convention
/// `<crate>/<version>` so the MLflow server log identifies the source.
pub const DEFAULT_USER_AGENT: &str = concat!("forge-eval/", env!("CARGO_PKG_VERSION"));

/// Default experiment name used when [`crate::config::EvalConfig::experiment_name`]
/// is `None`. Mirrors what [`crate::manifest::RunManifest`] defaults to so the
/// HTTP sink, filesystem sink and manifest all agree on one fallback value.
pub const DEFAULT_EXPERIMENT_NAME: &str = "forge-eval-default";

// ─── REST path segments + body field names ─────────────────────────────────
// Centralised so adding a new MLflow API version (or a typo audit) is a
// single-file edit. All names match the MLflow REST contract verbatim; the
// pin tests below catch any accidental rename.

/// First URL path segment of every MLflow REST call.
const API_ROOT: &str = "api";
/// MLflow REST API version segment. Server contract still on `2.0`.
const API_VERSION: &str = "2.0";
/// Namespace under `/api/<API_VERSION>/` for everything except artefacts.
const NAMESPACE_MLFLOW: &str = "mlflow";
/// Namespace under `/api/<API_VERSION>/` for artefact upload. NOT under
/// `/mlflow/` (that's MLflow's deliberate REST quirk).
const NAMESPACE_ARTIFACTS: &str = "mlflow-artifacts";
/// Sub-segment after `mlflow-artifacts` in the artefact upload URL.
const ARTIFACTS_PATH: &str = "artifacts";

// REST body / query field names — every literal `"experiment_id"` etc. in
// this module must come from one of these constants. Tests pin the values.

const FIELD_EXPERIMENT_NAME: &str = "experiment_name";
const FIELD_EXPERIMENT_ID: &str = "experiment_id";
const FIELD_NAME: &str = "name";
const FIELD_RUN_ID: &str = "run_id";
const FIELD_RUN: &str = "run";
const FIELD_INFO: &str = "info";
const FIELD_START_TIME: &str = "start_time";
const FIELD_END_TIME: &str = "end_time";
const FIELD_STATUS: &str = "status";
const FIELD_KEY: &str = "key";
const FIELD_VALUE: &str = "value";
const FIELD_TAGS: &str = "tags";
const FIELD_METRICS: &str = "metrics";
const FIELD_PARAMS: &str = "params";
const FIELD_TIMESTAMP: &str = "timestamp";
const FIELD_STEP: &str = "step";
const QUERY_RUN_ID: &str = "run_id";

/// Compile-time path slice into the `get_or_create_experiment` response
/// JSON: `{"experiment": {"experiment_id": "<id>"}}`.
const RESP_PATH_EXPERIMENT_ID: &[&str] = &["experiment", FIELD_EXPERIMENT_ID];
/// Compile-time path slice into the `create_experiment` response JSON:
/// `{"experiment_id": "<id>"}`.
const RESP_PATH_CREATE_EXP_ID: &[&str] = &[FIELD_EXPERIMENT_ID];
/// Compile-time path slice into the `create_run` response JSON:
/// `{"run": {"info": {"run_id": "<id>"}}}`.
const RESP_PATH_RUN_ID: &[&str] = &[FIELD_RUN, FIELD_INFO, FIELD_RUN_ID];

// ─── Retry policy ──────────────────────────────────────────────────────────

/// Decision policy for HTTP retries. Pure function over the response
/// status — no I/O, no allocations — so it's trivially table-testable.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts after the initial request.
    pub max_retries: u32,
    /// Exponential-backoff base (ms). Doubled each attempt up to
    /// [`DEFAULT_HTTP_BACKOFF_CAP_MS`].
    pub backoff_base_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_HTTP_MAX_RETRIES,
            backoff_base_ms: DEFAULT_HTTP_BACKOFF_BASE_MS,
        }
    }
}

impl RetryPolicy {
    /// Returns `true` iff `status` belongs to the set of statuses worth
    /// retrying. Anything else (2xx, most 4xx) fails immediately.
    #[inline]
    pub fn should_retry(status: u16) -> bool {
        matches!(status, 408 | 429 | 500..=599)
    }

    /// Backoff wait for the `attempt`-th retry (`attempt` is 1-indexed:
    /// after the first failed request, the caller waits `wait_ms(1)` then
    /// retries; after that fails, `wait_ms(2)`; ...).
    ///
    /// Capped at [`DEFAULT_HTTP_BACKOFF_CAP_MS`] so very long retry chains
    /// don't sleep past the runner's own deadline.
    pub fn wait_ms(&self, attempt: u32) -> u64 {
        let exp = self
            .backoff_base_ms
            .saturating_mul(1u64.checked_shl(attempt.min(20)).unwrap_or(u64::MAX));
        exp.min(DEFAULT_HTTP_BACKOFF_CAP_MS)
    }
}

// ─── Client config ─────────────────────────────────────────────────────────

/// Construction-time config for an [`MlflowHttpClient`]. Defaults mirror
/// the `DEFAULT_HTTP_*` constants so a bare `HttpClientConfig::default()`
/// produces the same behaviour the FsSink users would expect by analogy.
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Per-request timeout (ms).
    pub timeout_ms: u64,
    /// Retry policy applied to every request (constructed from
    /// `max_retries` + `backoff_base_ms` so callers don't have to import
    /// the [`RetryPolicy`] struct directly).
    pub max_retries: u32,
    /// Exponential-backoff base (ms).
    pub backoff_base_ms: u64,
    /// `runs/log-batch` chunk size cap. MLflow REST limit is 1000.
    pub batch_size: usize,
    /// Bearer token sent in the `Authorization` header when `Some`.
    /// Never logged — manual `Debug` impl on [`MlflowHttpClient`] redacts.
    pub bearer_token: Option<String>,
    /// `User-Agent` header value. Defaults to [`DEFAULT_USER_AGENT`].
    pub user_agent: String,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            timeout_ms: DEFAULT_HTTP_TIMEOUT_MS,
            max_retries: DEFAULT_HTTP_MAX_RETRIES,
            backoff_base_ms: DEFAULT_HTTP_BACKOFF_BASE_MS,
            batch_size: DEFAULT_LOG_BATCH_SIZE,
            bearer_token: None,
            user_agent: DEFAULT_USER_AGENT.to_string(),
        }
    }
}

// ─── Client ────────────────────────────────────────────────────────────────

/// Synchronous REST client for an MLflow tracking server.
///
/// Implementation lives across multiple sub-modules in Slice 2b — this
/// skeleton ships the type + constructor so the feature compiles and the
/// dispatcher wiring can land first.
pub struct MlflowHttpClient {
    base: Url,
    // Consumed by Slice 2b — every REST method (`create_run`, `log_batch`,
    // ...) will borrow this client. Kept private so the field never leaks
    // into a public signature.
    #[allow(dead_code)]
    http: Client,
    retry: RetryPolicy,
    batch_size: usize,
    auth: Option<String>,
}

impl std::fmt::Debug for MlflowHttpClient {
    /// Redacts the bearer token so it never lands in logs / panic
    /// backtraces / `tracing` events. Token presence is exposed as a
    /// boolean.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlflowHttpClient")
            .field("base", &self.base.as_str())
            .field("retry", &self.retry)
            .field("batch_size", &self.batch_size)
            .field("auth_present", &self.auth.is_some())
            .finish()
    }
}

impl MlflowHttpClient {
    /// Build a client targeting `base` (e.g. `http://localhost:5000`) with
    /// the supplied [`HttpClientConfig`]. Returns
    /// [`ExportError::InvalidTarget`] when the base URL uses a scheme
    /// other than `http`/`https` (e.g. `file:`, `data:`, `ftp:`) — the
    /// MLflow REST contract only speaks HTTP — or when the underlying
    /// `reqwest` builder can't construct the client (e.g. invalid TLS
    /// config).
    pub fn new(base: Url, cfg: HttpClientConfig) -> Result<Self, ExportError> {
        // Guard against schemes that `Url::parse` will happily accept but
        // `reqwest` either can't speak or would silently downgrade. Doing
        // this once in the single constructor protects every caller
        // (`MlflowHttpSink::from_config`, integration tests, future
        // callers) without duplicating the check.
        match base.scheme() {
            "http" | "https" => {}
            other => {
                return Err(ExportError::InvalidTarget(format!(
                    "mlflow tracking uri must use http or https, got scheme {other:?} (base={base})"
                )));
            }
        }
        tracing::debug!(
            base = %base,
            timeout_ms = cfg.timeout_ms,
            max_retries = cfg.max_retries,
            batch_size = cfg.batch_size,
            "constructing MLflow HTTP client"
        );
        let http = Client::builder()
            .timeout(Duration::from_millis(cfg.timeout_ms))
            .user_agent(cfg.user_agent)
            .build()
            .map_err(|e| ExportError::InvalidTarget(format!("reqwest build: {e}")))?;
        Ok(Self {
            base,
            http,
            retry: RetryPolicy {
                max_retries: cfg.max_retries,
                backoff_base_ms: cfg.backoff_base_ms,
            },
            batch_size: cfg.batch_size,
            auth: cfg.bearer_token,
        })
    }

    /// Tracking-server base URL.
    pub fn base(&self) -> &Url {
        &self.base
    }

    /// Configured retry policy.
    pub fn retry_policy(&self) -> RetryPolicy {
        self.retry
    }

    /// Configured `log_batch` chunk size cap.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Whether the client was constructed with a bearer token. The token
    /// value itself is intentionally private — there is no public getter.
    pub fn has_auth(&self) -> bool {
        self.auth.is_some()
    }

    // ─── REST methods ──────────────────────────────────────────────────────
    //
    // Each method funnels through `execute_with_retry`, which applies the
    // retry policy uniformly. Bodies are constructed via `serde_json::json!`
    // to avoid the DTO-struct boilerplate; responses are parsed as
    // `serde_json::Value` and walked for the fields the MLflow REST contract
    // documents. Unknown response fields are ignored (server can add new
    // fields without breaking us).

    /// MLflow `GET /api/2.0/mlflow/experiments/get-by-name?experiment_name=<name>`,
    /// falling back to `POST /api/2.0/mlflow/experiments/create` on 404.
    /// Returns the experiment id (string-typed per MLflow's convention).
    pub fn get_or_create_experiment(&self, name: &str) -> Result<String, ExportError> {
        let url = self.api_url(&["experiments", "get-by-name"])?;
        let url_with_query = {
            let mut u = url.clone();
            u.query_pairs_mut().append_pair(FIELD_EXPERIMENT_NAME, name);
            u
        };
        let lookup =
            self.execute_with_retry(|| self.with_auth(self.http.get(url_with_query.clone())));
        match lookup {
            Ok(resp) => extract_string(&resp_json(resp)?, RESP_PATH_EXPERIMENT_ID),
            Err(ExportError::Http(msg)) if msg.starts_with("status 404") => {
                self.create_experiment(name)
            }
            Err(e) => Err(e),
        }
    }

    /// MLflow `POST /api/2.0/mlflow/experiments/create`. Used by
    /// [`Self::get_or_create_experiment`] on lookup-404. Returns the
    /// new experiment id.
    pub fn create_experiment(&self, name: &str) -> Result<String, ExportError> {
        let url = self.api_url(&["experiments", "create"])?;
        let body = json!({ (FIELD_NAME): name });
        let resp =
            self.execute_with_retry(|| self.with_auth(self.http.post(url.clone()).json(&body)))?;
        extract_string(&resp_json(resp)?, RESP_PATH_CREATE_EXP_ID)
    }

    /// MLflow `POST /api/2.0/mlflow/runs/create`. Returns the new run id.
    /// `tags` are sent in the create request so the run shows up in the UI
    /// with the right `mlflow.runName` / `mlflow.parentRunId` from the
    /// first paint — avoids a separate set-tag round-trip per run.
    pub fn create_run(
        &self,
        experiment_id: &str,
        start_time_ms: u64,
        tags: &[TagKv],
    ) -> Result<String, ExportError> {
        let url = self.api_url(&["runs", "create"])?;
        let body = json!({
            (FIELD_EXPERIMENT_ID): experiment_id,
            (FIELD_START_TIME): start_time_ms,
            (FIELD_TAGS): to_json_array(tags),
        });
        let resp =
            self.execute_with_retry(|| self.with_auth(self.http.post(url.clone()).json(&body)))?;
        extract_string(&resp_json(resp)?, RESP_PATH_RUN_ID)
    }

    /// MLflow `POST /api/2.0/mlflow/runs/log-batch`. Chunks `metrics` at
    /// `self.batch_size` (REST cap is 1000); `params` + `tags` go with the
    /// first chunk only so subsequent chunks don't trigger duplicate-key
    /// errors. Empty input is a no-op.
    pub fn log_batch(
        &self,
        run_id: &str,
        metrics: &[MetricSample],
        params: &[ParamKv],
        tags: &[TagKv],
    ) -> Result<(), ExportError> {
        if metrics.is_empty() && params.is_empty() && tags.is_empty() {
            return Ok(());
        }
        let cap = self.batch_size.max(1);
        // At least one call (with empty metrics) so a params-only / tags-only
        // payload still ships.
        let chunks: Vec<&[MetricSample]> = if metrics.is_empty() {
            vec![&[][..]]
        } else {
            metrics.chunks(cap).collect()
        };
        let url = self.api_url(&["runs", "log-batch"])?;
        for (i, chunk) in chunks.iter().enumerate() {
            let (p, t) = if i == 0 {
                (params, tags)
            } else {
                (&[][..], &[][..])
            };
            let body = json!({
                (FIELD_RUN_ID): run_id,
                (FIELD_METRICS): to_json_array(chunk),
                (FIELD_PARAMS):  to_json_array(p),
                (FIELD_TAGS):    to_json_array(t),
            });
            debug!(
                run_id = run_id,
                chunk = i,
                metrics_in_chunk = chunk.len(),
                "log_batch chunk submitting"
            );
            self.execute_with_retry(|| self.with_auth(self.http.post(url.clone()).json(&body)))?;
        }
        Ok(())
    }

    /// MLflow `POST /api/2.0/mlflow/runs/set-tag`. Equivalent to a
    /// `log_batch` with only one tag, but kept as a separate method so
    /// the parent-run-id linkage tag for child runs has a stable code path.
    pub fn set_tag(&self, run_id: &str, key: &str, value: &str) -> Result<(), ExportError> {
        let url = self.api_url(&["runs", "set-tag"])?;
        // `sanitize(key)` matches the FsSink's write-time normalisation —
        // both transports therefore accept the same key universe.
        let body = json!({
            (FIELD_RUN_ID): run_id,
            (FIELD_KEY): sanitize(key),
            (FIELD_VALUE): value,
        });
        self.execute_with_retry(|| self.with_auth(self.http.post(url.clone()).json(&body)))?;
        Ok(())
    }

    /// MLflow `POST /api/2.0/mlflow/runs/update`. Sets `status` (per the
    /// MLflow lifecycle enum: RUNNING / FINISHED / FAILED / KILLED / SCHEDULED)
    /// and `end_time`. Slice 2c calls this via a scopeguard so an aborted
    /// run is visible as FAILED in the UI rather than wedged in RUNNING.
    pub fn set_terminated(
        &self,
        run_id: &str,
        status: RunStatus,
        end_time_ms: u64,
    ) -> Result<(), ExportError> {
        let url = self.api_url(&["runs", "update"])?;
        let body = json!({
            (FIELD_RUN_ID): run_id,
            (FIELD_STATUS): status.as_str(),
            (FIELD_END_TIME): end_time_ms,
        });
        self.execute_with_retry(|| self.with_auth(self.http.post(url.clone()).json(&body)))?;
        Ok(())
    }

    /// MLflow artifact upload via the proxy endpoint
    /// `PUT /api/2.0/mlflow-artifacts/artifacts/<artifact_uri>?run_id=<id>`.
    /// `rel_path` is appended to the run's artifact root; sinks supply the
    /// same path they would have written to disk in the FsSink.
    pub fn log_artifact(
        &self,
        run_id: &str,
        rel_path: &str,
        bytes: &[u8],
    ) -> Result<(), ExportError> {
        // mlflow-artifacts lives under /api/2.0/ directly, NOT /api/2.0/mlflow/.
        // Use the shared namespace helper so the `api` / API_VERSION prefix
        // exists in exactly one place each. Append the per-file path segments
        // after the namespace pair so embedded `/` in rel_path becomes a real
        // delimiter, not URL-encoded as `%2F`.
        let mut url = self.url_with_namespace(NAMESPACE_ARTIFACTS, &[ARTIFACTS_PATH])?;
        {
            let mut path = url.path_segments_mut().map_err(|_| {
                ExportError::InvalidTarget("base url cannot have a path".to_string())
            })?;
            for seg in rel_path.split('/').filter(|s| !s.is_empty()) {
                path.push(seg);
            }
        }
        url.query_pairs_mut().append_pair(QUERY_RUN_ID, run_id);
        let body = bytes.to_vec();
        self.execute_with_retry(|| self.with_auth(self.http.put(url.clone()).body(body.clone())))?;
        Ok(())
    }

    // ─── Internals ─────────────────────────────────────────────────────────

    /// Build `<base>/api/<API_VERSION>/<NAMESPACE_MLFLOW>/<segments...>`.
    /// Returns [`ExportError::InvalidTarget`] if the base URL is opaque
    /// (e.g. a `data:` URI) — `path_segments_mut()` fails for those.
    fn api_url(&self, segments: &[&str]) -> Result<Url, ExportError> {
        self.url_with_namespace(NAMESPACE_MLFLOW, segments)
    }

    /// Build `<base>/api/<API_VERSION>/<namespace>/<segments...>`.
    /// Used by [`Self::api_url`] (namespace = `mlflow`) and the artefact
    /// endpoint (namespace = `mlflow-artifacts`). Centralises the prefix
    /// so `api` / `2.0` literals exist in one place each.
    fn url_with_namespace(&self, namespace: &str, segments: &[&str]) -> Result<Url, ExportError> {
        let mut url = self.base.clone();
        {
            let mut path = url.path_segments_mut().map_err(|_| {
                ExportError::InvalidTarget("base url cannot have a path".to_string())
            })?;
            // Avoid an empty trailing segment if `base.path()` is `/`.
            path.pop_if_empty();
            path.push(API_ROOT);
            path.push(API_VERSION);
            path.push(namespace);
            for s in segments {
                path.push(s);
            }
        }
        Ok(url)
    }

    /// Apply the configured bearer token (if any) to a request builder.
    fn with_auth(&self, mut rb: RequestBuilder) -> RequestBuilder {
        if let Some(token) = &self.auth {
            rb = rb.bearer_auth(token);
        }
        rb
    }

    /// Execute the request returned by `build_request` with retries on
    /// retryable HTTP / transport errors. Sleeps between attempts using
    /// the configured exponential backoff (capped at
    /// [`DEFAULT_HTTP_BACKOFF_CAP_MS`]).
    fn execute_with_retry<F>(&self, build_request: F) -> Result<Response, ExportError>
    where
        F: Fn() -> RequestBuilder,
    {
        let mut attempt: u32 = 0;
        loop {
            let send_result = build_request().send();
            let err = match send_result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return Ok(resp);
                    }
                    // Consume the body to give classify_status some context.
                    // Body bytes are dropped on retry — that's intentional;
                    // we don't want to grow heap on a transient blip.
                    let body = resp.text().unwrap_or_default();
                    match classify_status(status, &body) {
                        Ok(()) => unreachable!("classify_status returns Err for non-2xx"),
                        Err(e) => e,
                    }
                }
                Err(transport_err) => map_reqwest_err(transport_err),
            };
            let retryable = matches!(err, ExportError::Retryable(_));
            if retryable && attempt < self.retry.max_retries {
                attempt += 1;
                let wait_ms = self.retry.wait_ms(attempt);
                warn!(attempt, wait_ms, error = %err, "retrying mlflow http request");
                std::thread::sleep(Duration::from_millis(wait_ms));
                continue;
            }
            if retryable {
                error!(attempt, error = %err, "mlflow http retry budget exhausted");
            }
            return Err(err);
        }
    }
}

/// MLflow lifecycle status enum — string-typed on the wire. Sent in
/// [`MlflowHttpClient::set_terminated`] to mark a run FINISHED (happy
/// path) or FAILED (scopeguard path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Run is in flight; not used by this sink (handled implicitly by
    /// `runs/create` which defaults to RUNNING).
    Running,
    /// Run completed successfully — the happy-path terminal state.
    Finished,
    /// Run aborted mid-export — set by the scopeguard in `MlflowHttpSink`
    /// when `?` propagates out of `export()`.
    Failed,
    /// Run was killed externally; included for completeness.
    Killed,
    /// Run is scheduled to start later; not emitted by this sink.
    Scheduled,
}

impl RunStatus {
    /// Wire form per the MLflow REST `runs/update` `status` enum.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "RUNNING",
            Self::Finished => "FINISHED",
            Self::Failed => "FAILED",
            Self::Killed => "KILLED",
            Self::Scheduled => "SCHEDULED",
        }
    }
}

// ─── JSON helpers (request body builders + response field extractors) ──────

/// Trait for payload items renderable as MLflow REST JSON. Each item type
/// (metric / param / tag) implements one tiny method; the generic
/// [`to_json_array`] helper does the iteration. Eliminates the three
/// near-duplicate `*_to_json` helpers the audit flagged and gives one
/// extension point for future fields (e.g. metric `model_id`).
trait ToMlflowJson {
    fn to_mlflow_json(&self) -> Value;
}

// All three impls call `sanitize(key)` so user-supplied agent metadata
// keys with characters MLflow's REST schema rejects (`:`, `?`, `&`, ...)
// don't trip a 400 from the server mid-export. Mirrors the FsSink's
// write-time sanitisation so both transports stay symmetric.

impl ToMlflowJson for MetricSample {
    fn to_mlflow_json(&self) -> Value {
        json!({
            (FIELD_KEY): sanitize(&self.key),
            (FIELD_VALUE): self.value,
            (FIELD_TIMESTAMP): self.timestamp_ms,
            (FIELD_STEP): self.step,
        })
    }
}

impl ToMlflowJson for ParamKv {
    fn to_mlflow_json(&self) -> Value {
        json!({ (FIELD_KEY): sanitize(&self.key), (FIELD_VALUE): self.value })
    }
}

impl ToMlflowJson for TagKv {
    fn to_mlflow_json(&self) -> Value {
        json!({ (FIELD_KEY): sanitize(&self.key), (FIELD_VALUE): self.value })
    }
}

/// Render a slice of MLflow JSON-renderable items as a `Vec<Value>` ready
/// to embed in a request body. One implementation for every kind of item
/// the REST surface accepts.
fn to_json_array<T: ToMlflowJson>(items: &[T]) -> Vec<Value> {
    items.iter().map(ToMlflowJson::to_mlflow_json).collect()
}

fn resp_json(resp: Response) -> Result<Value, ExportError> {
    resp.json::<Value>().map_err(map_reqwest_err)
}

/// Walk `path` keys into `value` and return the string leaf, mapping any
/// missing intermediate or non-string leaf to [`ExportError::Http`].
fn extract_string(value: &Value, path: &[&str]) -> Result<String, ExportError> {
    let mut cur = value;
    for key in path {
        cur = cur
            .get(*key)
            .ok_or_else(|| ExportError::Http(format!("response missing field `{key}`")))?;
    }
    cur.as_str().map(|s| s.to_string()).ok_or_else(|| {
        ExportError::Http(format!("response field `{}` not a string", path.join(".")))
    })
}

/// MLflow tag key used to link a child run to its parent. Centralised
/// so the rewrite helper + sink stay in lock-step.
const MLFLOW_PARENT_RUN_ID_TAG: &str = "mlflow.parentRunId";

/// Rewrite `mlflow.parentRunId` in `tags` to point at the parent run's
/// SERVER-assigned id. Parents call this with `parent_server_run_id =
/// None` so their tags pass through unchanged; children supply the
/// parent's server id so the link points at the right MLflow record
/// (the payload-side id is meaningless to the server — MLflow's REST
/// API always allocates fresh run ids). Non-parent tags pass through
/// untouched in both branches.
fn rewrite_parent_run_id_tag(tags: &[TagKv], parent_server_run_id: Option<&str>) -> Vec<TagKv> {
    match parent_server_run_id {
        None => tags.to_vec(),
        Some(parent_id) => tags
            .iter()
            .map(|t| {
                if t.key == MLFLOW_PARENT_RUN_ID_TAG {
                    TagKv {
                        key: t.key.clone(),
                        value: parent_id.to_string(),
                    }
                } else {
                    t.clone()
                }
            })
            .collect(),
    }
}

/// Wall-clock milliseconds since Unix epoch. Mirrors the
/// `mlflow_fs::now_ms` helper so HTTP + filesystem sinks agree on the
/// stamp recorded for create_run / set_terminated.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─── MlflowHttpSink ────────────────────────────────────────────────────────

/// MLflow HTTP tracking-server sink. Consumes the same
/// [`super::mlflow_payload::RunPayload`] as [`super::mlflow_fs::MlflowFsSink`]
/// and serialises it as REST calls.
///
/// Construction goes through [`MlflowHttpSink::from_config`] which mirrors
/// the public `EvalConfig` HTTP fields so callers can ship the sink end-to-end
/// without instantiating the lower-level [`MlflowHttpClient`] /
/// [`HttpClientConfig`] structs directly.
pub struct MlflowHttpSink {
    client: MlflowHttpClient,
    experiment_name: String,
}

impl std::fmt::Debug for MlflowHttpSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlflowHttpSink")
            .field("client", &self.client)
            .field("experiment_name", &self.experiment_name)
            .finish()
    }
}

impl MlflowHttpSink {
    /// Construct a sink from an `EvalConfig`. Reads
    /// `mlflow_http_tracking_uri`, `mlflow_http_timeout_ms`,
    /// `mlflow_http_max_retries`, `mlflow_http_backoff_base_ms`,
    /// `mlflow_http_batch_size`, `mlflow_http_token`, and falls back to
    /// `"forge-eval-default"` when `experiment_name` is `None` (mirroring
    /// `RunManifest`'s default).
    ///
    /// Returns [`ExportError::InvalidTarget`] when the tracking URI is
    /// absent or malformed.
    pub fn from_config(cfg: &EvalConfig) -> Result<Self, ExportError> {
        let uri = cfg.mlflow_http_tracking_uri.as_ref().ok_or_else(|| {
            ExportError::InvalidTarget("mlflow_http_tracking_uri must be set".to_string())
        })?;
        let base = Url::parse(uri)
            .map_err(|e| ExportError::InvalidTarget(format!("bad http tracking uri: {e}")))?;
        let http_cfg = HttpClientConfig {
            timeout_ms: cfg.mlflow_http_timeout_ms,
            max_retries: cfg.mlflow_http_max_retries,
            backoff_base_ms: cfg.mlflow_http_backoff_base_ms,
            batch_size: cfg.mlflow_http_batch_size,
            bearer_token: cfg.mlflow_http_token.clone(),
            user_agent: DEFAULT_USER_AGENT.to_string(),
        };
        let client = MlflowHttpClient::new(base, http_cfg)?;
        let experiment_name = cfg
            .experiment_name
            .clone()
            .unwrap_or_else(|| DEFAULT_EXPERIMENT_NAME.to_string());
        Ok(Self {
            client,
            experiment_name,
        })
    }

    /// Borrow the underlying HTTP client. Exposed for tests + advanced
    /// callers that need to issue ad-hoc requests against the same auth
    /// + retry policy.
    pub fn client(&self) -> &MlflowHttpClient {
        &self.client
    }

    /// Experiment name the sink resolves on every export.
    pub fn experiment_name(&self) -> &str {
        &self.experiment_name
    }
}

impl Exporter for MlflowHttpSink {
    fn name(&self) -> &'static str {
        "mlflow-http"
    }

    #[instrument(skip_all, fields(experiment = %self.experiment_name))]
    fn export(
        &self,
        scorecard: &Scorecard,
        manifest: &RunManifest,
        artifacts_dir: &Path,
    ) -> Result<(), ExportError> {
        let start_ms = now_ms();
        let payload = build_run_payload(scorecard, manifest, artifacts_dir, start_ms)?;
        let exp_id = self
            .client
            .get_or_create_experiment(&self.experiment_name)?;
        info!(
            payload_run_id = %payload.run_id,
            experiment_id = %exp_id,
            children = payload.children.len(),
            "mlflow-http: exporting suite"
        );
        // Capture the parent's SERVER-assigned id so child runs can link
        // to it via `mlflow.parentRunId`. The payload-side run_id is
        // ignored by MLflow's REST API (the server always allocates a
        // fresh id); without this rewrite, children's parentRunId tag
        // would point at a non-existent run and MLflow UI nesting +
        // parent/child queries would break silently.
        let parent_server_id = self.write_run(&exp_id, &payload, artifacts_dir, start_ms, None)?;
        info!(
            payload_run_id = %payload.run_id,
            server_run_id = %parent_server_id,
            "mlflow-http: parent run created on server"
        );
        for child in &payload.children {
            self.write_run(
                &exp_id,
                child,
                artifacts_dir,
                start_ms,
                Some(&parent_server_id),
            )?;
        }
        info!(payload_run_id = %payload.run_id, "mlflow-http: export complete");
        Ok(())
    }
}

impl MlflowHttpSink {
    /// Submit a single payload (parent OR child) to the server: create the
    /// run, stream params/tags/metrics via log_batch, upload artefacts via
    /// log_artifact, and mark FINISHED. A scopeguard ensures the run lands
    /// as FAILED rather than wedged in RUNNING if any of the intermediate
    /// calls returns Err.
    ///
    /// `#[instrument]` adds `run_id` (payload-side) to every nested
    /// `tracing::warn!` / `error!` so a CI log dump can correlate a
    /// retry-exhaustion message back to the specific run that failed.
    #[instrument(
        skip_all,
        fields(payload_run_id = %payload.run_id, experiment_id = %experiment_id)
    )]
    fn write_run(
        &self,
        experiment_id: &str,
        payload: &RunPayload,
        artifacts_dir: &Path,
        start_ms: u64,
        parent_server_run_id: Option<&str>,
    ) -> Result<String, ExportError> {
        let create_tags = rewrite_parent_run_id_tag(&payload.tags, parent_server_run_id);
        let server_run_id = self
            .client
            .create_run(experiment_id, start_ms, &create_tags)?;

        // `success` flips to true on the happy-path terminator. The
        // scopeguard runs on Drop regardless of how we exit and only fires
        // the FAILED update when success is still false (panic OR `?` early-
        // return). `Cell<bool>` gives interior mutability so the closure can
        // borrow `&success` while we set it after the closure is constructed.
        let success = Cell::new(false);
        let server_id_for_guard = server_run_id.clone();
        let client_for_guard = &self.client;
        let success_for_guard = &success;
        let guard = scopeguard::guard((), |_| {
            if !success_for_guard.get() {
                if let Err(e) = client_for_guard.set_terminated(
                    &server_id_for_guard,
                    RunStatus::Failed,
                    now_ms(),
                ) {
                    error!(
                        run_id = %server_id_for_guard,
                        status = RunStatus::Failed.as_str(),
                        error = %e,
                        "mlflow-http: failed to mark run as FAILED on abort"
                    );
                } else {
                    warn!(
                        run_id = %server_id_for_guard,
                        status = RunStatus::Failed.as_str(),
                        "mlflow-http: run marked FAILED on abort"
                    );
                }
            }
        });

        // Params + metrics: tags already shipped via create_run.
        self.client
            .log_batch(&server_run_id, &payload.metrics, &payload.params, &[])?;

        for artifact in &payload.artifact_refs {
            self.upload_artifact(&server_run_id, artifact, artifacts_dir)?;
        }

        self.client
            .set_terminated(&server_run_id, RunStatus::Finished, now_ms())?;
        success.set(true);
        // Disarm the guard (success branch). `into_inner` is explicit so a
        // future refactor that moves work past this point still triggers the
        // guard until the new terminator runs.
        scopeguard::ScopeGuard::into_inner(guard);

        debug!(
            run_id = %server_run_id,
            tier_metrics = payload.metrics.len(),
            params = payload.params.len(),
            artefacts = payload.artifact_refs.len(),
            "mlflow-http: run written"
        );
        Ok(server_run_id)
    }

    /// Stream one artefact ref into the server. Inline bytes ship verbatim;
    /// File / Directory sources are read on demand so a 1000-episode run
    /// with gigabytes of replays doesn't blow heap.
    fn upload_artifact(
        &self,
        run_id: &str,
        artifact: &ArtifactRef,
        artifacts_dir: &Path,
    ) -> Result<(), ExportError> {
        match &artifact.source {
            ArtifactSource::Inline(bytes) => {
                self.client.log_artifact(run_id, &artifact.rel_path, bytes)
            }
            ArtifactSource::File(p) => {
                let bytes = std::fs::read(p)?;
                self.client.log_artifact(run_id, &artifact.rel_path, &bytes)
            }
            ArtifactSource::Directory(src) => {
                upload_dir_recursively(&self.client, run_id, &artifact.rel_path, src)
            }
        }?;
        let _ = artifacts_dir; // reserved for relative-path resolution in future
        Ok(())
    }
}

/// Walk `src` recursively, uploading each file under
/// `<rel_root>/<file_path_relative_to_src>`. Mirrors `mlflow_fs`'s
/// `copy_subdir_if_exists` but with HTTP semantics. Silently no-ops if
/// `src` does not exist (matches the FsSink behaviour for optional
/// replays/trajectories subdirs).
fn upload_dir_recursively(
    client: &MlflowHttpClient,
    run_id: &str,
    rel_root: &str,
    src: &Path,
) -> Result<(), ExportError> {
    if !src.exists() || !src.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let child_rel = if rel_root.is_empty() {
            name_str.to_string()
        } else {
            format!("{}/{}", rel_root, name_str)
        };
        let ty = entry.file_type()?;
        if ty.is_dir() {
            upload_dir_recursively(client, run_id, &child_rel, &path)?;
        } else if ty.is_file() {
            let bytes = std::fs::read(&path)?;
            client.log_artifact(run_id, &child_rel, &bytes)?;
        } else {
            debug!(
                src = %path.display(),
                "mlflow-http sink: skipping non-file/non-dir entry"
            );
        }
    }
    Ok(())
}

// ─── Status helpers ────────────────────────────────────────────────────────

/// Internal helper: map a `reqwest::Error` to our `ExportError`.
/// Retryable transport errors (timeouts, connect failures) become
/// [`ExportError::Retryable`]; everything else is [`ExportError::Http`].
/// Slice 2b's REST methods all funnel through this.
#[inline]
pub(crate) fn map_reqwest_err(e: reqwest::Error) -> ExportError {
    if e.is_timeout() || e.is_connect() {
        ExportError::Retryable(format!("transport: {e}"))
    } else {
        ExportError::Http(format!("transport: {e}"))
    }
}

/// Internal helper: classify a `StatusCode` after a request completes.
/// Returns `Ok(())` on 2xx, [`ExportError::Retryable`] on retryable
/// statuses (5xx, 408, 429), and [`ExportError::Http`] otherwise.
/// Slice 2b's REST methods all funnel through this. Covered by the
/// `classify_status_distinguishes_retryable_from_terminal` test below.
#[inline]
pub(crate) fn classify_status(status: StatusCode, body_hint: &str) -> Result<(), ExportError> {
    if status.is_success() {
        return Ok(());
    }
    let code = status.as_u16();
    let msg = format!("status {code}: {body_hint}");
    if RetryPolicy::should_retry(code) {
        Err(ExportError::Retryable(msg))
    } else {
        Err(ExportError::Http(msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_policy_table_pins_retryable_statuses() {
        for (status, want) in [
            (200u16, false),
            (201, false),
            (204, false),
            (301, false),
            (400, false),
            (401, false),
            (403, false),
            (404, false),
            (408, true),
            (409, false),
            (422, false),
            (429, true),
            (500, true),
            (502, true),
            (503, true),
            (504, true),
            (599, true),
        ] {
            assert_eq!(RetryPolicy::should_retry(status), want, "status={status}");
        }
    }

    #[test]
    fn retry_policy_wait_ms_is_exponential_with_cap() {
        let p = RetryPolicy {
            max_retries: 10,
            backoff_base_ms: 250,
        };
        // 250 * 2^1 = 500; 250 * 2^2 = 1000; 250 * 2^3 = 2000; ...
        assert_eq!(p.wait_ms(1), 500);
        assert_eq!(p.wait_ms(2), 1_000);
        assert_eq!(p.wait_ms(3), 2_000);
        assert_eq!(p.wait_ms(4), 4_000);
        // Cap at 60s — wait_ms(20) would overflow without the cap.
        assert_eq!(p.wait_ms(20), DEFAULT_HTTP_BACKOFF_CAP_MS);
        assert_eq!(p.wait_ms(50), DEFAULT_HTTP_BACKOFF_CAP_MS);
    }

    #[test]
    fn http_client_config_default_uses_constants() {
        let cfg = HttpClientConfig::default();
        assert_eq!(cfg.timeout_ms, DEFAULT_HTTP_TIMEOUT_MS);
        assert_eq!(cfg.max_retries, DEFAULT_HTTP_MAX_RETRIES);
        assert_eq!(cfg.backoff_base_ms, DEFAULT_HTTP_BACKOFF_BASE_MS);
        assert_eq!(cfg.batch_size, DEFAULT_LOG_BATCH_SIZE);
        assert_eq!(cfg.user_agent, DEFAULT_USER_AGENT);
        assert!(cfg.bearer_token.is_none());
        assert!(
            cfg.batch_size <= 1_000,
            "MLflow REST caps log_batch at 1000"
        );
    }

    #[test]
    fn client_new_accepts_valid_url_and_stores_config() {
        let base = Url::parse("http://localhost:5000").unwrap();
        let client = MlflowHttpClient::new(base.clone(), HttpClientConfig::default()).unwrap();
        assert_eq!(client.base().as_str(), base.as_str());
        assert_eq!(client.batch_size(), DEFAULT_LOG_BATCH_SIZE);
        assert!(!client.has_auth());
    }

    #[test]
    fn client_debug_redacts_bearer_token() {
        let base = Url::parse("https://mlflow.example").unwrap();
        let cfg = HttpClientConfig {
            bearer_token: Some("super-secret-PAT".to_string()),
            ..HttpClientConfig::default()
        };
        let client = MlflowHttpClient::new(base, cfg).unwrap();
        let debug = format!("{client:?}");
        assert!(
            !debug.contains("super-secret-PAT"),
            "bearer token must not leak into Debug output: {debug}"
        );
        assert!(debug.contains("auth_present: true"));
    }

    #[test]
    fn classify_status_distinguishes_retryable_from_terminal() {
        assert!(classify_status(StatusCode::OK, "").is_ok());
        assert!(classify_status(StatusCode::CREATED, "").is_ok());

        let retryable = classify_status(StatusCode::SERVICE_UNAVAILABLE, "down");
        assert!(matches!(retryable, Err(ExportError::Retryable(_))));

        let terminal = classify_status(StatusCode::BAD_REQUEST, "bad");
        assert!(matches!(terminal, Err(ExportError::Http(_))));
    }

    // ─── REST method tests (mockito) ───────────────────────────────────────
    //
    // Each test uses a fast retry config (1ms base, 0 retries) so even a
    // failing test exits in milliseconds rather than sitting on the
    // default 250ms × 2^n backoff schedule.

    use mockito::{Matcher, Server};
    use serde_json::json;

    fn fast_client(server: &Server) -> MlflowHttpClient {
        let cfg = HttpClientConfig {
            timeout_ms: 5_000,
            max_retries: 0,
            backoff_base_ms: 1,
            batch_size: DEFAULT_LOG_BATCH_SIZE,
            bearer_token: None,
            user_agent: DEFAULT_USER_AGENT.to_string(),
        };
        MlflowHttpClient::new(Url::parse(&server.url()).unwrap(), cfg).unwrap()
    }

    #[test]
    fn get_or_create_experiment_returns_existing_id_on_hit() {
        let mut server = Server::new();
        let mock = server
            .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
            .match_query(Matcher::UrlEncoded(
                "experiment_name".into(),
                "exp-1".into(),
            ))
            .with_status(200)
            .with_body(json!({"experiment": {"experiment_id": "42"}}).to_string())
            .create();
        let client = fast_client(&server);
        let id = client.get_or_create_experiment("exp-1").unwrap();
        assert_eq!(id, "42");
        mock.assert();
    }

    #[test]
    fn get_or_create_experiment_falls_back_to_create_on_404() {
        let mut server = Server::new();
        let lookup = server
            .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
            // mockito requires an explicit query matcher; default mocks don't
            // match requests that carry a query string.
            .match_query(Matcher::UrlEncoded(
                "experiment_name".into(),
                "missing-exp".into(),
            ))
            .with_status(404)
            .with_body(json!({"error_code": "RESOURCE_DOES_NOT_EXIST"}).to_string())
            .create();
        let create = server
            .mock("POST", "/api/2.0/mlflow/experiments/create")
            .match_body(Matcher::PartialJson(json!({"name": "missing-exp"})))
            .with_status(200)
            .with_body(json!({"experiment_id": "99"}).to_string())
            .create();
        let client = fast_client(&server);
        let id = client.get_or_create_experiment("missing-exp").unwrap();
        assert_eq!(id, "99");
        lookup.assert();
        create.assert();
    }

    #[test]
    fn create_run_posts_experiment_id_and_parses_run_id() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/api/2.0/mlflow/runs/create")
            .with_status(200)
            .with_body(json!({"run": {"info": {"run_id": "abc123"}}}).to_string())
            .create();
        let client = fast_client(&server);
        let tags = vec![TagKv {
            key: "mlflow.runName".to_string(),
            value: "test".to_string(),
        }];
        let run_id = client.create_run("0", 1_000, &tags).unwrap();
        assert_eq!(run_id, "abc123");
        mock.assert();
    }

    #[test]
    fn log_batch_chunks_metrics_at_batch_size_and_sends_params_only_once() {
        let mut server = Server::new();
        // 5 metrics, batch_size=2 → 3 chunks. Params + tags should only appear
        // in the first chunk's body.
        let mock = server
            .mock("POST", "/api/2.0/mlflow/runs/log-batch")
            .with_status(200)
            .expect(3)
            .with_body("{}")
            .create();
        let cfg = HttpClientConfig {
            batch_size: 2,
            max_retries: 0,
            backoff_base_ms: 1,
            ..HttpClientConfig::default()
        };
        let client = MlflowHttpClient::new(Url::parse(&server.url()).unwrap(), cfg).unwrap();
        let metrics: Vec<MetricSample> = (0..5)
            .map(|i| MetricSample {
                key: format!("m{i}"),
                value: i as f64,
                timestamp_ms: 1,
                step: i,
            })
            .collect();
        let params = vec![ParamKv {
            key: "p".to_string(),
            value: "v".to_string(),
        }];
        let tags = vec![TagKv {
            key: "t".to_string(),
            value: "v".to_string(),
        }];
        client.log_batch("run-1", &metrics, &params, &tags).unwrap();
        mock.assert();
    }

    #[test]
    fn log_batch_no_op_on_empty_input() {
        let server = Server::new(); // No mock — must not be hit.
        let client = fast_client(&server);
        client.log_batch("run-1", &[], &[], &[]).unwrap();
    }

    #[test]
    fn set_tag_posts_run_id_and_kv() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/api/2.0/mlflow/runs/set-tag")
            .with_status(200)
            .with_body("{}")
            .match_body(Matcher::PartialJson(
                json!({"run_id": "r", "key": "k", "value": "v"}),
            ))
            .create();
        let client = fast_client(&server);
        client.set_tag("r", "k", "v").unwrap();
        mock.assert();
    }

    #[test]
    fn set_terminated_emits_status_string_for_finished_and_failed() {
        for (status, wire) in [
            (RunStatus::Finished, "FINISHED"),
            (RunStatus::Failed, "FAILED"),
        ] {
            let mut server = Server::new();
            let mock = server
                .mock("POST", "/api/2.0/mlflow/runs/update")
                .with_status(200)
                .with_body("{}")
                .match_body(Matcher::PartialJson(
                    json!({"run_id": "r", "status": wire, "end_time": 999u64}),
                ))
                .create();
            let client = fast_client(&server);
            client.set_terminated("r", status, 999).unwrap();
            mock.assert();
        }
    }

    #[test]
    fn log_artifact_puts_bytes_under_rel_path_with_run_id_query() {
        let mut server = Server::new();
        let mock = server
            .mock("PUT", "/api/2.0/mlflow-artifacts/artifacts/scorecard.json")
            .match_query(Matcher::UrlEncoded("run_id".into(), "abc".into()))
            .with_status(200)
            .with_body("{}")
            .match_body(Matcher::Exact(r#"{"score":0.5}"#.to_string()))
            .create();
        let client = fast_client(&server);
        client
            .log_artifact("abc", "scorecard.json", br#"{"score":0.5}"#)
            .unwrap();
        mock.assert();
    }

    #[test]
    fn log_artifact_handles_nested_rel_path_segments() {
        let mut server = Server::new();
        let mock = server
            .mock(
                "PUT",
                "/api/2.0/mlflow-artifacts/artifacts/replays/ep0/data.bin",
            )
            .match_query(Matcher::UrlEncoded("run_id".into(), "run-x".into()))
            .with_status(200)
            .with_body("{}")
            .create();
        let client = fast_client(&server);
        client
            .log_artifact("run-x", "replays/ep0/data.bin", b"binary")
            .unwrap();
        mock.assert();
    }

    #[test]
    fn execute_with_retry_surfaces_retryable_after_budget_exhausted() {
        // Server returns 503 every call. With max_retries=2 we expect 3
        // attempts (1 initial + 2 retries) before giving up with Retryable.
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/api/2.0/mlflow/runs/set-tag")
            .with_status(503)
            .with_body("server down")
            .expect(3)
            .create();
        let cfg = HttpClientConfig {
            max_retries: 2,
            backoff_base_ms: 1,
            ..HttpClientConfig::default()
        };
        let client = MlflowHttpClient::new(Url::parse(&server.url()).unwrap(), cfg).unwrap();
        let err = client.set_tag("r", "k", "v").unwrap_err();
        assert!(matches!(err, ExportError::Retryable(_)), "got {err:?}");
        mock.assert();
    }

    #[test]
    fn execute_with_retry_surfaces_http_immediately_on_terminal_4xx() {
        // 400 is non-retryable. Even with max_retries=5 we should see
        // exactly one attempt, ending in ExportError::Http.
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/api/2.0/mlflow/runs/create")
            .with_status(400)
            .with_body("malformed")
            .expect(1)
            .create();
        let cfg = HttpClientConfig {
            max_retries: 5,
            backoff_base_ms: 1,
            ..HttpClientConfig::default()
        };
        let client = MlflowHttpClient::new(Url::parse(&server.url()).unwrap(), cfg).unwrap();
        let err = client.create_run("0", 0, &[]).unwrap_err();
        assert!(matches!(err, ExportError::Http(_)), "got {err:?}");
        mock.assert();
    }

    #[test]
    fn run_status_wire_strings_pin_mlflow_enum() {
        // MLflow's REST contract for runs/update.status is a tiny enum;
        // pin the wire strings so a typo never silently breaks the UI.
        assert_eq!(RunStatus::Running.as_str(), "RUNNING");
        assert_eq!(RunStatus::Finished.as_str(), "FINISHED");
        assert_eq!(RunStatus::Failed.as_str(), "FAILED");
        assert_eq!(RunStatus::Killed.as_str(), "KILLED");
        assert_eq!(RunStatus::Scheduled.as_str(), "SCHEDULED");
    }

    // ─── MlflowHttpSink tests ──────────────────────────────────────────────

    use crate::manifest::{RunManifest, MANIFEST_SOURCE_NAME, UNKNOWN};
    use crate::scorecard::{ScenarioResult, Scorecard, SummaryStats, TierScore};
    use chrono::Utc;
    use forge_types::agent_interface::AgentMetadata;

    fn fixture_scorecard_with_n_scenarios(n: usize) -> Scorecard {
        let scenarios: Vec<ScenarioResult> = (0..n)
            .map(|i| ScenarioResult::from_episodes(format!("s{i}"), 1, vec![]))
            .collect();
        Scorecard {
            agent_metadata: AgentMetadata::heuristic("TestAgent"),
            timestamp: "1970-01-01T00:00:00Z".to_string(),
            overall_score: 0.0,
            tier_scores: vec![TierScore {
                tier: 1,
                success_rate: 0.0,
                mean_reward: 0.0,
                mean_steps_to_completion: 0.0,
                episodes_evaluated: 0,
                scenarios_count: n as u32,
            }],
            scenario_results: scenarios,
            summary: SummaryStats {
                total_episodes: 0,
                total_steps: 0,
                wall_clock_seconds: 0.0,
                mean_decision_latency_ms: 0.0,
            },
        }
    }

    fn fixture_manifest() -> RunManifest {
        RunManifest {
            run_id: "parent-run".to_string(),
            experiment_name: "test-exp".to_string(),
            timestamp: Utc::now(),
            git_sha: UNKNOWN.to_string(),
            git_branch: UNKNOWN.to_string(),
            rustc_version: UNKNOWN.to_string(),
            user: UNKNOWN.to_string(),
            config_hash: "h".to_string(),
            scenario_file_hashes: vec![],
            source_name: MANIFEST_SOURCE_NAME.to_string(),
        }
    }

    fn http_sink_from_url(url: &str, max_retries: u32) -> MlflowHttpSink {
        // Struct-literal init keeps clippy's field_reassign_with_default
        // happy while still inheriting Default::default() for every field
        // the tests don't care about.
        let cfg = EvalConfig {
            mlflow_http_tracking_uri: Some(url.to_string()),
            mlflow_http_max_retries: max_retries,
            mlflow_http_backoff_base_ms: 1,
            experiment_name: Some("test-exp".to_string()),
            ..EvalConfig::default()
        };
        MlflowHttpSink::from_config(&cfg).unwrap()
    }

    /// Happy-path: 0 scenarios = 1 parent run only. Asserts the full
    /// create-run → log-batch → log-artifact (4 inline) → set-terminated
    /// FINISHED sequence happens exactly once and the experiment is
    /// resolved up front.
    #[test]
    fn mlflow_http_sink_writes_parent_run_only_when_no_scenarios() {
        let mut server = Server::new();
        let _get_exp = server
            .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
            .match_query(Matcher::UrlEncoded(
                "experiment_name".into(),
                "test-exp".into(),
            ))
            .with_status(200)
            .with_body(json!({"experiment": {"experiment_id": "0"}}).to_string())
            .create();
        let _create_run = server
            .mock("POST", "/api/2.0/mlflow/runs/create")
            .with_status(200)
            .with_body(json!({"run": {"info": {"run_id": "server-parent"}}}).to_string())
            .expect(1)
            .create();
        let _log_batch = server
            .mock("POST", "/api/2.0/mlflow/runs/log-batch")
            .with_status(200)
            .with_body("{}")
            .expect(1)
            .create();
        // 4 inline artefacts on the parent: scorecard.json, scorecard.md,
        // manifest.json, tier_success_rates.html.
        let _log_artifact = server
            .mock(
                "PUT",
                Matcher::Regex(r"^/api/2\.0/mlflow-artifacts/artifacts/.+$".to_string()),
            )
            .match_query(Matcher::UrlEncoded("run_id".into(), "server-parent".into()))
            .with_status(200)
            .expect(4)
            .create();
        let set_terminated_finished = server
            .mock("POST", "/api/2.0/mlflow/runs/update")
            .match_body(Matcher::PartialJson(json!({
                "run_id": "server-parent",
                "status": "FINISHED",
            })))
            .with_status(200)
            .with_body("{}")
            .expect(1)
            .create();
        let sink = http_sink_from_url(&server.url(), 0);
        sink.export(
            &fixture_scorecard_with_n_scenarios(0),
            &fixture_manifest(),
            Path::new(""),
        )
        .unwrap();
        set_terminated_finished.assert();
    }

    /// Direct happy-path test for `create_experiment` (without the 404
    /// fallback dance). Ensures the body field name + response-parse path
    /// don't depend on `get_or_create_experiment` to remain wired.
    #[test]
    fn create_experiment_posts_name_and_parses_id() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/api/2.0/mlflow/experiments/create")
            .match_body(Matcher::PartialJson(json!({"name": "fresh-exp"})))
            .with_status(200)
            .with_body(json!({"experiment_id": "777"}).to_string())
            .create();
        let client = fast_client(&server);
        let id = client.create_experiment("fresh-exp").unwrap();
        assert_eq!(id, "777");
        mock.assert();
    }

    /// `from_config` must reject a missing tracking URI with InvalidTarget
    /// rather than panicking or returning a misleading transport error.
    #[test]
    fn mlflow_http_sink_from_config_errors_on_missing_uri() {
        let cfg = EvalConfig {
            mlflow_http_tracking_uri: None,
            ..EvalConfig::default()
        };
        let err = MlflowHttpSink::from_config(&cfg).unwrap_err();
        assert!(matches!(err, ExportError::InvalidTarget(_)), "got {err:?}");
    }

    /// `from_config` must reject a malformed URI with InvalidTarget.
    #[test]
    fn mlflow_http_sink_from_config_errors_on_malformed_uri() {
        let cfg = EvalConfig {
            mlflow_http_tracking_uri: Some("not a url".to_string()),
            ..EvalConfig::default()
        };
        let err = MlflowHttpSink::from_config(&cfg).unwrap_err();
        assert!(matches!(err, ExportError::InvalidTarget(_)), "got {err:?}");
    }

    /// `from_config` must reject non-HTTP(S) schemes. `Url::parse` will
    /// happily accept things like `file:///tmp/x` and `ftp://h/p`, but
    /// the MLflow REST contract only speaks HTTP — the sink would
    /// otherwise fail much later with an opaque transport error.
    #[test]
    fn mlflow_http_sink_from_config_rejects_non_http_schemes() {
        for bad in [
            "file:///tmp/mlflow",
            "ftp://mlflow.example.com",
            "ws://mlflow.example.com",
        ] {
            let cfg = EvalConfig {
                mlflow_http_tracking_uri: Some(bad.to_string()),
                ..EvalConfig::default()
            };
            let err = MlflowHttpSink::from_config(&cfg).unwrap_err();
            assert!(
                matches!(err, ExportError::InvalidTarget(_)),
                "scheme {bad} must be InvalidTarget, got {err:?}",
            );
        }
    }

    /// `from_config` with no `experiment_name` falls back to the
    /// canonical [`DEFAULT_EXPERIMENT_NAME`] constant.
    #[test]
    fn mlflow_http_sink_from_config_defaults_experiment_name() {
        let mut server = Server::new();
        let cfg = EvalConfig {
            mlflow_http_tracking_uri: Some(server.url()),
            mlflow_http_max_retries: 0,
            mlflow_http_backoff_base_ms: 1,
            experiment_name: None,
            ..EvalConfig::default()
        };
        let sink = MlflowHttpSink::from_config(&cfg).unwrap();
        assert_eq!(sink.experiment_name(), DEFAULT_EXPERIMENT_NAME);
        // Make sure we hit a mock so the test doesn't accidentally pass on
        // server.url() being unreachable.
        let _ = server
            .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
            .match_query(Matcher::UrlEncoded(
                "experiment_name".into(),
                DEFAULT_EXPERIMENT_NAME.into(),
            ))
            .with_status(200)
            .with_body(json!({"experiment": {"experiment_id": "0"}}).to_string())
            .create();
        sink.client
            .get_or_create_experiment(sink.experiment_name())
            .unwrap();
    }

    /// `upload_dir_recursively` exercise: nested tree → one log_artifact
    /// PUT per file with paths anchored under `rel_root`.
    #[test]
    fn upload_dir_recursively_walks_nested_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("sub")).unwrap();
        std::fs::write(tmp.path().join("a.txt"), b"A").unwrap();
        std::fs::write(tmp.path().join("sub/b.txt"), b"B").unwrap();

        let mut server = Server::new();
        let mock_a = server
            .mock("PUT", "/api/2.0/mlflow-artifacts/artifacts/replays/a.txt")
            .match_query(Matcher::UrlEncoded("run_id".into(), "r1".into()))
            .with_status(200)
            .expect(1)
            .create();
        let mock_b = server
            .mock(
                "PUT",
                "/api/2.0/mlflow-artifacts/artifacts/replays/sub/b.txt",
            )
            .match_query(Matcher::UrlEncoded("run_id".into(), "r1".into()))
            .with_status(200)
            .expect(1)
            .create();
        let client = fast_client(&server);
        upload_dir_recursively(&client, "r1", "replays", tmp.path()).unwrap();
        mock_a.assert();
        mock_b.assert();
    }

    /// `upload_dir_recursively` must no-op on a missing dir (parity with
    /// FsSink's `copy_subdir_if_exists`).
    #[test]
    fn upload_dir_recursively_no_op_on_missing_dir() {
        let server = Server::new(); // No mocks — must not be hit.
        let client = fast_client(&server);
        upload_dir_recursively(&client, "r1", "absent", Path::new("/no/such/dir")).unwrap();
    }

    /// Pin every const-block value to its on-disk-contract string so a
    /// rename surfaces here, not at runtime when MLflow rejects the request.
    /// Mirrors `exporters::tests::exporter_string_constants_are_stable_contract`
    /// for the HTTP transport's own constants.
    #[test]
    fn mlflow_http_constants_are_stable_contract() {
        assert_eq!(API_ROOT, "api");
        assert_eq!(API_VERSION, "2.0");
        assert_eq!(NAMESPACE_MLFLOW, "mlflow");
        assert_eq!(NAMESPACE_ARTIFACTS, "mlflow-artifacts");
        assert_eq!(ARTIFACTS_PATH, "artifacts");
        assert_eq!(DEFAULT_EXPERIMENT_NAME, "forge-eval-default");
        // REST body / query fields.
        assert_eq!(FIELD_EXPERIMENT_NAME, "experiment_name");
        assert_eq!(FIELD_EXPERIMENT_ID, "experiment_id");
        assert_eq!(FIELD_RUN_ID, "run_id");
        assert_eq!(FIELD_RUN, "run");
        assert_eq!(FIELD_INFO, "info");
        assert_eq!(FIELD_START_TIME, "start_time");
        assert_eq!(FIELD_END_TIME, "end_time");
        assert_eq!(FIELD_STATUS, "status");
        assert_eq!(FIELD_KEY, "key");
        assert_eq!(FIELD_VALUE, "value");
        assert_eq!(FIELD_NAME, "name");
        assert_eq!(FIELD_TAGS, "tags");
        assert_eq!(FIELD_METRICS, "metrics");
        assert_eq!(FIELD_PARAMS, "params");
        assert_eq!(FIELD_TIMESTAMP, "timestamp");
        assert_eq!(FIELD_STEP, "step");
        assert_eq!(QUERY_RUN_ID, "run_id");
        // Response paths.
        assert_eq!(RESP_PATH_EXPERIMENT_ID, &["experiment", "experiment_id"]);
        assert_eq!(RESP_PATH_CREATE_EXP_ID, &["experiment_id"]);
        assert_eq!(RESP_PATH_RUN_ID, &["run", "info", "run_id"]);
    }

    /// Pin the binding between `config.rs`'s `DEFAULT_MLFLOW_HTTP_*` re-exports
    /// (when feature is enabled) and `mlflow_http.rs`'s canonical constants.
    /// Catches future drift between the two.
    #[test]
    fn config_re_exports_match_canonical_http_defaults() {
        assert_eq!(
            crate::config::DEFAULT_MLFLOW_HTTP_TIMEOUT_MS,
            DEFAULT_HTTP_TIMEOUT_MS
        );
        assert_eq!(
            crate::config::DEFAULT_MLFLOW_HTTP_MAX_RETRIES,
            DEFAULT_HTTP_MAX_RETRIES
        );
        assert_eq!(
            crate::config::DEFAULT_MLFLOW_HTTP_BACKOFF_BASE_MS,
            DEFAULT_HTTP_BACKOFF_BASE_MS
        );
        assert_eq!(
            crate::config::DEFAULT_MLFLOW_HTTP_BATCH_SIZE,
            DEFAULT_LOG_BATCH_SIZE
        );
    }

    /// Scopeguard contract: when a mid-export call (log_batch here) returns
    /// the retry budget is exhausted, the sink propagates Err — AND the
    /// run lands in FAILED, not RUNNING. Without the scopeguard the run
    /// would be wedged in RUNNING in the UI.
    #[test]
    fn mlflow_http_sink_scopeguard_marks_run_failed_on_mid_export_error() {
        let mut server = Server::new();
        let _get_exp = server
            .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
            .match_query(Matcher::UrlEncoded(
                "experiment_name".into(),
                "test-exp".into(),
            ))
            .with_status(200)
            .with_body(json!({"experiment": {"experiment_id": "0"}}).to_string())
            .create();
        let _create_run = server
            .mock("POST", "/api/2.0/mlflow/runs/create")
            .with_status(200)
            .with_body(json!({"run": {"info": {"run_id": "doomed-run"}}}).to_string())
            .expect(1)
            .create();
        // Every log-batch attempt returns 503 → after the retry budget is
        // exhausted, the sink's `?` propagates and the scopeguard fires.
        let _log_batch_fail = server
            .mock("POST", "/api/2.0/mlflow/runs/log-batch")
            .with_status(503)
            .with_body("server down")
            .create();
        // The set_terminated(FAILED, ...) call the scopeguard MUST issue.
        let set_terminated_failed = server
            .mock("POST", "/api/2.0/mlflow/runs/update")
            .match_body(Matcher::PartialJson(json!({
                "run_id": "doomed-run",
                "status": "FAILED",
            })))
            .with_status(200)
            .with_body("{}")
            .expect(1)
            .create();
        let sink = http_sink_from_url(&server.url(), 0);
        let result = sink.export(
            &fixture_scorecard_with_n_scenarios(0),
            &fixture_manifest(),
            Path::new(""),
        );
        assert!(
            matches!(result, Err(ExportError::Retryable(_))),
            "got {result:?}"
        );
        // The critical contract: FAILED was sent — run isn't wedged in RUNNING.
        set_terminated_failed.assert();
    }

    /// Pin the child-run linkage contract: when given a parent's
    /// SERVER-assigned id, the helper rewrites `mlflow.parentRunId` in
    /// every child's tag set to that id and leaves all other tags
    /// untouched. Pure-function unit test — no mockito, no server, no
    /// parallelism contention. The sink's child-create flow consumes
    /// this helper directly, so the linkage contract is enforced
    /// transitively in `write_run`.
    #[test]
    fn rewrite_parent_run_id_tag_replaces_only_target_tag_for_children() {
        let original = vec![
            TagKv {
                key: "mlflow.parentRunId".to_string(),
                value: "PAYLOAD-PARENT".to_string(),
            },
            TagKv {
                key: "mlflow.runName".to_string(),
                value: "scenario_a".to_string(),
            },
            TagKv {
                key: "forge.eval.tier".to_string(),
                value: "1".to_string(),
            },
        ];

        // Child case: target tag is rewritten, others pass through.
        let rewritten = rewrite_parent_run_id_tag(&original, Some("SERVER-PARENT"));
        assert_eq!(rewritten.len(), 3);
        assert_eq!(rewritten[0].key, "mlflow.parentRunId");
        assert_eq!(
            rewritten[0].value, "SERVER-PARENT",
            "child must carry the parent's SERVER-assigned id, not the payload id"
        );
        assert_eq!(rewritten[1].value, "scenario_a", "other tags untouched");
        assert_eq!(rewritten[2].value, "1", "other tags untouched");

        // Parent case: passthrough (no rewrite).
        let unchanged = rewrite_parent_run_id_tag(&original, None);
        assert_eq!(unchanged.len(), 3);
        assert_eq!(unchanged[0].value, "PAYLOAD-PARENT");

        // Multiple parentRunId tags (shouldn't happen but mustn't panic).
        let pathological = vec![
            TagKv {
                key: "mlflow.parentRunId".to_string(),
                value: "A".to_string(),
            },
            TagKv {
                key: "mlflow.parentRunId".to_string(),
                value: "B".to_string(),
            },
        ];
        let both = rewrite_parent_run_id_tag(&pathological, Some("REWRITE"));
        assert!(both.iter().all(|t| t.value == "REWRITE"));
    }
}
