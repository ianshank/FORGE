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

use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::StatusCode;
use url::Url;

use super::ExportError;

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
    /// [`ExportError::InvalidTarget`] when the underlying `reqwest`
    /// builder can't construct the client (e.g. invalid TLS config).
    pub fn new(base: Url, cfg: HttpClientConfig) -> Result<Self, ExportError> {
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
}

// ─── Status helpers ────────────────────────────────────────────────────────

/// Internal helper: map a `reqwest::Error` to our `ExportError`.
/// Retryable transport errors (timeouts, connect failures) become
/// [`ExportError::Retryable`]; everything else is [`ExportError::Http`].
/// Slice 2b's REST methods all funnel through this.
#[inline]
#[allow(dead_code)]
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
#[allow(dead_code)]
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
        assert!(cfg.batch_size <= 1_000, "MLflow REST caps log_batch at 1000");
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
}
