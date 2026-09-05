//! Router assembly: route table, middleware stack, and the split
//! between read-only and mutating endpoints.
//!
//! Lives in the library rather than in `main.rs` so the middleware
//! actually gets tested — the binary used to build the router inline
//! with exactly one layer (CORS), which left the timeout, body-size and
//! authentication behaviour untestable and, as it turned out, absent.
//!
//! ## Layer stack
//!
//! ```text
//! CORS                        (browser-origin policy)
//!  └── TimeoutLayer           (config.request_timeout_ms -> 408)
//!       └── DefaultBodyLimit  (config.max_body_bytes    -> 413)
//!            ├── read-only routes
//!            └── bearer-auth  (config.auth_token        -> 401)
//!                 └── mutating routes
//! ```
//!
//! The WebSocket upgrade at `/ws` is merged in *after* the timeout and
//! body-limit layers on purpose: `/ws` is a long-lived connection whose
//! lifetime is deliberately unbounded, and per-request caps have no
//! meaning for it.

use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tracing::{debug, info};

use crate::api::{
    config_handler, decision_traces_handler, health_handler, metrics_handler, remix_handler,
    runs_handler, traces_history_handler, training_history_handler, training_metrics_handler,
};
use crate::config::ServerConfig;
use crate::ws_handler::{ws_upgrade_handler, AppState};

/// Status returned when a request exceeds `config.request_timeout_ms`.
pub const TIMEOUT_STATUS: axum::http::StatusCode = axum::http::StatusCode::REQUEST_TIMEOUT;

/// Per-request timeout layer built from `config.request_timeout_ms`.
///
/// Exposed so tests can assert the configured duration is what actually
/// reaches the stack, rather than inferring it from the router.
pub fn request_timeout_layer(config: &ServerConfig) -> TimeoutLayer {
    TimeoutLayer::with_status_code(
        TIMEOUT_STATUS,
        Duration::from_millis(config.request_timeout_ms),
    )
}

/// Request-body size cap built from `config.max_body_bytes`.
pub fn body_limit_layer(config: &ServerConfig) -> DefaultBodyLimit {
    DefaultBodyLimit::max(config.max_body_bytes)
}

/// CORS layer built from `config.allowed_origins`.
///
/// Origins that do not parse as a header value are dropped with a
/// `debug!` rather than aborting startup — a typo in one entry should
/// not take the server down.
pub fn cors_layer(config: &ServerConfig) -> CorsLayer {
    let origins: Vec<_> = config
        .allowed_origins
        .iter()
        .filter_map(|o| match o.parse::<axum::http::HeaderValue>() {
            Ok(v) => Some(v),
            Err(e) => {
                debug!(origin = %o, error = %e, "Ignoring unparseable CORS origin");
                None
            }
        })
        .collect();
    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
        ])
}

/// Routes that only read server state. No authentication.
fn read_only_routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health_handler))
        .route("/api/config", get(config_handler))
        .route("/api/metrics", get(metrics_handler))
        .route(
            "/api/training-metrics/history",
            get(training_history_handler),
        )
        .route("/api/decision-traces/history", get(traces_history_handler))
        .route("/api/runs", get(runs_handler))
        .route("/api/env/render", get(crate::env::render_handler))
}

/// Routes that mutate simulation state or write to disk.
///
/// These are the ones [`crate::auth::protect`] guards when a token is
/// configured. Adding a new write endpoint here picks up the guard
/// automatically; adding it to [`read_only_routes`] would not, which is
/// the one thing to get right when extending this table.
fn mutating_routes() -> Router<AppState> {
    Router::new()
        .route("/api/scenario/remix", post(remix_handler))
        .route("/api/training-metrics", post(training_metrics_handler))
        .route("/api/decision-traces", post(decision_traces_handler))
        .route("/api/env/reset", post(crate::env::reset_handler))
        .route("/api/env/step", post(crate::env::step_handler))
}

/// Apply the per-request HTTP middleware — body-size cap then request
/// timeout — to `router`.
///
/// Factored out of [`build_router`] so tests can drive the *same* stack
/// over a deliberately slow or oversize route. Not applied to `/ws`.
pub fn apply_http_middleware<S>(router: Router<S>, config: &ServerConfig) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(body_limit_layer(config))
        .layer(request_timeout_layer(config))
}

/// Build the complete application router for `state` and `config`.
///
/// Applies, in order: bearer auth on the mutating routes (only when
/// `config.auth_token` is set), then the body limit, the request
/// timeout, and CORS across the HTTP surface; `/ws` is merged outside
/// the timeout/body layers.
#[tracing::instrument(skip_all, fields(auth = config.auth_enabled()))]
pub fn build_router(state: AppState, config: &ServerConfig) -> Router {
    let mutating = match config.auth_token.as_deref() {
        Some(token) => {
            info!("Bearer-token auth enabled on mutating endpoints");
            crate::auth::protect(mutating_routes(), token)
        }
        None => mutating_routes(),
    };

    let http = apply_http_middleware(read_only_routes().merge(mutating), config);

    http.merge(Router::new().route("/ws", get(ws_upgrade_handler)))
        .layer(cors_layer(config))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    //! Behavioural coverage of the assembled stack (401 / 408 / 413 per
    //! route) lives in `tests/hardening_integration.rs`, where the real
    //! router can be driven with `tower::ServiceExt::oneshot`. These
    //! unit tests cover only what is reachable without an `AppState`.
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use tower::ServiceExt as _;

    /// A well-formed origin the tests configure and expect to be echoed.
    const GOOD_ORIGIN: &str = "http://ok.example";
    /// A well-formed origin the tests deliberately never configure.
    const UNCONFIGURED_ORIGIN: &str = "http://evil.example";
    /// `\u{7f}` is DEL, which `HeaderValue::from_str` rejects — this is
    /// the entry [`cors_layer`] must skip rather than panic on.
    const UNPARSEABLE_ORIGIN: &str = "\u{7f}bad";
    /// Path of the throwaway route the CORS probes are sent to.
    const PROBE_PATH: &str = "/probe";

    fn config_with_origins(origins: &[&str]) -> ServerConfig {
        ServerConfig {
            allowed_origins: origins.iter().map(|o| (*o).to_string()).collect(),
            ..ServerConfig::default()
        }
    }

    /// A stateless router carrying nothing but [`cors_layer`], so the
    /// layer's real behaviour can be driven without an `AppState`.
    fn cors_only_router(config: &ServerConfig) -> Router {
        Router::new()
            .route(PROBE_PATH, get(|| async { "ok" }))
            .layer(cors_layer(config))
    }

    /// Send a simple (non-preflight) `GET` carrying `origin` and return
    /// the `Access-Control-Allow-Origin` the layer answered with, if any.
    async fn allow_origin_for(config: &ServerConfig, origin: &str) -> Option<String> {
        let response = cors_only_router(config)
            .oneshot(
                Request::builder()
                    .uri(PROBE_PATH)
                    .header(header::ORIGIN, origin)
                    .body(Body::empty())
                    .expect("probe request builds"),
            )
            .await
            .expect("probe request is served");
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "the probe route itself must succeed; CORS is advisory to the browser"
        );
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .map(|v| {
                v.to_str()
                    .expect("an echoed origin is always valid ASCII")
                    .to_string()
            })
    }

    /// A malformed entry in `allowed_origins` must be skipped, not panic
    /// the server at startup — and skipping it must not take its
    /// well-formed neighbours down with it.
    ///
    /// The assertion is the point: an earlier version of this test bound
    /// the layer to `_layer` and asserted nothing, so it proved only
    /// "does not panic". Rewriting `cors_layer` to return a bare
    /// `CorsLayer::new()` — allowing no origin at all — kept it green.
    #[tokio::test]
    async fn cors_layer_skips_unparseable_origins_but_keeps_the_valid_ones() {
        let config = config_with_origins(&[GOOD_ORIGIN, UNPARSEABLE_ORIGIN]);
        assert_eq!(
            allow_origin_for(&config, GOOD_ORIGIN).await.as_deref(),
            Some(GOOD_ORIGIN),
            "a well-formed origin must survive an unparseable entry beside it"
        );
    }

    /// The allowlist has to actually be an allowlist: an origin nobody
    /// configured gets no `Access-Control-Allow-Origin` back. Without
    /// this, widening the layer to `AllowOrigin::any()` is invisible.
    #[tokio::test]
    async fn cors_layer_does_not_allow_an_unconfigured_origin() {
        let config = config_with_origins(&[GOOD_ORIGIN]);
        assert_eq!(
            allow_origin_for(&config, UNCONFIGURED_ORIGIN).await,
            None,
            "an unconfigured origin must not be echoed back as allowed"
        );
    }

    /// The preflight answer must carry the methods and headers the
    /// dashboard actually sends. `Authorization` matters specifically:
    /// dropping it from `allow_headers` silently breaks every browser
    /// call to a mutating endpoint once `auth_token` is set, and the
    /// failure surfaces as a CORS error in the client, not here.
    #[tokio::test]
    async fn cors_preflight_allows_the_methods_and_headers_the_dashboard_needs() {
        let config = config_with_origins(&[GOOD_ORIGIN]);
        let response = cors_only_router(&config)
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri(PROBE_PATH)
                    .header(header::ORIGIN, GOOD_ORIGIN)
                    .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                    .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "authorization")
                    .body(Body::empty())
                    .expect("preflight request builds"),
            )
            .await
            .expect("preflight is served");

        // The probe route is GET-only, so a 200 here also proves the CORS
        // layer intercepted the preflight rather than falling through to a 405.
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "the CORS layer must answer the preflight itself"
        );

        let header_value = |name: header::HeaderName| -> String {
            response
                .headers()
                .get(&name)
                .unwrap_or_else(|| panic!("preflight answer is missing {name}"))
                .to_str()
                .expect("CORS response headers are ASCII")
                .to_ascii_lowercase()
        };

        let methods = header_value(header::ACCESS_CONTROL_ALLOW_METHODS);
        assert!(
            methods.contains("get") && methods.contains("post"),
            "the dashboard reads over GET and writes over POST; got {methods:?}"
        );

        let headers = header_value(header::ACCESS_CONTROL_ALLOW_HEADERS);
        assert!(
            headers.contains("authorization"),
            "bearer auth is unusable from a browser without Authorization; got {headers:?}"
        );
        assert!(
            headers.contains("content-type"),
            "every POST body here is JSON; got {headers:?}"
        );
    }
}
