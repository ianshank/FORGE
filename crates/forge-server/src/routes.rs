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

    /// A malformed entry in `allowed_origins` must be skipped, not
    /// panic the server at startup.
    #[test]
    fn cors_layer_skips_unparseable_origins() {
        let config = ServerConfig {
            allowed_origins: vec!["http://ok.example".into(), "\u{7f}bad".into()],
            ..ServerConfig::default()
        };
        let _layer = cors_layer(&config);
    }
}
