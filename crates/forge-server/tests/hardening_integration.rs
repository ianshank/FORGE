//! Integration coverage for the forge-server network hardening.
//!
//! Drives the **real** router from [`forge_server::routes::build_router`]
//! via `tower::ServiceExt::oneshot` (no sockets, no HTTP client) and
//! asserts the middleware stack actually behaves:
//!
//! - bearer auth guards exactly the mutating routes (401 vs 200),
//! - the body limit rejects oversize payloads (413),
//! - the request timeout fires (408),
//! - `?limit=` is clamped to the configured cap.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::get;
use axum::Router;
use forge_server::config::ServerConfig;
use forge_server::history::{InMemoryHistoryStore, TrainingRecord};
use forge_server::metrics::MetricsCollector;
use forge_server::routes::{apply_http_middleware, build_router};
use forge_server::state::SharedState;
use forge_server::ws_handler::{AppState, SubscriptionManager, WsMessage};
use tokio::sync::{broadcast, mpsc};
use tower::ServiceExt; // for `oneshot`

/// The routes the security review requires auth on.
const MUTATING_ROUTES: &[(&str, &str)] = &[
    ("POST", "/api/scenario/remix"),
    ("POST", "/api/training-metrics"),
    ("POST", "/api/decision-traces"),
    ("POST", "/api/env/reset"),
    ("POST", "/api/env/step"),
];

/// A valid JSON body for each mutating route, so a 200 is reachable
/// once the request is authenticated.
fn body_for(path: &str) -> &'static str {
    match path {
        "/api/scenario/remix" => r#"{"seed":1}"#,
        "/api/training-metrics" => r#"{"episode":1,"meanReward":0.5}"#,
        "/api/decision-traces" => r#"[{"tick":1,"intentLabel":"explore"}]"#,
        "/api/env/reset" => r#"{"seed":1}"#,
        "/api/env/step" => r#"{"action":0}"#,
        other => panic!("no test body defined for {other}"),
    }
}

/// Build `AppState` backed by an in-memory history store.
fn app_state(query_limit: usize) -> AppState {
    let (tx, _rx) = broadcast::channel::<WsMessage>(16);
    let (replacement_tx, _replacement_rx) = mpsc::channel(1);
    AppState {
        tx,
        subscriptions: Arc::new(Mutex::new(SubscriptionManager::new())),
        shared_state: SharedState::new(),
        next_client_id: Arc::new(AtomicU64::new(0)),
        metrics_collector: Arc::new(Mutex::new(MetricsCollector::default())),
        start_time: Instant::now(),
        world_replacement_tx: Arc::new(replacement_tx),
        rest_world: forge_server::env::new_session_world(),
        history: Arc::new(InMemoryHistoryStore::new(10_000)),
        run_id: Arc::from("session-run"),
        history_query_limit: query_limit,
    }
}

/// Build the real router with `config`.
fn app(config: &ServerConfig) -> Router {
    build_router(app_state(config.history_query_limit), config)
}

/// Config with auth enabled and everything else at defaults.
fn config_with_auth(token: &str) -> ServerConfig {
    ServerConfig {
        auth_token: Some(token.to_string()),
        ..ServerConfig::default()
    }
}

async fn send(
    app: &Router,
    method: &str,
    uri: &str,
    auth: Option<&str>,
    body: &str,
) -> axum::http::Response<Body> {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(value) = auth {
        builder = builder.header("authorization", value);
    }
    let req = builder.body(Body::from(body.to_string())).unwrap();
    app.clone().oneshot(req).await.unwrap()
}

// ──────────────────────────────────────────────────────────────────────
// Bearer auth
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn mutating_routes_reject_requests_without_a_token() {
    let app = app(&config_with_auth("s3cret"));
    for (method, path) in MUTATING_ROUTES {
        let resp = send(&app, method, path, None, body_for(path)).await;
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "{method} {path} must be 401 without an Authorization header"
        );
    }
}

#[tokio::test]
async fn mutating_routes_reject_wrong_token_and_wrong_scheme() {
    let app = app(&config_with_auth("s3cret"));
    for bad in ["Bearer wrong", "Bearer ", "Basic s3cret", "s3cret"] {
        let resp = send(
            &app,
            "POST",
            "/api/training-metrics",
            Some(bad),
            body_for("/api/training-metrics"),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "Authorization: {bad:?} must be rejected"
        );
    }
}

#[tokio::test]
async fn mutating_routes_accept_the_configured_token() {
    let app = app(&config_with_auth("s3cret"));
    for (method, path) in MUTATING_ROUTES {
        let resp = send(&app, method, path, Some("Bearer s3cret"), body_for(path)).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "{method} {path} must succeed with a valid bearer token"
        );
    }
}

/// `/api/env/step` needs a reset first; with a token, the pair must work
/// end-to-end so auth is not silently breaking sequencing.
#[tokio::test]
async fn authenticated_reset_then_step_succeeds() {
    let app = app(&config_with_auth("s3cret"));
    let reset = send(
        &app,
        "POST",
        "/api/env/reset",
        Some("Bearer s3cret"),
        body_for("/api/env/reset"),
    )
    .await;
    assert_eq!(reset.status(), StatusCode::OK);
    let step = send(
        &app,
        "POST",
        "/api/env/step",
        Some("Bearer s3cret"),
        body_for("/api/env/step"),
    )
    .await;
    assert_eq!(step.status(), StatusCode::OK);
}

#[tokio::test]
async fn read_only_routes_are_not_gated_by_auth() {
    let app = app(&config_with_auth("s3cret"));
    for path in [
        "/health",
        "/api/config",
        "/api/metrics",
        "/api/training-metrics/history",
        "/api/decision-traces/history",
        "/api/runs",
    ] {
        let resp = send(&app, "GET", path, None, "").await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET {path} must stay open when auth is enabled"
        );
    }
}

/// Backwards compatibility: with no token configured, every mutating
/// route behaves exactly as it did before the change.
#[tokio::test]
async fn without_a_configured_token_mutating_routes_stay_open() {
    let app = app(&ServerConfig::default());
    for (method, path) in MUTATING_ROUTES {
        let resp = send(&app, method, path, None, body_for(path)).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "{method} {path} must stay open when no token is configured"
        );
    }
}

/// A stray `Authorization` header must be ignored, not rejected, when
/// auth is off — otherwise enabling a proxy that adds one would break
/// existing deployments.
#[tokio::test]
async fn without_a_configured_token_a_stray_header_is_ignored() {
    let app = app(&ServerConfig::default());
    let resp = send(
        &app,
        "POST",
        "/api/training-metrics",
        Some("Bearer anything"),
        body_for("/api/training-metrics"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

// ──────────────────────────────────────────────────────────────────────
// Body limit
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn oversize_body_is_rejected_with_413() {
    let config = ServerConfig {
        max_body_bytes: 64,
        ..ServerConfig::default()
    };
    let app = app(&config);
    // Well-formed JSON, but far larger than the configured cap.
    let big = format!(
        r#"{{"episode":1,"meanReward":0.5,"pad":"{}"}}"#,
        "x".repeat(4096)
    );
    let resp = send(&app, "POST", "/api/training-metrics", None, &big).await;
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn body_within_limit_is_accepted() {
    let config = ServerConfig {
        max_body_bytes: 4096,
        ..ServerConfig::default()
    };
    let app = app(&config);
    let resp = send(
        &app,
        "POST",
        "/api/training-metrics",
        None,
        body_for("/api/training-metrics"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

/// The default cap must be generous enough for a realistic decision-trace
/// batch — a limit that rejects normal traffic is a availability bug.
#[tokio::test]
async fn default_body_limit_accepts_a_realistic_trace_batch() {
    let app = app(&ServerConfig::default());
    let traces: Vec<String> = (0..500)
        .map(|i| format!(r#"{{"tick":{i},"intentLabel":"explore"}}"#))
        .collect();
    let body = format!("[{}]", traces.join(","));
    assert!(body.len() > 16 * 1024, "test payload should be non-trivial");
    let resp = send(&app, "POST", "/api/decision-traces", None, &body).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

// ──────────────────────────────────────────────────────────────────────
// Request timeout
// ──────────────────────────────────────────────────────────────────────

/// Stub router carrying a handler that outlives any sane timeout,
/// wrapped in the *same* middleware stack `build_router` installs.
///
/// None of the real handlers can be made to block deterministically, so
/// the timeout is exercised over `apply_http_middleware` — the exact
/// function the production router composes — rather than over a
/// hand-rolled copy of the layer stack.
fn slow_app(config: &ServerConfig) -> Router {
    apply_http_middleware(
        Router::new().route(
            "/slow",
            get(|| async {
                tokio::time::sleep(Duration::from_secs(30)).await;
                "never delivered"
            }),
        ),
        config,
    )
}

#[tokio::test]
async fn slow_request_is_cut_off_with_408() {
    let config = ServerConfig {
        request_timeout_ms: 20,
        ..ServerConfig::default()
    };
    let req = Request::builder().uri("/slow").body(Body::empty()).unwrap();
    let resp = slow_app(&config).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
}

/// The configured value is what governs: a budget longer than the
/// handler's own delay must let the response through.
#[tokio::test]
async fn fast_request_is_not_timed_out() {
    let config = ServerConfig {
        request_timeout_ms: 10_000,
        ..ServerConfig::default()
    };
    let fast = apply_http_middleware(
        Router::new().route(
            "/fast",
            get(|| async {
                tokio::time::sleep(Duration::from_millis(5)).await;
                "ok"
            }),
        ),
        &config,
    );
    let req = Request::builder().uri("/fast").body(Body::empty()).unwrap();
    let resp = fast.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ──────────────────────────────────────────────────────────────────────
// History query-limit clamp
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn history_limit_is_clamped_to_the_configured_cap() {
    let config = ServerConfig {
        history_query_limit: 2,
        ..ServerConfig::default()
    };
    let app = app(&config);
    for episode in 0..5 {
        let body = format!(r#"{{"episode":{episode},"meanReward":0.0}}"#);
        let resp = send(&app, "POST", "/api/training-metrics", None, &body).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // Asking for far more than the cap yields exactly the cap, newest first.
    let resp = send(
        &app,
        "GET",
        "/api/training-metrics/history?limit=1000000",
        None,
        "",
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let records: Vec<TrainingRecord> = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(records.len(), 2, "limit must clamp to history_query_limit");
    assert_eq!(records[0].metrics.episode, 3);
    assert_eq!(records[1].metrics.episode, 4);

    // A request below the cap is still honoured.
    let resp = send(
        &app,
        "GET",
        "/api/training-metrics/history?limit=1",
        None,
        "",
    )
    .await;
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let records: Vec<TrainingRecord> = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].metrics.episode, 4);
}
