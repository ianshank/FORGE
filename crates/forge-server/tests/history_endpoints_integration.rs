//! Integration coverage for the history REST endpoints.
//!
//! Builds the real router with an in-memory [`HistoryStore`] and drives it via
//! `tower::ServiceExt::oneshot` (no network, no extra HTTP-client dependency):
//! POST training metrics / decision traces, then GET the history + run summary
//! and assert persistence, filtering, ordering, and the `limit` cap.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use forge_server::api::{
    decision_traces_handler, runs_handler, traces_history_handler, training_history_handler,
    training_metrics_handler,
};
use forge_server::history::{InMemoryHistoryStore, RunSummary, TrainingRecord};
use forge_server::metrics::MetricsCollector;
use forge_server::state::SharedState;
use forge_server::ws_handler::{AppState, SubscriptionManager, WsMessage};
use tokio::sync::{broadcast, mpsc};
use tower::ServiceExt; // for `oneshot`

/// Build a router + state backed by an in-memory history store.
fn build_app(query_limit: usize) -> Router {
    let (tx, _rx) = broadcast::channel::<WsMessage>(16);
    let (replacement_tx, _replacement_rx) = mpsc::channel(1);
    let state = AppState {
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
    };
    Router::new()
        .route("/api/training-metrics", post(training_metrics_handler))
        .route(
            "/api/training-metrics/history",
            get(training_history_handler),
        )
        .route("/api/decision-traces", post(decision_traces_handler))
        .route("/api/decision-traces/history", get(traces_history_handler))
        .route("/api/runs", get(runs_handler))
        .with_state(state)
}

/// POST a JSON body to `uri` (optionally with an `X-Forge-Run-Id` header) and
/// assert a 200 response.
async fn post_json(app: &Router, uri: &str, run_id: Option<&str>, body: &str) {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(id) = run_id {
        builder = builder.header("x-forge-run-id", id);
    }
    let req = builder.body(Body::from(body.to_string())).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "POST {uri} should be 200");
}

/// GET `uri` and deserialize the JSON response body into `T`.
async fn get_json<T: serde::de::DeserializeOwned>(app: &Router, uri: &str) -> T {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET {uri} should be 200");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn training_metrics_persist_and_query() {
    let app = build_app(500);

    // Two samples for run-a (with explicit header), one for run-b (session id).
    post_json(
        &app,
        "/api/training-metrics",
        Some("run-a"),
        r#"{"episode":1,"meanReward":0.5}"#,
    )
    .await;
    post_json(
        &app,
        "/api/training-metrics",
        Some("run-a"),
        r#"{"episode":2,"meanReward":1.5}"#,
    )
    .await;
    post_json(
        &app,
        "/api/training-metrics",
        None,
        r#"{"episode":1,"meanReward":-0.5}"#,
    )
    .await;

    // All records, chronological (newest last).
    let all: Vec<TrainingRecord> = get_json(&app, "/api/training-metrics/history").await;
    assert_eq!(all.len(), 3);

    // Filter by run id via query param.
    let run_a: Vec<TrainingRecord> =
        get_json(&app, "/api/training-metrics/history?runId=run-a").await;
    assert_eq!(run_a.len(), 2);
    assert_eq!(run_a[1].metrics.mean_reward, 1.5);

    // limit keeps the newest N.
    let last: Vec<TrainingRecord> =
        get_json(&app, "/api/training-metrics/history?runId=run-a&limit=1").await;
    assert_eq!(last.len(), 1);
    assert_eq!(last[0].metrics.episode, 2);
}

#[tokio::test]
async fn traces_persist_and_runs_summarize() {
    let app = build_app(500);

    post_json(
        &app,
        "/api/training-metrics",
        Some("run-a"),
        r#"{"episode":1,"meanReward":2.0}"#,
    )
    .await;
    post_json(
        &app,
        "/api/decision-traces",
        Some("run-a"),
        r#"[{"tick":1,"intentLabel":"explore"},{"tick":2,"intentLabel":"mine"}]"#,
    )
    .await;

    let traces: Vec<serde_json::Value> =
        get_json(&app, "/api/decision-traces/history?runId=run-a").await;
    assert_eq!(traces.len(), 2);

    let runs: Vec<RunSummary> = get_json(&app, "/api/runs").await;
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].run_id, "run-a");
    assert_eq!(runs[0].episodes, 1);
    assert_eq!(runs[0].latest_mean_reward, 2.0);
}

#[tokio::test]
async fn empty_history_returns_empty_arrays() {
    let app = build_app(500);
    let training: Vec<TrainingRecord> = get_json(&app, "/api/training-metrics/history").await;
    assert!(training.is_empty());
    let runs: Vec<RunSummary> = get_json(&app, "/api/runs").await;
    assert!(runs.is_empty());
}
