//! End-to-end WebSocket handler coverage.
//!
//! Boots a small Axum router that uses [`forge_server::ws_handler::ws_upgrade_handler`]
//! on a random local port, drives a [`tokio_tungstenite`] client through the full
//! lifecycle (upgrade → initial snapshot → broadcast → close), and asserts the
//! handler's bookkeeping side-effects. This is the cheapest way to exercise the
//! async paths inside `handle_ws_connection` which can't be unit-tested in
//! isolation.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::routing::get;
use axum::Router;
use forge_server::metrics::MetricsCollector;
use forge_server::state::{SharedState, SimulationSnapshot};
use forge_server::ws_handler::{ws_upgrade_handler, AppState, SubscriptionManager, WsMessage};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::tungstenite::Message;

/// Default broadcast buffer for integration tests. Larger than the handler
/// will ever fill, so we never see lag in the happy-path tests.
const TEST_BROADCAST_CAPACITY: usize = 32;

/// Maximum time we'll wait for the initial snapshot before failing the test.
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(3);

/// Maximum time we'll wait for the handler to register a client.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(3);

/// Test fixture: build an `AppState` with a fresh broadcast channel and
/// trivial dependents.
fn build_state() -> (AppState, broadcast::Sender<WsMessage>) {
    let (tx, _rx) = broadcast::channel::<WsMessage>(TEST_BROADCAST_CAPACITY);
    let (replacement_tx, _replacement_rx) = mpsc::channel(1);
    let state = AppState {
        tx: tx.clone(),
        subscriptions: Arc::new(Mutex::new(SubscriptionManager::new())),
        shared_state: SharedState::new(),
        next_client_id: Arc::new(AtomicU64::new(0)),
        metrics_collector: Arc::new(Mutex::new(MetricsCollector::default())),
        start_time: Instant::now(),
        world_replacement_tx: Arc::new(replacement_tx),
    };
    (state, tx)
}

/// Spawns the handler on a random local port and returns the `ws://` URL plus
/// the cloneable `AppState` for assertions.
async fn spawn_server() -> (String, AppState, broadcast::Sender<WsMessage>) {
    let (state, tx) = build_state();
    let app = Router::new()
        .route("/ws", get(ws_upgrade_handler))
        .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        // Ignore errors — the test will tear down the server by dropping
        // the runtime when it finishes.
        let _ = axum::serve(listener, app).await;
    });
    (format!("ws://{addr}/ws"), state, tx)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_handler_sends_initial_snapshot_on_connect() {
    let (url, state, _tx) = spawn_server().await;

    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("WebSocket upgrade should succeed");
    let (_write, mut read) = ws_stream.split();

    let first = tokio::time::timeout(SNAPSHOT_TIMEOUT, read.next())
        .await
        .expect("snapshot must arrive within timeout")
        .expect("stream must yield at least one message")
        .expect("first message must not be an error");

    let text = match first {
        Message::Text(t) => t,
        other => panic!("unexpected message kind on connect: {other:?}"),
    };
    let parsed: WsMessage =
        serde_json::from_str(&text).expect("initial snapshot must be valid JSON WsMessage");
    matches!(parsed, WsMessage::StateUpdate(_))
        .then_some(())
        .expect("first message must be a StateUpdate");

    // The handler must have registered exactly one client.
    let started = Instant::now();
    while started.elapsed() < REGISTRATION_TIMEOUT {
        let count = state.subscriptions.lock().unwrap().active_clients();
        if count >= 1 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("handler did not register the client within {REGISTRATION_TIMEOUT:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_handler_forwards_broadcast_messages() {
    let (url, _state, tx) = spawn_server().await;

    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (_write, mut read) = ws_stream.split();

    // Consume the initial snapshot first.
    let _initial = tokio::time::timeout(SNAPSHOT_TIMEOUT, read.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    // Publish a broadcast message.
    let snapshot = SimulationSnapshot {
        tick: 1234,
        ..Default::default()
    };
    tx.send(WsMessage::StateUpdate(snapshot)).unwrap();

    // Subsequent frames may include other broadcast deliveries — search until
    // we find one whose tick matches what we sent, or time out.
    let mut found = false;
    let deadline = Instant::now() + SNAPSHOT_TIMEOUT;
    while Instant::now() < deadline {
        let Some(Ok(Message::Text(text))) =
            tokio::time::timeout(Duration::from_millis(500), read.next())
                .await
                .ok()
                .flatten()
        else {
            continue;
        };
        if let Ok(WsMessage::StateUpdate(snap)) = serde_json::from_str::<WsMessage>(&text) {
            if snap.tick == 1234 {
                found = true;
                break;
            }
        }
    }
    assert!(found, "broadcast snapshot was not forwarded to client");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_handler_unregisters_client_on_close() {
    let (url, state, _tx) = spawn_server().await;

    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut write, mut read) = ws_stream.split();

    // Drain the snapshot, then send a close frame.
    let _initial = tokio::time::timeout(SNAPSHOT_TIMEOUT, read.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    write.send(Message::Close(None)).await.unwrap();
    drop(write);
    drop(read);

    // The handler should drop the client from the manager. Allow generous
    // settle time — async teardown isn't synchronous with the close frame.
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(5) {
        let count = state.subscriptions.lock().unwrap().active_clients();
        if count == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("handler did not unregister client after close");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_handler_assigns_distinct_client_ids() {
    let (url, state, _tx) = spawn_server().await;

    let (a, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (b, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    // Give the handler a moment to register both.
    let started = Instant::now();
    while started.elapsed() < REGISTRATION_TIMEOUT {
        let n = state.subscriptions.lock().unwrap().active_clients();
        if n == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let ids = state.subscriptions.lock().unwrap().client_ids();
    assert_eq!(ids.len(), 2, "expected two registered clients, got {ids:?}");
    // IDs must be unique (the handler assigns from a monotonic counter).
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 2, "client IDs collided: {ids:?}");

    drop(a);
    drop(b);
}
