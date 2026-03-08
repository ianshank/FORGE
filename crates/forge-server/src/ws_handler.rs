//! WebSocket subscription handler types and logic.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures::SinkExt;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::instrument;

use crate::metrics::ServerMetrics;
use crate::state::SimulationSnapshot;

/// Messages sent over WebSocket connections to clients.
///
/// Serialized with `{ "type": "...", "payload": ... }` envelope for
/// backwards-compatible message dispatch on the client side.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum WsMessage {
    /// A full simulation state update.
    StateUpdate(SimulationSnapshot),
    /// Current server metrics.
    Metrics(ServerMetrics),
    /// An error description sent to the client.
    Error(String),
}

/// Tracks a single client's subscription metadata.
#[derive(Clone, Debug)]
pub struct ClientSubscription {
    /// Unique client identifier.
    pub client_id: u64,
    /// Tick at which the client subscribed.
    pub subscribed_at: u64,
    /// Tick at which the client was last seen.
    pub last_seen: u64,
}

/// Manages the set of active WebSocket client subscriptions.
#[derive(Debug)]
pub struct SubscriptionManager {
    clients: HashMap<u64, ClientSubscription>,
}

impl SubscriptionManager {
    /// Creates a new empty `SubscriptionManager`.
    #[instrument]
    pub fn new() -> Self {
        tracing::debug!("Creating new SubscriptionManager");
        Self {
            clients: HashMap::new(),
        }
    }

    /// Registers a new client subscription at the given tick.
    #[instrument(skip(self))]
    pub fn add_client(&mut self, client_id: u64, tick: u64) {
        tracing::info!(client_id, tick, "Adding client subscription");
        self.clients.insert(
            client_id,
            ClientSubscription {
                client_id,
                subscribed_at: tick,
                last_seen: tick,
            },
        );
    }

    /// Removes a client subscription. Returns `true` if the client was present.
    #[instrument(skip(self))]
    pub fn remove_client(&mut self, client_id: u64) -> bool {
        let removed = self.clients.remove(&client_id).is_some();
        tracing::info!(client_id, removed, "Removing client subscription");
        removed
    }

    /// Returns the number of currently active client subscriptions.
    #[instrument(skip(self))]
    pub fn active_clients(&self) -> usize {
        self.clients.len()
    }

    /// Returns the IDs of all currently connected clients.
    #[instrument(skip(self))]
    pub fn client_ids(&self) -> Vec<u64> {
        self.clients.keys().copied().collect()
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared application state passed to WebSocket and REST handlers.
#[derive(Clone)]
pub struct AppState {
    /// Broadcast channel for simulation state updates.
    pub tx: broadcast::Sender<WsMessage>,
    /// Subscription manager for tracking connected clients.
    pub subscriptions: Arc<Mutex<SubscriptionManager>>,
    /// Shared simulation state for initial snapshot.
    pub shared_state: crate::state::SharedState,
    /// Next client ID counter.
    pub next_client_id: Arc<std::sync::atomic::AtomicU64>,
    /// Server metrics collector.
    pub metrics_collector: Arc<Mutex<crate::metrics::MetricsCollector>>,
    /// Server start time for uptime calculation.
    pub start_time: std::time::Instant,
    /// Channel to send a replacement world to the simulation loop (e.g. from remix).
    pub world_replacement_tx: Arc<tokio::sync::mpsc::Sender<forge_core::WorldState>>,
}

/// Serializes a `WsMessage` to a JSON string for sending over WebSocket.
///
/// This helper centralizes serialization to avoid repeating the pattern
/// across initial-snapshot and broadcast code paths.
fn serialize_ws_message(msg: &WsMessage) -> Result<String, serde_json::Error> {
    serde_json::to_string(msg)
}

/// Axum handler that upgrades an HTTP request to a WebSocket connection.
#[instrument(skip_all)]
pub async fn ws_upgrade_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    tracing::info!("WebSocket upgrade requested");
    ws.on_upgrade(move |socket| handle_ws_connection(socket, state))
}

/// Handles an individual WebSocket connection lifecycle.
///
/// 1. Registers the client with the `SubscriptionManager`
/// 2. Sends the current simulation snapshot immediately
/// 3. Forwards broadcast messages until the client disconnects
/// 4. Unregisters the client on disconnect
async fn handle_ws_connection(socket: WebSocket, state: AppState) {
    let client_id = state
        .next_client_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Register client
    {
        let current_tick = state.shared_state.read().tick;
        if let Ok(mut subs) = state.subscriptions.lock() {
            subs.add_client(client_id, current_tick);
        }
    }

    tracing::info!(client_id, "WebSocket client connected");

    let (mut ws_tx, mut ws_rx) = socket.split();

    // Subscribe to broadcast channel *before* reading the initial snapshot,
    // so any ticks that occur while sending the snapshot are queued.
    let mut rx = state.tx.subscribe();

    // Send current state immediately on connect
    let initial_snapshot = state.shared_state.read();
    let initial_msg = WsMessage::StateUpdate(initial_snapshot);
    if let Ok(json) = serialize_ws_message(&initial_msg) {
        if let Err(e) = ws_tx.send(Message::Text(json)).await {
            tracing::warn!(client_id, error = %e, "Failed to send initial snapshot");
        }
    }

    // Forward broadcast messages to this client
    let mut send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(msg) => match serialize_ws_message(&msg) {
                    Ok(json) => {
                        if ws_tx.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to serialize WsMessage: {}", e);
                    }
                },
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(lagged = n, "Broadcast receiver lagged, skipping messages");
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    });

    // Consume incoming messages (keepalive / close detection)
    let mut recv_task = tokio::spawn(async move {
        while let Some(result) = ws_rx.next().await {
            match result {
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {} // Ignore other messages
            }
        }
    });

    // Wait for either task to complete, then abort the other.
    tokio::select! {
        _ = &mut send_task => { recv_task.abort(); },
        _ = &mut recv_task => { send_task.abort(); },
    }

    // Unregister client
    {
        if let Ok(mut subs) = state.subscriptions.lock() {
            subs.remove_client(client_id);
        }
    }

    tracing::info!(client_id, "WebSocket client disconnected");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_remove_clients() {
        let mut manager = SubscriptionManager::new();

        manager.add_client(1, 0);
        manager.add_client(2, 5);
        assert_eq!(manager.active_clients(), 2);

        assert!(manager.remove_client(1));
        assert_eq!(manager.active_clients(), 1);

        // Removing a non-existent client returns false.
        assert!(!manager.remove_client(99));
        assert_eq!(manager.active_clients(), 1);
    }

    #[test]
    fn test_active_count() {
        let mut manager = SubscriptionManager::new();
        assert_eq!(manager.active_clients(), 0);

        manager.add_client(10, 0);
        manager.add_client(20, 0);
        manager.add_client(30, 0);
        assert_eq!(manager.active_clients(), 3);

        let mut ids = manager.client_ids();
        ids.sort();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn test_ws_message_serialization() {
        let snapshot = SimulationSnapshot::default();
        let msg = WsMessage::StateUpdate(snapshot);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("StateUpdate"));
        assert!(json.contains("\"type\""));
        assert!(json.contains("\"payload\""));
    }

    #[test]
    fn test_serialize_ws_message_helper() {
        let msg = WsMessage::Error("test error".to_string());
        let result = serialize_ws_message(&msg);
        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json.contains("Error"));
        assert!(json.contains("test error"));
    }

    #[test]
    fn test_ws_message_metrics_variant() {
        let metrics = ServerMetrics::default();
        let msg = WsMessage::Metrics(metrics);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("Metrics"));
        assert!(json.contains("\"type\""));
    }

    #[test]
    fn test_ws_message_roundtrip() {
        let snapshot = SimulationSnapshot::default();
        let msg = WsMessage::StateUpdate(snapshot);
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: WsMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            WsMessage::StateUpdate(s) => assert_eq!(s.tick, 0),
            _ => panic!("Expected StateUpdate variant"),
        }
    }
}
