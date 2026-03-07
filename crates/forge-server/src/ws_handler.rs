//! WebSocket subscription handler types and logic.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::metrics::ServerMetrics;
use crate::state::SimulationSnapshot;

/// Messages sent over WebSocket connections to clients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WsMessage {
    /// A full simulation state update.
    StateUpdate(SimulationSnapshot),
    /// Current server metrics.
    Metrics(ServerMetrics),
    /// An error message.
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
}
