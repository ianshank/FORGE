//! Shared simulation state for broadcasting to connected clients.
//!
//! Types in this module are serialized as JSON and sent to dashboard clients.
//! All optional/newer fields use `#[serde(default)]` for backwards compatibility
//! — older clients that don't know about new fields will still parse correctly.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tracing::instrument;

use crate::SCHEMA_VERSION;

/// A snapshot of a single agent's state for visualization.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSnapshot {
    /// Unique agent identifier.
    pub id: u32,
    /// X coordinate on the grid.
    pub x: u16,
    /// Y coordinate on the grid.
    pub y: u16,
    /// Current health points.
    pub health: i32,
    /// Whether the agent is alive.
    pub alive: bool,
    /// Optional team identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team_id: Option<u32>,
    /// Optional description of the agent's current intent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    /// Sensor/vision radius in grid cells.
    #[serde(default)]
    pub vision_radius: u16,
}

/// A point-in-time snapshot of the entire simulation state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationSnapshot {
    /// Current simulation tick.
    pub tick: u64,
    /// Snapshots of all agents.
    pub agents: Vec<AgentSnapshot>,
    /// Width of the simulation grid.
    pub grid_width: u16,
    /// Height of the simulation grid.
    pub grid_height: u16,
    /// Recent simulation events.
    #[serde(default)]
    pub events: Vec<String>,
    /// Schema version for client compatibility.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
}

/// Returns the current schema version (used as serde default).
fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

impl Default for SimulationSnapshot {
    fn default() -> Self {
        Self {
            tick: 0,
            agents: Vec::new(),
            grid_width: 0,
            grid_height: 0,
            events: Vec::new(),
            schema_version: SCHEMA_VERSION,
        }
    }
}

/// Thread-safe shared simulation state for broadcasting updates.
///
/// Uses `Arc<RwLock<...>>` to allow concurrent reads from multiple
/// WebSocket handler tasks while serializing writes from the simulation loop.
#[derive(Clone, Debug)]
pub struct SharedState {
    inner: Arc<RwLock<SimulationSnapshot>>,
}

impl SharedState {
    /// Creates a new `SharedState` with a default snapshot.
    #[instrument]
    pub fn new() -> Self {
        tracing::debug!("Creating new SharedState");
        Self {
            inner: Arc::new(RwLock::new(SimulationSnapshot::default())),
        }
    }

    /// Replaces the current snapshot with the provided one.
    #[instrument(skip(self, snapshot), fields(tick = snapshot.tick))]
    pub fn update(&self, snapshot: SimulationSnapshot) {
        tracing::debug!(
            tick = snapshot.tick,
            agents = snapshot.agents.len(),
            "Updating shared simulation state"
        );
        match self.inner.write() {
            Ok(mut state) => *state = snapshot,
            Err(e) => tracing::error!("SharedState write lock poisoned: {}", e),
        }
    }

    /// Returns a clone of the current simulation snapshot.
    #[instrument(skip(self))]
    pub fn read(&self) -> SimulationSnapshot {
        tracing::trace!("Reading shared simulation state");
        match self.inner.read() {
            Ok(state) => state.clone(),
            Err(e) => {
                tracing::error!("SharedState read lock poisoned: {}", e);
                SimulationSnapshot::default()
            }
        }
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_and_read() {
        let shared = SharedState::new();
        let snapshot = SimulationSnapshot {
            tick: 42,
            agents: vec![AgentSnapshot {
                id: 1,
                x: 10,
                y: 20,
                health: 100,
                alive: true,
                team_id: Some(0),
                intent: Some("move_north".to_string()),
                vision_radius: 5,
            }],
            grid_width: 64,
            grid_height: 64,
            events: vec!["agent_spawned".to_string()],
            schema_version: SCHEMA_VERSION,
        };

        shared.update(snapshot);
        let read_back = shared.read();

        assert_eq!(read_back.tick, 42);
        assert_eq!(read_back.agents.len(), 1);
        assert_eq!(read_back.agents[0].id, 1);
        assert_eq!(read_back.agents[0].x, 10);
        assert_eq!(read_back.agents[0].y, 20);
        assert_eq!(read_back.grid_width, 64);
        assert_eq!(read_back.grid_height, 64);
        assert_eq!(read_back.events.len(), 1);
    }

    #[test]
    fn test_default_snapshot() {
        let snapshot = SimulationSnapshot::default();
        assert_eq!(snapshot.tick, 0);
        assert!(snapshot.agents.is_empty());
        assert_eq!(snapshot.grid_width, 0);
        assert_eq!(snapshot.grid_height, 0);
        assert!(snapshot.events.is_empty());
        assert_eq!(snapshot.schema_version, 1);
    }

    #[test]
    fn test_camel_case_serialization() {
        let snapshot = SimulationSnapshot {
            tick: 1,
            agents: vec![AgentSnapshot {
                id: 0,
                x: 5,
                y: 10,
                health: 80,
                alive: true,
                team_id: None,
                intent: None,
                vision_radius: 7,
            }],
            grid_width: 32,
            grid_height: 32,
            events: vec![],
            schema_version: SCHEMA_VERSION,
        };
        let json = serde_json::to_string(&snapshot).unwrap();
        // Verify camelCase keys are used
        assert!(json.contains("gridWidth"));
        assert!(json.contains("gridHeight"));
        assert!(json.contains("schemaVersion"));
        assert!(json.contains("visionRadius"));
        // Verify snake_case keys are NOT used
        assert!(!json.contains("grid_width"));
        assert!(!json.contains("grid_height"));
        assert!(!json.contains("schema_version"));
    }

    #[test]
    fn test_optional_fields_skipped_when_none() {
        let agent = AgentSnapshot {
            id: 1,
            x: 0,
            y: 0,
            health: 100,
            alive: true,
            team_id: None,
            intent: None,
            vision_radius: 5,
        };
        let json = serde_json::to_string(&agent).unwrap();
        // None fields should be omitted
        assert!(!json.contains("teamId"));
        assert!(!json.contains("intent"));
    }

    #[test]
    fn test_backwards_compatible_deserialization() {
        // Simulate an older payload missing newer fields
        let json = r#"{"tick":10,"agents":[],"gridWidth":64,"gridHeight":64}"#;
        let snapshot: SimulationSnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(snapshot.tick, 10);
        assert_eq!(snapshot.grid_width, 64);
        // Defaults should be applied
        assert!(snapshot.events.is_empty());
        assert_eq!(snapshot.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn test_agent_backwards_compatible_deserialization() {
        // Older payload without visionRadius
        let json = r#"{"id":1,"x":5,"y":10,"health":100,"alive":true}"#;
        let agent: AgentSnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(agent.id, 1);
        assert_eq!(agent.vision_radius, 0); // default
        assert!(agent.team_id.is_none());
    }
}
