//! REST API route definitions and handler functions.

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::state::{AgentSnapshot, SimulationSnapshot};
use crate::ws_handler::{AppState, WsMessage};
use crate::SCHEMA_VERSION;

/// Response payload for the `/api/config` endpoint.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigResponse {
    /// Server version string.
    pub version: String,
    /// Schema version for client compatibility.
    pub schema_version: u32,
}

/// Response payload for the `/health` endpoint.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    /// Current server status.
    pub status: String,
    /// Server uptime in seconds.
    pub uptime_seconds: u64,
}

/// Returns configuration and version information.
#[instrument]
pub async fn config_handler() -> Json<ConfigResponse> {
    tracing::debug!("Handling config request");
    Json(ConfigResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        schema_version: SCHEMA_VERSION,
    })
}

/// Returns the current health status of the server.
#[instrument]
pub async fn health_handler() -> Json<HealthResponse> {
    tracing::debug!("Handling health request");
    Json(HealthResponse {
        status: "ok".to_string(),
        uptime_seconds: 0,
    })
}

/// Returns current server metrics.
#[instrument(skip_all)]
pub async fn metrics_handler(State(state): State<AppState>) -> Json<crate::metrics::ServerMetrics> {
    let subs = state.subscriptions.lock().expect("lock poisoned");
    let snapshot = state.shared_state.read();
    Json(crate::metrics::ServerMetrics {
        simulation_ticks: snapshot.tick,
        steps_per_second: 0.0,
        ws_connections: subs.active_clients() as u32,
        uptime_seconds: 0,
    })
}

/// Request body for the scenario remix endpoint.
#[derive(Debug, Deserialize)]
pub struct RemixRequest {
    /// Optional seed override. If absent, a random seed is generated.
    pub seed: Option<u64>,
}

/// Response from the scenario remix endpoint.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemixResponse {
    /// The seed used for the new scenario.
    pub seed: u64,
    /// Grid width of the new scenario.
    pub grid_width: u16,
    /// Grid height of the new scenario.
    pub grid_height: u16,
}

/// Generates a new scenario with a fresh seed and resets the simulation.
#[instrument(skip_all)]
pub async fn remix_handler(
    State(state): State<AppState>,
    body: Option<Json<RemixRequest>>,
) -> Json<RemixResponse> {
    let seed = body.and_then(|b| b.seed).unwrap_or_else(rand::random);

    tracing::info!(seed, "Remixing scenario");

    // Create a new world with the given seed
    let mut config = forge_types::config::ForgeConfig::default();
    config.world.seed = seed;

    let grid_width = config.world.width;
    let grid_height = config.world.height;

    match forge_core::WorldState::new(config) {
        Ok(world) => {
            // Build snapshot from the new world
            let snapshot = build_snapshot_from_world(&world);
            state.shared_state.update(snapshot.clone());

            // Broadcast the new state to all connected clients
            let _ = state.tx.send(WsMessage::StateUpdate(snapshot));

            Json(RemixResponse {
                seed,
                grid_width,
                grid_height,
            })
        }
        Err(e) => {
            tracing::error!("Failed to create world: {}", e);
            Json(RemixResponse {
                seed,
                grid_width,
                grid_height,
            })
        }
    }
}

/// Builds a `SimulationSnapshot` from a `WorldState`.
pub fn build_snapshot_from_world(world: &forge_core::WorldState) -> SimulationSnapshot {
    let agents: Vec<AgentSnapshot> = world
        .agents
        .iter()
        .map(|a| AgentSnapshot {
            id: a.id,
            x: a.position.x,
            y: a.position.y,
            health: a.health,
            alive: a.alive,
            team_id: None,
            intent: None,
            vision_radius: u16::from(a.vision_radius),
        })
        .collect();

    SimulationSnapshot {
        tick: world.tick,
        agents,
        grid_width: world.grid.width,
        grid_height: world.grid.height,
        events: Vec::new(),
        schema_version: SCHEMA_VERSION,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_response_serialization() {
        let resp = ConfigResponse {
            version: "0.1.0".to_string(),
            schema_version: 1,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("schemaVersion"));
    }

    #[test]
    fn test_remix_response_serialization() {
        let resp = RemixResponse {
            seed: 42,
            grid_width: 64,
            grid_height: 64,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("gridWidth"));
        assert!(json.contains("gridHeight"));
    }
}
