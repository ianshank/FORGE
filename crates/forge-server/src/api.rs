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
#[instrument(skip_all)]
pub async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    tracing::debug!("Handling health request");
    let uptime = state.start_time.elapsed().as_secs();
    Json(HealthResponse {
        status: "ok".to_string(),
        uptime_seconds: uptime,
    })
}

/// Returns current server metrics from the `MetricsCollector`.
#[instrument(skip_all)]
pub async fn metrics_handler(State(state): State<AppState>) -> Json<crate::metrics::ServerMetrics> {
    let mut metrics = if let Ok(mc) = state.metrics_collector.lock() {
        mc.snapshot()
    } else {
        tracing::warn!("MetricsCollector lock poisoned, returning defaults");
        crate::metrics::ServerMetrics::default()
    };

    // Populate live values
    metrics.uptime_seconds = state.start_time.elapsed().as_secs();
    if let Ok(subs) = state.subscriptions.lock() {
        metrics.ws_connections = subs.active_clients() as u32;
    }

    Json(metrics)
}

/// Request body for the scenario remix endpoint.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemixRequest {
    /// Optional seed override. If absent, a random seed is generated.
    pub seed: Option<u64>,
    /// Optional grid size override. Applied to both width and height.
    pub grid_size: Option<u16>,
}

/// Response from the scenario remix endpoint.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemixResponse {
    /// Whether the remix was successful.
    pub success: bool,
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
    let (seed, grid_size) = match body {
        Some(Json(req)) => (req.seed, req.grid_size),
        None => (None, None),
    };
    let seed = seed.unwrap_or_else(rand::random);

    tracing::info!(seed, ?grid_size, "Remixing scenario");

    // Create a new world with the given seed (and optional grid size)
    let mut config = forge_types::config::ForgeConfig::default();
    config.world.seed = seed;
    if let Some(gs) = grid_size {
        config.world.width = gs;
        config.world.height = gs;
    }

    let grid_width = config.world.width;
    let grid_height = config.world.height;

    match forge_core::WorldState::new(config) {
        Ok(world) => {
            // Build snapshot from the new world
            let snapshot = build_snapshot_from_world(&world);
            state.shared_state.update(snapshot.clone());

            // Broadcast the new state to all connected clients
            if state.tx.send(WsMessage::StateUpdate(snapshot)).is_err() {
                tracing::trace!("No active subscribers for remix broadcast");
            }

            // Send the new world to the simulation loop so it adopts it
            // on the next tick instead of continuing with the old world.
            if state.world_replacement_tx.try_send(world).is_err() {
                tracing::warn!("Failed to send replacement world to simulation loop");
            }

            Json(RemixResponse {
                success: true,
                seed,
                grid_width,
                grid_height,
            })
        }
        Err(e) => {
            tracing::error!(seed, error = %e, "Failed to create world for remix");
            Json(RemixResponse {
                success: false,
                seed,
                grid_width,
                grid_height,
            })
        }
    }
}

/// Builds a `SimulationSnapshot` from a `WorldState`.
///
/// Maps each agent in the world to an `AgentSnapshot` for serialization
/// to dashboard clients.
#[instrument(skip_all, fields(tick = world.tick, agents = world.agents.len()))]
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
            success: true,
            seed: 42,
            grid_width: 64,
            grid_height: 64,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("gridWidth"));
        assert!(json.contains("gridHeight"));
        assert!(json.contains("\"success\":true"));
    }

    #[test]
    fn test_remix_response_failure() {
        let resp = RemixResponse {
            success: false,
            seed: 0,
            grid_width: 32,
            grid_height: 32,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"success\":false"));
    }

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".to_string(),
            uptime_seconds: 120,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("uptimeSeconds"));
        assert!(json.contains("120"));
    }

    #[test]
    fn test_build_snapshot_from_world() {
        let config = forge_types::config::ForgeConfig::default();
        let world = forge_core::WorldState::new(config).unwrap();
        let snapshot = build_snapshot_from_world(&world);
        assert_eq!(snapshot.tick, 0);
        assert_eq!(snapshot.schema_version, SCHEMA_VERSION);
        assert!(snapshot.grid_width > 0);
        assert!(snapshot.grid_height > 0);
    }
}
