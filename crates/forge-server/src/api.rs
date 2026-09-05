//! REST API route definitions and handler functions.

use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::history::{RunSummary, TraceRecord, TrainingRecord};
use crate::state::{AgentSnapshot, SimulationSnapshot};
use crate::ws_handler::{AppState, WsMessage};
use crate::SCHEMA_VERSION;

/// HTTP header a client may set to tag pushed metrics/traces with a run id.
const RUN_ID_HEADER: &str = "x-forge-run-id";

/// Query parameters shared by the history GET endpoints.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryQuery {
    /// Restrict results to a single run id.
    pub run_id: Option<String>,
    /// Maximum records to return (defaults to the server's configured limit).
    pub limit: Option<usize>,
}

/// Resolve the effective run id for a write: `?runId=` wins, then the
/// `X-Forge-Run-Id` header, then the server-session run id on `AppState`.
fn resolve_run_id(state: &AppState, headers: &HeaderMap, query: &HistoryQuery) -> String {
    if let Some(id) = query.run_id.as_deref().filter(|s| !s.is_empty()) {
        return id.to_string();
    }
    if let Some(id) = headers
        .get(RUN_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
    {
        return id.to_string();
    }
    state.run_id.to_string()
}

/// Resolve a request's `limit` against the server's `configured` cap.
///
/// Absent → the configured limit. Present → the smaller of the two, so
/// `?limit=1000000` cannot make the server read and serialize an
/// unbounded slice of history on an unauthenticated GET. (This
/// previously only supplied a default, despite the name.)
///
/// Takes the cap rather than the whole `AppState` so it is directly
/// unit-testable.
fn effective_limit(configured: usize, query: &HistoryQuery) -> usize {
    query
        .limit
        .map_or(configured, |requested| requested.min(configured))
}

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
    let mut metrics = if let Ok(mut mc) = state.metrics_collector.lock() {
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
            // Send the new world to the simulation loop first — if the
            // channel is full a prior remix is still pending and we bail.
            if state.world_replacement_tx.try_send(world).is_err() {
                tracing::warn!("Replacement channel full — a prior remix is still pending");
                return Json(RemixResponse {
                    success: false,
                    seed,
                    grid_width,
                    grid_height,
                });
            }

            // Build snapshot from the new world config (world was moved,
            // but we can reconstruct a minimal snapshot for immediate broadcast).
            let mut snapshot_config = forge_types::config::ForgeConfig::default();
            snapshot_config.world.seed = seed;
            if let Some(gs) = grid_size {
                snapshot_config.world.width = gs;
                snapshot_config.world.height = gs;
            }
            if let Ok(snapshot_world) = forge_core::WorldState::new(snapshot_config) {
                let snapshot = build_snapshot_from_world(&snapshot_world);
                state.shared_state.update(snapshot.clone());
                if state.tx.send(WsMessage::StateUpdate(snapshot)).is_err() {
                    tracing::trace!("No active subscribers for remix broadcast");
                }
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

/// Accepts training metrics from the Python training loop and broadcasts
/// them to all WebSocket clients.
#[instrument(skip_all)]
pub async fn training_metrics_handler(
    State(state): State<AppState>,
    Query(query): Query<HistoryQuery>,
    headers: HeaderMap,
    Json(metrics): Json<crate::metrics::TrainingMetrics>,
) -> Json<AckResponse> {
    let run_id = resolve_run_id(&state, &headers, &query);
    tracing::debug!(
        run_id = %run_id,
        episode = metrics.episode,
        mean_reward = metrics.mean_reward,
        "Received training metrics"
    );

    // Persist first so the history reflects the sample even if no clients are
    // currently subscribed; persistence errors are logged, never fatal.
    if let Err(e) = state.history.append_training(&run_id, &metrics) {
        tracing::warn!(error = %e, "Failed to persist training metrics");
    }

    if state.tx.send(WsMessage::TrainingMetrics(metrics)).is_err() {
        tracing::trace!("No active subscribers for training metrics broadcast");
    }

    Json(AckResponse { accepted: true })
}

/// Returns stored training-metric history, optionally filtered by `runId` and
/// capped at `limit` (defaults to the server's configured query limit).
#[instrument(skip_all)]
pub async fn training_history_handler(
    State(state): State<AppState>,
    Query(query): Query<HistoryQuery>,
) -> Json<Vec<TrainingRecord>> {
    let limit = effective_limit(state.history_query_limit, &query);
    match state
        .history
        .training_history(query.run_id.as_deref(), limit)
    {
        Ok(records) => Json(records),
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read training history");
            Json(Vec::new())
        }
    }
}

/// Accepts decision trace entries from agent planning and broadcasts
/// them to all WebSocket clients.
#[instrument(skip_all)]
pub async fn decision_traces_handler(
    State(state): State<AppState>,
    Query(query): Query<HistoryQuery>,
    headers: HeaderMap,
    Json(traces): Json<Vec<crate::metrics::DecisionTraceEntry>>,
) -> Json<AckResponse> {
    let run_id = resolve_run_id(&state, &headers, &query);
    tracing::debug!(run_id = %run_id, count = traces.len(), "Received decision traces");

    if let Err(e) = state.history.append_traces(&run_id, &traces) {
        tracing::warn!(error = %e, "Failed to persist decision traces");
    }

    if state.tx.send(WsMessage::DecisionTraces(traces)).is_err() {
        tracing::trace!("No active subscribers for decision traces broadcast");
    }

    Json(AckResponse { accepted: true })
}

/// Returns stored decision-trace history, optionally filtered by `runId` and
/// capped at `limit` (defaults to the server's configured query limit).
#[instrument(skip_all)]
pub async fn traces_history_handler(
    State(state): State<AppState>,
    Query(query): Query<HistoryQuery>,
) -> Json<Vec<TraceRecord>> {
    let limit = effective_limit(state.history_query_limit, &query);
    match state.history.traces_history(query.run_id.as_deref(), limit) {
        Ok(records) => Json(records),
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read trace history");
            Json(Vec::new())
        }
    }
}

/// Returns a summary of all runs seen across stored training/trace history.
#[instrument(skip_all)]
pub async fn runs_handler(State(state): State<AppState>) -> Json<Vec<RunSummary>> {
    match state.history.runs() {
        Ok(runs) => Json(runs),
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read run summaries");
            Json(Vec::new())
        }
    }
}

/// Simple acknowledgement response for POST endpoints.
#[derive(Debug, Serialize)]
pub struct AckResponse {
    /// Whether the payload was accepted.
    pub accepted: bool,
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

    /// Absent `limit` falls back to the server's configured cap.
    #[test]
    fn effective_limit_defaults_to_configured_cap() {
        let query = HistoryQuery::default();
        assert_eq!(effective_limit(500, &query), 500);
    }

    /// A request below the cap is honoured verbatim.
    #[test]
    fn effective_limit_honours_smaller_request() {
        let query = HistoryQuery {
            limit: Some(10),
            ..HistoryQuery::default()
        };
        assert_eq!(effective_limit(500, &query), 10);
    }

    /// The bug this fixes: an oversized `?limit=` used to pass straight
    /// through, so an unauthenticated GET could ask for every record.
    #[test]
    fn effective_limit_clamps_oversized_request() {
        let query = HistoryQuery {
            limit: Some(1_000_000),
            ..HistoryQuery::default()
        };
        assert_eq!(effective_limit(500, &query), 500);
        // Exactly at the cap is not clamped down.
        let at_cap = HistoryQuery {
            limit: Some(500),
            ..HistoryQuery::default()
        };
        assert_eq!(effective_limit(500, &at_cap), 500);
    }

    /// `usize::MAX` is the shape a caller would use to try to defeat the
    /// cap; it must clamp like any other oversized value.
    #[test]
    fn effective_limit_clamps_usize_max() {
        let query = HistoryQuery {
            limit: Some(usize::MAX),
            ..HistoryQuery::default()
        };
        assert_eq!(effective_limit(7, &query), 7);
    }

    #[test]
    fn test_history_query_deserializes_camel_case() {
        let q: HistoryQuery = serde_json::from_str(r#"{"runId":"r1","limit":5}"#).unwrap();
        assert_eq!(q.run_id.as_deref(), Some("r1"));
        assert_eq!(q.limit, Some(5));
        // Both fields are optional.
        let empty: HistoryQuery = serde_json::from_str("{}").unwrap();
        assert!(empty.run_id.is_none() && empty.limit.is_none());
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

    #[test]
    fn test_build_snapshot_agent_count_matches() {
        let mut config = forge_types::config::ForgeConfig::default();
        config.agents.num_agents = 4;
        let world = forge_core::WorldState::new(config).unwrap();
        let snapshot = build_snapshot_from_world(&world);
        assert_eq!(snapshot.agents.len(), 4);
        for agent in &snapshot.agents {
            assert!(agent.alive);
            assert!(agent.x < snapshot.grid_width);
            assert!(agent.y < snapshot.grid_height);
        }
    }

    #[test]
    fn test_build_snapshot_grid_dimensions() {
        let mut config = forge_types::config::ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 32;
        let world = forge_core::WorldState::new(config).unwrap();
        let snapshot = build_snapshot_from_world(&world);
        assert_eq!(snapshot.grid_width, 16);
        assert_eq!(snapshot.grid_height, 32);
    }

    #[test]
    fn test_ack_response_serialization() {
        let ack = AckResponse { accepted: true };
        let json = serde_json::to_string(&ack).unwrap();
        assert!(json.contains("\"accepted\":true"));

        let ack_false = AckResponse { accepted: false };
        let json = serde_json::to_string(&ack_false).unwrap();
        assert!(json.contains("\"accepted\":false"));
    }

    #[test]
    fn test_remix_request_deserialization() {
        let json = r#"{"seed":42,"gridSize":16}"#;
        let req: RemixRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.grid_size, Some(16));
    }

    #[test]
    fn test_remix_request_optional_fields() {
        let json = r#"{}"#;
        let req: RemixRequest = serde_json::from_str(json).unwrap();
        assert!(req.seed.is_none());
        assert!(req.grid_size.is_none());
    }

    #[test]
    fn test_remix_request_partial_fields() {
        let json = r#"{"seed":99}"#;
        let req: RemixRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.seed, Some(99));
        assert!(req.grid_size.is_none());
    }

    #[test]
    fn test_config_response_fields() {
        let resp = ConfigResponse {
            version: "1.2.3".to_string(),
            schema_version: 5,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"version\":\"1.2.3\""));
        assert!(json.contains("\"schemaVersion\":5"));
    }

    #[test]
    fn test_health_response_fields() {
        let resp = HealthResponse {
            status: "degraded".to_string(),
            uptime_seconds: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"status\":\"degraded\""));
        assert!(json.contains("\"uptimeSeconds\":0"));
    }

    #[test]
    fn test_build_snapshot_events_empty() {
        let config = forge_types::config::ForgeConfig::default();
        let world = forge_core::WorldState::new(config).unwrap();
        let snapshot = build_snapshot_from_world(&world);
        assert!(snapshot.events.is_empty());
    }

    #[test]
    fn test_build_snapshot_after_step() {
        let config = forge_types::config::ForgeConfig::default();
        let mut world = forge_core::WorldState::new(config).unwrap();
        let actions: Vec<_> = (0..world.agents.len())
            .map(|_| forge_types::Action::Noop)
            .collect();
        let _result = world.step(&actions);
        let snapshot = build_snapshot_from_world(&world);
        assert_eq!(snapshot.tick, 1);
    }

    #[test]
    fn test_build_snapshot_correctness() {
        let mut config = forge_types::config::ForgeConfig::default();
        config.agents.num_agents = 3;
        config.world.width = 20;
        config.world.height = 25;
        let world = forge_core::WorldState::new(config).unwrap();
        let snapshot = build_snapshot_from_world(&world);
        assert_eq!(snapshot.tick, 0);
        assert_eq!(snapshot.agents.len(), 3);
        assert_eq!(snapshot.grid_width, 20);
        assert_eq!(snapshot.grid_height, 25);
        assert_eq!(snapshot.schema_version, SCHEMA_VERSION);
        assert!(snapshot.events.is_empty());
        for agent in &snapshot.agents {
            assert!(agent.alive);
            assert!(agent.health > 0);
            assert!(agent.x < 20);
            assert!(agent.y < 25);
            assert!(agent.team_id.is_none());
            assert!(agent.intent.is_none());
        }
    }

    #[test]
    fn test_remix_request_both_fields_none() {
        let json = r#"{}"#;
        let req: RemixRequest = serde_json::from_str(json).unwrap();
        assert!(req.seed.is_none());
        assert!(req.grid_size.is_none());
    }
}
