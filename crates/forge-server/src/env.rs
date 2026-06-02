//! REST endpoints for driving a session [`WorldState`](forge_core::WorldState).
//!
//! Exposes a small gym-style surface — `/api/env/{reset,step,render}` — backed
//! by a **session world** held in [`AppState::rest_world`]. This world is
//! deliberately decoupled from the demo broadcast loop (which owns its own world
//! and only accepts whole-world replacement), so REST clients get clean
//! request/response semantics without racing the ticker.
//!
//! All responses reuse the existing [`SimulationSnapshot`] DTO (camelCase,
//! `schemaVersion`); no new wire types are invented for the world state.

use std::sync::{Arc, Mutex, MutexGuard};

use axum::extract::State;
use axum::Json;
use forge_types::action::Action;
use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};

use crate::api::build_snapshot_from_world;
use crate::error::ApiError;
use crate::state::SimulationSnapshot;
use crate::ws_handler::AppState;

/// Default number of agents in a REST session world (gym-style single agent).
const DEFAULT_REST_NUM_AGENTS: u32 = 1;
/// Communication vocabulary size used when decoding REST actions.
const REST_COMM_VOCAB_SIZE: u16 = 0;
/// Whether drone actions are enabled when decoding REST actions.
const REST_DRONE_ACTIONS_ENABLED: bool = false;

/// A REST-managed session world, decoupled from the demo broadcast loop.
pub type SessionWorld = Arc<Mutex<Option<forge_core::WorldState>>>;

/// Creates a fresh, empty session world handle.
pub fn new_session_world() -> SessionWorld {
    Arc::new(Mutex::new(None))
}

/// Request body for `POST /api/env/reset`. All fields are optional; absent
/// fields fall back to [`ForgeConfig::default`] / the module defaults.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ResetRequest {
    /// World RNG seed.
    pub seed: Option<u64>,
    /// Square grid size (applied to both width and height).
    pub grid_size: Option<u16>,
    /// Number of agents to spawn.
    pub num_agents: Option<u32>,
}

/// Request body for `POST /api/env/step`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepRequest {
    /// Discrete action id applied to agent 0 (other agents idle).
    pub action: u32,
}

/// Response from `POST /api/env/step`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StepResponse {
    /// Post-step world snapshot.
    pub snapshot: SimulationSnapshot,
    /// Reward for agent 0 on this step.
    pub reward: f32,
    /// Whether the episode terminated.
    pub terminated: bool,
    /// Whether the episode was truncated.
    pub truncated: bool,
}

/// Response from `GET /api/env/render`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderResponse {
    /// Current world snapshot.
    pub snapshot: SimulationSnapshot,
    /// ASCII rendering of the grid.
    pub ascii: String,
}

/// Locks the session world, mapping a poisoned lock to an internal error.
fn lock_world(
    world: &SessionWorld,
) -> Result<MutexGuard<'_, Option<forge_core::WorldState>>, ApiError> {
    world
        .lock()
        .map_err(|_| ApiError::Internal("session world lock poisoned".to_string()))
}

/// `POST /api/env/reset` — builds a new session world and returns its snapshot.
#[instrument(skip_all)]
pub async fn reset_handler(
    State(state): State<AppState>,
    body: Option<Json<ResetRequest>>,
) -> Result<Json<SimulationSnapshot>, ApiError> {
    let req = body.map(|Json(r)| r).unwrap_or_default();

    let mut config = ForgeConfig::default();
    if let Some(seed) = req.seed {
        config.world.seed = seed;
    }
    if let Some(grid_size) = req.grid_size {
        config.world.width = grid_size;
        config.world.height = grid_size;
    }
    config.agents.num_agents = req.num_agents.unwrap_or(DEFAULT_REST_NUM_AGENTS);
    let seed = config.world.seed;

    let world = forge_core::WorldState::new(config).map_err(|e| ApiError::Config(e.to_string()))?;
    let snapshot = build_snapshot_from_world(&world);

    *lock_world(&state.rest_world)? = Some(world);
    info!(
        seed,
        agents = snapshot.agents.len(),
        "REST session world reset"
    );
    Ok(Json(snapshot))
}

/// `POST /api/env/step` — steps the session world by one tick.
#[instrument(skip_all)]
pub async fn step_handler(
    State(state): State<AppState>,
    Json(req): Json<StepRequest>,
) -> Result<Json<StepResponse>, ApiError> {
    let action =
        Action::from_discrete(req.action, REST_COMM_VOCAB_SIZE, REST_DRONE_ACTIONS_ENABLED)
            .ok_or(ApiError::InvalidAction(req.action))?;

    let mut guard = lock_world(&state.rest_world)?;
    let world = guard.as_mut().ok_or(ApiError::NotReset)?;

    let num_agents = world.agents.len();
    let mut actions = vec![Action::Noop; num_agents];
    if let Some(first) = actions.first_mut() {
        *first = action;
    }

    let result = world.step(&actions);
    let reward = result.rewards.first().copied().unwrap_or(0.0);
    let snapshot = build_snapshot_from_world(world);

    Ok(Json(StepResponse {
        snapshot,
        reward,
        terminated: result.terminated,
        truncated: result.truncated,
    }))
}

/// `GET /api/env/render` — returns the snapshot plus an ASCII grid.
#[instrument(skip_all)]
pub async fn render_handler(
    State(state): State<AppState>,
) -> Result<Json<RenderResponse>, ApiError> {
    let guard = lock_world(&state.rest_world)?;
    let world = guard.as_ref().ok_or(ApiError::NotReset)?;
    Ok(Json(RenderResponse {
        snapshot: build_snapshot_from_world(world),
        ascii: world.to_debug_grid(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MetricsCollector;
    use crate::state::SharedState;
    use crate::ws_handler::{SubscriptionManager, WsMessage};
    use axum::response::IntoResponse;
    use std::sync::atomic::AtomicU64;
    use std::time::Instant;
    use tokio::sync::broadcast;

    fn test_state() -> AppState {
        let (tx, _rx) = broadcast::channel::<WsMessage>(8);
        let (world_tx, _world_rx) = tokio::sync::mpsc::channel(1);
        AppState {
            tx,
            subscriptions: Arc::new(Mutex::new(SubscriptionManager::new())),
            shared_state: SharedState::new(),
            next_client_id: Arc::new(AtomicU64::new(1)),
            metrics_collector: Arc::new(Mutex::new(MetricsCollector::new())),
            start_time: Instant::now(),
            world_replacement_tx: Arc::new(world_tx),
            rest_world: new_session_world(),
        }
    }

    #[tokio::test]
    async fn test_reset_creates_world() {
        let state = test_state();
        let body = Some(Json(ResetRequest {
            seed: Some(7),
            grid_size: Some(16),
            num_agents: Some(2),
        }));
        let Json(snapshot) = reset_handler(State(state.clone()), body).await.unwrap();
        assert_eq!(snapshot.agents.len(), 2);
        assert_eq!(snapshot.grid_width, 16);
        assert_eq!(snapshot.schema_version, crate::SCHEMA_VERSION);
        assert!(state.rest_world.lock().unwrap().is_some());
    }

    #[tokio::test]
    async fn test_reset_with_no_body_uses_defaults() {
        let state = test_state();
        let Json(snapshot) = reset_handler(State(state), None).await.unwrap();
        assert_eq!(snapshot.agents.len(), DEFAULT_REST_NUM_AGENTS as usize);
    }

    #[tokio::test]
    async fn test_step_before_reset_is_conflict() {
        let state = test_state();
        let err = step_handler(State(state), Json(StepRequest { action: 0 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ApiError::NotReset));
        assert_eq!(
            err.into_response().status(),
            axum::http::StatusCode::CONFLICT
        );
    }

    #[tokio::test]
    async fn test_reset_then_step_advances_tick() {
        let state = test_state();
        let Json(_) = reset_handler(State(state.clone()), None).await.unwrap();
        let Json(resp) = step_handler(State(state), Json(StepRequest { action: 0 }))
            .await
            .unwrap();
        assert_eq!(resp.snapshot.tick, 1);
    }

    #[tokio::test]
    async fn test_step_invalid_action_is_unprocessable() {
        let state = test_state();
        let Json(_) = reset_handler(State(state.clone()), None).await.unwrap();
        let err = step_handler(State(state), Json(StepRequest { action: u32::MAX }))
            .await
            .unwrap_err();
        assert!(matches!(err, ApiError::InvalidAction(_)));
        assert_eq!(
            err.into_response().status(),
            axum::http::StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    #[tokio::test]
    async fn test_render_before_reset_is_conflict() {
        let state = test_state();
        let err = render_handler(State(state)).await.unwrap_err();
        assert!(matches!(err, ApiError::NotReset));
    }

    #[tokio::test]
    async fn test_render_returns_ascii_and_snapshot() {
        let state = test_state();
        let Json(_) = reset_handler(
            State(state.clone()),
            Some(Json(ResetRequest {
                seed: Some(1),
                grid_size: Some(12),
                num_agents: Some(1),
            })),
        )
        .await
        .unwrap();
        let Json(render) = render_handler(State(state)).await.unwrap();
        assert_eq!(render.snapshot.grid_width, 12);
        assert!(!render.ascii.is_empty());
    }

    #[tokio::test]
    async fn test_reset_is_deterministic_for_same_seed() {
        let run = || async {
            let state = test_state();
            let Json(_) = reset_handler(
                State(state.clone()),
                Some(Json(ResetRequest {
                    seed: Some(99),
                    grid_size: Some(16),
                    num_agents: Some(1),
                })),
            )
            .await
            .unwrap();
            let Json(resp) = step_handler(State(state), Json(StepRequest { action: 1 }))
                .await
                .unwrap();
            (resp.snapshot.agents[0].x, resp.snapshot.agents[0].y)
        };
        assert_eq!(run().await, run().await);
    }
}
