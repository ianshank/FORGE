//! FORGE simulation server binary.
//!
//! Runs an Axum HTTP+WebSocket server that:
//! - Hosts a background simulation loop stepping a `WorldState`
//! - Broadcasts `SimulationSnapshot` to WebSocket clients each tick
//! - Serves REST endpoints for health, config, metrics, and scenario remix

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::routing::{get, post};
use axum::Router;
use forge_observability::{init_tracing, TracingOptions};
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

use forge_server::api::{
    build_snapshot_from_world, config_handler, decision_traces_handler, health_handler,
    metrics_handler, remix_handler, training_metrics_handler,
};
use forge_server::config::ServerConfig;
use forge_server::metrics::MetricsCollector;
use forge_server::state::SharedState;
use forge_server::ws_handler::{ws_upgrade_handler, AppState, SubscriptionManager, WsMessage};

#[tokio::main]
async fn main() {
    // Initialize tracing first so ServerConfig::from_env() logs are captured.
    // Format (text/JSON) is env-driven via FORGE_LOG_FORMAT; the default filter
    // preserves the historical per-binary directive.
    init_tracing(TracingOptions::new("forge_server=info,forge_core=info"));

    let config = ServerConfig::from_env();

    info!(?config, "FORGE server starting");

    // Create shared state
    let shared_state = SharedState::new();
    let metrics_collector = Arc::new(Mutex::new(MetricsCollector::new()));
    let (tx, _rx) = broadcast::channel::<WsMessage>(config.broadcast_capacity);
    let (world_tx, world_rx) = tokio::sync::mpsc::channel::<forge_core::WorldState>(1);

    let app_state = AppState {
        tx: tx.clone(),
        subscriptions: Arc::new(Mutex::new(SubscriptionManager::new())),
        shared_state: shared_state.clone(),
        next_client_id: Arc::new(AtomicU64::new(1)),
        metrics_collector: metrics_collector.clone(),
        start_time: Instant::now(),
        world_replacement_tx: Arc::new(world_tx),
        rest_world: forge_server::env::new_session_world(),
    };

    // Create initial world
    let forge_config = forge_types::config::ForgeConfig::default();
    let world = match forge_core::WorldState::new(forge_config) {
        Ok(w) => {
            info!(agents = w.agents.len(), "Initial world created");
            w
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to create initial world");
            std::process::exit(1);
        }
    };

    let initial_snapshot = build_snapshot_from_world(&world);
    shared_state.update(initial_snapshot);

    // Spawn background simulation loop
    let sim_state = shared_state.clone();
    let sim_tx = tx.clone();
    let sim_metrics = metrics_collector.clone();
    let tick_interval_ms = config.tick_interval_ms;
    tokio::spawn(async move {
        simulation_loop(
            world,
            sim_state,
            sim_tx,
            tick_interval_ms,
            sim_metrics,
            world_rx,
        )
        .await;
    });

    // CORS layer — restrict to configured origins
    let origins: Vec<_> = config
        .allowed_origins
        .iter()
        .filter_map(|o| o.parse::<axum::http::HeaderValue>().ok())
        .collect();
    let cors = CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    // Build router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/config", get(config_handler))
        .route("/api/metrics", get(metrics_handler))
        .route("/api/scenario/remix", post(remix_handler))
        .route("/api/training-metrics", post(training_metrics_handler))
        .route("/api/decision-traces", post(decision_traces_handler))
        .route("/api/env/reset", post(forge_server::env::reset_handler))
        .route("/api/env/step", post(forge_server::env::step_handler))
        .route("/api/env/render", get(forge_server::env::render_handler))
        .route("/ws", get(ws_upgrade_handler))
        .layer(cors)
        .with_state(app_state);

    info!(
        bind = %config.bind_addr,
        tick_ms = config.tick_interval_ms,
        "Server ready"
    );

    let listener = tokio::net::TcpListener::bind(config.bind_addr)
        .await
        .unwrap_or_else(|e| {
            tracing::error!(
                bind = %config.bind_addr,
                error = %e,
                "Failed to bind — is the port already in use?"
            );
            std::process::exit(1);
        });

    axum::serve(listener, app).await.expect("Server error");
}

/// Background loop that steps the simulation and broadcasts state.
///
/// On each tick this function:
/// 1. Checks for a replacement world from the remix endpoint
/// 2. Steps the `WorldState` with no-op actions (demo/visualization mode)
/// 3. Records tick metrics
/// 4. Builds a `SimulationSnapshot` and updates shared state
/// 5. Broadcasts the snapshot to all WebSocket clients
/// 6. Resets the world if the simulation ended
#[tracing::instrument(skip_all, fields(tick_interval_ms))]
async fn simulation_loop(
    mut world: forge_core::WorldState,
    shared_state: SharedState,
    tx: broadcast::Sender<WsMessage>,
    tick_interval_ms: u64,
    metrics_collector: Arc<Mutex<MetricsCollector>>,
    mut world_rx: tokio::sync::mpsc::Receiver<forge_core::WorldState>,
) {
    let interval = Duration::from_millis(tick_interval_ms);
    let mut tick_timer = tokio::time::interval(interval);

    info!("Simulation loop started");

    loop {
        tick_timer.tick().await;

        // Check for a replacement world from the remix endpoint
        if let Ok(new_world) = world_rx.try_recv() {
            info!(
                "Adopting remixed world (seed={})",
                new_world.config.world.seed
            );
            world = new_world;
        }

        // Step the world with no-op actions (demo/visualization mode)
        let num_agents = world.agents.len();
        let actions = vec![forge_types::Action::Noop; num_agents];
        let _result = world.step(&actions);

        // Record tick in metrics
        if let Ok(mut mc) = metrics_collector.lock() {
            mc.record_tick();
        }

        // Build snapshot and broadcast
        let snapshot = build_snapshot_from_world(&world);
        shared_state.update(snapshot.clone());

        if tx.send(WsMessage::StateUpdate(snapshot)).is_err() {
            tracing::trace!("No active WebSocket subscribers");
        }

        // Reset if simulation ended
        if world.terminated || world.truncated {
            info!(tick = world.tick, "Simulation ended, resetting");
            let config = forge_types::config::ForgeConfig::default();
            match forge_core::WorldState::new(config) {
                Ok(new_world) => world = new_world,
                Err(e) => {
                    warn!(error = %e, "Failed to reset world, retrying next tick");
                }
            }
        }
    }
}
