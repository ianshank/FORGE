//! FORGE simulation server binary.
//!
//! Runs an Axum HTTP+WebSocket server that:
//! - Hosts a background simulation loop stepping a `WorldState`
//! - Broadcasts `SimulationSnapshot` to WebSocket clients each tick
//! - Serves REST endpoints for health, config, metrics, and scenario remix

use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use forge_server::api::{
    build_snapshot_from_world, config_handler, health_handler, metrics_handler, remix_handler,
};
use forge_server::state::SharedState;
use forge_server::ws_handler::{ws_upgrade_handler, AppState, SubscriptionManager, WsMessage};

/// Default bind address for the server.
const DEFAULT_BIND: &str = "0.0.0.0:8080";
/// Broadcast channel capacity for WebSocket fan-out.
const BROADCAST_CAPACITY: usize = 64;
/// Default simulation tick interval in milliseconds.
const DEFAULT_TICK_INTERVAL_MS: u64 = 100;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("forge_server=info,forge_core=info")),
        )
        .with_target(true)
        .init();

    let bind_addr: SocketAddr = std::env::var("FORGE_SERVER_BIND")
        .unwrap_or_else(|_| DEFAULT_BIND.to_string())
        .parse()
        .unwrap_or_else(|_| {
            warn!("Invalid FORGE_SERVER_BIND, using default");
            DEFAULT_BIND.parse().expect("default bind is valid")
        });

    let tick_interval_ms: u64 = std::env::var("FORGE_SERVER_TICK_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_TICK_INTERVAL_MS);

    // Create shared state
    let shared_state = SharedState::new();
    let (tx, _rx) = broadcast::channel::<WsMessage>(BROADCAST_CAPACITY);

    let app_state = AppState {
        tx: tx.clone(),
        subscriptions: Arc::new(Mutex::new(SubscriptionManager::new())),
        shared_state: shared_state.clone(),
        next_client_id: Arc::new(AtomicU64::new(1)),
    };

    // Create initial world
    let config = forge_types::config::ForgeConfig::default();
    let world = forge_core::WorldState::new(config).expect("Failed to create initial world");
    let initial_snapshot = build_snapshot_from_world(&world);
    shared_state.update(initial_snapshot);

    // Spawn background simulation loop
    let sim_state = shared_state.clone();
    let sim_tx = tx.clone();
    tokio::spawn(async move {
        simulation_loop(world, sim_state, sim_tx, tick_interval_ms).await;
    });

    // CORS layer for dashboard dev server
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/config", get(config_handler))
        .route("/api/metrics", get(metrics_handler))
        .route("/api/scenario/remix", post(remix_handler))
        .route("/ws", get(ws_upgrade_handler))
        .layer(cors)
        .with_state(app_state);

    info!("FORGE server starting on {}", bind_addr);
    info!("Tick interval: {}ms", tick_interval_ms);

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app).await.expect("Server error");
}

/// Background loop that steps the simulation and broadcasts state.
async fn simulation_loop(
    mut world: forge_core::WorldState,
    shared_state: SharedState,
    tx: broadcast::Sender<WsMessage>,
    tick_interval_ms: u64,
) {
    let interval = Duration::from_millis(tick_interval_ms);
    let mut tick_timer = tokio::time::interval(interval);

    info!("Simulation loop started");

    loop {
        tick_timer.tick().await;

        // Step the world with no-op actions (demo/visualization mode)
        let num_agents = world.agents.len();
        let actions = vec![forge_types::Action::Noop; num_agents];
        let _result = world.step(&actions);

        // Build snapshot and broadcast
        let snapshot = build_snapshot_from_world(&world);
        shared_state.update(snapshot.clone());

        let _ = tx.send(WsMessage::StateUpdate(snapshot));

        // Reset if simulation ended
        if world.terminated || world.truncated {
            info!(tick = world.tick, "Simulation ended, resetting");
            let config = forge_types::config::ForgeConfig::default();
            match forge_core::WorldState::new(config) {
                Ok(new_world) => world = new_world,
                Err(e) => {
                    warn!("Failed to reset world: {}", e);
                    break;
                }
            }
        }
    }
}
