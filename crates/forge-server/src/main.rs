//! FORGE simulation server binary.
//!
//! Runs an Axum HTTP+WebSocket server that:
//! - Hosts a background simulation loop stepping a `WorldState`
//! - Broadcasts `SimulationSnapshot` to WebSocket clients each tick
//! - Serves REST endpoints for health, config, metrics, and scenario remix

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use forge_observability::{init_tracing, TracingOptions};
use tokio::sync::broadcast;
use tracing::{info, warn};

use forge_server::api::build_snapshot_from_world;
use forge_server::config::ServerConfig;
use forge_server::metrics::MetricsCollector;
use forge_server::routes::build_router;
use forge_server::state::SharedState;
use forge_server::ws_handler::{AppState, SubscriptionManager, WsMessage};

#[tokio::main]
async fn main() {
    // Initialize tracing first so ServerConfig::from_env() logs are captured.
    // Format (text/JSON) is env-driven via FORGE_LOG_FORMAT; the default filter
    // preserves the historical per-binary directive.
    init_tracing(TracingOptions::new("forge_server=info,forge_core=info"));

    let config = ServerConfig::from_env();

    // `ServerConfig`'s Debug impl redacts `auth_token`.
    info!(?config, "FORGE server starting");

    // Surface the network-exposure posture before anything binds, so
    // the warning is the first thing in the log rather than buried
    // after the simulation-loop chatter.
    for advisory in config.security_advisories() {
        warn!(advisory = ?advisory, "{}", advisory.message());
    }

    // Create shared state
    let shared_state = SharedState::new();
    let metrics_collector = Arc::new(Mutex::new(MetricsCollector::new()));
    let (tx, _rx) = broadcast::channel::<WsMessage>(config.broadcast_capacity);
    let (world_tx, world_rx) = tokio::sync::mpsc::channel::<forge_core::WorldState>(1);

    // Persistent training/trace history. A failure to open the store is
    // non-fatal: fall back to an in-memory store so the server still serves.
    let history: Arc<dyn forge_server::history::HistoryStore> =
        match forge_server::history::JsonlHistoryStore::open(
            &config.history_dir,
            config.history_retention,
        ) {
            Ok(store) => Arc::new(store),
            Err(e) => {
                warn!(error = %e, "Failed to open history store; using in-memory fallback");
                Arc::new(forge_server::history::InMemoryHistoryStore::new(
                    config.history_retention,
                ))
            }
        };
    // Server-session run id (ms since epoch) used when a request supplies none.
    let run_id: Arc<str> = Arc::from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis().to_string())
            .unwrap_or_else(|_| "0".to_string())
            .as_str(),
    );

    let app_state = AppState {
        tx: tx.clone(),
        subscriptions: Arc::new(Mutex::new(SubscriptionManager::new())),
        shared_state: shared_state.clone(),
        next_client_id: Arc::new(AtomicU64::new(1)),
        metrics_collector: metrics_collector.clone(),
        start_time: Instant::now(),
        world_replacement_tx: Arc::new(world_tx),
        rest_world: forge_server::env::new_session_world(),
        history,
        run_id,
        history_query_limit: config.history_query_limit,
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

    // Router + middleware stack (CORS, timeout, body limit, optional
    // bearer auth on the mutating routes) live in `forge_server::routes`
    // so they are exercised by tests rather than only by the binary.
    let app = build_router(app_state, &config);

    info!(
        bind = %config.bind_addr,
        tick_ms = config.tick_interval_ms,
        request_timeout_ms = config.request_timeout_ms,
        max_body_bytes = config.max_body_bytes,
        auth = config.auth_enabled(),
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

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        tracing::error!(error = %e, "Server error");
        std::process::exit(1);
    }
    info!("Server shut down cleanly");
}

/// Resolves on SIGINT (Ctrl-C) or, on Unix, SIGTERM.
///
/// Handed to `axum::serve(..).with_graceful_shutdown(..)` so in-flight
/// requests finish and the listener closes instead of the process being
/// torn down mid-response. Mirrors the pattern
/// `forge_mc_runner::metrics::serve_metrics` already uses.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            warn!(error = %e, "Failed to install Ctrl-C handler; SIGINT will not shut down cleanly");
            // Never resolve: leave SIGTERM as the shutdown path rather
            // than shutting down immediately on a handler-install error.
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                warn!(error = %e, "Failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("SIGINT received, shutting down"),
        _ = terminate => info!("SIGTERM received, shutting down"),
    }
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
