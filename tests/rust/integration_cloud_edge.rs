//! Integration tests for the FORGE cloud-edge data pipeline (Phase 5/6).
//!
//! These tests exercise the full cloud-edge loop: replay generation, storage,
//! trajectory reconstruction, edge agent inference, telemetry collection,
//! batch transport, worker registry lifecycle, config backward compatibility,
//! and model version management.
//!
//! To run: `cargo test --test integration_cloud_edge`

use std::collections::VecDeque;
use std::sync::Mutex;

use forge_agent::latent_mcts::model::StubLatentModel;
use forge_agent::latent_mcts::search::LatentMctsConfig;
use forge_agent::mcts::tree::MctsConfig;
use forge_cloud::traits::{ModelStore, ReplayStore, WorkerManager};
use forge_cloud::{
    InMemoryWorkerRegistry, LocalModelStore, LocalReplayStore, ReplayBatch,
    TrajectoryReconstructor, WorkerMetadata,
};
use forge_data::edge_replay::EdgeReplayLoader;
use forge_data::loader::DatasetLoader;
use forge_edge::{EdgeAgent, TelemetryCollector};
use forge_replay::compact::CompactReplay;
use forge_types::agent_interface::AgentInterface;
use forge_types::config::ForgeConfig;
use forge_types::constants::{self, OBS_EMPTY_SLOT_ITEM};
use forge_types::error::ForgeResult;
use forge_types::observation::{InventoryObservation, Observation, TileObservation};
use forge_types::transport::ReplayTransport;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Creates a small, fast-to-test ForgeConfig.
fn make_config() -> ForgeConfig {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.seed = 42;
    config.agents.num_agents = 1;
    config.agents.comm_vocab_size = 0;
    config.task.max_episode_length = 100;
    config
}

/// Builds a CompactReplay with the given seed and number of noop ticks.
fn make_replay(seed: u64, ticks: usize) -> CompactReplay {
    let config = make_config();
    let mut builder = CompactReplay::builder(config, seed);
    for _ in 0..ticks {
        builder.record_tick(vec![0]); // Noop action
    }
    builder
        .agent_names(vec!["TestAgent".to_string()])
        .final_rewards(vec![0.0])
        .build()
}

/// Builds a minimal observation for edge agent testing.
fn make_test_observation() -> Observation {
    Observation {
        grid_view: vec![TileObservation::default()],
        view_width: 1,
        view_height: 1,
        inventory: InventoryObservation {
            slots: vec![(OBS_EMPTY_SLOT_ITEM, 0)],
        },
        health: 1.0,
        stamina: 1.0,
        position: (5, 5),
        messages: vec![],
        day_phase: 1,
        task_progress: vec![],
        altitude: 0,
        battery: 1.0,
        morphology: 0,
        heading: 0,
        crop_scan_results: vec![],
        soil_readings: vec![],
        disease_detections: 0,
        report_ready: false,
    }
}

/// Creates an EdgeAgent backed by a StubLatentModel.
fn make_edge_agent() -> EdgeAgent<StubLatentModel> {
    let model = StubLatentModel::new(8, 64);
    let edge_cfg = forge_types::config::EdgeConfig {
        mcts_latency_budget_ms: 100,
        mcts_min_simulations: 4,
        mcts_max_simulations: 50,
        latency_ema_alpha: 0.3,
        telemetry_buffer_bytes: 1_048_576,
        ..Default::default()
    };
    let mcts_cfg = LatentMctsConfig {
        base: MctsConfig {
            num_simulations: 10,
            action_space: 8,
            max_depth: 10,
            ..MctsConfig::default()
        },
        ..LatentMctsConfig::default()
    };
    EdgeAgent::new(model, &edge_cfg, mcts_cfg, "test-v1".to_string())
}

/// Mock transport that stores sent payloads in memory.
struct MockTransport {
    queue: Mutex<VecDeque<(String, Vec<u8>)>>,
}

impl MockTransport {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    fn sent_count(&self) -> usize {
        self.queue.lock().unwrap().len()
    }
}

impl ReplayTransport for MockTransport {
    fn send(&self, key: &str, payload: &[u8]) -> ForgeResult<()> {
        self.queue
            .lock()
            .unwrap()
            .push_back((key.to_string(), payload.to_vec()));
        Ok(())
    }

    fn receive(&self) -> ForgeResult<Option<(String, Vec<u8>)>> {
        Ok(self.queue.lock().unwrap().pop_front())
    }

    fn backend_name(&self) -> &str {
        "mock"
    }
}

// ---------------------------------------------------------------------------
// 1. Worker generates replay, store loads it back
// ---------------------------------------------------------------------------

/// Generate a CompactReplay, store it via LocalReplayStore, load it back,
/// and verify the loaded replay matches the original.
#[test]
fn test_worker_generates_replay_and_store_loads() {
    let dir = tempdir().unwrap();
    let store = LocalReplayStore::new(dir.path());

    // Generate a replay with 5 ticks
    let replay = make_replay(42, 5);
    assert_eq!(replay.actions.len(), 5);
    assert!(replay.validate_config());

    // Store and load back
    store.store(&replay, "episode_42").unwrap();
    let loaded = store.load("episode_42").unwrap();

    // Verify equality
    assert_eq!(loaded.seed, replay.seed);
    assert_eq!(loaded.actions.len(), replay.actions.len());
    assert_eq!(loaded.config_hash, replay.config_hash);
    assert!(loaded.validate_config());
    assert_eq!(loaded.metadata.total_ticks, replay.metadata.total_ticks);
    assert_eq!(loaded.metadata.agent_names, replay.metadata.agent_names);
    assert_eq!(loaded.metadata.final_rewards, replay.metadata.final_rewards);

    // Verify the stored bytes round-trip identically
    let original_bytes = replay.to_bytes().unwrap();
    let loaded_bytes = loaded.to_bytes().unwrap();
    assert_eq!(original_bytes, loaded_bytes);
}

// ---------------------------------------------------------------------------
// 2. Trajectory reconstruction determinism
// ---------------------------------------------------------------------------

/// Generate a CompactReplay, reconstruct via TrajectoryReconstructor twice,
/// and verify the outputs are identical (determinism guarantee).
#[test]
fn test_trajectory_reconstruction_determinism() {
    let replay = make_replay(42, 5);

    let traj_a = TrajectoryReconstructor::reconstruct(&replay).unwrap();
    let traj_b = TrajectoryReconstructor::reconstruct(&replay).unwrap();

    // Same number of steps
    assert_eq!(traj_a.len(), traj_b.len());
    assert_eq!(traj_a.len(), 5);

    // Each step must have identical observations and rewards
    for (step_a, step_b) in traj_a.steps.iter().zip(traj_b.steps.iter()) {
        assert_eq!(step_a.tick, step_b.tick);
        assert_eq!(step_a.observations.len(), step_b.observations.len());
        assert_eq!(step_a.rewards, step_b.rewards);
        assert_eq!(step_a.terminated, step_b.terminated);
        assert_eq!(step_a.truncated, step_b.truncated);

        // Verify position determinism for each agent
        for (obs_a, obs_b) in step_a.observations.iter().zip(step_b.observations.iter()) {
            assert_eq!(obs_a.position, obs_b.position);
            assert_eq!(obs_a.health, obs_b.health);
        }
    }

    // Metadata must match
    assert_eq!(traj_a.metadata.seed, traj_b.metadata.seed);
    assert_eq!(traj_a.metadata.total_steps, traj_b.metadata.total_steps);
}

// ---------------------------------------------------------------------------
// 3. Edge replay loader end-to-end
// ---------------------------------------------------------------------------

/// Write CompactReplay files to a tempdir, load via EdgeReplayLoader,
/// and verify the OfflineDataset has the correct trajectory count.
#[test]
fn test_edge_replay_loader_end_to_end() {
    let dir = tempdir().unwrap();

    // Write 3 replays as .bin files
    for i in 0..3u64 {
        let replay = make_replay(100 + i, 5);
        let bytes = replay.to_bytes().unwrap();
        let path = dir.path().join(format!("replay_{i}.bin"));
        std::fs::write(path, bytes).unwrap();
    }

    let loader = EdgeReplayLoader::new();
    let dataset = loader.load(dir.path().to_str().unwrap()).unwrap();

    // Should have 3 trajectories, each with 5 steps
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.total_steps(), 15);
    assert_eq!(dataset.metadata.source, "EdgeReplayLoader");
}

// ---------------------------------------------------------------------------
// 4. Edge agent with eval pattern
// ---------------------------------------------------------------------------

/// Create an EdgeAgent with a StubLatentModel, run select_action 10 times,
/// and verify all responses have valid action IDs.
#[test]
fn test_edge_agent_with_eval_pattern() {
    let mut agent = make_edge_agent();
    let obs = make_test_observation();

    for i in 0..10 {
        let resp = agent.select_action(&obs, 0);
        assert!(
            resp.action_id < agent.action_space_size(),
            "Action {} out of range at iteration {}",
            resp.action_id,
            i
        );
    }

    // Verify agent metadata
    assert_eq!(agent.name(), "EdgeAgent");
    assert_eq!(agent.model_version(), "test-v1");

    // Reset and verify clean state
    agent.reset();
    let resp = agent.select_action(&obs, 0);
    assert!(resp.action_id < agent.action_space_size());
}

// ---------------------------------------------------------------------------
// 5. Telemetry collect and flush
// ---------------------------------------------------------------------------

/// Create a TelemetryCollector, record replays, flush via MockTransport,
/// and verify all replays are delivered.
#[test]
fn test_telemetry_collect_and_flush() {
    let edge_cfg = forge_types::config::EdgeConfig {
        telemetry_buffer_bytes: 10_485_760, // 10 MB
        compress_telemetry: false,
        ..Default::default()
    };
    let mut collector = TelemetryCollector::new(&edge_cfg);
    let transport = MockTransport::new();

    // Record 5 replays
    for i in 0..5u64 {
        let replay = make_replay(i, 3);
        collector.record(replay).unwrap();
    }

    // Verify pending count
    assert_eq!(collector.pending_count(), 5);
    let snap = collector.snapshot();
    assert_eq!(snap.total_replays_recorded, 5);
    assert!(snap.buffer_bytes > 0);

    // Flush
    let sent = collector.flush(&transport).unwrap();
    assert_eq!(sent, 5);
    assert_eq!(transport.sent_count(), 5);
    assert_eq!(collector.pending_count(), 0);

    // Verify counters
    let snap = collector.snapshot();
    assert_eq!(snap.total_replays_flushed, 5);
    assert_eq!(snap.total_flush_failures, 0);
}

// ---------------------------------------------------------------------------
// 6. Replay batch transport
// ---------------------------------------------------------------------------

/// Create a ReplayBatch, serialize/deserialize via JSON, and verify
/// the contents are preserved.
#[test]
fn test_replay_batch_transport() {
    let mut batch = ReplayBatch::new("worker-001".to_string(), Some(3));

    // Add 4 replays with different seeds
    for i in 0..4u64 {
        batch.push(make_replay(i * 10, 3));
    }

    assert_eq!(batch.len(), 4);
    assert_eq!(batch.worker_id, "worker-001");
    assert_eq!(batch.model_version, Some(3));
    assert!(!batch.timestamp.is_empty());

    // JSON roundtrip
    let json = serde_json::to_string(&batch).expect("JSON serialize failed");
    let deserialized: ReplayBatch = serde_json::from_str(&json).expect("JSON deserialize failed");

    assert_eq!(deserialized.len(), 4);
    assert_eq!(deserialized.worker_id, "worker-001");
    assert_eq!(deserialized.model_version, Some(3));
    assert!(!deserialized.timestamp.is_empty());

    // Verify each replay preserved its seed
    for (orig, deser) in batch.replays.iter().zip(deserialized.replays.iter()) {
        assert_eq!(orig.seed, deser.seed);
        assert_eq!(orig.actions.len(), deser.actions.len());
    }
}

// ---------------------------------------------------------------------------
// 7. Worker registry lifecycle
// ---------------------------------------------------------------------------

/// Register workers, heartbeat, assign seeds, deregister, and verify
/// the full lifecycle works correctly.
#[test]
fn test_worker_registry_lifecycle() {
    let config = forge_cloud::WorkerConfig {
        max_workers: 5,
        ..Default::default()
    };
    let registry = InMemoryWorkerRegistry::new(config);

    // Register 3 workers
    for i in 1..=3 {
        let meta = WorkerMetadata {
            hostname: format!("host-{i}"),
            capabilities: vec!["cpu".to_string()],
            max_parallel_episodes: 4,
        };
        registry.register(&format!("w-{i:03}"), meta).unwrap();
    }
    assert_eq!(registry.worker_count(), 3);

    // Heartbeat all workers
    for i in 1..=3 {
        registry.heartbeat(&format!("w-{i:03}")).unwrap();
    }

    // Verify active workers
    let active = registry.active_workers().unwrap();
    assert_eq!(active.len(), 3);

    // Assign seeds to worker 1
    let assignment = registry.assign_seeds("w-001", 5).unwrap();
    assert_eq!(assignment.seeds.len(), 5);
    assert_eq!(
        assignment.seeds[0],
        forge_cloud::constants::DEFAULT_SEED_RANGE_START
    );

    // Second assignment picks up where the first left off
    let assignment2 = registry.assign_seeds("w-001", 3).unwrap();
    assert_eq!(assignment2.seeds[0], assignment.seeds[4] + 1);

    // Deregister worker 2
    registry.deregister("w-002").unwrap();
    assert_eq!(registry.worker_count(), 2);

    // Deregistering unknown worker fails
    assert!(registry.deregister("w-999").is_err());

    // Heartbeat on deregistered worker fails
    assert!(registry.heartbeat("w-002").is_err());

    // Register a replacement
    let meta = WorkerMetadata {
        hostname: "host-new".to_string(),
        capabilities: vec!["gpu".to_string()],
        max_parallel_episodes: 8,
    };
    registry.register("w-004", meta).unwrap();
    assert_eq!(registry.worker_count(), 3);
}

// ---------------------------------------------------------------------------
// 8. Cloud/edge config backward compatibility
// ---------------------------------------------------------------------------

/// Deserialize the existing forge.toml (which has no cloud/edge sections)
/// into ForgeConfig, and verify cloud.enabled=false and edge.enabled=false.
#[test]
fn test_cloud_edge_config_backward_compat() {
    // Read the existing forge.toml
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let forge_toml_path = std::path::Path::new(manifest_dir).join("forge.toml");
    let toml_content = std::fs::read_to_string(&forge_toml_path)
        .expect("forge.toml should exist at workspace root");

    let config: ForgeConfig =
        toml::from_str(&toml_content).expect("Failed to deserialize forge.toml into ForgeConfig");

    // Cloud and edge should default to disabled
    assert!(
        !config.cloud.enabled,
        "cloud.enabled should be false by default"
    );
    assert!(
        !config.edge.enabled,
        "edge.enabled should be false by default"
    );

    // Verify other cloud defaults are populated
    assert_eq!(
        config.cloud.num_workers,
        constants::DEFAULT_CLOUD_NUM_WORKERS
    );
    assert_eq!(
        config.cloud.coordinator_port,
        constants::DEFAULT_CLOUD_COORDINATOR_PORT
    );

    // Verify edge defaults are populated
    assert_eq!(
        config.edge.mcts_latency_budget_ms,
        constants::DEFAULT_EDGE_MCTS_LATENCY_BUDGET_MS
    );
    assert_eq!(
        config.edge.mcts_min_simulations,
        constants::DEFAULT_EDGE_MCTS_MIN_SIMULATIONS
    );

    // Test that a TOML with explicit cloud/edge sections also parses correctly
    let extended_toml = format!(
        "{}\n\n[cloud]\nenabled = true\nnum_workers = 8\n\n[edge]\nenabled = true\nmcts_latency_budget_ms = 100\n",
        toml_content
    );
    let extended_config: ForgeConfig =
        toml::from_str(&extended_toml).expect("Failed to deserialize extended config");
    assert!(extended_config.cloud.enabled);
    assert_eq!(extended_config.cloud.num_workers, 8);
    assert!(extended_config.edge.enabled);
    assert_eq!(extended_config.edge.mcts_latency_budget_ms, 100);
}

// ---------------------------------------------------------------------------
// 9. Model store version management
// ---------------------------------------------------------------------------

/// Store multiple model versions, verify latest_version and list_versions
/// work correctly.
#[test]
fn test_model_store_version_management() {
    let dir = tempdir().unwrap();
    let store = LocalModelStore::new(dir.path());

    // No versions initially
    assert_eq!(ModelStore::latest_version(&store, "policy").unwrap(), None);
    assert!(ModelStore::list_versions(&store, "policy")
        .unwrap()
        .is_empty());

    // Store versions 1, 2, 3
    ModelStore::store_model(&store, "policy", 1, b"weights_v1").unwrap();
    ModelStore::store_model(&store, "policy", 2, b"weights_v2").unwrap();
    ModelStore::store_model(&store, "policy", 3, b"weights_v3").unwrap();

    // Latest should be 3
    assert_eq!(
        ModelStore::latest_version(&store, "policy").unwrap(),
        Some(3)
    );

    // All versions listed in sorted order
    let versions = ModelStore::list_versions(&store, "policy").unwrap();
    assert_eq!(versions, vec![1, 2, 3]);

    // Load specific version and verify data
    let data = ModelStore::load_model(&store, "policy", 2).unwrap();
    assert_eq!(data, b"weights_v2");

    // Different model names are independent
    ModelStore::store_model(&store, "value_net", 1, b"value_v1").unwrap();
    assert_eq!(
        ModelStore::latest_version(&store, "value_net").unwrap(),
        Some(1)
    );
    assert_eq!(
        ModelStore::latest_version(&store, "policy").unwrap(),
        Some(3)
    );

    // Overwrite a version
    ModelStore::store_model(&store, "policy", 2, b"weights_v2_updated").unwrap();
    let updated = ModelStore::load_model(&store, "policy", 2).unwrap();
    assert_eq!(updated, b"weights_v2_updated");

    // Loading nonexistent version fails
    assert!(ModelStore::load_model(&store, "policy", 99).is_err());
}
