//! Default constants for the FORGE cloud training pipeline and edge deployment.
//!
//! These provide default values for all cloud/edge configuration structs.
//! Every constant is overridable via the corresponding config field.

// ---------- Worker defaults ----------

/// Default interval between worker heartbeat messages in milliseconds.
pub const DEFAULT_WORKER_HEARTBEAT_INTERVAL_MS: u64 = 5000;
/// Default timeout for worker liveness detection in milliseconds.
pub const DEFAULT_WORKER_TIMEOUT_MS: u64 = 30000;
/// Default maximum number of concurrent workers.
pub const DEFAULT_MAX_WORKERS: u32 = 1000;
/// Default minimum number of workers to maintain.
pub const DEFAULT_MIN_WORKERS: u32 = 1;
/// Default number of episodes each worker produces per batch.
pub const DEFAULT_EPISODES_PER_BATCH: u32 = 8;
/// Default start of the seed range assigned to workers.
pub const DEFAULT_SEED_RANGE_START: u64 = 0;
/// Default end of the seed range assigned to workers.
pub const DEFAULT_SEED_RANGE_END: u64 = 1_000_000;

// ---------- Replay transport ----------

/// Default number of replays per transport batch.
pub const DEFAULT_REPLAY_BATCH_SIZE: u32 = 64;
/// Whether replay compression is enabled by default.
pub const DEFAULT_REPLAY_COMPRESSION_ENABLED: bool = true;
/// Default maximum replay payload size in bytes (10 MB).
pub const DEFAULT_REPLAY_MAX_PAYLOAD_BYTES: usize = 10 * 1024 * 1024;

// ---------- Edge defaults ----------

/// Default latency budget for edge inference in milliseconds.
pub const DEFAULT_EDGE_LATENCY_BUDGET_MS: u64 = 50;
/// Default number of MCTS simulations on edge devices.
pub const DEFAULT_EDGE_MCTS_SIMULATIONS: u32 = 50;
/// Default model cache size on edge devices in megabytes.
pub const DEFAULT_EDGE_MODEL_CACHE_SIZE_MB: u32 = 256;
/// Default telemetry batch size for edge upload.
pub const DEFAULT_EDGE_TELEMETRY_BATCH_SIZE: u32 = 16;
/// Default number of retries for edge replay upload.
pub const DEFAULT_EDGE_UPLOAD_RETRY_COUNT: u32 = 4;
/// Default base delay between edge upload retries in milliseconds.
pub const DEFAULT_EDGE_UPLOAD_RETRY_BASE_MS: u64 = 2000;

// ---------- Coordinator defaults ----------

/// Default port for the training coordinator service.
pub const DEFAULT_COORDINATOR_PORT: u16 = 9090;
/// Default number of training steps between checkpoints.
pub const DEFAULT_CHECKPOINT_INTERVAL_STEPS: u64 = 1000;
/// Default number of model versions to retain.
pub const DEFAULT_MODEL_VERSION_RETENTION: u32 = 10;

// ---------- Storage ----------

/// Default storage backend identifier.
pub const DEFAULT_STORAGE_BACKEND: &str = "local";
/// Default path for replay archive storage.
pub const DEFAULT_REPLAY_ARCHIVE_PATH: &str = "replays";
/// Default path for model registry storage.
pub const DEFAULT_MODEL_REGISTRY_PATH: &str = "models";
/// Default path for checkpoint storage.
pub const DEFAULT_CHECKPOINT_PATH: &str = "checkpoints";

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_timeout_greater_than_heartbeat() {
        assert!(DEFAULT_WORKER_TIMEOUT_MS > DEFAULT_WORKER_HEARTBEAT_INTERVAL_MS);
    }

    #[test]
    fn test_min_workers_less_than_max() {
        assert!(DEFAULT_MIN_WORKERS <= DEFAULT_MAX_WORKERS);
    }

    #[test]
    fn test_min_workers_nonzero() {
        assert!(DEFAULT_MIN_WORKERS >= 1);
    }

    #[test]
    fn test_replay_batch_size_nonzero() {
        assert!(DEFAULT_REPLAY_BATCH_SIZE > 0);
    }

    #[test]
    fn test_replay_max_payload_is_10mb() {
        assert_eq!(DEFAULT_REPLAY_MAX_PAYLOAD_BYTES, 10 * 1024 * 1024);
    }

    #[test]
    fn test_edge_latency_budget_positive() {
        assert!(DEFAULT_EDGE_LATENCY_BUDGET_MS > 0);
    }

    #[test]
    fn test_edge_mcts_simulations_positive() {
        assert!(DEFAULT_EDGE_MCTS_SIMULATIONS > 0);
    }

    #[test]
    fn test_edge_model_cache_positive() {
        assert!(DEFAULT_EDGE_MODEL_CACHE_SIZE_MB > 0);
    }

    #[test]
    fn test_edge_retry_count_positive() {
        assert!(DEFAULT_EDGE_UPLOAD_RETRY_COUNT > 0);
    }

    #[test]
    fn test_edge_retry_base_ms_positive() {
        assert!(DEFAULT_EDGE_UPLOAD_RETRY_BASE_MS > 0);
    }

    #[test]
    fn test_coordinator_port_non_privileged() {
        assert!(DEFAULT_COORDINATOR_PORT >= 1024);
    }

    #[test]
    fn test_checkpoint_interval_positive() {
        assert!(DEFAULT_CHECKPOINT_INTERVAL_STEPS > 0);
    }

    #[test]
    fn test_model_version_retention_positive() {
        assert!(DEFAULT_MODEL_VERSION_RETENTION > 0);
    }

    #[test]
    fn test_storage_backend_not_empty() {
        assert!(!DEFAULT_STORAGE_BACKEND.is_empty());
    }

    #[test]
    fn test_storage_paths_not_empty() {
        assert!(!DEFAULT_REPLAY_ARCHIVE_PATH.is_empty());
        assert!(!DEFAULT_MODEL_REGISTRY_PATH.is_empty());
        assert!(!DEFAULT_CHECKPOINT_PATH.is_empty());
    }

    #[test]
    fn test_seed_range_valid() {
        assert!(DEFAULT_SEED_RANGE_START < DEFAULT_SEED_RANGE_END);
    }

    #[test]
    fn test_episodes_per_batch_positive() {
        assert!(DEFAULT_EPISODES_PER_BATCH > 0);
    }

    #[test]
    fn test_edge_telemetry_batch_size_positive() {
        assert!(DEFAULT_EDGE_TELEMETRY_BATCH_SIZE > 0);
    }

    #[test]
    fn test_coordinator_port_distinct_from_server() {
        // Coordinator port should not collide with forge-server default (8080)
        assert_ne!(DEFAULT_COORDINATOR_PORT, 8080);
    }
}
