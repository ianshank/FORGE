//! Configuration types for the FORGE cloud training pipeline.
//!
//! All cloud/edge parameters are configurable through these structs.
//! No hard-coded values -- defaults are provided via `Default` trait
//! and can be overridden at construction time.

use serde::{Deserialize, Serialize};

use crate::constants;

/// Top-level configuration for the FORGE cloud training pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CloudConfig {
    /// Worker pool configuration.
    pub worker: WorkerConfig,
    /// Training coordinator configuration.
    pub coordinator: CoordinatorConfig,
    /// Storage backend configuration.
    pub storage: StorageConfig,
    /// Edge deployment configuration.
    pub edge: EdgeConfig,
    /// Model registry configuration.
    pub model_registry: ModelRegistryConfig,
    /// Replay transport configuration.
    pub replay_transport: ReplayTransportConfig,
}

/// Configuration for the distributed worker pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerConfig {
    /// Interval between heartbeat messages in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Timeout for worker liveness detection in milliseconds.
    pub timeout_ms: u64,
    /// Maximum number of concurrent workers.
    pub max_workers: u32,
    /// Minimum number of workers to maintain.
    pub min_workers: u32,
    /// Start of the seed range assigned to workers.
    pub seed_range_start: u64,
    /// End of the seed range assigned to workers (exclusive).
    pub seed_range_end: u64,
    /// Number of episodes each worker produces per batch.
    pub episodes_per_batch: u32,
    /// Whether workers run episodes in parallel.
    pub parallel: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_ms: constants::DEFAULT_WORKER_HEARTBEAT_INTERVAL_MS,
            timeout_ms: constants::DEFAULT_WORKER_TIMEOUT_MS,
            max_workers: constants::DEFAULT_MAX_WORKERS,
            min_workers: constants::DEFAULT_MIN_WORKERS,
            seed_range_start: constants::DEFAULT_SEED_RANGE_START,
            seed_range_end: constants::DEFAULT_SEED_RANGE_END,
            episodes_per_batch: constants::DEFAULT_EPISODES_PER_BATCH,
            parallel: false,
        }
    }
}

/// Configuration for the training coordinator service.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CoordinatorConfig {
    /// Port the coordinator listens on.
    pub port: u16,
    /// Number of training steps between checkpoints.
    pub checkpoint_interval_steps: u64,
    /// Number of model versions to retain before pruning.
    pub model_version_retention: u32,
    /// Strategy for aggregating results from multiple workers.
    pub aggregation_strategy: AggregationStrategy,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            port: constants::DEFAULT_COORDINATOR_PORT,
            checkpoint_interval_steps: constants::DEFAULT_CHECKPOINT_INTERVAL_STEPS,
            model_version_retention: constants::DEFAULT_MODEL_VERSION_RETENTION,
            aggregation_strategy: AggregationStrategy::default(),
        }
    }
}

/// Strategy for aggregating training results from workers.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AggregationStrategy {
    /// Process worker results sequentially.
    #[default]
    Sequential,
    /// Process worker results in parallel.
    Parallel,
    /// Use federated averaging with a fixed number of rounds.
    Federated {
        /// Number of federated averaging rounds.
        num_rounds: u32,
    },
}

/// Configuration for the storage backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Which storage backend to use.
    pub backend: StorageBackend,
    /// Path for replay archive storage.
    pub replay_archive_path: String,
    /// Path for model registry storage.
    pub model_registry_path: String,
    /// Path for checkpoint storage.
    pub checkpoint_path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::default(),
            replay_archive_path: constants::DEFAULT_REPLAY_ARCHIVE_PATH.to_string(),
            model_registry_path: constants::DEFAULT_MODEL_REGISTRY_PATH.to_string(),
            checkpoint_path: constants::DEFAULT_CHECKPOINT_PATH.to_string(),
        }
    }
}

/// Available storage backends.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StorageBackend {
    /// Local filesystem storage.
    #[default]
    Local,
    /// Google Cloud Storage.
    Gcs {
        /// GCS bucket name.
        bucket: String,
        /// Key prefix within the bucket.
        prefix: String,
    },
}

/// Configuration for edge deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EdgeConfig {
    /// Maximum latency budget for edge inference in milliseconds.
    pub latency_budget_ms: u64,
    /// Number of MCTS simulations on edge devices.
    pub mcts_simulations: u32,
    /// Model cache size on edge devices in megabytes.
    pub model_cache_size_mb: u32,
    /// Number of telemetry events to batch before uploading.
    pub telemetry_batch_size: u32,
    /// Number of retries for replay upload.
    pub upload_retry_count: u32,
    /// Base delay between retries in milliseconds (exponential backoff).
    pub upload_retry_base_ms: u64,
    /// Policy to use when the primary model is unavailable.
    pub fallback_policy: FallbackPolicy,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            latency_budget_ms: constants::DEFAULT_EDGE_LATENCY_BUDGET_MS,
            mcts_simulations: constants::DEFAULT_EDGE_MCTS_SIMULATIONS,
            model_cache_size_mb: constants::DEFAULT_EDGE_MODEL_CACHE_SIZE_MB,
            telemetry_batch_size: constants::DEFAULT_EDGE_TELEMETRY_BATCH_SIZE,
            upload_retry_count: constants::DEFAULT_EDGE_UPLOAD_RETRY_COUNT,
            upload_retry_base_ms: constants::DEFAULT_EDGE_UPLOAD_RETRY_BASE_MS,
            fallback_policy: FallbackPolicy::default(),
        }
    }
}

/// Fallback policy when the primary model is unavailable on edge.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FallbackPolicy {
    /// Do nothing (no-op actions).
    #[default]
    Noop,
    /// Select random actions with the given seed.
    Random {
        /// RNG seed for reproducible random actions.
        seed: u64,
    },
    /// Use the last successfully loaded model.
    LastKnownGood,
}

/// Configuration for replay transport between workers and coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplayTransportConfig {
    /// Number of replays per transport batch.
    pub batch_size: u32,
    /// Whether to compress replays during transport.
    pub compression_enabled: bool,
    /// Maximum payload size in bytes.
    pub max_payload_bytes: usize,
}

impl Default for ReplayTransportConfig {
    fn default() -> Self {
        Self {
            batch_size: constants::DEFAULT_REPLAY_BATCH_SIZE,
            compression_enabled: constants::DEFAULT_REPLAY_COMPRESSION_ENABLED,
            max_payload_bytes: constants::DEFAULT_REPLAY_MAX_PAYLOAD_BYTES,
        }
    }
}

/// Configuration for the model registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelRegistryConfig {
    /// Path for storing model artifacts.
    pub path: String,
    /// Maximum number of versions to retain per model.
    pub max_versions: u32,
}

impl Default for ModelRegistryConfig {
    fn default() -> Self {
        Self {
            path: constants::DEFAULT_MODEL_REGISTRY_PATH.to_string(),
            max_versions: constants::DEFAULT_MODEL_VERSION_RETENTION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_config_default() {
        let config = CloudConfig::default();
        assert_eq!(
            config.worker.heartbeat_interval_ms,
            constants::DEFAULT_WORKER_HEARTBEAT_INTERVAL_MS
        );
        assert_eq!(
            config.worker.timeout_ms,
            constants::DEFAULT_WORKER_TIMEOUT_MS
        );
        assert_eq!(config.worker.max_workers, constants::DEFAULT_MAX_WORKERS);
        assert_eq!(config.worker.min_workers, constants::DEFAULT_MIN_WORKERS);
        assert_eq!(config.coordinator.port, constants::DEFAULT_COORDINATOR_PORT);
        assert_eq!(
            config.coordinator.checkpoint_interval_steps,
            constants::DEFAULT_CHECKPOINT_INTERVAL_STEPS
        );
        assert_eq!(
            config.edge.latency_budget_ms,
            constants::DEFAULT_EDGE_LATENCY_BUDGET_MS
        );
        assert_eq!(
            config.edge.mcts_simulations,
            constants::DEFAULT_EDGE_MCTS_SIMULATIONS
        );
        assert_eq!(
            config.replay_transport.batch_size,
            constants::DEFAULT_REPLAY_BATCH_SIZE
        );
        assert_eq!(
            config.replay_transport.compression_enabled,
            constants::DEFAULT_REPLAY_COMPRESSION_ENABLED
        );
        assert_eq!(
            config.replay_transport.max_payload_bytes,
            constants::DEFAULT_REPLAY_MAX_PAYLOAD_BYTES
        );
    }

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(
            config.heartbeat_interval_ms,
            constants::DEFAULT_WORKER_HEARTBEAT_INTERVAL_MS
        );
        assert_eq!(config.timeout_ms, constants::DEFAULT_WORKER_TIMEOUT_MS);
        assert_eq!(config.max_workers, constants::DEFAULT_MAX_WORKERS);
        assert_eq!(config.min_workers, constants::DEFAULT_MIN_WORKERS);
        assert_eq!(config.seed_range_start, constants::DEFAULT_SEED_RANGE_START);
        assert_eq!(config.seed_range_end, constants::DEFAULT_SEED_RANGE_END);
        assert_eq!(
            config.episodes_per_batch,
            constants::DEFAULT_EPISODES_PER_BATCH
        );
        assert!(!config.parallel);
    }

    #[test]
    fn test_coordinator_config_default() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.port, constants::DEFAULT_COORDINATOR_PORT);
        assert_eq!(
            config.checkpoint_interval_steps,
            constants::DEFAULT_CHECKPOINT_INTERVAL_STEPS
        );
        assert_eq!(
            config.model_version_retention,
            constants::DEFAULT_MODEL_VERSION_RETENTION
        );
        assert_eq!(config.aggregation_strategy, AggregationStrategy::Sequential);
    }

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.backend, StorageBackend::Local);
        assert_eq!(
            config.replay_archive_path,
            constants::DEFAULT_REPLAY_ARCHIVE_PATH
        );
        assert_eq!(
            config.model_registry_path,
            constants::DEFAULT_MODEL_REGISTRY_PATH
        );
        assert_eq!(config.checkpoint_path, constants::DEFAULT_CHECKPOINT_PATH);
    }

    #[test]
    fn test_edge_config_default() {
        let config = EdgeConfig::default();
        assert_eq!(
            config.latency_budget_ms,
            constants::DEFAULT_EDGE_LATENCY_BUDGET_MS
        );
        assert_eq!(
            config.mcts_simulations,
            constants::DEFAULT_EDGE_MCTS_SIMULATIONS
        );
        assert_eq!(
            config.model_cache_size_mb,
            constants::DEFAULT_EDGE_MODEL_CACHE_SIZE_MB
        );
        assert_eq!(
            config.telemetry_batch_size,
            constants::DEFAULT_EDGE_TELEMETRY_BATCH_SIZE
        );
        assert_eq!(
            config.upload_retry_count,
            constants::DEFAULT_EDGE_UPLOAD_RETRY_COUNT
        );
        assert_eq!(
            config.upload_retry_base_ms,
            constants::DEFAULT_EDGE_UPLOAD_RETRY_BASE_MS
        );
        assert_eq!(config.fallback_policy, FallbackPolicy::Noop);
    }

    #[test]
    fn test_replay_transport_config_default() {
        let config = ReplayTransportConfig::default();
        assert_eq!(config.batch_size, constants::DEFAULT_REPLAY_BATCH_SIZE);
        assert_eq!(
            config.compression_enabled,
            constants::DEFAULT_REPLAY_COMPRESSION_ENABLED
        );
        assert_eq!(
            config.max_payload_bytes,
            constants::DEFAULT_REPLAY_MAX_PAYLOAD_BYTES
        );
    }

    #[test]
    fn test_model_registry_config_default() {
        let config = ModelRegistryConfig::default();
        assert_eq!(config.path, constants::DEFAULT_MODEL_REGISTRY_PATH);
        assert_eq!(
            config.max_versions,
            constants::DEFAULT_MODEL_VERSION_RETENTION
        );
    }

    #[test]
    fn test_cloud_config_serde_json_roundtrip() {
        let config = CloudConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: CloudConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.worker.max_workers, config.worker.max_workers);
        assert_eq!(deserialized.coordinator.port, config.coordinator.port);
        assert_eq!(
            deserialized.edge.latency_budget_ms,
            config.edge.latency_budget_ms
        );
    }

    #[test]
    fn test_worker_config_serde_json_roundtrip() {
        let config = WorkerConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: WorkerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            deserialized.heartbeat_interval_ms,
            config.heartbeat_interval_ms
        );
        assert_eq!(deserialized.timeout_ms, config.timeout_ms);
        assert_eq!(deserialized.max_workers, config.max_workers);
    }

    #[test]
    fn test_aggregation_strategy_serde_roundtrip() {
        let strategies = vec![
            AggregationStrategy::Sequential,
            AggregationStrategy::Parallel,
            AggregationStrategy::Federated { num_rounds: 5 },
        ];
        for strategy in strategies {
            let json = serde_json::to_string(&strategy).expect("serialize");
            let deserialized: AggregationStrategy =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, strategy);
        }
    }

    #[test]
    fn test_storage_backend_serde_roundtrip() {
        let backends = vec![
            StorageBackend::Local,
            StorageBackend::Gcs {
                bucket: "my-bucket".to_string(),
                prefix: "forge/".to_string(),
            },
        ];
        for backend in backends {
            let json = serde_json::to_string(&backend).expect("serialize");
            let deserialized: StorageBackend = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, backend);
        }
    }

    #[test]
    fn test_fallback_policy_serde_roundtrip() {
        let policies = vec![
            FallbackPolicy::Noop,
            FallbackPolicy::Random { seed: 42 },
            FallbackPolicy::LastKnownGood,
        ];
        for policy in policies {
            let json = serde_json::to_string(&policy).expect("serialize");
            let deserialized: FallbackPolicy = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, policy);
        }
    }

    #[test]
    fn test_replay_transport_config_bincode_roundtrip() {
        // Note: CloudConfig uses internally-tagged enums (#[serde(tag = "type")])
        // which are incompatible with bincode v1. Leaf configs without tagged
        // enums still support bincode roundtrips.
        let config = ReplayTransportConfig::default();
        let bytes = bincode::serialize(&config).expect("serialize");
        let deserialized: ReplayTransportConfig =
            bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(deserialized.batch_size, config.batch_size);
        assert_eq!(deserialized.compression_enabled, config.compression_enabled);
        assert_eq!(deserialized.max_payload_bytes, config.max_payload_bytes);
    }

    #[test]
    fn test_cloud_config_deserializes_from_empty_json() {
        let config: CloudConfig = serde_json::from_str("{}").expect("deserialize");
        assert_eq!(config.worker.max_workers, constants::DEFAULT_MAX_WORKERS);
        assert_eq!(config.coordinator.port, constants::DEFAULT_COORDINATOR_PORT);
    }

    #[test]
    fn test_cloud_config_deserializes_partial_json() {
        let json = r#"{"worker": {"max_workers": 42}}"#;
        let config: CloudConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(config.worker.max_workers, 42);
        // All other fields should be defaults
        assert_eq!(
            config.worker.heartbeat_interval_ms,
            constants::DEFAULT_WORKER_HEARTBEAT_INTERVAL_MS
        );
        assert_eq!(config.coordinator.port, constants::DEFAULT_COORDINATOR_PORT);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_worker_config_roundtrip(
                heartbeat in 100u64..60000,
                timeout in 1000u64..120000,
                max_w in 1u32..10000,
                min_w in 1u32..100,
            ) {
                let config = WorkerConfig {
                    heartbeat_interval_ms: heartbeat,
                    timeout_ms: timeout,
                    max_workers: max_w,
                    min_workers: min_w,
                    ..WorkerConfig::default()
                };
                let json = serde_json::to_string(&config).unwrap();
                let rt: WorkerConfig = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(rt.heartbeat_interval_ms, heartbeat);
                prop_assert_eq!(rt.timeout_ms, timeout);
                prop_assert_eq!(rt.max_workers, max_w);
                prop_assert_eq!(rt.min_workers, min_w);
            }

            #[test]
            fn test_edge_config_roundtrip(
                latency in 1u64..1000,
                sims in 1u32..500,
                cache in 1u32..4096,
            ) {
                let config = EdgeConfig {
                    latency_budget_ms: latency,
                    mcts_simulations: sims,
                    model_cache_size_mb: cache,
                    ..EdgeConfig::default()
                };
                let json = serde_json::to_string(&config).unwrap();
                let rt: EdgeConfig = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(rt.latency_budget_ms, latency);
                prop_assert_eq!(rt.mcts_simulations, sims);
                prop_assert_eq!(rt.model_cache_size_mb, cache);
            }

            #[test]
            fn test_replay_transport_config_roundtrip(
                batch in 1u32..1000,
                compress in proptest::bool::ANY,
                max_bytes in 1024usize..100_000_000,
            ) {
                let config = ReplayTransportConfig {
                    batch_size: batch,
                    compression_enabled: compress,
                    max_payload_bytes: max_bytes,
                };
                let json = serde_json::to_string(&config).unwrap();
                let rt: ReplayTransportConfig = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(rt.batch_size, batch);
                prop_assert_eq!(rt.compression_enabled, compress);
                prop_assert_eq!(rt.max_payload_bytes, max_bytes);
            }
        }
    }
}
