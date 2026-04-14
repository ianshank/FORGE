//! Core traits for the FORGE cloud training pipeline.
//!
//! These traits define the interfaces for replay storage, model storage,
//! and worker lifecycle management. Concrete implementations live in
//! separate modules or downstream crates.

use forge_replay::compact::CompactReplay;
use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};

use crate::error::CloudResult;

/// Trait for replay storage backends.
///
/// Implementations handle persistence of compact replays to local disk,
/// cloud object storage, or other backends.
pub trait ReplayStore: Send + Sync {
    /// Stores a compact replay under the given key.
    fn store(&self, replay: &CompactReplay, key: &str) -> CloudResult<()>;

    /// Loads a compact replay by key.
    fn load(&self, key: &str) -> CloudResult<CompactReplay>;

    /// Lists all keys with the given prefix.
    fn list(&self, prefix: &str) -> CloudResult<Vec<String>>;

    /// Deletes a replay by key.
    fn delete(&self, key: &str) -> CloudResult<()>;

    /// Checks whether a replay with the given key exists.
    fn exists(&self, key: &str) -> CloudResult<bool>;
}

/// Trait for model artifact storage.
///
/// Implementations manage versioned storage of trained model weights
/// and associated metadata.
pub trait ModelStore: Send + Sync {
    /// Stores model data for the given model ID and version.
    fn store_model(&self, model_id: &str, version: u32, data: &[u8]) -> CloudResult<()>;

    /// Loads model data for the given model ID and version.
    fn load_model(&self, model_id: &str, version: u32) -> CloudResult<Vec<u8>>;

    /// Returns the latest version number for a model, or `None` if not found.
    fn latest_version(&self, model_id: &str) -> CloudResult<Option<u32>>;

    /// Lists all available versions for a model.
    fn list_versions(&self, model_id: &str) -> CloudResult<Vec<u32>>;
}

/// Trait for worker lifecycle management.
///
/// Implementations handle worker registration, heartbeats, and seed
/// assignment for distributed episode collection.
pub trait WorkerManager: Send + Sync {
    /// Registers a new worker with the given metadata.
    fn register(&self, worker_id: &str, metadata: WorkerMetadata) -> CloudResult<()>;

    /// Deregisters a worker, removing it from the pool.
    fn deregister(&self, worker_id: &str) -> CloudResult<()>;

    /// Records a heartbeat from a worker.
    fn heartbeat(&self, worker_id: &str) -> CloudResult<()>;

    /// Returns information about all active workers.
    fn active_workers(&self) -> CloudResult<Vec<WorkerInfo>>;

    /// Assigns a batch of seeds to a worker for episode generation.
    fn assign_seeds(&self, worker_id: &str, count: u32) -> CloudResult<SeedAssignment>;
}

/// Metadata about a worker node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetadata {
    /// Hostname or identifier of the worker machine.
    pub hostname: String,
    /// List of capabilities this worker supports (e.g., "gpu", "mcts").
    pub capabilities: Vec<String>,
    /// Maximum number of episodes this worker can run in parallel.
    pub max_parallel_episodes: u32,
}

/// Current state of a registered worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Unique identifier for this worker.
    pub worker_id: String,
    /// Metadata about the worker node.
    pub metadata: WorkerMetadata,
    /// Current operational status.
    pub status: WorkerStatus,
    /// Time since last heartbeat in milliseconds.
    pub last_heartbeat_ms: u64,
    /// Total number of episodes completed by this worker.
    pub episodes_completed: u64,
    /// Total number of replays submitted by this worker.
    pub replays_submitted: u64,
}

/// Operational status of a worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkerStatus {
    /// Worker is idle and available for work.
    Idle,
    /// Worker is actively running episodes with the given seeds.
    Running {
        /// Seeds currently being processed.
        current_seeds: Vec<u64>,
    },
    /// Worker is finishing current work and will not accept new assignments.
    Draining,
    /// Worker has been disconnected.
    Disconnected {
        /// Time since disconnection in milliseconds.
        since_ms: u64,
    },
}

/// Seed assignment for a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedAssignment {
    /// Seeds to use for episode generation.
    pub seeds: Vec<u64>,
    /// Simulation configuration to use for these episodes.
    pub config: ForgeConfig,
    /// Model version to use, if any.
    pub model_version: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_metadata_serde_roundtrip() {
        let metadata = WorkerMetadata {
            hostname: "worker-01.local".to_string(),
            capabilities: vec!["gpu".to_string(), "mcts".to_string()],
            max_parallel_episodes: 8,
        };
        let json = serde_json::to_string(&metadata).expect("serialize");
        let rt: WorkerMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt.hostname, "worker-01.local");
        assert_eq!(rt.capabilities.len(), 2);
        assert_eq!(rt.max_parallel_episodes, 8);
    }

    #[test]
    fn test_worker_info_serde_roundtrip() {
        let info = WorkerInfo {
            worker_id: "w-001".to_string(),
            metadata: WorkerMetadata {
                hostname: "host-a".to_string(),
                capabilities: vec![],
                max_parallel_episodes: 4,
            },
            status: WorkerStatus::Idle,
            last_heartbeat_ms: 1000,
            episodes_completed: 50,
            replays_submitted: 45,
        };
        let json = serde_json::to_string(&info).expect("serialize");
        let rt: WorkerInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt.worker_id, "w-001");
        assert_eq!(rt.episodes_completed, 50);
        assert_eq!(rt.replays_submitted, 45);
    }

    #[test]
    fn test_worker_status_idle_serde() {
        let status = WorkerStatus::Idle;
        let json = serde_json::to_string(&status).expect("serialize");
        let rt: WorkerStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt, WorkerStatus::Idle);
    }

    #[test]
    fn test_worker_status_running_serde() {
        let status = WorkerStatus::Running {
            current_seeds: vec![1, 2, 3],
        };
        let json = serde_json::to_string(&status).expect("serialize");
        let rt: WorkerStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            rt,
            WorkerStatus::Running {
                current_seeds: vec![1, 2, 3]
            }
        );
    }

    #[test]
    fn test_worker_status_draining_serde() {
        let status = WorkerStatus::Draining;
        let json = serde_json::to_string(&status).expect("serialize");
        let rt: WorkerStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt, WorkerStatus::Draining);
    }

    #[test]
    fn test_worker_status_disconnected_serde() {
        let status = WorkerStatus::Disconnected { since_ms: 5000 };
        let json = serde_json::to_string(&status).expect("serialize");
        let rt: WorkerStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt, WorkerStatus::Disconnected { since_ms: 5000 });
    }

    #[test]
    fn test_seed_assignment_serde_roundtrip() {
        let assignment = SeedAssignment {
            seeds: vec![100, 200, 300],
            config: ForgeConfig::default(),
            model_version: Some(5),
        };
        let json = serde_json::to_string(&assignment).expect("serialize");
        let rt: SeedAssignment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt.seeds, vec![100, 200, 300]);
        assert_eq!(rt.model_version, Some(5));
    }

    #[test]
    fn test_seed_assignment_no_model_version() {
        let assignment = SeedAssignment {
            seeds: vec![42],
            config: ForgeConfig::default(),
            model_version: None,
        };
        let json = serde_json::to_string(&assignment).expect("serialize");
        let rt: SeedAssignment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt.model_version, None);
    }

    #[test]
    fn test_worker_metadata_empty_capabilities() {
        let metadata = WorkerMetadata {
            hostname: "bare-metal".to_string(),
            capabilities: vec![],
            max_parallel_episodes: 1,
        };
        let json = serde_json::to_string(&metadata).expect("serialize");
        let rt: WorkerMetadata = serde_json::from_str(&json).expect("deserialize");
        assert!(rt.capabilities.is_empty());
    }
}
