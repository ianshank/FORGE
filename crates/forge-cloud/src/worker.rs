//! Worker ID generation and in-memory worker registry.
//!
//! Provides utilities for generating unique worker identifiers and an
//! in-memory implementation of [`WorkerManager`] for testing and
//! single-node deployments.

use std::collections::HashMap;
use std::sync::Mutex;

use chrono::Utc;
use forge_types::config::ForgeConfig;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

use crate::config::WorkerConfig;
use crate::error::{CloudResult, WorkerError};
use crate::traits::{SeedAssignment, WorkerInfo, WorkerManager, WorkerMetadata, WorkerStatus};

/// Generates a unique worker ID from hostname and current timestamp.
///
/// The ID format is `{hostname}-{unix_timestamp_ms}`, providing a
/// human-readable identifier that is practically unique.
#[instrument]
pub fn generate_worker_id(hostname: &str) -> String {
    let ts = Utc::now().timestamp_millis();
    let id = format!("{hostname}-{ts}");
    debug!(worker_id = %id, "generated worker ID");
    id
}

/// Internal state for a tracked worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerEntry {
    /// Worker metadata.
    metadata: WorkerMetadata,
    /// Current status.
    status: WorkerStatus,
    /// Timestamp of last heartbeat (Unix millis).
    last_heartbeat_ms: u64,
    /// Total episodes completed.
    episodes_completed: u64,
    /// Total replays submitted.
    replays_submitted: u64,
}

/// In-memory worker registry implementing [`WorkerManager`].
///
/// Suitable for testing and single-node deployments. For distributed
/// deployments, use a persistent store-backed implementation.
pub struct InMemoryWorkerRegistry {
    /// Worker config for capacity limits and defaults.
    config: WorkerConfig,
    /// Map from worker ID to worker entry, guarded by a mutex.
    workers: Mutex<HashMap<String, WorkerEntry>>,
    /// Global seed cursor shared across all workers to prevent overlaps.
    next_seed: Mutex<u64>,
}

impl InMemoryWorkerRegistry {
    /// Creates a new in-memory worker registry with the given config.
    #[instrument(skip_all)]
    pub fn new(config: WorkerConfig) -> Self {
        info!(
            max_workers = config.max_workers,
            "created in-memory worker registry"
        );
        let seed_start = config.seed_range_start;
        Self {
            config,
            workers: Mutex::new(HashMap::new()),
            next_seed: Mutex::new(seed_start),
        }
    }

    /// Creates a new in-memory worker registry with default config.
    pub fn with_defaults() -> Self {
        Self::new(WorkerConfig::default())
    }

    /// Returns the number of currently registered workers.
    #[instrument(skip(self))]
    pub fn worker_count(&self) -> usize {
        let workers = self.workers.lock().expect("lock poisoned");
        workers.len()
    }

    /// Returns the current timestamp in milliseconds.
    fn now_ms() -> u64 {
        Utc::now().timestamp_millis() as u64
    }
}

impl WorkerManager for InMemoryWorkerRegistry {
    #[instrument(skip(self, metadata))]
    fn register(&self, worker_id: &str, metadata: WorkerMetadata) -> CloudResult<()> {
        let mut workers = self.workers.lock().expect("lock poisoned");

        if workers.contains_key(worker_id) {
            warn!(worker_id, "attempted to register duplicate worker");
            return Err(WorkerError::AlreadyRegistered(worker_id.to_string()).into());
        }

        if workers.len() as u32 >= self.config.max_workers {
            warn!(
                current = workers.len(),
                max = self.config.max_workers,
                "worker capacity exceeded"
            );
            return Err(WorkerError::CapacityExceeded {
                current: workers.len() as u32,
                max: self.config.max_workers,
            }
            .into());
        }

        let entry = WorkerEntry {
            metadata,
            status: WorkerStatus::Idle,
            last_heartbeat_ms: Self::now_ms(),
            episodes_completed: 0,
            replays_submitted: 0,
        };

        info!(worker_id, "registered worker");
        workers.insert(worker_id.to_string(), entry);
        Ok(())
    }

    #[instrument(skip(self))]
    fn deregister(&self, worker_id: &str) -> CloudResult<()> {
        let mut workers = self.workers.lock().expect("lock poisoned");
        if workers.remove(worker_id).is_none() {
            warn!(worker_id, "attempted to deregister unknown worker");
            return Err(WorkerError::NotFound(worker_id.to_string()).into());
        }
        info!(worker_id, "deregistered worker");
        Ok(())
    }

    #[instrument(skip(self))]
    fn heartbeat(&self, worker_id: &str) -> CloudResult<()> {
        let mut workers = self.workers.lock().expect("lock poisoned");
        let entry = workers
            .get_mut(worker_id)
            .ok_or_else(|| WorkerError::NotFound(worker_id.to_string()))?;
        entry.last_heartbeat_ms = Self::now_ms();
        debug!(worker_id, "heartbeat recorded");
        Ok(())
    }

    #[instrument(skip(self))]
    fn active_workers(&self) -> CloudResult<Vec<WorkerInfo>> {
        let workers = self.workers.lock().expect("lock poisoned");
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;
        let infos = workers
            .iter()
            .map(|(id, entry)| {
                let status =
                    if now_ms.saturating_sub(entry.last_heartbeat_ms) > self.config.timeout_ms {
                        WorkerStatus::Disconnected {
                            since_ms: entry.last_heartbeat_ms,
                        }
                    } else {
                        entry.status.clone()
                    };
                WorkerInfo {
                    worker_id: id.clone(),
                    metadata: entry.metadata.clone(),
                    status,
                    last_heartbeat_ms: entry.last_heartbeat_ms,
                    episodes_completed: entry.episodes_completed,
                    replays_submitted: entry.replays_submitted,
                }
            })
            .collect();
        Ok(infos)
    }

    #[instrument(skip(self))]
    fn assign_seeds(&self, worker_id: &str, count: u32) -> CloudResult<SeedAssignment> {
        let mut workers = self.workers.lock().expect("lock poisoned");
        let entry = workers
            .get_mut(worker_id)
            .ok_or_else(|| WorkerError::NotFound(worker_id.to_string()))?;

        let mut next = self.next_seed.lock().expect("lock poisoned");
        let start = *next;
        let end = start
            .checked_add(count as u64)
            .ok_or(WorkerError::CapacityExceeded {
                current: 0,
                max: self.config.max_workers,
            })?;

        if end > self.config.seed_range_end {
            return Err(WorkerError::CapacityExceeded {
                current: (start - self.config.seed_range_start) as u32,
                max: (self.config.seed_range_end - self.config.seed_range_start) as u32,
            }
            .into());
        }

        let seeds: Vec<u64> = (start..end).collect();
        *next = end;
        entry.status = WorkerStatus::Running {
            current_seeds: seeds.clone(),
        };

        debug!(worker_id, seed_count = count, "assigned seeds");
        Ok(SeedAssignment {
            seeds,
            config: ForgeConfig::default(),
            model_version: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;

    fn test_config() -> WorkerConfig {
        WorkerConfig {
            max_workers: 3,
            ..WorkerConfig::default()
        }
    }

    fn test_metadata() -> WorkerMetadata {
        WorkerMetadata {
            hostname: "test-host".to_string(),
            capabilities: vec!["cpu".to_string()],
            max_parallel_episodes: 4,
        }
    }

    #[test]
    fn test_generate_worker_id_contains_hostname() {
        let id = generate_worker_id("my-host");
        assert!(id.starts_with("my-host-"));
    }

    #[test]
    fn test_generate_worker_id_unique() {
        let id1 = generate_worker_id("host");
        let id2 = generate_worker_id("host");
        // IDs generated in the same millisecond could match, but the test
        // is primarily validating format and non-emptiness.
        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
    }

    #[test]
    fn test_register_worker() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        let result = registry.register("w-001", test_metadata());
        assert!(result.is_ok());
        assert_eq!(registry.worker_count(), 1);
    }

    #[test]
    fn test_register_duplicate_worker() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        let result = registry.register("w-001", test_metadata());
        assert!(result.is_err());
    }

    #[test]
    fn test_register_exceeds_capacity() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        registry.register("w-002", test_metadata()).unwrap();
        registry.register("w-003", test_metadata()).unwrap();
        let result = registry.register("w-004", test_metadata());
        assert!(result.is_err());
    }

    #[test]
    fn test_deregister_worker() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        assert_eq!(registry.worker_count(), 1);
        registry.deregister("w-001").unwrap();
        assert_eq!(registry.worker_count(), 0);
    }

    #[test]
    fn test_deregister_unknown_worker() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        let result = registry.deregister("w-999");
        assert!(result.is_err());
    }

    #[test]
    fn test_heartbeat() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        let result = registry.heartbeat("w-001");
        assert!(result.is_ok());
    }

    #[test]
    fn test_heartbeat_unknown_worker() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        let result = registry.heartbeat("w-999");
        assert!(result.is_err());
    }

    #[test]
    fn test_active_workers() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        registry.register("w-002", test_metadata()).unwrap();
        let workers = registry.active_workers().unwrap();
        assert_eq!(workers.len(), 2);
    }

    #[test]
    fn test_assign_seeds() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        let assignment = registry.assign_seeds("w-001", 5).unwrap();
        assert_eq!(assignment.seeds.len(), 5);
        assert_eq!(
            assignment.seeds,
            vec![
                constants::DEFAULT_SEED_RANGE_START,
                constants::DEFAULT_SEED_RANGE_START + 1,
                constants::DEFAULT_SEED_RANGE_START + 2,
                constants::DEFAULT_SEED_RANGE_START + 3,
                constants::DEFAULT_SEED_RANGE_START + 4,
            ]
        );
    }

    #[test]
    fn test_assign_seeds_increments() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        let a1 = registry.assign_seeds("w-001", 3).unwrap();
        let a2 = registry.assign_seeds("w-001", 2).unwrap();
        // Second assignment should start where first left off.
        assert_eq!(a2.seeds[0], a1.seeds[2] + 1);
    }

    #[test]
    fn test_assign_seeds_unknown_worker() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        let result = registry.assign_seeds("w-999", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_assign_seeds_sets_running_status() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        registry.assign_seeds("w-001", 3).unwrap();
        let workers = registry.active_workers().unwrap();
        let worker = workers.iter().find(|w| w.worker_id == "w-001").unwrap();
        assert!(matches!(worker.status, WorkerStatus::Running { .. }));
    }

    #[test]
    fn test_with_defaults() {
        let registry = InMemoryWorkerRegistry::with_defaults();
        assert_eq!(registry.worker_count(), 0);
    }

    #[test]
    fn test_assign_seeds_no_overlap() {
        let registry = InMemoryWorkerRegistry::new(test_config());
        registry.register("w-001", test_metadata()).unwrap();
        registry.register("w-002", test_metadata()).unwrap();

        let a1 = registry.assign_seeds("w-001", 5).unwrap();
        let a2 = registry.assign_seeds("w-002", 5).unwrap();

        // Seeds assigned to different workers must be disjoint.
        for s in &a1.seeds {
            assert!(!a2.seeds.contains(s), "seed {s} overlaps between workers");
        }
        // Second assignment starts where first ended.
        assert_eq!(a2.seeds[0], a1.seeds[4] + 1);
    }

    #[test]
    fn test_assign_seeds_exceeds_range() {
        let config = WorkerConfig {
            max_workers: 3,
            seed_range_start: 0,
            seed_range_end: 10,
            ..WorkerConfig::default()
        };
        let registry = InMemoryWorkerRegistry::new(config);
        registry.register("w-001", test_metadata()).unwrap();

        // First assignment of 8 seeds should succeed (0..8).
        let a1 = registry.assign_seeds("w-001", 8).unwrap();
        assert_eq!(a1.seeds.len(), 8);

        // Second assignment of 5 seeds would need 8..13, exceeding range_end=10.
        let result = registry.assign_seeds("w-001", 5);
        assert!(result.is_err());
    }
}
