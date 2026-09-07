#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-cloud
//!
//! > **Maturity**: `[Experimental]` — Experimental Component — distributed cloud training orchestration.
//!
//! Cloud training pipeline and edge deployment for the FORGE platform.
//!
//! This crate defines the foundational types for distributed training
//! and edge inference:
//! - Configuration structs for workers, coordinator, edge, and storage
//! - Error types for all cloud subsystems
//! - Default constants (all overridable via config)
//! - Core traits for replay storage, model storage, and worker management
//! - Worker ID generation and in-memory registry
//! - Replay compression and batching for transport

pub mod backend;
pub mod config;
pub mod constants;
pub mod error;
pub mod reconstruct;
pub mod replay_transport;
pub mod storage;
pub mod traits;
pub mod worker;

#[cfg(feature = "gcs")]
pub mod gcs_storage;

// Re-export commonly used types at crate root
pub use backend::{create_model_store, create_replay_store, create_replay_transport};
pub use config::{
    AggregationStrategy, CloudConfig, CoordinatorConfig, EdgeConfig, FallbackPolicy,
    ModelRegistryConfig, ReplayTransportConfig, StorageBackend, StorageConfig, WorkerConfig,
};
pub use error::{
    CloudError, CloudResult, ModelRegistryError, StorageError, TransportError, WorkerError,
};
pub use reconstruct::TrajectoryReconstructor;
pub use replay_transport::ReplayBatch;
pub use storage::{LocalModelStore, LocalReplayStore};
pub use traits::{
    ModelStore, ReplayStore, SeedAssignment, WorkerInfo, WorkerManager, WorkerMetadata,
    WorkerStatus,
};

#[cfg(feature = "gcs")]
pub use gcs_storage::{GcsModelStore, GcsReplayStore, GcsReplayTransport};
pub use worker::{generate_worker_id, InMemoryWorkerRegistry};
