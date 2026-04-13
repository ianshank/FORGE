#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-cloud
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

pub mod config;
pub mod constants;
pub mod error;
pub mod replay_transport;
pub mod traits;
pub mod worker;

// Re-export commonly used types at crate root
pub use config::{
    AggregationStrategy, CloudConfig, CoordinatorConfig, EdgeConfig, FallbackPolicy,
    ModelRegistryConfig, ReplayTransportConfig, StorageBackend, StorageConfig, WorkerConfig,
};
pub use error::{
    CloudError, CloudResult, ModelRegistryError, StorageError, TransportError, WorkerError,
};
pub use replay_transport::ReplayBatch;
pub use traits::{
    ModelStore, ReplayStore, SeedAssignment, WorkerInfo, WorkerManager, WorkerMetadata,
    WorkerStatus,
};
pub use worker::{generate_worker_id, InMemoryWorkerRegistry};
