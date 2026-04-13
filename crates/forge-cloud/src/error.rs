//! Error types for the FORGE cloud training pipeline.
//!
//! All errors are structured using `thiserror` for clean error propagation
//! across crate boundaries.

use thiserror::Error;

/// Top-level error type for cloud operations.
#[derive(Debug, Error)]
pub enum CloudError {
    /// An error occurred with a worker.
    #[error("worker error: {0}")]
    Worker(#[from] WorkerError),

    /// An error occurred during replay transport.
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),

    /// An error occurred in storage.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// An error occurred in the model registry.
    #[error("model registry error: {0}")]
    ModelRegistry(#[from] ModelRegistryError),

    /// An error occurred with configuration.
    #[error("configuration error: {0}")]
    Config(#[from] forge_types::error::ConfigError),

    /// Serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// Errors that can occur with worker management.
#[derive(Debug, Error)]
pub enum WorkerError {
    /// A worker timed out.
    #[error("worker '{worker_id}' timed out after {timeout_ms}ms")]
    Timeout {
        /// ID of the timed-out worker.
        worker_id: String,
        /// Timeout threshold in milliseconds.
        timeout_ms: u64,
    },

    /// A worker missed its heartbeat.
    #[error("worker '{worker_id}' missed heartbeat, last seen {last_seen_ms}ms ago")]
    HeartbeatMissed {
        /// ID of the unresponsive worker.
        worker_id: String,
        /// Time since last heartbeat in milliseconds.
        last_seen_ms: u64,
    },

    /// Worker capacity has been exceeded.
    #[error("worker capacity exceeded: {current}/{max}")]
    CapacityExceeded {
        /// Current number of workers.
        current: u32,
        /// Maximum allowed workers.
        max: u32,
    },

    /// A worker was not found.
    #[error("worker not found: {0}")]
    NotFound(String),

    /// A worker with this ID is already registered.
    #[error("worker already registered: {0}")]
    AlreadyRegistered(String),
}

/// Errors that can occur during replay transport.
#[derive(Debug, Error)]
pub enum TransportError {
    /// The payload exceeds the maximum allowed size.
    #[error("payload too large: {size} bytes (max: {max} bytes)")]
    PayloadTooLarge {
        /// Actual payload size.
        size: usize,
        /// Maximum allowed size.
        max: usize,
    },

    /// Serialization of a replay failed.
    #[error("serialization failed: {0}")]
    SerializationFailed(String),

    /// Deserialization of a replay failed.
    #[error("deserialization failed: {0}")]
    DeserializationFailed(String),

    /// Connection to the transport endpoint failed.
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    /// A transport operation timed out.
    #[error("transport timeout: {operation} after {timeout_ms}ms")]
    Timeout {
        /// Description of the operation that timed out.
        operation: String,
        /// Timeout threshold in milliseconds.
        timeout_ms: u64,
    },
}

/// Errors that can occur in storage operations.
#[derive(Debug, Error)]
pub enum StorageError {
    /// The requested resource was not found.
    #[error("not found: {path}")]
    NotFound {
        /// Path of the missing resource.
        path: String,
    },

    /// A write operation failed.
    #[error("write failed for '{path}': {reason}")]
    WriteFailed {
        /// Path of the resource being written.
        path: String,
        /// Reason the write failed.
        reason: String,
    },

    /// A read operation failed.
    #[error("read failed for '{path}': {reason}")]
    ReadFailed {
        /// Path of the resource being read.
        path: String,
        /// Reason the read failed.
        reason: String,
    },

    /// The resource has an invalid format.
    #[error("invalid format for '{path}': {reason}")]
    InvalidFormat {
        /// Path of the resource with bad format.
        path: String,
        /// Description of the format issue.
        reason: String,
    },
}

/// Errors that can occur in the model registry.
#[derive(Debug, Error)]
pub enum ModelRegistryError {
    /// The requested model version was not found.
    #[error("model '{model_id}' version {version} not found")]
    VersionNotFound {
        /// ID of the model.
        model_id: String,
        /// Version that was requested.
        version: u32,
    },

    /// The model artifact is invalid.
    #[error("invalid model: {reason}")]
    InvalidModel {
        /// Description of why the model is invalid.
        reason: String,
    },

    /// Model registration failed.
    #[error("registration failed for model '{model_id}': {reason}")]
    RegistrationFailed {
        /// ID of the model.
        model_id: String,
        /// Reason registration failed.
        reason: String,
    },
}

/// Result type alias for cloud operations.
pub type CloudResult<T> = Result<T, CloudError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_timeout_display() {
        let err = WorkerError::Timeout {
            worker_id: "w-001".to_string(),
            timeout_ms: 30000,
        };
        let msg = err.to_string();
        assert!(msg.contains("w-001"));
        assert!(msg.contains("30000"));
    }

    #[test]
    fn test_worker_heartbeat_missed_display() {
        let err = WorkerError::HeartbeatMissed {
            worker_id: "w-002".to_string(),
            last_seen_ms: 15000,
        };
        let msg = err.to_string();
        assert!(msg.contains("w-002"));
        assert!(msg.contains("15000"));
    }

    #[test]
    fn test_worker_capacity_exceeded_display() {
        let err = WorkerError::CapacityExceeded {
            current: 100,
            max: 50,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_worker_not_found_display() {
        let err = WorkerError::NotFound("w-missing".to_string());
        assert!(err.to_string().contains("w-missing"));
    }

    #[test]
    fn test_worker_already_registered_display() {
        let err = WorkerError::AlreadyRegistered("w-dup".to_string());
        assert!(err.to_string().contains("w-dup"));
    }

    #[test]
    fn test_transport_payload_too_large_display() {
        let err = TransportError::PayloadTooLarge {
            size: 20_000_000,
            max: 10_000_000,
        };
        let msg = err.to_string();
        assert!(msg.contains("20000000"));
        assert!(msg.contains("10000000"));
    }

    #[test]
    fn test_transport_serialization_failed_display() {
        let err = TransportError::SerializationFailed("bad data".to_string());
        assert!(err.to_string().contains("bad data"));
    }

    #[test]
    fn test_transport_deserialization_failed_display() {
        let err = TransportError::DeserializationFailed("corrupt".to_string());
        assert!(err.to_string().contains("corrupt"));
    }

    #[test]
    fn test_transport_connection_failed_display() {
        let err = TransportError::ConnectionFailed("host unreachable".to_string());
        assert!(err.to_string().contains("host unreachable"));
    }

    #[test]
    fn test_transport_timeout_display() {
        let err = TransportError::Timeout {
            operation: "upload".to_string(),
            timeout_ms: 5000,
        };
        let msg = err.to_string();
        assert!(msg.contains("upload"));
        assert!(msg.contains("5000"));
    }

    #[test]
    fn test_storage_not_found_display() {
        let err = StorageError::NotFound {
            path: "/data/replay.bin".to_string(),
        };
        assert!(err.to_string().contains("/data/replay.bin"));
    }

    #[test]
    fn test_storage_write_failed_display() {
        let err = StorageError::WriteFailed {
            path: "/data/model.bin".to_string(),
            reason: "disk full".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/data/model.bin"));
        assert!(msg.contains("disk full"));
    }

    #[test]
    fn test_storage_read_failed_display() {
        let err = StorageError::ReadFailed {
            path: "/data/model.bin".to_string(),
            reason: "permission denied".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/data/model.bin"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn test_storage_invalid_format_display() {
        let err = StorageError::InvalidFormat {
            path: "/data/model.bin".to_string(),
            reason: "header mismatch".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/data/model.bin"));
        assert!(msg.contains("header mismatch"));
    }

    #[test]
    fn test_model_registry_version_not_found_display() {
        let err = ModelRegistryError::VersionNotFound {
            model_id: "policy-v1".to_string(),
            version: 42,
        };
        let msg = err.to_string();
        assert!(msg.contains("policy-v1"));
        assert!(msg.contains("42"));
    }

    #[test]
    fn test_model_registry_invalid_model_display() {
        let err = ModelRegistryError::InvalidModel {
            reason: "missing weights".to_string(),
        };
        assert!(err.to_string().contains("missing weights"));
    }

    #[test]
    fn test_model_registry_registration_failed_display() {
        let err = ModelRegistryError::RegistrationFailed {
            model_id: "policy-v2".to_string(),
            reason: "duplicate version".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("policy-v2"));
        assert!(msg.contains("duplicate version"));
    }

    #[test]
    fn test_worker_error_converts_to_cloud_error() {
        let worker_err = WorkerError::NotFound("w-001".to_string());
        let cloud_err: CloudError = worker_err.into();
        assert!(matches!(cloud_err, CloudError::Worker(_)));
    }

    #[test]
    fn test_transport_error_converts_to_cloud_error() {
        let transport_err = TransportError::ConnectionFailed("fail".to_string());
        let cloud_err: CloudError = transport_err.into();
        assert!(matches!(cloud_err, CloudError::Transport(_)));
    }

    #[test]
    fn test_storage_error_converts_to_cloud_error() {
        let storage_err = StorageError::NotFound {
            path: "/tmp".to_string(),
        };
        let cloud_err: CloudError = storage_err.into();
        assert!(matches!(cloud_err, CloudError::Storage(_)));
    }

    #[test]
    fn test_model_registry_error_converts_to_cloud_error() {
        let reg_err = ModelRegistryError::InvalidModel {
            reason: "bad".to_string(),
        };
        let cloud_err: CloudError = reg_err.into();
        assert!(matches!(cloud_err, CloudError::ModelRegistry(_)));
    }

    #[test]
    fn test_config_error_converts_to_cloud_error() {
        let config_err = forge_types::error::ConfigError::ParseError("bad toml".to_string());
        let cloud_err: CloudError = config_err.into();
        assert!(matches!(cloud_err, CloudError::Config(_)));
    }

    #[test]
    fn test_serialization_error_display() {
        let err = CloudError::Serialization("invalid JSON at line 5".to_string());
        let msg = err.to_string();
        assert!(msg.contains("serialization error"));
        assert!(msg.contains("invalid JSON at line 5"));
    }

    #[test]
    fn test_cloud_result_ok() {
        let result: CloudResult<u32> = Ok(42);
        assert!(result.is_ok());
        assert!(matches!(result, Ok(42)));
    }

    #[test]
    fn test_cloud_result_err() {
        let result: CloudResult<u32> = Err(CloudError::Serialization("bad data".to_string()));
        assert!(result.is_err());
        assert!(matches!(result, Err(CloudError::Serialization(_))));
    }
}
