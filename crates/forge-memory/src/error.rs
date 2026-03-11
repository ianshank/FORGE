//! Error types for the memory system.

use thiserror::Error;

/// Errors that can occur in the memory system.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// I/O error during persistence operations.
    #[error("memory I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization error.
    #[error("memory serialization error: {0}")]
    Serialize(String),
    /// Deserialization error.
    #[error("memory deserialization error: {0}")]
    Deserialize(String),
}
