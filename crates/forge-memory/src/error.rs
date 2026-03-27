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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_error_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = MemoryError::from(io_err);
        let msg = err.to_string();
        assert!(msg.contains("memory I/O error"));
        assert!(msg.contains("file missing"));
    }

    #[test]
    fn test_serialize_error_display() {
        let err = MemoryError::Serialize("bad data".into());
        assert!(err.to_string().contains("serialization error"));
        assert!(err.to_string().contains("bad data"));
    }

    #[test]
    fn test_deserialize_error_display() {
        let err = MemoryError::Deserialize("corrupt".into());
        assert!(err.to_string().contains("deserialization error"));
        assert!(err.to_string().contains("corrupt"));
    }

    #[test]
    fn test_io_error_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let mem_err: MemoryError = io_err.into();
        assert!(matches!(mem_err, MemoryError::Io(_)));
    }
}
