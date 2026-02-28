//! Error types for the FORGE simulation platform.
//!
//! All errors are structured using `thiserror` for clean error propagation
//! across crate boundaries.

use thiserror::Error;

/// Top-level error type for FORGE operations.
#[derive(Debug, Error)]
pub enum ForgeError {
    /// An error occurred during world generation.
    #[error("world generation error: {0}")]
    WorldGen(#[from] WorldGenError),

    /// An error occurred during simulation stepping.
    #[error("simulation error: {0}")]
    Simulation(#[from] SimulationError),

    /// An error occurred with configuration.
    #[error("configuration error: {0}")]
    Config(#[from] ConfigError),

    /// An error occurred during serialization/deserialization.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// An error occurred in the task system.
    #[error("task error: {0}")]
    Task(#[from] TaskError),
}

/// Errors that can occur during world generation.
#[derive(Debug, Error)]
pub enum WorldGenError {
    /// The world dimensions are invalid.
    #[error("invalid world dimensions: {width}x{height} (min: {min_size}, max: {max_size})")]
    InvalidDimensions {
        width: u16,
        height: u16,
        min_size: u16,
        max_size: u16,
    },

    /// Resource density is out of range.
    #[error("resource density {0} out of range [0.0, 1.0]")]
    InvalidResourceDensity(f32),

    /// Too many entities requested for the world size.
    #[error("max entities {requested} exceeds capacity for {width}x{height} world")]
    TooManyEntities {
        requested: u16,
        width: u16,
        height: u16,
    },
}

/// Errors that can occur during simulation stepping.
#[derive(Debug, Error)]
pub enum SimulationError {
    /// An invalid action was submitted.
    #[error("invalid action for agent {agent_id}: {reason}")]
    InvalidAction { agent_id: u32, reason: String },

    /// Wrong number of actions provided.
    #[error("expected {expected} actions, got {got}")]
    ActionCountMismatch { expected: usize, got: usize },

    /// Agent referenced does not exist.
    #[error("agent {0} not found")]
    AgentNotFound(u32),

    /// The simulation is in a terminal state.
    #[error("simulation has terminated, call reset()")]
    AlreadyTerminated,

    /// Inventory operation failed.
    #[error("inventory error for agent {agent_id}: {reason}")]
    InventoryError { agent_id: u32, reason: String },
}

/// Errors related to configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// A configuration value is out of its valid range.
    #[error("config field '{field}': value {value} out of range [{min}, {max}]")]
    OutOfRange {
        field: String,
        value: String,
        min: String,
        max: String,
    },

    /// Configuration deserialization failed.
    #[error("failed to parse config: {0}")]
    ParseError(String),
}

/// Errors in the task system.
#[derive(Debug, Error)]
pub enum TaskError {
    /// A predicate references an invalid entity.
    #[error("predicate references unknown entity: {0}")]
    UnknownEntity(String),

    /// Task composition is invalid.
    #[error("invalid task composition: {0}")]
    InvalidComposition(String),

    /// Task difficulty is out of range.
    #[error("task difficulty {0} out of range [1, 6]")]
    InvalidDifficulty(u8),
}

/// Result type alias for FORGE operations.
pub type ForgeResult<T> = Result<T, ForgeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SimulationError::InvalidAction {
            agent_id: 0,
            reason: "cannot move into wall".to_string(),
        };
        assert!(err.to_string().contains("agent 0"));
        assert!(err.to_string().contains("cannot move into wall"));
    }

    #[test]
    fn test_error_conversion() {
        let sim_err = SimulationError::AgentNotFound(5);
        let forge_err: ForgeError = sim_err.into();
        assert!(matches!(forge_err, ForgeError::Simulation(_)));
    }

    #[test]
    fn test_config_error_display() {
        let config_err = ConfigError::OutOfRange {
            field: "width".to_string(),
            value: "999".to_string(),
            min: "8".to_string(),
            max: "256".to_string(),
        };
        let forge_err: ForgeError = config_err.into();
        let msg = forge_err.to_string();
        assert!(msg.contains("configuration error"));
        assert!(msg.contains("width"));
        assert!(msg.contains("999"));
        assert!(msg.contains("8"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn test_serialization_error_display() {
        let forge_err = ForgeError::Serialization("invalid JSON at line 5".to_string());
        let msg = forge_err.to_string();
        assert!(msg.contains("serialization error"));
        assert!(msg.contains("invalid JSON at line 5"));
    }

    #[test]
    fn test_forge_result_ok() {
        let result: ForgeResult<u32> = Ok(42);
        assert!(result.is_ok());
        assert!(matches!(result, Ok(42)));
    }

    #[test]
    fn test_forge_result_err() {
        let result: ForgeResult<u32> = Err(ForgeError::Serialization("bad data".to_string()));
        assert!(result.is_err());
        assert!(matches!(result, Err(ForgeError::Serialization(_))));
    }
}
