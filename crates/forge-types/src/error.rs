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

    /// An error occurred in the cloud training pipeline.
    #[error("cloud error: {0}")]
    Cloud(#[from] CloudError),

    /// An error occurred in the edge runtime.
    #[error("edge error: {0}")]
    Edge(#[from] EdgeError),

    /// An error occurred while encoding an [`crate::action::Action`] into its
    /// discrete integer representation.
    #[error("action encoding error: {0}")]
    ActionEncoding(#[from] ActionEncodingError),
}

/// Errors that can occur during world generation.
#[derive(Debug, Error)]
pub enum WorldGenError {
    /// The world dimensions are invalid.
    #[error("invalid world dimensions: {width}x{height} (min: {min_size}, max: {max_size})")]
    InvalidDimensions {
        /// Requested world width.
        width: u16,
        /// Requested world height.
        height: u16,
        /// Minimum allowed dimension.
        min_size: u16,
        /// Maximum allowed dimension.
        max_size: u16,
    },

    /// Resource density is out of range.
    #[error("resource density {0} out of range [0.0, 1.0]")]
    InvalidResourceDensity(f32),

    /// Too many entities requested for the world size.
    #[error("max entities {requested} exceeds capacity for {width}x{height} world")]
    TooManyEntities {
        /// Number of entities requested.
        requested: u16,
        /// World width in tiles.
        width: u16,
        /// World height in tiles.
        height: u16,
    },
}

/// Errors that can occur during simulation stepping.
#[derive(Debug, Error)]
pub enum SimulationError {
    /// An invalid action was submitted.
    #[error("invalid action for agent {agent_id}: {reason}")]
    InvalidAction {
        /// ID of the agent that submitted the action.
        agent_id: u32,
        /// Description of why the action was invalid.
        reason: String,
    },

    /// Wrong number of actions provided.
    #[error("expected {expected} actions, got {got}")]
    ActionCountMismatch {
        /// Expected number of actions.
        expected: usize,
        /// Actual number of actions received.
        got: usize,
    },

    /// Agent referenced does not exist.
    #[error("agent {0} not found")]
    AgentNotFound(u32),

    /// The simulation is in a terminal state.
    #[error("simulation has terminated, call reset()")]
    AlreadyTerminated,

    /// Inventory operation failed.
    #[error("inventory error for agent {agent_id}: {reason}")]
    InventoryError {
        /// ID of the agent whose inventory operation failed.
        agent_id: u32,
        /// Description of the inventory error.
        reason: String,
    },
}

/// Errors related to configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// A configuration value is out of its valid range.
    #[error("config field '{field}': value {value} out of range [{min}, {max}]")]
    OutOfRange {
        /// Name of the configuration field.
        field: String,
        /// The invalid value that was provided.
        value: String,
        /// Minimum allowed value.
        min: String,
        /// Maximum allowed value.
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

/// Errors in the cloud training pipeline.
#[derive(Debug, Error)]
pub enum CloudError {
    /// An error occurred with cloud storage operations.
    #[error("storage error: {0}")]
    Storage(String),

    /// An error occurred during replay transport.
    #[error("transport error: {0}")]
    Transport(String),

    /// An error occurred with worker management.
    #[error("worker error: {0}")]
    Worker(String),

    /// An error occurred in the model registry.
    #[error("model registry error: {0}")]
    ModelRegistry(String),

    /// An error occurred in the training coordinator.
    #[error("coordinator error: {0}")]
    Coordinator(String),

    /// An error occurred during replay compression.
    #[error("compression error: {0}")]
    Compression(String),

    /// Replay payload exceeds the configured maximum size.
    #[error("replay payload too large: {size} bytes exceeds limit of {max} bytes")]
    PayloadTooLarge {
        /// Actual payload size in bytes.
        size: u64,
        /// Maximum allowed payload size in bytes.
        max: u64,
    },
}

/// Errors in the edge runtime.
#[derive(Debug, Error)]
pub enum EdgeError {
    /// An error occurred during neural network inference.
    #[error("inference error: {0}")]
    Inference(String),

    /// An error occurred in the telemetry subsystem.
    #[error("telemetry error: {0}")]
    Telemetry(String),

    /// An error occurred during model update.
    #[error("model update error: {0}")]
    ModelUpdate(String),

    /// MCTS planning exceeded the configured latency budget.
    #[error("latency budget exceeded: budget={budget_ms}ms, actual={actual_ms}ms")]
    LatencyBudgetExceeded {
        /// Configured latency budget in milliseconds.
        budget_ms: u32,
        /// Actual latency in milliseconds.
        actual_ms: u32,
    },

    /// Telemetry buffer is full and cannot accept more data.
    #[error("telemetry buffer full: {current_bytes} bytes of {max_bytes} bytes used")]
    TelemetryBufferFull {
        /// Current buffer usage in bytes.
        current_bytes: u64,
        /// Maximum buffer capacity in bytes.
        max_bytes: u64,
    },
}

/// Errors that can occur when encoding an [`crate::action::Action`] into its
/// discrete integer representation.
///
/// These errors replace the legacy `panic!` paths in
/// [`crate::action::Action::to_discrete`] /
/// [`crate::action::Action::to_discrete_configured`]. Use the `try_*` variants
/// to recover gracefully from a configuration / action mismatch instead of
/// crashing the process.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum ActionEncodingError {
    /// A drone action was submitted to an encoder that does not advertise the
    /// drone communication offset (e.g. [`crate::action::Action::to_discrete`]).
    ///
    /// The fix is to use [`crate::action::Action::to_discrete_full`] or
    /// [`crate::action::Action::to_discrete_configured`] with
    /// `drone_actions_enabled = true`.
    #[error(
        "drone action {action_name} requires the drone-aware encoder \
        (call to_discrete_full or to_discrete_configured with drone_actions_enabled=true)"
    )]
    DroneActionRequiresFullEncoder {
        /// Static, debug-style name of the offending action variant.
        action_name: &'static str,
    },

    /// An agricultural action was submitted but either drone or agricultural
    /// support is disabled in the active action space layout. Agricultural
    /// actions are layered on top of drone infrastructure, so they require
    /// both flags.
    #[error(
        "agricultural action {action_name} requires drone_actions_enabled=true \
        AND agri_actions_enabled=true (got drone={drone_actions_enabled}, agri={agri_actions_enabled})"
    )]
    AgriActionUnsupported {
        /// Static, debug-style name of the offending action variant.
        action_name: &'static str,
        /// Whether drone actions were enabled at the call site.
        drone_actions_enabled: bool,
        /// Whether agricultural actions were enabled at the call site.
        agri_actions_enabled: bool,
    },

    /// A hex-grid movement action was submitted but hex actions are disabled
    /// in the active action space layout.
    #[error(
        "hex move action requires hex_actions_enabled=true \
        (got hex_actions_enabled={hex_actions_enabled})"
    )]
    HexActionUnsupported {
        /// Whether hex actions were enabled at the call site.
        hex_actions_enabled: bool,
    },
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

    // ---- Cloud error tests ----

    #[test]
    fn test_cloud_error_display() {
        let err = CloudError::Storage("bucket not found".to_string());
        assert!(err.to_string().contains("storage error"));
        assert!(err.to_string().contains("bucket not found"));
    }

    #[test]
    fn test_cloud_error_conversion_to_forge() {
        let cloud_err = CloudError::Transport("connection refused".to_string());
        let forge_err: ForgeError = cloud_err.into();
        assert!(matches!(forge_err, ForgeError::Cloud(_)));
        assert!(forge_err.to_string().contains("transport error"));
    }

    #[test]
    fn test_cloud_payload_too_large_display() {
        let err = CloudError::PayloadTooLarge {
            size: 20_000_000,
            max: 10_485_760,
        };
        let msg = err.to_string();
        assert!(msg.contains("20000000"));
        assert!(msg.contains("10485760"));
    }

    #[test]
    fn test_cloud_error_all_variants() {
        let variants: Vec<CloudError> = vec![
            CloudError::Storage("s".to_string()),
            CloudError::Transport("t".to_string()),
            CloudError::Worker("w".to_string()),
            CloudError::ModelRegistry("m".to_string()),
            CloudError::Coordinator("c".to_string()),
            CloudError::Compression("z".to_string()),
            CloudError::PayloadTooLarge { size: 1, max: 0 },
        ];
        for err in &variants {
            // All variants must produce non-empty display strings.
            assert!(!err.to_string().is_empty());
        }
    }

    // ---- Edge error tests ----

    #[test]
    fn test_edge_error_display() {
        let err = EdgeError::Inference("ONNX model not loaded".to_string());
        assert!(err.to_string().contains("inference error"));
        assert!(err.to_string().contains("ONNX model not loaded"));
    }

    #[test]
    fn test_edge_error_conversion_to_forge() {
        let edge_err = EdgeError::Telemetry("buffer overflow".to_string());
        let forge_err: ForgeError = edge_err.into();
        assert!(matches!(forge_err, ForgeError::Edge(_)));
        assert!(forge_err.to_string().contains("telemetry error"));
    }

    #[test]
    fn test_edge_latency_budget_exceeded_display() {
        let err = EdgeError::LatencyBudgetExceeded {
            budget_ms: 50,
            actual_ms: 120,
        };
        let msg = err.to_string();
        assert!(msg.contains("budget=50ms"));
        assert!(msg.contains("actual=120ms"));
    }

    #[test]
    fn test_edge_telemetry_buffer_full_display() {
        let err = EdgeError::TelemetryBufferFull {
            current_bytes: 1_048_576,
            max_bytes: 1_048_576,
        };
        let msg = err.to_string();
        assert!(msg.contains("1048576"));
    }

    #[test]
    fn test_edge_error_all_variants() {
        let variants: Vec<EdgeError> = vec![
            EdgeError::Inference("i".to_string()),
            EdgeError::Telemetry("t".to_string()),
            EdgeError::ModelUpdate("m".to_string()),
            EdgeError::LatencyBudgetExceeded {
                budget_ms: 50,
                actual_ms: 100,
            },
            EdgeError::TelemetryBufferFull {
                current_bytes: 500,
                max_bytes: 1000,
            },
        ];
        for err in &variants {
            assert!(!err.to_string().is_empty());
        }
    }
}
