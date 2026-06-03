//! Error types for the MangoMAS integration layer.
//!
//! All errors are structured via `thiserror` derive macros and compose
//! with the FORGE error hierarchy.

use thiserror::Error;

/// Top-level error type for MangoMAS integration operations.
#[derive(Debug, Error)]
pub enum MangoMasError {
    /// An action adapter operation failed.
    #[error("action adapter error: {0}")]
    ActionAdapter(String),

    /// An observation adapter operation failed.
    #[error("observation adapter error: {0}")]
    ObservationAdapter(String),

    /// A configuration translation error occurred.
    #[error("config adapter error: {0}")]
    ConfigAdapter(String),

    /// A batch runner error occurred.
    #[error("batch runner error: {0}")]
    BatchRunner(String),

    /// An MCTS sweep error occurred.
    #[error("mcts sweep error: {0}")]
    MctsSweep(String),

    /// A BDI collection error occurred.
    #[error("bdi collector error: {0}")]
    BdiCollector(String),

    /// A constitutional constraint mapping error occurred.
    #[error("constitutional error: {0}")]
    Constitutional(String),

    /// An RSSM adapter error occurred.
    #[error("rssm adapter error: {0}")]
    RssmAdapter(String),

    /// A curriculum error occurred.
    #[error("curriculum error: {0}")]
    Curriculum(String),

    /// A multi-agent swarm coordination error occurred (e.g. agent/observation
    /// count mismatch, out-of-range agent index, or joint-planner failure).
    #[error("swarm coordination error: {0}")]
    SwarmCoordination(String),

    /// A weight export error occurred.
    #[error("export error: {0}")]
    Export(String),

    /// An underlying FORGE simulation error.
    #[error("forge error: {0}")]
    Forge(#[from] forge_types::error::ForgeError),

    /// A serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// Result type alias for MangoMAS operations.
pub type MangoMasResult<T> = Result<T, MangoMasError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MangoMasError::ActionAdapter("out of range".to_string());
        assert_eq!(err.to_string(), "action adapter error: out of range");
    }

    #[test]
    fn test_error_variants_display() {
        let cases: Vec<(MangoMasError, &str)> = vec![
            (
                MangoMasError::ObservationAdapter("dim mismatch".into()),
                "observation adapter error: dim mismatch",
            ),
            (
                MangoMasError::ConfigAdapter("missing field".into()),
                "config adapter error: missing field",
            ),
            (
                MangoMasError::BatchRunner("timeout".into()),
                "batch runner error: timeout",
            ),
            (
                MangoMasError::MctsSweep("invalid range".into()),
                "mcts sweep error: invalid range",
            ),
            (
                MangoMasError::BdiCollector("no episodes".into()),
                "bdi collector error: no episodes",
            ),
            (
                MangoMasError::Constitutional("unmapped".into()),
                "constitutional error: unmapped",
            ),
            (
                MangoMasError::RssmAdapter("shape error".into()),
                "rssm adapter error: shape error",
            ),
            (
                MangoMasError::Curriculum("invalid tier".into()),
                "curriculum error: invalid tier",
            ),
            (
                MangoMasError::SwarmCoordination("agent count mismatch".into()),
                "swarm coordination error: agent count mismatch",
            ),
            (
                MangoMasError::Export("io error".into()),
                "export error: io error",
            ),
            (
                MangoMasError::Serialization("parse failed".into()),
                "serialization error: parse failed",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }
}
