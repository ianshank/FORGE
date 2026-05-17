//! Default error type for environments that don't need a richer variant.
//!
//! Implementors are free to define their own `Error` associated type; this
//! is provided as a convenience for tests and simple wrappers.

use thiserror::Error;

/// Generic environment error.
///
/// Concrete environments (`forge-env-forge`, `forge-env-mc`) define their
/// own `thiserror`-derived enums; this type exists for stub/mock envs in
/// tests that don't care to define a bespoke error.
#[derive(Debug, Error)]
pub enum EnvError {
    /// Reset was called with parameters the env cannot satisfy.
    #[error("invalid reset: {0}")]
    InvalidReset(String),
    /// Step was called with an action outside the declared action space.
    #[error("invalid action: {action_id} (action_space.n = {space_n})")]
    InvalidAction {
        /// The offending action id.
        action_id: u32,
        /// The declared action-space cardinality.
        space_n: u32,
    },
    /// Env was used after [`crate::Env::close`] was called.
    #[error("env is closed")]
    Closed,
    /// Catch-all for env-specific errors that don't warrant a new variant.
    #[error("env error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_format_cleanly() {
        let e = EnvError::InvalidAction {
            action_id: 99,
            space_n: 40,
        };
        assert_eq!(e.to_string(), "invalid action: 99 (action_space.n = 40)");
    }

    #[test]
    fn closed_is_distinct_variant() {
        let e = EnvError::Closed;
        assert_eq!(e.to_string(), "env is closed");
    }
}
