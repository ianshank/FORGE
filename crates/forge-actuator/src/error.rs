//! Error type for the actuator bridge.

use thiserror::Error;

/// Errors that can occur while loading or dispatching through an actuator
/// bridge.
///
/// All variants carry enough context that callers can decide whether to (a)
/// fall back to an EdgeAgent-configured safe-pose action, (b) trip a hardware
/// E-stop, or (c) surface the error to the user. The variants are
/// non-exhaustive so additional failure modes (firmware version mismatch,
/// torque-limit trip) can be added without breaking downstream `match`
/// statements.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ActuatorError {
    /// The action id has no entry in the [`crate::ActionMapping`] and the
    /// mapping was loaded in `strict` mode (no default fallback).
    ///
    /// Non-strict mappings will silently route unknown ids to the configured
    /// default sequence and never raise this error.
    #[error("no mapping entry for action id {action_id} (strict mode)")]
    UnknownActionId {
        /// Discrete FORGE action id that was not found in the mapping.
        action_id: u32,
    },

    /// The underlying [`crate::ActuatorDriver`] reported a failure executing
    /// a command (serial timeout, torque trip, malformed packet).
    #[error("actuator driver failure: {0}")]
    DriverFailure(String),

    /// The mapping TOML was structurally invalid (e.g. duplicate action ids,
    /// missing `default` block when `strict = false`).
    #[error("invalid mapping: {0}")]
    InvalidMapping(String),

    /// The mapping TOML failed to parse.
    #[error("failed to parse actuator mapping TOML: {0}")]
    ParseToml(#[from] toml::de::Error),

    /// The mapping TOML file could not be read from disk.
    #[error("failed to read actuator mapping file '{path}': {source}")]
    Io {
        /// Path the loader attempted to read.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_action_id_message_includes_id() {
        let err = ActuatorError::UnknownActionId { action_id: 42 };
        let msg = err.to_string();
        assert!(msg.contains("42"), "expected id in error message: {msg}");
    }

    #[test]
    fn driver_failure_message_includes_payload() {
        let err = ActuatorError::DriverFailure("torque limit exceeded".to_string());
        let msg = err.to_string();
        assert!(msg.contains("torque limit exceeded"));
    }

    #[test]
    fn parse_toml_via_from_impl() {
        // Force the From<toml::de::Error> conversion path.
        let parse_err = toml::from_str::<crate::ActuatorCommand>("not [valid toml")
            .expect_err("malformed toml must error");
        let err: ActuatorError = parse_err.into();
        assert!(matches!(err, ActuatorError::ParseToml(_)));
    }
}
