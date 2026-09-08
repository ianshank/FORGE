//! Error types for the Minecraft env.

use thiserror::Error;

/// Display prefix for [`McEnvError::Transient`].
///
/// `forge-mc-runner` string-matches this prefix because `Runner` is
/// generic over `Env::Error` and cannot downcast to [`McEnvError`]
/// when the `mc-live` feature is off. Changing it requires a
/// coordinated bump of `TRANSIENT_ENV_DISPLAY_PREFIX` in
/// `crates/forge-mc-runner/src/error.rs`.
pub const TRANSIENT_PROTOCOL_ERROR_DISPLAY_PREFIX: &str = "transient protocol error [";

/// Error returned by [`crate::MinecraftEnv`] and supporting types.
#[derive(Debug, Error)]
pub enum McEnvError {
    /// Underlying WebSocket transport failed.
    #[error("websocket error: {0}")]
    WebSocket(String),
    /// JSON (de)serialisation failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Bot replied with a protocol-level error.
    #[error("protocol error [{code}]: {message}")]
    Protocol {
        /// Bot-supplied error code.
        code: String,
        /// Bot-supplied human message.
        message: String,
    },
    /// Bot replied with a **transient** protocol error (`RECONNECTING`
    /// or `BUSY`). The request–response pair is complete; the episode
    /// is not resumable. Display is parseable by `forge-mc-runner`
    /// (which cannot downcast to this type when `mc-live` is off).
    ///
    /// Prefix is pinned to [`TRANSIENT_PROTOCOL_ERROR_DISPLAY_PREFIX`].
    #[error("transient protocol error [{code}]: {message}")]
    Transient {
        /// Bot-supplied error code (`RECONNECTING` or `BUSY`).
        code: String,
        /// Bot-supplied human message.
        message: String,
    },
    /// Bot's reported schema_version, action_count, obs_dim, or schema_id
    /// disagreed with the client's compiled-in expectations.
    #[error("handshake mismatch: client expected {client}, server reported {server}")]
    HandshakeMismatch {
        /// Human description of the client's expectation.
        client: String,
        /// Human description of the server's reply.
        server: String,
    },
    /// Bot returned an observation vector whose length changed after
    /// handshake.
    #[error("observation dimension mismatch: expected {expected}, got {got}")]
    ObsDimMismatch {
        /// Dimension declared by the bot during handshake.
        expected: usize,
        /// Dimension returned by a reset/step observation.
        got: usize,
    },
    /// Action id outside the declared action space.
    #[error("invalid action id: {action_id} (action_space.n = {space_n})")]
    InvalidAction {
        /// The offending id.
        action_id: u32,
        /// The space cardinality.
        space_n: u32,
    },
    /// Configuration file failed to load or validate.
    #[error("config error: {0}")]
    Config(String),
    /// Env was used after [`crate::MinecraftEnv::close`] returned `Ok`.
    #[error("env is closed")]
    Closed,
    /// Bot sent an unexpected message kind for the current state.
    #[error("unexpected message: {0}")]
    Unexpected(String),
}

impl McEnvError {
    /// Map a bot `Error` frame onto [`Self::Transient`] or [`Self::Protocol`]
    /// using [`crate::protocol::is_transient_error_code`].
    #[must_use]
    pub fn from_protocol_error(code: impl Into<String>, message: impl Into<String>) -> Self {
        let code = code.into();
        let message = message.into();
        if crate::protocol::is_transient_error_code(&code) {
            Self::Transient { code, message }
        } else {
            Self::Protocol { code, message }
        }
    }

    /// True when this error ends the episode without failing the run.
    #[must_use]
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::Transient { .. })
    }
}

impl From<tungstenite::Error> for McEnvError {
    fn from(value: tungstenite::Error) -> Self {
        Self::WebSocket(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_tungstenite_error_maps_to_websocket_variant() {
        let inner = tungstenite::Error::ConnectionClosed;
        let mapped: McEnvError = inner.into();
        assert!(matches!(mapped, McEnvError::WebSocket(_)));
    }

    #[test]
    fn closed_error_display() {
        assert_eq!(McEnvError::Closed.to_string(), "env is closed");
    }

    #[test]
    fn unexpected_error_display() {
        let e = McEnvError::Unexpected("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn obs_dim_mismatch_display() {
        let e = McEnvError::ObsDimMismatch {
            expected: 4,
            got: 7,
        };
        assert!(e.to_string().contains("expected 4"));
        assert!(e.to_string().contains("got 7"));
    }

    #[test]
    fn transient_display_uses_pinned_prefix() {
        let e = McEnvError::Transient {
            code: "RECONNECTING".into(),
            message: "bot rebuilding".into(),
        };
        let displayed = e.to_string();
        assert!(
            displayed.starts_with(TRANSIENT_PROTOCOL_ERROR_DISPLAY_PREFIX),
            "Transient Display must start with the runner-parseable prefix, got {displayed}"
        );
        assert_eq!(
            displayed,
            "transient protocol error [RECONNECTING]: bot rebuilding"
        );
        assert!(e.is_transient());
    }

    #[test]
    fn from_protocol_error_classifies_transient_vs_fatal() {
        let t = McEnvError::from_protocol_error(crate::protocol::ERROR_CODE_RECONNECTING, "x");
        assert!(matches!(t, McEnvError::Transient { .. }));
        let busy = McEnvError::from_protocol_error(crate::protocol::ERROR_CODE_BUSY, "x");
        assert!(
            matches!(busy, McEnvError::Transient { ref code, .. } if code == crate::protocol::ERROR_CODE_BUSY)
        );
        let p = McEnvError::from_protocol_error("INTERNAL", "x");
        assert!(matches!(p, McEnvError::Protocol { .. }));
        assert!(!p.is_transient());
    }
}
