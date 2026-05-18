//! Error types for the Minecraft env.

use thiserror::Error;

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
}
