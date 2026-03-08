//! REST API route definitions and handler functions.

use serde::Serialize;
use tracing::instrument;

use crate::SCHEMA_VERSION;

/// Response payload for the `/config` endpoint.
#[derive(Debug, Serialize)]
pub struct ConfigResponse {
    /// Server version string.
    pub version: String,
    /// Schema version for client compatibility.
    pub schema_version: u32,
}

/// Response payload for the `/health` endpoint.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Current server status.
    pub status: String,
    /// Server uptime in seconds.
    pub uptime_seconds: u64,
}

/// Returns configuration and version information.
#[instrument]
pub fn config_handler() -> ConfigResponse {
    tracing::debug!("Handling config request");
    ConfigResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        schema_version: SCHEMA_VERSION,
    }
}

/// Returns the current health status of the server.
#[instrument]
pub fn health_handler() -> HealthResponse {
    tracing::debug!("Handling health request");
    HealthResponse {
        status: "ok".to_string(),
        uptime_seconds: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_handler() {
        let response = config_handler();
        assert!(!response.version.is_empty());
        assert_eq!(response.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn test_health_handler() {
        let response = health_handler();
        assert_eq!(response.status, "ok");
        // uptime_seconds is a valid field (starts at 0).
        assert_eq!(response.uptime_seconds, 0);
    }
}
