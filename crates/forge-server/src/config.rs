//! Server configuration with sensible defaults and env-var overrides.

use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

/// Default bind address for the server.
const DEFAULT_BIND_ADDR: &str = "0.0.0.0:8080";
/// Default broadcast channel capacity for WebSocket fan-out.
const DEFAULT_BROADCAST_CAPACITY: usize = 64;
/// Default simulation tick interval in milliseconds.
const DEFAULT_TICK_INTERVAL_MS: u64 = 100;
/// Default tracing filter for the server.
const DEFAULT_LOG_FILTER: &str = "forge_server=info,forge_core=info";
/// Default allowed CORS origins.
const DEFAULT_ALLOWED_ORIGINS: &str = "http://localhost:5173";

/// Configuration for the FORGE server binary.
///
/// All values have sensible defaults and can be overridden via environment
/// variables prefixed with `FORGE_SERVER_`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Socket address to bind the HTTP/WebSocket server to.
    pub bind_addr: SocketAddr,
    /// Broadcast channel capacity for WebSocket fan-out.
    pub broadcast_capacity: usize,
    /// Simulation tick interval in milliseconds.
    pub tick_interval_ms: u64,
    /// Tracing env filter string.
    pub log_filter: String,
    /// Allowed CORS origins (comma-separated).
    pub allowed_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: DEFAULT_BIND_ADDR.parse().expect("valid default addr"),
            broadcast_capacity: DEFAULT_BROADCAST_CAPACITY,
            tick_interval_ms: DEFAULT_TICK_INTERVAL_MS,
            log_filter: DEFAULT_LOG_FILTER.to_string(),
            allowed_origins: DEFAULT_ALLOWED_ORIGINS
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
        }
    }
}

impl ServerConfig {
    /// Build a `ServerConfig` from environment variables with defaults.
    ///
    /// Reads:
    /// - `FORGE_SERVER_BIND` — bind address (default `0.0.0.0:8080`)
    /// - `FORGE_SERVER_TICK_MS` — tick interval in ms (default `100`)
    /// - `FORGE_SERVER_BROADCAST_CAPACITY` — WS channel size (default `64`)
    /// - `FORGE_SERVER_LOG_FILTER` — tracing filter (default `forge_server=info,forge_core=info`)
    /// - `FORGE_SERVER_ALLOWED_ORIGINS` — allowed CORS origins, comma-separated (default `http://localhost:5173`)
    #[tracing::instrument]
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("FORGE_SERVER_BIND") {
            match val.parse() {
                Ok(addr) => config.bind_addr = addr,
                Err(e) => {
                    tracing::warn!(
                        value = %val,
                        error = %e,
                        "Invalid FORGE_SERVER_BIND, using default"
                    );
                }
            }
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_TICK_MS") {
            match val.parse() {
                Ok(ms) => config.tick_interval_ms = ms,
                Err(e) => {
                    tracing::warn!(
                        value = %val,
                        error = %e,
                        "Invalid FORGE_SERVER_TICK_MS, using default"
                    );
                }
            }
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_BROADCAST_CAPACITY") {
            match val.parse() {
                Ok(cap) => config.broadcast_capacity = cap,
                Err(e) => {
                    tracing::warn!(
                        value = %val,
                        error = %e,
                        "Invalid FORGE_SERVER_BROADCAST_CAPACITY, using default"
                    );
                }
            }
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_LOG_FILTER") {
            config.log_filter = val;
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_ALLOWED_ORIGINS") {
            config.allowed_origins = val.split(',').map(|s| s.trim().to_string()).collect();
        }

        tracing::debug!(?config, "Server config loaded from environment");
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.bind_addr.port(), 8080);
        assert_eq!(config.broadcast_capacity, DEFAULT_BROADCAST_CAPACITY);
        assert_eq!(config.tick_interval_ms, DEFAULT_TICK_INTERVAL_MS);
        assert!(!config.log_filter.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = ServerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.bind_addr, config.bind_addr);
        assert_eq!(parsed.tick_interval_ms, config.tick_interval_ms);
    }

    #[test]
    fn test_from_env_defaults() {
        // Without env vars set, should use defaults
        let config = ServerConfig::from_env();
        assert_eq!(config.bind_addr.port(), 8080);
    }
}
