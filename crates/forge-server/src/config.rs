//! Server configuration with sensible defaults and env-var overrides.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use forge_types::constants::{DEFAULT_FRONTEND_DEV_PORT, DEFAULT_SERVER_PORT};
use serde::{Deserialize, Serialize};

/// Default broadcast channel capacity for WebSocket fan-out.
const DEFAULT_BROADCAST_CAPACITY: usize = 64;
/// Default simulation tick interval in milliseconds.
const DEFAULT_TICK_INTERVAL_MS: u64 = 100;
/// Default tracing filter for the server.
const DEFAULT_LOG_FILTER: &str = "forge_server=info,forge_core=info";

/// Configuration for the FORGE server binary.
///
/// All values have sensible defaults and can be overridden via environment
/// variables prefixed with `FORGE_SERVER_`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Socket address to bind the HTTP/WebSocket server to.
    #[serde(default = "default_bind_addr")]
    pub bind_addr: SocketAddr,
    /// Broadcast channel capacity for WebSocket fan-out.
    #[serde(default = "default_broadcast_capacity")]
    pub broadcast_capacity: usize,
    /// Simulation tick interval in milliseconds.
    #[serde(default = "default_tick_interval_ms")]
    pub tick_interval_ms: u64,
    /// Tracing env filter string.
    #[serde(default = "default_log_filter")]
    pub log_filter: String,
    /// Allowed CORS origins (comma-separated).
    #[serde(default = "default_allowed_origins")]
    pub allowed_origins: Vec<String>,
}

fn default_bind_addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), DEFAULT_SERVER_PORT)
}

fn default_allowed_origins_str() -> String {
    format!("http://localhost:{DEFAULT_FRONTEND_DEV_PORT}")
}

fn default_broadcast_capacity() -> usize {
    DEFAULT_BROADCAST_CAPACITY
}

fn default_tick_interval_ms() -> u64 {
    DEFAULT_TICK_INTERVAL_MS
}

fn default_log_filter() -> String {
    DEFAULT_LOG_FILTER.to_string()
}

fn default_allowed_origins() -> Vec<String> {
    vec![default_allowed_origins_str()]
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: default_bind_addr(),
            broadcast_capacity: DEFAULT_BROADCAST_CAPACITY,
            tick_interval_ms: DEFAULT_TICK_INTERVAL_MS,
            log_filter: DEFAULT_LOG_FILTER.to_string(),
            allowed_origins: default_allowed_origins(),
        }
    }
}

impl ServerConfig {
    /// Build a `ServerConfig` from environment variables with defaults.
    ///
    /// Reads:
    /// - `FORGE_SERVER_BIND` — bind address (default `0.0.0.0:8080`)
    /// - `FORGE_SERVER_PORT` — port only override (default `8080`); ignored if `FORGE_SERVER_BIND` is set
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
        } else if let Ok(val) = std::env::var("FORGE_SERVER_PORT") {
            match val.parse::<u16>() {
                Ok(port) => config.bind_addr.set_port(port),
                Err(e) => {
                    tracing::warn!(
                        value = %val,
                        error = %e,
                        "Invalid FORGE_SERVER_PORT, using default"
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
        assert_eq!(
            config.bind_addr.port(),
            forge_types::constants::DEFAULT_SERVER_PORT
        );
        assert_eq!(config.broadcast_capacity, DEFAULT_BROADCAST_CAPACITY);
        assert_eq!(config.tick_interval_ms, DEFAULT_TICK_INTERVAL_MS);
        assert!(!config.log_filter.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        forge_types::assert_config_serde_roundtrip!(ServerConfig);
    }

    #[test]
    fn test_defaults_valid() {
        forge_types::assert_config_defaults_valid!(ServerConfig);
    }

    #[test]
    fn test_from_env_defaults() {
        // Env vars are process-global; without the lock + scope this test fails
        // under any ambient `FORGE_SERVER_PORT` set in the dev shell or by a
        // sibling cargo test. Reuses the `ENV_LOCK`/`EnvScope`/`ALL_KEYS`
        // machinery declared below in this same `mod tests`.
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        let config = ServerConfig::from_env();
        assert_eq!(
            config.bind_addr.port(),
            forge_types::constants::DEFAULT_SERVER_PORT
        );
    }

    #[test]
    fn test_default_allowed_origins() {
        let config = ServerConfig::default();
        assert_eq!(config.allowed_origins.len(), 1);
        assert!(config.allowed_origins[0].contains("localhost"));
    }

    #[test]
    fn test_default_log_filter() {
        let config = ServerConfig::default();
        assert!(config.log_filter.contains("forge_server"));
        assert!(config.log_filter.contains("forge_core"));
    }

    #[test]
    fn test_default_tick_interval() {
        let config = ServerConfig::default();
        assert_eq!(config.tick_interval_ms, DEFAULT_TICK_INTERVAL_MS);
        assert!(config.tick_interval_ms > 0);
    }

    #[test]
    fn test_default_broadcast_capacity() {
        let config = ServerConfig::default();
        assert_eq!(config.broadcast_capacity, DEFAULT_BROADCAST_CAPACITY);
        assert!(config.broadcast_capacity > 0);
    }

    #[test]
    fn test_config_clone() {
        let config = ServerConfig::default();
        let cloned = config.clone();
        assert_eq!(config.bind_addr, cloned.bind_addr);
        assert_eq!(config.tick_interval_ms, cloned.tick_interval_ms);
        assert_eq!(config.broadcast_capacity, cloned.broadcast_capacity);
    }

    #[test]
    fn test_config_debug() {
        let config = ServerConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ServerConfig"));
        assert!(debug.contains(&forge_types::constants::DEFAULT_SERVER_PORT.to_string()));
    }

    #[test]
    fn test_from_env_defaults_when_no_env_vars() {
        // Without any FORGE_SERVER_ env vars, from_env should return defaults.
        // Lock + scope clear ambient pollution so this is deterministic regardless
        // of dev-shell state. See note on `test_from_env_defaults` above.
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        let config = ServerConfig::from_env();
        let default_config = ServerConfig::default();
        assert_eq!(config.bind_addr, default_config.bind_addr);
        assert_eq!(config.broadcast_capacity, default_config.broadcast_capacity);
        assert_eq!(config.tick_interval_ms, default_config.tick_interval_ms);
        assert_eq!(config.log_filter, default_config.log_filter);
        assert_eq!(config.allowed_origins, default_config.allowed_origins);
    }

    #[test]
    fn test_custom_bind_addr() {
        let config = ServerConfig {
            bind_addr: "127.0.0.1:9090".parse().unwrap(),
            ..ServerConfig::default()
        };
        assert_eq!(config.bind_addr.port(), 9090);
        assert_eq!(config.bind_addr.ip().to_string(), "127.0.0.1");
    }

    // ──────────────────────────────────────────────────────────────────
    // from_env override coverage. Env vars are process-global, so we
    // serialize these tests under a single mutex and a scoped guard
    // that restores prior state on drop — they can run alongside the
    // wider test suite safely.
    // ──────────────────────────────────────────────────────────────────

    use std::sync::Mutex;

    /// Process-wide guard so env-var tests don't race each other or
    /// other tests that read `FORGE_SERVER_*`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that records, then restores, the chosen env vars.
    struct EnvScope {
        previous: Vec<(&'static str, Option<String>)>,
    }
    impl EnvScope {
        fn new(keys: &[&'static str]) -> Self {
            let previous = keys
                .iter()
                .map(|k| (*k, std::env::var(k).ok()))
                .collect::<Vec<_>>();
            // Diagnostic trace — visible under `cargo test -- --nocapture` so a
            // future flake (e.g. a new test that forgets to acquire this scope)
            // is debuggable from the test output alone.
            for (k, original) in &previous {
                eprintln!("[env-scope] clearing {k} (was: {original:?})");
            }
            for k in keys {
                std::env::remove_var(k);
            }
            Self { previous }
        }
    }
    impl Drop for EnvScope {
        fn drop(&mut self) {
            for (k, v) in &self.previous {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    const ALL_KEYS: &[&str] = &[
        "FORGE_SERVER_BIND",
        "FORGE_SERVER_PORT",
        "FORGE_SERVER_TICK_MS",
        "FORGE_SERVER_BROADCAST_CAPACITY",
        "FORGE_SERVER_LOG_FILTER",
        "FORGE_SERVER_ALLOWED_ORIGINS",
    ];

    #[test]
    fn test_from_env_bind_override_valid() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_BIND", "127.0.0.1:7777");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.bind_addr.port(), 7777);
        assert_eq!(cfg.bind_addr.ip().to_string(), "127.0.0.1");
    }

    #[test]
    fn test_from_env_bind_override_invalid_falls_back() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_BIND", "not a socket addr");
        let cfg = ServerConfig::from_env();
        // On parse failure, defaults must apply.
        assert_eq!(cfg.bind_addr, ServerConfig::default().bind_addr);
    }

    #[test]
    fn test_from_env_port_only_override() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_PORT", "9001");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.bind_addr.port(), 9001);
    }

    #[test]
    fn test_from_env_port_invalid_falls_back() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_PORT", "abc");
        let cfg = ServerConfig::from_env();
        assert_eq!(
            cfg.bind_addr.port(),
            forge_types::constants::DEFAULT_SERVER_PORT
        );
    }

    #[test]
    fn test_from_env_bind_takes_precedence_over_port() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_BIND", "127.0.0.1:1111");
        std::env::set_var("FORGE_SERVER_PORT", "2222");
        let cfg = ServerConfig::from_env();
        // BIND wins; PORT is ignored.
        assert_eq!(cfg.bind_addr.port(), 1111);
    }

    #[test]
    fn test_from_env_tick_ms_override_valid() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_TICK_MS", "33");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.tick_interval_ms, 33);
    }

    #[test]
    fn test_from_env_tick_ms_invalid_falls_back() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_TICK_MS", "fast");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.tick_interval_ms, DEFAULT_TICK_INTERVAL_MS);
    }

    #[test]
    fn test_from_env_broadcast_capacity_override_valid() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_BROADCAST_CAPACITY", "512");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.broadcast_capacity, 512);
    }

    #[test]
    fn test_from_env_broadcast_capacity_invalid_falls_back() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_BROADCAST_CAPACITY", "huge");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.broadcast_capacity, DEFAULT_BROADCAST_CAPACITY);
    }

    #[test]
    fn test_from_env_log_filter_override() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_LOG_FILTER", "warn");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.log_filter, "warn");
    }

    #[test]
    fn test_from_env_allowed_origins_parses_comma_separated() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var(
            "FORGE_SERVER_ALLOWED_ORIGINS",
            "https://a.example, https://b.example ,https://c.example",
        );
        let cfg = ServerConfig::from_env();
        assert_eq!(
            cfg.allowed_origins,
            vec![
                "https://a.example".to_string(),
                "https://b.example".to_string(),
                "https://c.example".to_string(),
            ]
        );
    }

    #[test]
    fn test_serde_default_helpers_round_trip_through_empty_toml() {
        // Force every serde-default helper to run by deserializing an
        // empty struct literal — covers the `default_*` free functions.
        let toml_str = ""; // every field uses its serde default
        let cfg: ServerConfig = toml::from_str(toml_str).unwrap();
        let baseline = ServerConfig::default();
        assert_eq!(cfg.bind_addr, baseline.bind_addr);
        assert_eq!(cfg.broadcast_capacity, baseline.broadcast_capacity);
        assert_eq!(cfg.tick_interval_ms, baseline.tick_interval_ms);
        assert_eq!(cfg.log_filter, baseline.log_filter);
        assert_eq!(cfg.allowed_origins, baseline.allowed_origins);
    }
}
