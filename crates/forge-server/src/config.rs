//! Server configuration with sensible defaults and env-var overrides.
//!
//! ## Bind-address default (breaking change)
//!
//! The default bind is **loopback** (`127.0.0.1:8080`). It used to be
//! `0.0.0.0:8080`, which put every mutating endpoint — simulation
//! reset/step, scenario remix, and the history writers — in reach of
//! anything that could route to the port, with no timeout, no body cap,
//! and no authentication in front of them.
//!
//! Deployments that genuinely need to listen on all interfaces set
//! `FORGE_SERVER_BIND=0.0.0.0:8080` explicitly (see
//! [`ServerConfig::from_env`]). Doing so without also setting
//! [`FORGE_SERVER_AUTH_TOKEN`](ServerConfig::auth_token) raises a
//! startup [`SecurityAdvisory`].

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use forge_types::constants::{DEFAULT_FRONTEND_DEV_PORT, DEFAULT_SERVER_PORT};
use serde::{Deserialize, Serialize};

/// Default broadcast channel capacity for WebSocket fan-out.
const DEFAULT_BROADCAST_CAPACITY: usize = 64;
/// Default simulation tick interval in milliseconds.
const DEFAULT_TICK_INTERVAL_MS: u64 = 100;
/// Default tracing filter for the server.
const DEFAULT_LOG_FILTER: &str = "forge_server=info,forge_core=info";
/// Default directory for persisted training/trace history (JSONL files).
const DEFAULT_HISTORY_DIR: &str = "forge-history";
/// Default maximum number of records retained per history file.
const DEFAULT_HISTORY_RETENTION: usize = 10_000;
/// Default maximum number of records returned by a history GET endpoint.
const DEFAULT_HISTORY_QUERY_LIMIT: usize = 500;
/// Default per-request timeout in milliseconds.
///
/// Applied to every HTTP route (not the WebSocket upgrade — see
/// [`crate::routes::build_router`]). Generous enough for a large
/// history query on a cold file, short enough that a stalled client
/// cannot pin a connection indefinitely.
const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 30_000;
/// Default maximum accepted request-body size, in bytes (1 MiB).
///
/// The largest legitimate body is a batch of decision traces; 1 MiB
/// holds thousands of them. Axum's own built-in default is 2 MiB, so
/// this is a tightening, and it is explicit rather than implicit.
const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;

/// A startup security condition worth telling the operator about.
///
/// Produced by [`ServerConfig::security_advisories`] and logged at
/// `warn!` by the binary. Modelled as data rather than inline logging
/// so the conditions are unit-testable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityAdvisory {
    /// No bearer token is configured, so the mutating endpoints accept
    /// any caller that can reach the port.
    NoAuthToken,
    /// The server is bound to a non-loopback address *and* has no
    /// bearer token configured — the mutating endpoints are reachable
    /// from the network unauthenticated.
    PubliclyBoundWithoutAuth,
}

impl SecurityAdvisory {
    /// Operator-facing description of the condition and its remedy.
    pub fn message(self) -> &'static str {
        match self {
            Self::NoAuthToken => {
                "No FORGE_SERVER_AUTH_TOKEN configured: /api/env/reset, /api/env/step, \
                 /api/scenario/remix and the training-metrics / decision-traces writers \
                 accept unauthenticated requests. Set FORGE_SERVER_AUTH_TOKEN to require \
                 an `Authorization: Bearer <token>` header on those routes."
            }
            Self::PubliclyBoundWithoutAuth => {
                "Server is bound to a non-loopback address with no FORGE_SERVER_AUTH_TOKEN \
                 set: every host that can route to this port can reset and step the \
                 simulation, replace the broadcast world, and write records to disk. \
                 Bind to 127.0.0.1 or set FORGE_SERVER_AUTH_TOKEN."
            }
        }
    }
}

/// Configuration for the FORGE server binary.
///
/// All values have sensible defaults and can be overridden via environment
/// variables prefixed with `FORGE_SERVER_`.
///
/// `Debug` is implemented by hand so [`Self::auth_token`] is redacted —
/// the binary logs the whole config at startup.
#[derive(Clone, Serialize, Deserialize)]
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
    /// Directory where training/trace history JSONL files are persisted.
    #[serde(default = "default_history_dir")]
    pub history_dir: PathBuf,
    /// Maximum number of records retained per history file (oldest pruned).
    #[serde(default = "default_history_retention")]
    pub history_retention: usize,
    /// Default maximum number of records returned by a history GET endpoint.
    #[serde(default = "default_history_query_limit")]
    pub history_query_limit: usize,
    /// Per-request timeout in milliseconds, applied to every HTTP route
    /// as a `tower_http` `TimeoutLayer`. A request that has not produced
    /// a response by then is answered `408 Request Timeout`.
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    /// Maximum accepted request-body size in bytes, applied as an axum
    /// `DefaultBodyLimit`. Larger bodies are rejected with
    /// `413 Payload Too Large` before a handler sees them.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Optional bearer token required on the mutating routes.
    ///
    /// `None` (the default) keeps the historical behaviour — those
    /// routes stay open — but raises a [`SecurityAdvisory`] at startup.
    /// When set, a request to a mutating route must carry
    /// `Authorization: Bearer <token>`; otherwise it is rejected with
    /// `401 Unauthorized`. Read-only routes and the WebSocket upgrade
    /// are unaffected.
    #[serde(default)]
    pub auth_token: Option<String>,
}

/// Redacts [`ServerConfig::auth_token`]; everything else is verbatim.
/// `main` logs `info!(?config, ...)` at startup, and a bearer token in
/// the log stream is the same leak as one in a response body.
impl std::fmt::Debug for ServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerConfig")
            .field("bind_addr", &self.bind_addr)
            .field("broadcast_capacity", &self.broadcast_capacity)
            .field("tick_interval_ms", &self.tick_interval_ms)
            .field("log_filter", &self.log_filter)
            .field("allowed_origins", &self.allowed_origins)
            .field("history_dir", &self.history_dir)
            .field("history_retention", &self.history_retention)
            .field("history_query_limit", &self.history_query_limit)
            .field("request_timeout_ms", &self.request_timeout_ms)
            .field("max_body_bytes", &self.max_body_bytes)
            .field(
                "auth_token",
                &self.auth_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Loopback by default — see the module docs for the migration note.
fn default_bind_addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), DEFAULT_SERVER_PORT)
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

fn default_history_dir() -> PathBuf {
    PathBuf::from(DEFAULT_HISTORY_DIR)
}

fn default_history_retention() -> usize {
    DEFAULT_HISTORY_RETENTION
}

fn default_history_query_limit() -> usize {
    DEFAULT_HISTORY_QUERY_LIMIT
}

fn default_request_timeout_ms() -> u64 {
    DEFAULT_REQUEST_TIMEOUT_MS
}

fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: default_bind_addr(),
            broadcast_capacity: DEFAULT_BROADCAST_CAPACITY,
            tick_interval_ms: DEFAULT_TICK_INTERVAL_MS,
            log_filter: DEFAULT_LOG_FILTER.to_string(),
            allowed_origins: default_allowed_origins(),
            history_dir: default_history_dir(),
            history_retention: DEFAULT_HISTORY_RETENTION,
            history_query_limit: DEFAULT_HISTORY_QUERY_LIMIT,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            auth_token: None,
        }
    }
}

impl ServerConfig {
    /// Build a `ServerConfig` from environment variables with defaults.
    ///
    /// Reads:
    /// - `FORGE_SERVER_BIND` — bind address (default `127.0.0.1:8080`;
    ///   set `0.0.0.0:8080` to restore the pre-hardening behaviour)
    /// - `FORGE_SERVER_PORT` — port only override (default `8080`); ignored if `FORGE_SERVER_BIND` is set
    /// - `FORGE_SERVER_TICK_MS` — tick interval in ms (default `100`)
    /// - `FORGE_SERVER_BROADCAST_CAPACITY` — WS channel size (default `64`)
    /// - `FORGE_SERVER_LOG_FILTER` — tracing filter (default `forge_server=info,forge_core=info`)
    /// - `FORGE_SERVER_ALLOWED_ORIGINS` — allowed CORS origins, comma-separated (default `http://localhost:5173`)
    /// - `FORGE_SERVER_HISTORY_DIR` — history JSONL directory (default `forge-history`)
    /// - `FORGE_SERVER_HISTORY_RETENTION` — max records kept per history file (default `10000`)
    /// - `FORGE_SERVER_HISTORY_QUERY_LIMIT` — default GET history limit (default `500`)
    /// - `FORGE_SERVER_REQUEST_TIMEOUT_MS` — per-request timeout in ms (default `30000`)
    /// - `FORGE_SERVER_MAX_BODY_BYTES` — max request body in bytes (default `1048576`)
    /// - `FORGE_SERVER_AUTH_TOKEN` — bearer token required on mutating
    ///   routes (default unset = routes stay open, with a startup warning)
    ///
    /// Values that fail to parse fall back to the default with a
    /// `warn!`. An empty `FORGE_SERVER_AUTH_TOKEN` is treated as unset
    /// rather than as "the empty token", so exporting a blank variable
    /// cannot silently produce an accept-anything credential.
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

        if let Ok(val) = std::env::var("FORGE_SERVER_HISTORY_DIR") {
            config.history_dir = PathBuf::from(val);
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_HISTORY_RETENTION") {
            match val.parse() {
                Ok(n) => config.history_retention = n,
                Err(e) => {
                    tracing::warn!(
                        value = %val,
                        error = %e,
                        "Invalid FORGE_SERVER_HISTORY_RETENTION, using default"
                    );
                }
            }
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_HISTORY_QUERY_LIMIT") {
            match val.parse() {
                Ok(n) => config.history_query_limit = n,
                Err(e) => {
                    tracing::warn!(
                        value = %val,
                        error = %e,
                        "Invalid FORGE_SERVER_HISTORY_QUERY_LIMIT, using default"
                    );
                }
            }
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_REQUEST_TIMEOUT_MS") {
            match val.parse() {
                Ok(ms) => config.request_timeout_ms = ms,
                Err(e) => {
                    tracing::warn!(
                        value = %val,
                        error = %e,
                        "Invalid FORGE_SERVER_REQUEST_TIMEOUT_MS, using default"
                    );
                }
            }
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_MAX_BODY_BYTES") {
            match val.parse() {
                Ok(n) => config.max_body_bytes = n,
                Err(e) => {
                    tracing::warn!(
                        value = %val,
                        error = %e,
                        "Invalid FORGE_SERVER_MAX_BODY_BYTES, using default"
                    );
                }
            }
        }

        if let Ok(val) = std::env::var("FORGE_SERVER_AUTH_TOKEN") {
            // An exported-but-empty variable is a common shell mishap;
            // treating it as a token would mean "Bearer " authenticates.
            if val.is_empty() {
                tracing::warn!("FORGE_SERVER_AUTH_TOKEN is set but empty; treating as unset");
            } else {
                config.auth_token = Some(val);
            }
        }

        tracing::debug!(?config, "Server config loaded from environment");
        config
    }

    /// `true` when the configured bind address is a loopback address
    /// and therefore unreachable from other hosts.
    pub fn binds_loopback(&self) -> bool {
        self.bind_addr.ip().is_loopback()
    }

    /// `true` when bearer-token auth is configured for mutating routes.
    pub fn auth_enabled(&self) -> bool {
        self.auth_token.is_some()
    }

    /// Security conditions the operator should know about at startup.
    ///
    /// Empty when the server is either loopback-bound *and*
    /// token-protected, or token-protected on a public bind. The binary
    /// logs each returned advisory's [`SecurityAdvisory::message`] at
    /// `warn!`.
    pub fn security_advisories(&self) -> Vec<SecurityAdvisory> {
        let mut out = Vec::new();
        if !self.auth_enabled() {
            out.push(SecurityAdvisory::NoAuthToken);
            if !self.binds_loopback() {
                out.push(SecurityAdvisory::PubliclyBoundWithoutAuth);
            }
        }
        out
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
            // is debuggable from the test output alone. Only print when the var
            // was actually set (`is_some()`); otherwise the noise would drown
            // out real pollution signals (10+ tests × 6 keys = 60 spurious
            // lines per test run with --nocapture).
            for (k, original) in &previous {
                if let Some(val) = original {
                    eprintln!("[env-scope] clearing {k} (was: {val:?})");
                }
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
        "FORGE_SERVER_HISTORY_DIR",
        "FORGE_SERVER_HISTORY_RETENTION",
        "FORGE_SERVER_HISTORY_QUERY_LIMIT",
        "FORGE_SERVER_REQUEST_TIMEOUT_MS",
        "FORGE_SERVER_MAX_BODY_BYTES",
        "FORGE_SERVER_AUTH_TOKEN",
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
    fn test_history_defaults() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.history_dir, default_history_dir());
        assert_eq!(cfg.history_retention, DEFAULT_HISTORY_RETENTION);
        assert_eq!(cfg.history_query_limit, DEFAULT_HISTORY_QUERY_LIMIT);
    }

    #[test]
    fn test_from_env_history_overrides() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_HISTORY_DIR", "/tmp/forge-hist");
        std::env::set_var("FORGE_SERVER_HISTORY_RETENTION", "42");
        std::env::set_var("FORGE_SERVER_HISTORY_QUERY_LIMIT", "7");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.history_dir, PathBuf::from("/tmp/forge-hist"));
        assert_eq!(cfg.history_retention, 42);
        assert_eq!(cfg.history_query_limit, 7);
    }

    #[test]
    fn test_from_env_history_invalid_falls_back() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_HISTORY_RETENTION", "not-a-number");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.history_retention, DEFAULT_HISTORY_RETENTION);
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

    // ──────────────────────────────────────────────────────────────────
    // Network-hardening defaults + overrides.
    // ──────────────────────────────────────────────────────────────────

    /// The deliberate breaking default change: loopback, not 0.0.0.0.
    #[test]
    fn default_bind_is_loopback() {
        let cfg = ServerConfig::default();
        assert!(
            cfg.binds_loopback(),
            "default bind must be loopback; got {}",
            cfg.bind_addr
        );
        assert_eq!(cfg.bind_addr.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_ne!(cfg.bind_addr.ip(), IpAddr::V4(Ipv4Addr::UNSPECIFIED));
    }

    /// Operators can still restore the old behaviour explicitly.
    #[test]
    fn bind_env_override_can_restore_all_interfaces() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_BIND", "0.0.0.0:8080");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.bind_addr.ip(), IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        assert!(!cfg.binds_loopback());
    }

    #[test]
    fn hardening_defaults() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.request_timeout_ms, DEFAULT_REQUEST_TIMEOUT_MS);
        assert_eq!(cfg.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
        assert!(cfg.auth_token.is_none());
        assert!(!cfg.auth_enabled());
    }

    #[test]
    fn from_env_hardening_overrides() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_REQUEST_TIMEOUT_MS", "1500");
        std::env::set_var("FORGE_SERVER_MAX_BODY_BYTES", "2048");
        std::env::set_var("FORGE_SERVER_AUTH_TOKEN", "s3cret");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.request_timeout_ms, 1500);
        assert_eq!(cfg.max_body_bytes, 2048);
        assert_eq!(cfg.auth_token.as_deref(), Some("s3cret"));
        assert!(cfg.auth_enabled());
    }

    #[test]
    fn from_env_hardening_invalid_values_fall_back() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_REQUEST_TIMEOUT_MS", "soon");
        std::env::set_var("FORGE_SERVER_MAX_BODY_BYTES", "-1");
        let cfg = ServerConfig::from_env();
        assert_eq!(cfg.request_timeout_ms, DEFAULT_REQUEST_TIMEOUT_MS);
        assert_eq!(cfg.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
    }

    /// An exported-but-blank token must not become an accept-anything
    /// credential (`Authorization: Bearer ` would otherwise match).
    #[test]
    fn from_env_empty_auth_token_is_treated_as_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _scope = EnvScope::new(ALL_KEYS);
        std::env::set_var("FORGE_SERVER_AUTH_TOKEN", "");
        let cfg = ServerConfig::from_env();
        assert!(cfg.auth_token.is_none());
        assert!(!cfg.auth_enabled());
    }

    #[test]
    fn advisories_flag_missing_token_on_loopback() {
        let cfg = ServerConfig::default();
        assert_eq!(
            cfg.security_advisories(),
            vec![SecurityAdvisory::NoAuthToken]
        );
    }

    #[test]
    fn advisories_escalate_for_public_bind_without_auth() {
        let cfg = ServerConfig {
            bind_addr: "0.0.0.0:8080".parse().unwrap(),
            ..ServerConfig::default()
        };
        assert_eq!(
            cfg.security_advisories(),
            vec![
                SecurityAdvisory::NoAuthToken,
                SecurityAdvisory::PubliclyBoundWithoutAuth
            ]
        );
    }

    #[test]
    fn advisories_empty_when_token_configured() {
        let cfg = ServerConfig {
            bind_addr: "0.0.0.0:8080".parse().unwrap(),
            auth_token: Some("token".to_string()),
            ..ServerConfig::default()
        };
        assert!(cfg.security_advisories().is_empty());
    }

    #[test]
    fn advisory_messages_are_actionable() {
        for advisory in [
            SecurityAdvisory::NoAuthToken,
            SecurityAdvisory::PubliclyBoundWithoutAuth,
        ] {
            assert!(
                advisory.message().contains("FORGE_SERVER_AUTH_TOKEN"),
                "advisory {advisory:?} must name the remedy env var"
            );
        }
    }

    /// The startup log prints the whole config; the token must not be
    /// in it.
    #[test]
    fn debug_redacts_auth_token() {
        let cfg = ServerConfig {
            auth_token: Some("super-secret-value".to_string()),
            ..ServerConfig::default()
        };
        let debug = format!("{cfg:?}");
        assert!(
            !debug.contains("super-secret-value"),
            "auth token leaked into Debug output: {debug}"
        );
        assert!(debug.contains("<redacted>"), "got: {debug}");
        // Non-secret fields are still visible for diagnostics.
        assert!(debug.contains("ServerConfig"));
        assert!(debug.contains("request_timeout_ms"));
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
        // The hardening fields are `#[serde(default)]` too, so configs
        // written before they existed keep parsing.
        assert_eq!(cfg.request_timeout_ms, baseline.request_timeout_ms);
        assert_eq!(cfg.max_body_bytes, baseline.max_body_bytes);
        assert_eq!(cfg.auth_token, baseline.auth_token);
    }
}
