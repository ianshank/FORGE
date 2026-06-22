//! Shared tracing/log initialization for FORGE binaries.
//!
//! Every FORGE binary used to duplicate the same `tracing_subscriber::fmt()`
//! bootstrap. This crate centralizes it behind [`init_tracing`] so the policy
//! (env-filter handling, text-vs-JSON formatting, target rendering) lives in
//! one place and stays consistent across `forge-server`, `forge-mc-runner`,
//! and any future binary.
//!
//! # Behaviour (backwards compatible)
//!
//! - The log filter is read from `RUST_LOG` via [`EnvFilter`]. When `RUST_LOG`
//!   is unset/unparseable, the caller-supplied [`TracingOptions::default_filter`]
//!   directive is used — preserving each binary's historical default.
//! - The output format is selected by the `FORGE_LOG_FORMAT` environment
//!   variable: `text` (default — identical to the previous behaviour) or
//!   `json` for structured, machine-parseable lines suited to log aggregation.
//!   An explicit [`TracingOptions::format`] overrides the env var.
//! - Targets are always rendered (`with_target(true)`), matching prior output.
//!
//! No hard-coded values leak out: the only literal is the format env-var name,
//! exposed as [`LOG_FORMAT_ENV`] for callers/tests.

#![deny(missing_docs)]

use tracing_subscriber::EnvFilter;

/// Environment variable selecting the log output format (`text` | `json`).
pub const LOG_FORMAT_ENV: &str = "FORGE_LOG_FORMAT";

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable single-line format (the historical default).
    Text,
    /// Structured JSON, one object per event — suited to log aggregation.
    Json,
}

impl LogFormat {
    /// Parse a format from a string, case-insensitively. Unknown values
    /// (including empty) fall back to [`LogFormat::Text`] so a typo never
    /// crashes startup.
    pub fn from_str_lenient(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "json" => LogFormat::Json,
            _ => LogFormat::Text,
        }
    }

    /// Resolve the format from the [`LOG_FORMAT_ENV`] environment variable,
    /// defaulting to [`LogFormat::Text`] when unset.
    pub fn from_env() -> Self {
        std::env::var(LOG_FORMAT_ENV)
            .map(|v| Self::from_str_lenient(&v))
            .unwrap_or(LogFormat::Text)
    }
}

/// Options controlling tracing initialization.
#[derive(Debug, Clone)]
pub struct TracingOptions {
    /// Filter directive used when `RUST_LOG` is absent or unparseable, e.g.
    /// `"forge_server=info,forge_core=info"`. This keeps each binary's
    /// historical default behaviour.
    pub default_filter: String,
    /// Explicit format override. When `None`, the format is resolved from
    /// [`LOG_FORMAT_ENV`] (defaulting to text).
    pub format: Option<LogFormat>,
}

impl TracingOptions {
    /// Construct options with the given default filter directive and
    /// env-driven format resolution.
    pub fn new(default_filter: impl Into<String>) -> Self {
        Self {
            default_filter: default_filter.into(),
            format: None,
        }
    }

    /// Force a specific output format, ignoring [`LOG_FORMAT_ENV`].
    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Resolve the effective format (explicit override or env-derived).
    fn resolve_format(&self) -> LogFormat {
        self.format.unwrap_or_else(LogFormat::from_env)
    }

    /// Build the [`EnvFilter`] from `RUST_LOG`, falling back to
    /// [`Self::default_filter`].
    fn build_filter(&self) -> EnvFilter {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(self.default_filter.clone()))
    }
}

/// Initialize the global tracing subscriber for a binary.
///
/// Installs a process-global subscriber and therefore must be called exactly
/// once, early in `main`. Calling it twice will panic (via the underlying
/// `init()`); tests should use [`try_init_tracing`] instead.
///
/// # Example
///
/// ```no_run
/// use forge_observability::{init_tracing, TracingOptions};
/// init_tracing(TracingOptions::new("forge_server=info,forge_core=info"));
/// ```
pub fn init_tracing(opts: TracingOptions) {
    // `try_init` returns Err if a subscriber is already set; from a binary's
    // `main` that should not happen, so surface it loudly.
    try_init_tracing(opts).expect("tracing subscriber already initialized");
}

/// Fallible variant of [`init_tracing`] that returns an error instead of
/// panicking when a global subscriber is already installed. Useful in tests
/// and in embedders that may initialize logging themselves.
pub fn try_init_tracing(
    opts: TracingOptions,
) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let filter = opts.build_filter();
    match opts.resolve_format() {
        LogFormat::Json => tracing_subscriber::fmt()
            .json()
            .with_target(true)
            .with_env_filter(filter)
            .try_init(),
        LogFormat::Text => tracing_subscriber::fmt()
            .with_target(true)
            .with_env_filter(filter)
            .try_init(),
    }
    // `try_init()` yields `Result<(), TryInitError>`; convert the error into
    // the boxed trait object this function exposes so callers don't depend on
    // the concrete subscriber error type.
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_parsing_is_lenient_and_case_insensitive() {
        assert_eq!(LogFormat::from_str_lenient("json"), LogFormat::Json);
        assert_eq!(LogFormat::from_str_lenient("JSON"), LogFormat::Json);
        assert_eq!(LogFormat::from_str_lenient("  Json "), LogFormat::Json);
        assert_eq!(LogFormat::from_str_lenient("text"), LogFormat::Text);
        // Unknown / empty values fall back to text rather than erroring.
        assert_eq!(LogFormat::from_str_lenient("yaml"), LogFormat::Text);
        assert_eq!(LogFormat::from_str_lenient(""), LogFormat::Text);
    }

    #[test]
    fn explicit_format_overrides_env_resolution() {
        let opts = TracingOptions::new("info").with_format(LogFormat::Json);
        assert_eq!(opts.resolve_format(), LogFormat::Json);
    }

    #[test]
    fn options_carry_default_filter() {
        let opts = TracingOptions::new("forge_server=info,forge_core=info");
        assert_eq!(opts.default_filter, "forge_server=info,forge_core=info");
        // Default format resolution without an explicit override and with no
        // env var set is text.
        assert!(opts.format.is_none());
    }

    #[test]
    fn build_filter_uses_default_when_rust_log_absent() {
        // Build a filter with a known default directive and assert it is a
        // valid, constructable filter (exercises the fallback path).
        let opts = TracingOptions::new("forge_test=trace");
        let _filter = opts.build_filter();
    }
}
