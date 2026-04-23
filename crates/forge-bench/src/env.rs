//! Environment-variable-driven benchmark configuration.
//!
//! Every bench target in this crate — plus the `allocation_audit` binary —
//! honours the same override surface so callers on Reference Hardware A or
//! B can point the harness at different sizes without recompilation.
//!
//! | Variable | Meaning | Parser |
//! |---|---|---|
//! | `FORGE_BENCH_WORLD` | World side length (tiles) | [`world_side`] |
//! | `FORGE_BENCH_SEED` | Deterministic RNG seed | [`seed_u64`] |
//! | `FORGE_BENCH_AGENT_COUNTS` | Comma-separated agent counts | [`agent_counts`] |
//!
//! Parsers return the provided default when the variable is unset or
//! unparseable, and emit a `tracing::warn!` on validation failure so
//! regressions surface when `RUST_LOG` is enabled.

use std::env;

use tracing::warn;

/// Environment variable for overriding the benchmark world side length.
pub const ENV_WORLD: &str = "FORGE_BENCH_WORLD";
/// Environment variable for overriding the comma-separated agent count sweep.
pub const ENV_AGENT_COUNTS: &str = "FORGE_BENCH_AGENT_COUNTS";
/// Environment variable for overriding the deterministic RNG seed.
pub const ENV_SEED: &str = "FORGE_BENCH_SEED";

/// Shared bench-level deterministic seed. Matches the value used in
/// `step_throughput.rs` so new bench files stay consistent with the
/// pre-existing reproducibility convention.
pub const BENCH_SEED: u64 = 42;

/// Reads a non-zero `u16` from `var`, falling back to `default` if unset or
/// unparseable. A warning is emitted when an override is rejected.
pub fn u16_from_env(var: &str, default: u16) -> u16 {
    match env::var(var) {
        Ok(raw) => match raw.parse::<u16>() {
            Ok(v) if v > 0 => v,
            _ => {
                warn!(
                    env = var,
                    value = %raw,
                    default,
                    "invalid u16 override; using default"
                );
                default
            }
        },
        Err(_) => default,
    }
}

/// Reads a `u64` from `var`, falling back to `default` if unset or unparseable.
pub fn u64_from_env(var: &str, default: u64) -> u64 {
    match env::var(var) {
        Ok(raw) => match raw.parse::<u64>() {
            Ok(v) => v,
            Err(_) => {
                warn!(
                    env = var,
                    value = %raw,
                    default,
                    "invalid u64 override; using default"
                );
                default
            }
        },
        Err(_) => default,
    }
}

/// Convenience: reads the world side length with the documented default.
pub fn world_side(default: u16) -> u16 {
    u16_from_env(ENV_WORLD, default)
}

/// Convenience: reads the deterministic seed with the documented default.
pub fn seed_u64(default: u64) -> u64 {
    u64_from_env(ENV_SEED, default)
}

/// Parses the comma-separated agent count sweep from `FORGE_BENCH_AGENT_COUNTS`.
///
/// Invalid tokens are dropped with a warning. Returns the caller-supplied
/// `default` list if the variable is unset or every token fails to parse.
pub fn agent_counts(default: &[u32]) -> Vec<u32> {
    let Ok(raw) = env::var(ENV_AGENT_COUNTS) else {
        return default.to_vec();
    };
    let parsed: Vec<u32> = raw
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            match trimmed.parse::<u32>() {
                Ok(n) if n > 0 => Some(n),
                _ => {
                    warn!(
                        env = ENV_AGENT_COUNTS,
                        token = trimmed,
                        "ignoring invalid agent-count token"
                    );
                    None
                }
            }
        })
        .collect();
    if parsed.is_empty() {
        warn!(
            env = ENV_AGENT_COUNTS,
            value = %raw,
            "no valid agent counts; using default"
        );
        default.to_vec()
    } else {
        parsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The tests here mutate process-wide environment state. `ENV_LOCK`
    // serialises only the tests inside this module that acquire it — it
    // does not prevent concurrent env-var access from tests elsewhere in
    // the workspace. State restoration is the responsibility of
    // `EnvironmentGuard` below, which mirrors the RAII pattern in
    // `crates/forge-types/src/config.rs:822`.
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const PROBE_WORLD: &str = "FORGE_BENCH_WORLD_TEST";
    const PROBE_SEED: &str = "FORGE_BENCH_SEED_TEST";
    const PROBE_AGENTS: &str = "FORGE_BENCH_AGENTS_TEST";

    /// RAII guard that snapshots an env var on construction and restores
    /// it on drop — including restoration after a panic. Pattern lifted
    /// from `forge_types::config` tests so this crate's tests do not
    /// leak env state into sibling tests sharing the same process.
    struct EnvironmentGuard {
        var_name: &'static str,
        original_value: Option<String>,
    }

    impl EnvironmentGuard {
        fn new(var_name: &'static str) -> Self {
            Self {
                var_name,
                original_value: env::var(var_name).ok(),
            }
        }

        fn set(&self, value: &str) {
            env::set_var(self.var_name, value);
        }

        fn remove(&self) {
            env::remove_var(self.var_name);
        }
    }

    impl Drop for EnvironmentGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.original_value {
                env::set_var(self.var_name, value);
            } else {
                env::remove_var(self.var_name);
            }
        }
    }

    #[test]
    fn u16_from_env_returns_default_when_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvironmentGuard::new(PROBE_WORLD);
        guard.remove();
        assert_eq!(u16_from_env(PROBE_WORLD, 128), 128);
    }

    #[test]
    fn u16_from_env_parses_valid_override() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvironmentGuard::new(PROBE_WORLD);
        guard.set("256");
        assert_eq!(u16_from_env(PROBE_WORLD, 128), 256);
    }

    #[test]
    fn u16_from_env_falls_back_on_zero() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvironmentGuard::new(PROBE_WORLD);
        guard.set("0");
        assert_eq!(u16_from_env(PROBE_WORLD, 64), 64);
    }

    #[test]
    fn u16_from_env_falls_back_on_garbage() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvironmentGuard::new(PROBE_WORLD);
        guard.set("not-a-number");
        assert_eq!(u16_from_env(PROBE_WORLD, 32), 32);
    }

    #[test]
    fn u64_from_env_parses_and_falls_back() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvironmentGuard::new(PROBE_SEED);
        guard.set("99");
        assert_eq!(u64_from_env(PROBE_SEED, 1), 99);
        guard.set("bad");
        assert_eq!(u64_from_env(PROBE_SEED, 1), 1);
    }

    #[test]
    fn agent_counts_returns_default_when_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        let probe_guard = EnvironmentGuard::new(PROBE_AGENTS);
        let real_guard = EnvironmentGuard::new(ENV_AGENT_COUNTS);
        probe_guard.remove();
        real_guard.remove();
        assert_eq!(agent_counts(&[1, 2, 4]), vec![1, 2, 4]);
    }

    #[test]
    fn agent_counts_parses_comma_separated() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvironmentGuard::new(ENV_AGENT_COUNTS);
        guard.set("1, 8,16, 64 ");
        assert_eq!(agent_counts(&[99]), vec![1, 8, 16, 64]);
    }

    #[test]
    fn agent_counts_drops_invalid_tokens() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvironmentGuard::new(ENV_AGENT_COUNTS);
        guard.set("1,abc,0,-3,4");
        assert_eq!(agent_counts(&[99]), vec![1, 4]);
    }

    #[test]
    fn agent_counts_falls_back_when_all_invalid() {
        let _g = ENV_LOCK.lock().unwrap();
        let guard = EnvironmentGuard::new(ENV_AGENT_COUNTS);
        guard.set("abc,def,0");
        assert_eq!(agent_counts(&[7]), vec![7]);
    }
}
