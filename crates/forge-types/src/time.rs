//! Shared wall-clock helpers.
//!
//! A single `now_ms` implementation used across crates that need to stamp
//! records with a capture time (history stores, MLflow exporters, worker
//! registries) — consolidates what were previously several independent
//! `SystemTime`/`chrono` reimplementations of the same millisecond-epoch
//! conversion.

use std::time::{SystemTime, UNIX_EPOCH};

/// Current wall-clock time in milliseconds since the Unix epoch.
///
/// Saturates to `0` if the system clock is set before the epoch, rather
/// than panicking — callers use this for best-effort record timestamps,
/// not for anything security- or ordering-critical.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
