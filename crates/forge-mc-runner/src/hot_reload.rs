//! [`HotReloadWatcher`] — observes [`ModelManifest`] for monotonic
//! version bumps and emits a [`ReloadEvent`] when one lands.
//!
//! ## Discipline
//!
//! The watcher **must only be polled between episodes** (plan §3.4).
//! Calling `poll` mid-episode is not unsafe but breaks the runner's
//! lock-order story: model swaps must happen when no inference call
//! is in flight. The watcher itself has no way to enforce this; the
//! `Runner` puts `poll` calls at the top of its outer episode loop.
//!
//! ## Semantics
//!
//! - If the manifest file is *missing*: returns `Ok(None)`, no error.
//!   First-run callers handle missing-manifest via the `bootstrap`
//!   command, not this watcher.
//! - If the manifest exists and parses but `version <= last_seen`:
//!   returns `Ok(None)` (no event).
//! - If `version > last_seen`: returns `Ok(Some(ReloadEvent))` and
//!   advances `last_seen`. The same version is *not* re-emitted.

use std::path::PathBuf;

use tracing::{debug, instrument};

use crate::error::RunnerError;
use crate::manifest::ModelManifest;

/// What the watcher emits when a fresh manifest version lands.
///
/// Returned by [`HotReloadWatcher::poll`] for the caller to feed into
/// `OnnxMuZeroModel::reload` (Phase 4 plumbing).
#[derive(Debug, Clone, PartialEq)]
pub struct ReloadEvent {
    /// New version observed on disk.
    pub new_version: u64,
    /// The previous version the watcher had cached, or `None` for the
    /// first observed event.
    pub previous_version: Option<u64>,
    /// The freshly-loaded manifest.
    pub manifest: ModelManifest,
}

/// Polls `model_manifest.json` for monotonic version bumps.
///
/// Construct with the manifest path. The watcher carries the last
/// successfully-observed `version` so subsequent identical polls return
/// `Ok(None)`.
#[derive(Debug, Clone)]
pub struct HotReloadWatcher {
    manifest_path: PathBuf,
    last_seen_version: Option<u64>,
}

impl HotReloadWatcher {
    /// Build a fresh watcher for the given manifest path.
    pub fn new(manifest_path: impl Into<PathBuf>) -> Self {
        Self {
            manifest_path: manifest_path.into(),
            last_seen_version: None,
        }
    }

    /// The path the watcher is observing.
    pub fn manifest_path(&self) -> &std::path::Path {
        &self.manifest_path
    }

    /// The last version the watcher emitted, if any.
    pub fn last_seen_version(&self) -> Option<u64> {
        self.last_seen_version
    }

    /// Pre-seed the watcher with a `version` to ignore — useful when
    /// the runner starts up against an already-bootstrapped manifest
    /// and the first reload should only fire on the *next* trainer
    /// export, not the one already on disk.
    pub fn prime_with(&mut self, version: u64) {
        self.last_seen_version = Some(version);
    }

    /// Poll for a fresh manifest version.
    ///
    /// **Contract:** call only between episodes. Calling mid-episode
    /// breaks the Phase 4 lock-order story (see
    /// `OnnxMuZeroModel::reload` rationale in plan §3.4).
    ///
    /// Returns:
    /// - `Ok(None)` if the manifest is missing or its version is not
    ///   strictly greater than the cached `last_seen_version`.
    /// - `Ok(Some(ReloadEvent))` and advances `last_seen_version`
    ///   when a strictly-greater version lands.
    /// - `Err(RunnerError)` for parse / validation errors only.
    #[instrument(skip(self), fields(path = %self.manifest_path.display()))]
    pub fn poll(&mut self) -> Result<Option<ReloadEvent>, RunnerError> {
        if !self.manifest_path.exists() {
            return Ok(None);
        }
        let m = ModelManifest::load_json(&self.manifest_path)?;
        let strictly_greater = match self.last_seen_version {
            Some(prev) => m.version > prev,
            None => true,
        };
        if !strictly_greater {
            debug!(
                seen = ?self.last_seen_version,
                got = m.version,
                "manifest version not greater; no reload"
            );
            return Ok(None);
        }
        let previous_version = self.last_seen_version;
        self.last_seen_version = Some(m.version);
        debug!(
            previous = ?previous_version,
            new = m.version,
            "manifest version bump; emitting reload event"
        );
        Ok(Some(ReloadEvent {
            new_version: m.version,
            previous_version,
            manifest: m,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelFileEntry, ModelManifestFiles, MANIFEST_SCHEMA_VERSION};

    /// Helper: build a writable manifest at `path` with the given version.
    fn write_manifest_at(path: &std::path::Path, version: u64) {
        let m = ModelManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            version,
            schema_id: "sid".into(),
            created_at: "2026-05-17T00:00:00Z".into(),
            files: ModelManifestFiles {
                representation: ModelFileEntry {
                    path: "r.onnx".into(),
                    sha256: "a".repeat(64),
                },
                dynamics: ModelFileEntry {
                    path: "d.onnx".into(),
                    sha256: "b".repeat(64),
                },
                prediction: ModelFileEntry {
                    path: "p.onnx".into(),
                    sha256: "c".repeat(64),
                },
            },
        };
        m.save_json(path).unwrap();
    }

    #[test]
    fn missing_manifest_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let mut w = HotReloadWatcher::new(tmp.path().join("no_such.json"));
        assert!(w.poll().unwrap().is_none());
        assert!(w.last_seen_version().is_none());
    }

    #[test]
    fn first_poll_after_write_emits_event_with_no_previous() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        write_manifest_at(&path, 1);
        let mut w = HotReloadWatcher::new(&path);
        let ev = w.poll().unwrap().expect("expected reload event");
        assert_eq!(ev.new_version, 1);
        assert!(ev.previous_version.is_none());
        assert_eq!(w.last_seen_version(), Some(1));
    }

    #[test]
    fn second_poll_at_same_version_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        write_manifest_at(&path, 1);
        let mut w = HotReloadWatcher::new(&path);
        let _ = w.poll().unwrap().unwrap();
        assert!(w.poll().unwrap().is_none(), "no duplicate event");
    }

    #[test]
    fn poll_after_version_bump_emits_event_with_previous() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        write_manifest_at(&path, 2);
        let mut w = HotReloadWatcher::new(&path);
        let _ = w.poll().unwrap().unwrap(); // observe v2
        write_manifest_at(&path, 5);
        let ev = w.poll().unwrap().expect("expected event on bump");
        assert_eq!(ev.previous_version, Some(2));
        assert_eq!(ev.new_version, 5);
        assert_eq!(w.last_seen_version(), Some(5));
    }

    #[test]
    fn poll_with_lower_version_returns_none_and_does_not_advance() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        write_manifest_at(&path, 10);
        let mut w = HotReloadWatcher::new(&path);
        let _ = w.poll().unwrap().unwrap();
        // Overwrite with a lower version — tooling regression, runner
        // must NOT downgrade.
        write_manifest_at(&path, 3);
        assert!(w.poll().unwrap().is_none());
        assert_eq!(w.last_seen_version(), Some(10));
    }

    #[test]
    fn prime_with_suppresses_initial_event_for_already_observed_version() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        write_manifest_at(&path, 4);
        let mut w = HotReloadWatcher::new(&path);
        w.prime_with(4);
        assert!(w.poll().unwrap().is_none());
        // ... but a later bump still fires.
        write_manifest_at(&path, 5);
        let ev = w.poll().unwrap().unwrap();
        assert_eq!(ev.new_version, 5);
        assert_eq!(ev.previous_version, Some(4));
    }

    #[test]
    fn invalid_manifest_surface_as_runner_error_not_none() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        std::fs::write(&path, b"{not valid json").unwrap();
        let mut w = HotReloadWatcher::new(&path);
        let err = w.poll().unwrap_err();
        // The watcher must propagate the JSON error rather than silently
        // hiding manifest corruption as "no update".
        assert!(matches!(err, RunnerError::Json(_)));
        // last_seen_version stays None: bad parse doesn't poison the
        // watcher's clock.
        assert!(w.last_seen_version().is_none());
    }

    #[test]
    fn manifest_path_accessor_returns_constructor_arg() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        let w = HotReloadWatcher::new(&path);
        assert_eq!(w.manifest_path(), path.as_path());
    }
}
