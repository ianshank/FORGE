//! [`ModelManifest`] — the hot-reload swap signal.
//!
//! The manifest lives next to the three ONNX files
//! (`representation.onnx`, `dynamics.onnx`, `prediction.onnx`) and
//! carries:
//!
//! - `schema_version` — pinned constant; bumping is a breaking change.
//! - `version` — strictly monotonic counter; bumped per trainer export.
//! - `schema_id` — sha256 the runner cross-checks against the env's
//!   `Hello.schema_id`. Mismatch = action/reward space drift, hard fail.
//! - `files.<role>.sha256` — content hash recorded by the trainer at
//!   export time. NOTE: [`ModelManifest::validate`] only checks that
//!   this field is non-empty; it does **not** re-hash the file on
//!   disk, so a corrupted bundle is not detected here. Partial-write
//!   protection comes from the atomic rename below, not from this hash.
//!
//! ## Atomic write
//!
//! [`ModelManifest::save_json`] writes to a `.tmp` sibling then renames
//! into place. This matches the trainer-side
//! `python/forge/training/muzero_mc/exporter.py` discipline so the
//! runner never observes a half-written manifest.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::error::RunnerError;

/// Current `schema_version` value for this crate's manifest.
///
/// Bumping is a breaking change — older readers will refuse the file
/// with [`RunnerError::ManifestSchemaMismatch`].
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Per-role file entry inside a manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelFileEntry {
    /// Path on disk, typically relative to the manifest's directory.
    pub path: String,
    /// sha256 hex digest of the file contents. The runner can recompute
    /// this and reject the bundle on mismatch.
    pub sha256: String,
}

/// The three files a MuZero bundle ships, by role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifestFiles {
    /// Observation → latent (the representation network).
    pub representation: ModelFileEntry,
    /// (latent, action) → next latent + reward (the dynamics network).
    pub dynamics: ModelFileEntry,
    /// Latent → policy logits + value (the prediction network).
    pub prediction: ModelFileEntry,
}

/// A model bundle manifest. One file per trained checkpoint export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// MUST equal [`MANIFEST_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Monotonically increasing per-export version counter.
    pub version: u64,
    /// sha256 of canonical (env_id, obs_dim, action_count, action_map,
    /// reward_config) — runner cross-checks against env `Hello.schema_id`.
    pub schema_id: String,
    /// RFC3339 timestamp the trainer set at export.
    pub created_at: String,
    /// The three ONNX file entries.
    pub files: ModelManifestFiles,
}

impl ModelManifest {
    /// Construct a new manifest with the compiled `schema_version`.
    pub fn new(
        version: u64,
        schema_id: impl Into<String>,
        created_at: impl Into<String>,
        files: ModelManifestFiles,
    ) -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            version,
            schema_id: schema_id.into(),
            created_at: created_at.into(),
            files,
        }
    }

    /// Cheap invariant check. Doesn't touch the filesystem.
    pub fn validate(&self) -> Result<(), RunnerError> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(RunnerError::ManifestSchemaMismatch {
                expected: MANIFEST_SCHEMA_VERSION,
                got: self.schema_version,
            });
        }
        if self.version == 0 {
            return Err(RunnerError::InvalidManifest(
                "version must be >= 1 (0 reserved for 'pre-bootstrap')".into(),
            ));
        }
        if self.schema_id.is_empty() {
            return Err(RunnerError::InvalidManifest(
                "schema_id must be non-empty".into(),
            ));
        }
        for (role, entry) in [
            ("representation", &self.files.representation),
            ("dynamics", &self.files.dynamics),
            ("prediction", &self.files.prediction),
        ] {
            if entry.path.is_empty() {
                return Err(RunnerError::InvalidManifest(format!(
                    "files.{role}.path must be non-empty"
                )));
            }
            if entry.sha256.is_empty() {
                return Err(RunnerError::InvalidManifest(format!(
                    "files.{role}.sha256 must be non-empty"
                )));
            }
        }
        Ok(())
    }

    /// Load a manifest JSON file. Validates after parse.
    #[instrument(skip_all, fields(path = %path.as_ref().display()))]
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, RunnerError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|e| RunnerError::io(path, e))?;
        let m: ModelManifest = serde_json::from_slice(&bytes)?;
        m.validate()?;
        debug!(version = m.version, "manifest loaded");
        Ok(m)
    }

    /// Save the manifest atomically (tmp file + rename in same dir).
    /// Validates before writing — a failing manifest never reaches disk.
    #[instrument(skip(self), fields(path = %path.as_ref().display(), version = self.version))]
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), RunnerError> {
        self.validate()?;
        let path = path.as_ref();
        let dir = path.parent().ok_or_else(|| {
            RunnerError::InvalidManifest(format!(
                "manifest path has no parent dir: {}",
                path.display()
            ))
        })?;
        // `path.parent()` returns `Some("")` for bare filenames like
        // `"model_manifest.json"`; `create_dir_all("")` errors on Windows,
        // so only create when there's a real directory component.
        if !dir.as_os_str().is_empty() && !dir.exists() {
            fs::create_dir_all(dir).map_err(|e| RunnerError::io(dir, e))?;
        }
        // Use the dotted-tmp pattern so a crash mid-write leaves a hidden
        // sibling, not a malformed manifest the watcher might pick up.
        let tmp: PathBuf = dir.join(format!(
            ".{}.tmp",
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model_manifest.json".into())
        ));
        let bytes = serde_json::to_vec_pretty(self)?;
        fs::write(&tmp, &bytes).map_err(|e| RunnerError::io(&tmp, e))?;
        fs::rename(&tmp, path).map_err(|e| RunnerError::io(path, e))?;
        debug!("manifest written");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a syntactically valid manifest for tests.
    fn sample_manifest(version: u64, schema_id: &str) -> ModelManifest {
        ModelManifest::new(
            version,
            schema_id,
            "2026-05-17T00:00:00Z",
            ModelManifestFiles {
                representation: ModelFileEntry {
                    path: "representation.onnx".into(),
                    sha256: "a".repeat(64),
                },
                dynamics: ModelFileEntry {
                    path: "dynamics.onnx".into(),
                    sha256: "b".repeat(64),
                },
                prediction: ModelFileEntry {
                    path: "prediction.onnx".into(),
                    sha256: "c".repeat(64),
                },
            },
        )
    }

    #[test]
    fn new_uses_pinned_schema_version() {
        let m = sample_manifest(1, "abc");
        assert_eq!(m.schema_version, MANIFEST_SCHEMA_VERSION);
    }

    #[test]
    fn validate_accepts_well_formed_manifest() {
        let m = sample_manifest(3, "abc");
        m.validate().unwrap();
    }

    #[test]
    fn validate_rejects_schema_version_drift() {
        let mut m = sample_manifest(1, "abc");
        m.schema_version = 99;
        let err = m.validate().unwrap_err();
        assert!(matches!(err, RunnerError::ManifestSchemaMismatch { .. }));
    }

    #[test]
    fn validate_rejects_version_zero() {
        let m = sample_manifest(0, "abc");
        let err = m.validate().unwrap_err();
        assert!(matches!(err, RunnerError::InvalidManifest(_)));
    }

    #[test]
    fn validate_rejects_empty_schema_id() {
        let m = sample_manifest(1, "");
        let err = m.validate().unwrap_err();
        assert!(matches!(err, RunnerError::InvalidManifest(_)));
    }

    #[test]
    fn validate_rejects_empty_file_path() {
        let mut m = sample_manifest(1, "abc");
        m.files.representation.path = String::new();
        let err = m.validate().unwrap_err();
        match err {
            RunnerError::InvalidManifest(msg) => assert!(msg.contains("representation")),
            other => panic!("expected InvalidManifest, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_empty_file_sha() {
        let mut m = sample_manifest(1, "abc");
        m.files.dynamics.sha256 = String::new();
        let err = m.validate().unwrap_err();
        match err {
            RunnerError::InvalidManifest(msg) => assert!(msg.contains("dynamics")),
            other => panic!("expected InvalidManifest, got {other:?}"),
        }
    }

    #[test]
    fn save_then_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        let m = sample_manifest(2, "abcd");
        m.save_json(&path).unwrap();
        let back = ModelManifest::load_json(&path).unwrap();
        assert_eq!(m, back);
    }

    /// The .tmp sibling must not remain after a successful save.
    /// This is what guarantees atomic visibility — the watcher only ever
    /// sees the final `model_manifest.json`.
    #[test]
    fn save_cleans_up_tmp_after_success() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        let m = sample_manifest(1, "abc");
        m.save_json(&path).unwrap();
        let leftover_tmp = tmp.path().join(".model_manifest.json.tmp");
        assert!(!leftover_tmp.exists());
        assert!(path.exists());
    }

    /// Loading a missing file must produce a path-bearing Io error
    /// (not a generic 'JSON parse failed').
    #[test]
    fn load_missing_file_produces_io_error_with_path() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nope.json");
        let err = ModelManifest::load_json(&path).unwrap_err();
        match err {
            RunnerError::Io { path: p, .. } => assert_eq!(p, path),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    /// Loading a syntactically invalid JSON file must produce a Json error,
    /// not Io (the file *was* readable; parse is what failed).
    #[test]
    fn load_invalid_json_produces_json_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.json");
        std::fs::write(&path, b"{not json").unwrap();
        let err = ModelManifest::load_json(&path).unwrap_err();
        assert!(matches!(err, RunnerError::Json(_)));
    }

    /// Loading a file that parses but violates invariants surfaces
    /// the same validation error as `validate()` directly.
    #[test]
    fn load_file_with_bad_invariants_surfaces_validation_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad_invariant.json");
        // version=0 → validate() rejects.
        let m = sample_manifest(0, "abc");
        let bytes = serde_json::to_vec_pretty(&m).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let err = ModelManifest::load_json(&path).unwrap_err();
        assert!(matches!(err, RunnerError::InvalidManifest(_)));
    }

    /// Saving an invalid manifest must not leave a file on disk —
    /// validate runs before any IO.
    #[test]
    fn save_invalid_manifest_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model_manifest.json");
        let bad = sample_manifest(0, "abc"); // version=0 is invalid
        let _ = bad.save_json(&path).unwrap_err();
        assert!(!path.exists());
    }

    /// `save_json` must create missing intermediate directories so callers
    /// don't have to mkdir `models/` themselves before the first export.
    #[test]
    fn save_json_creates_missing_parent_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("a/b/c/model_manifest.json");
        assert!(!path.parent().unwrap().exists());
        let m = sample_manifest(1, "abc");
        m.save_json(&path).unwrap();
        assert!(path.exists());
    }

    /// `Path::new("model_manifest.json").parent()` returns `Some("")`, and
    /// `fs::create_dir_all("")` errors on Windows. The bare-filename case
    /// must still succeed (writes to cwd).
    #[test]
    fn save_json_handles_bare_filename() {
        let tmp = tempfile::tempdir().unwrap();
        // Operate in the tempdir so we don't litter the repo root.
        let prev_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let result = sample_manifest(1, "abc").save_json("model_manifest.json");
        let exists = tmp.path().join("model_manifest.json").exists();
        // Restore cwd before any assertion so a failure doesn't leak state.
        std::env::set_current_dir(prev_cwd).unwrap();
        result.unwrap();
        assert!(exists);
    }
}
