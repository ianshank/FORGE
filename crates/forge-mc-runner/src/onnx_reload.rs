//! Helpers that bridge [`forge_agent::latent_mcts::onnx_model::OnnxMuZeroModel`]
//! into the runner's [`ReloadFn<M>`] callback shape.
//!
//! The trait the runner consumes — `ReloadFn<M>` — is intentionally
//! model-agnostic. This module is the per-backend glue that takes a
//! [`crate::ModelManifest`] (the on-disk swap signal the
//! [`crate::HotReloadWatcher`] surfaces) and turns it into an
//! [`forge_agent::latent_mcts::onnx_model::OnnxModelConfig`] the ONNX
//! model can [`OnnxMuZeroModel::reload`] against.
//!
//! Gated behind the `onnx-reload` Cargo feature so the runner stays
//! buildable without the ONNX Runtime dependency. Callers opt in by
//! compiling with `--features onnx-reload`.

use std::path::PathBuf;

use forge_agent::latent_mcts::onnx_model::{OnnxModelConfig, OnnxMuZeroModel, OnnxReloadError};
use tracing::instrument;

use crate::error::RunnerError;
use crate::integrity::verify_bundle;
use crate::manifest::ModelManifest;
use crate::runner::ReloadFn;

/// Build an [`OnnxModelConfig`] from a [`ModelManifest`] + the
/// invariant knobs the model carries across reloads (action space,
/// latent dim, thread count).
///
/// **This is the integrity choke point.** Before returning a config it
/// runs [`verify_bundle`], which
///
/// - resolves every manifest entry *inside* `bundle_dir` (absolute
///   paths, `..` components and symlink escapes are rejected), and
/// - re-hashes each ONNX file and compares it against the manifest's
///   recorded sha256.
///
/// Both callers — the initial bundle load in `crate::live` and the
/// between-episode [`into_reload_fn`] callback — go through here, so no
/// unverified file can reach the ONNX Runtime. The returned paths are
/// the canonical ones that were hashed.
///
/// # Errors
///
/// [`RunnerError::UnsafeModelPath`], [`RunnerError::MissingPath`],
/// [`RunnerError::Io`], or [`RunnerError::ModelDigestMismatch`] — see
/// [`verify_bundle`].
#[instrument(skip_all, fields(version = manifest.version, bundle_dir = %bundle_dir.display()))]
pub fn config_from_manifest(
    manifest: &ModelManifest,
    bundle_dir: &std::path::Path,
    action_space_size: u32,
    latent_dim: usize,
    num_threads: usize,
) -> Result<OnnxModelConfig, RunnerError> {
    let verified = verify_bundle(manifest, bundle_dir)?;
    Ok(OnnxModelConfig {
        representation_path: path_to_string(&verified.representation),
        dynamics_path: path_to_string(&verified.dynamics),
        prediction_path: path_to_string(&verified.prediction),
        action_space_size,
        latent_dim,
        num_threads,
    })
}

/// `OnnxModelConfig` stores paths as `String`; the verified paths are
/// already canonical, so this is only an encoding step.
fn path_to_string(path: &std::path::Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Wrap `OnnxMuZeroModel::reload` into the runner's
/// [`ReloadFn<OnnxMuZeroModel>`] callback shape.
///
/// `bundle_dir` is the directory the manifest's per-role paths
/// resolve against (typically the directory containing
/// `model_manifest.json`). `action_space_size`, `latent_dim`, and
/// `num_threads` are the invariant knobs that don't change across
/// reloads — they're captured at construction time and reused for
/// every reload.
///
/// `OnnxReloadError` is collapsed to its [`Display`] string before
/// being wrapped in [`RunnerError::Reload`] so the runner doesn't
/// need to depend on the `ort` types.
///
/// Integrity failures surface *un*collapsed: a bundle whose digests
/// don't match (or whose paths escape `bundle_dir`) fails inside
/// [`config_from_manifest`] before `reload` is called, so the caller
/// sees [`RunnerError::ModelDigestMismatch`] /
/// [`RunnerError::UnsafeModelPath`] rather than a generic reload
/// error — and the model keeps serving the previous bundle.
///
/// # Example (rust,no_run because it needs real ONNX files)
///
/// ```rust,no_run
/// # #[cfg(feature = "onnx-reload")]
/// # {
/// use std::path::PathBuf;
/// use forge_agent::latent_mcts::onnx_model::OnnxMuZeroModel;
/// use forge_mc_runner::onnx_reload::into_reload_fn;
/// # let model: OnnxMuZeroModel = unimplemented!();
/// let reload_fn = into_reload_fn(
///     PathBuf::from("models/"),
///     /* action_space_size */ 12,
///     /* latent_dim */ 256,
///     /* num_threads */ 1,
/// );
/// # }
/// ```
pub fn into_reload_fn(
    bundle_dir: PathBuf,
    action_space_size: u32,
    latent_dim: usize,
    num_threads: usize,
) -> ReloadFn<OnnxMuZeroModel> {
    Box::new(
        move |model: &mut OnnxMuZeroModel, manifest: &ModelManifest| {
            let new_config = config_from_manifest(
                manifest,
                &bundle_dir,
                action_space_size,
                latent_dim,
                num_threads,
            )?;
            model
                .reload(new_config)
                .map_err(|e: OnnxReloadError| RunnerError::Reload(e.to_string()))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrity::{file_sha256_hex, ROLE_REPRESENTATION};
    use crate::manifest::{ModelFileEntry, ModelManifestFiles, MANIFEST_SCHEMA_VERSION};

    /// Write `contents` to `dir/name` and return the file's sha256.
    fn write_file(dir: &std::path::Path, name: &str, contents: &[u8]) -> String {
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        file_sha256_hex(&path).unwrap()
    }

    /// A bundle dir with three real files and a manifest whose digests
    /// match them.
    fn make_bundle(dir: &std::path::Path) -> ModelManifest {
        ModelManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            version: 1,
            schema_id: "stub-sid".into(),
            created_at: "2026-05-21T00:00:00Z".into(),
            files: ModelManifestFiles {
                representation: ModelFileEntry {
                    path: "representation.onnx".into(),
                    sha256: write_file(dir, "representation.onnx", b"rep"),
                },
                dynamics: ModelFileEntry {
                    path: "dynamics.onnx".into(),
                    sha256: write_file(dir, "dynamics.onnx", b"dyn"),
                },
                prediction: ModelFileEntry {
                    path: "prediction.onnx".into(),
                    sha256: write_file(dir, "prediction.onnx", b"pred"),
                },
            },
        }
    }

    #[test]
    fn config_from_manifest_resolves_relative_paths_against_bundle_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = make_bundle(tmp.path());
        let cfg = config_from_manifest(&manifest, tmp.path(), 12, 256, 1).unwrap();
        // Paths come back canonical (the tempdir itself may be a
        // symlink, e.g. /tmp -> /private/tmp on macOS), so compare
        // against the canonicalized bundle dir.
        let base = tmp.path().canonicalize().unwrap();
        assert_eq!(
            PathBuf::from(&cfg.representation_path),
            base.join("representation.onnx")
        );
        assert_eq!(
            PathBuf::from(&cfg.dynamics_path),
            base.join("dynamics.onnx")
        );
        assert_eq!(
            PathBuf::from(&cfg.prediction_path),
            base.join("prediction.onnx")
        );
        assert_eq!(cfg.action_space_size, 12);
        assert_eq!(cfg.latent_dim, 256);
        assert_eq!(cfg.num_threads, 1);
    }

    /// Absolute manifest paths used to pass through verbatim, letting
    /// the manifest writer point the ONNX Runtime at any file on the
    /// host. They must now be rejected.
    #[test]
    fn config_from_manifest_rejects_absolute_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let mut manifest = make_bundle(tmp.path());
        let abs_path = tmp
            .path()
            .join("representation.onnx")
            .to_string_lossy()
            .into_owned();
        manifest.files.representation.path = abs_path.clone();
        let err = config_from_manifest(&manifest, tmp.path(), 1, 1, 1).unwrap_err();
        match err {
            RunnerError::UnsafeModelPath {
                role,
                entry,
                reason,
                ..
            } => {
                assert_eq!(role, ROLE_REPRESENTATION);
                assert_eq!(entry, abs_path);
                assert!(reason.contains("absolute"), "got reason: {reason}");
            }
            other => panic!("expected UnsafeModelPath, got {other:?}"),
        }
    }

    #[test]
    fn config_from_manifest_rejects_parent_dir_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("bundle");
        std::fs::create_dir_all(&bundle).unwrap();
        let mut manifest = make_bundle(&bundle);
        std::fs::write(tmp.path().join("evil.onnx"), b"evil").unwrap();
        manifest.files.dynamics.path = "../evil.onnx".into();
        match config_from_manifest(&manifest, &bundle, 1, 1, 1).unwrap_err() {
            RunnerError::UnsafeModelPath { reason, .. } => {
                assert!(reason.contains(".."), "got reason: {reason}");
            }
            other => panic!("expected UnsafeModelPath, got {other:?}"),
        }
    }

    /// The whole point of the change: a file swapped after the manifest
    /// recorded its digest must never reach `ort`.
    #[test]
    fn config_from_manifest_rejects_tampered_file() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = make_bundle(tmp.path());
        std::fs::write(tmp.path().join("prediction.onnx"), b"tampered").unwrap();
        match config_from_manifest(&manifest, tmp.path(), 1, 1, 1).unwrap_err() {
            RunnerError::ModelDigestMismatch { role, .. } => assert_eq!(role, "prediction"),
            other => panic!("expected ModelDigestMismatch, got {other:?}"),
        }
    }

    #[test]
    fn into_reload_fn_returns_callable_box() {
        // We can only confirm the box constructs and types match here
        // — the actual `model.reload` call needs real ONNX files,
        // which lives in `tests/onnx_reload_integration.rs`.
        let _reload_fn: ReloadFn<OnnxMuZeroModel> = into_reload_fn(PathBuf::from("."), 4, 8, 1);
    }
}
