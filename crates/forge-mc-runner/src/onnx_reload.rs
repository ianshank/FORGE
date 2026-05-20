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

use crate::error::RunnerError;
use crate::manifest::ModelManifest;
use crate::runner::ReloadFn;

/// Build an [`OnnxModelConfig`] from a [`ModelManifest`] + the
/// invariant knobs the model carries across reloads (action space,
/// latent dim, thread count).
///
/// File paths are resolved relative to `bundle_dir` if the manifest
/// entry's path is relative; absolute paths in the manifest pass
/// through verbatim.
pub fn config_from_manifest(
    manifest: &ModelManifest,
    bundle_dir: &std::path::Path,
    action_space_size: u32,
    latent_dim: usize,
    num_threads: usize,
) -> OnnxModelConfig {
    OnnxModelConfig {
        representation_path: resolve_relative(&manifest.files.representation.path, bundle_dir),
        dynamics_path: resolve_relative(&manifest.files.dynamics.path, bundle_dir),
        prediction_path: resolve_relative(&manifest.files.prediction.path, bundle_dir),
        action_space_size,
        latent_dim,
        num_threads,
    }
}

fn resolve_relative(entry_path: &str, bundle_dir: &std::path::Path) -> String {
    let p = PathBuf::from(entry_path);
    if p.is_absolute() {
        entry_path.to_string()
    } else {
        bundle_dir.join(&p).to_string_lossy().into_owned()
    }
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
            );
            model
                .reload(new_config)
                .map_err(|e: OnnxReloadError| RunnerError::Reload(e.to_string()))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelFileEntry, ModelManifestFiles, MANIFEST_SCHEMA_VERSION};

    fn make_manifest_with_relative_paths() -> ModelManifest {
        ModelManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            version: 1,
            schema_id: "stub-sid".into(),
            created_at: "2026-05-21T00:00:00Z".into(),
            files: ModelManifestFiles {
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
        }
    }

    #[test]
    fn config_from_manifest_resolves_relative_paths_against_bundle_dir() {
        let manifest = make_manifest_with_relative_paths();
        let bundle_dir = PathBuf::from("/some/bundle");
        let cfg = config_from_manifest(&manifest, &bundle_dir, 12, 256, 1);
        // Cross-platform path-equality check via PathBuf normalisation.
        assert_eq!(
            PathBuf::from(&cfg.representation_path),
            PathBuf::from("/some/bundle/representation.onnx")
        );
        assert_eq!(
            PathBuf::from(&cfg.dynamics_path),
            PathBuf::from("/some/bundle/dynamics.onnx")
        );
        assert_eq!(
            PathBuf::from(&cfg.prediction_path),
            PathBuf::from("/some/bundle/prediction.onnx")
        );
        assert_eq!(cfg.action_space_size, 12);
        assert_eq!(cfg.latent_dim, 256);
        assert_eq!(cfg.num_threads, 1);
    }

    #[test]
    fn config_from_manifest_preserves_absolute_paths() {
        let mut manifest = make_manifest_with_relative_paths();
        let abs_path = if cfg!(windows) {
            "C:/abs/rep.onnx".to_string()
        } else {
            "/abs/rep.onnx".to_string()
        };
        manifest.files.representation.path = abs_path.clone();
        let cfg = config_from_manifest(&manifest, &PathBuf::from("/other/dir"), 1, 1, 1);
        assert_eq!(cfg.representation_path, abs_path);
    }

    #[test]
    fn into_reload_fn_returns_callable_box() {
        // We can only confirm the box constructs and types match here
        // — the actual `model.reload` call needs real ONNX files,
        // which lives in `tests/onnx_reload_integration.rs`.
        let _reload_fn: ReloadFn<OnnxMuZeroModel> = into_reload_fn(PathBuf::from("."), 4, 8, 1);
    }
}
