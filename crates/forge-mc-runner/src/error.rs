//! Error type for the runner foundation.

use std::path::PathBuf;

use thiserror::Error;

/// Errors emitted by the runner, manifest loader, hot-reload watcher,
/// and trajectory writer.
///
/// Variants carry just enough structured context for callers to log
/// usefully. File-IO failures preserve the offending path; JSON failures
/// preserve the underlying parser diagnostic.
///
/// ## Why no `From<MetricsError> for RunnerError`?
///
/// Metrics failures are a *binary-level* concern (the runner is happy
/// to keep stepping even when the metrics endpoint can't bind), and
/// are surfaced by the binary's `main` directly via [`std::process::
/// ExitCode`]. The runner loop ([`crate::Runner::run`]) never returns
/// a metrics error because it never *interacts* with the metrics
/// server beyond pushing samples into an in-process [`MetricsRecorder`].
/// Keeping the two error surfaces separate avoids accidentally
/// collapsing "metrics endpoint won't bind" failures into the same
/// `RunnerError` channel that surfaces real per-episode runner
/// problems.
#[derive(Debug, Error)]
pub enum RunnerError {
    /// Configured path does not exist on disk.
    #[error("missing path: {path}")]
    MissingPath {
        /// Path the caller attempted to use.
        path: PathBuf,
    },

    /// Manifest `schema_version` field did not match the compiled constant.
    #[error("manifest schema_version mismatch: expected {expected}, got {got}")]
    ManifestSchemaMismatch {
        /// Constant compiled into this crate.
        expected: u32,
        /// Value read from the manifest file.
        got: u32,
    },

    /// Manifest contained invariants-violating fields (e.g. zero version
    /// or missing onnx file paths).
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),

    /// Trajectory writer was used in an unexpected order (e.g. push step
    /// before `start_episode`, finalize without any push).
    #[error("trajectory writer state error: {0}")]
    WriterState(String),

    /// File-system IO failure with the offending path.
    #[error("io error on {path}: {source}")]
    Io {
        /// Path involved.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// JSON (de)serialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Forwarded from `forge_replay::v2::TrajectoryError` during writer
    /// validation and saving.
    #[error("trajectory error: {0}")]
    Trajectory(#[from] forge_replay::v2::TrajectoryError),

    /// Wrapped error from a [`forge_env::Env`] implementation. The
    /// concrete env error is `Send + Sync + 'static + std::error::Error`
    /// but is not nameable from this crate, so we collapse to its
    /// `Display` form. Callers wanting structured access should keep
    /// their own copy.
    #[error("env error: {0}")]
    Env(String),

    /// Wrapped error from the latent-MCTS planner (`anyhow::Error` from
    /// `forge_agent::latent_mcts::search::LatentMctsSearch::search`).
    #[error("planner error: {0}")]
    Planner(String),

    /// Reload callback returned an error.
    #[error("model reload failed: {0}")]
    Reload(String),
}

impl RunnerError {
    /// Wrap a `std::io::Error` with the path it was raised against.
    /// Kept as a helper because `From` can't carry the path through.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
