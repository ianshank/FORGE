//! Phase B exporters that fan the [`Scorecard`] + [`RunManifest`] out to
//! external visualization / dataset systems.
//!
//! Implementations live in submodules and all satisfy the [`Exporter`]
//! trait. The harness dispatches each exporter independently and treats
//! a failed export as a `tracing::warn!` event — never as a run failure.
//!
//! Concrete exporters:
//!
//! - [`mlflow::MlflowExporter`] — writes MLflow's `mlruns/` filesystem
//!   layout (no HTTP, no running tracking server required).
//! - [`huggingface::HuggingFaceExporter`] — writes an HF-`datasets`
//!   `DatasetDict` directory loadable via `load_from_disk(...)`.
//!
//! [`Scorecard`]: crate::scorecard::Scorecard
//! [`RunManifest`]: crate::manifest::RunManifest

use std::path::Path;

use thiserror::Error;

use crate::manifest::RunManifest;
use crate::scorecard::Scorecard;

pub mod huggingface;
pub mod mlflow;

/// Failure modes an [`Exporter`] can surface to the harness.
///
/// Marked `#[non_exhaustive]` so downstream `match` sites stay
/// forwards-compatible when new variants land (e.g. HTTP transport
/// errors from the planned MLflow HTTP exporter). Downstream code MUST
/// include a `_ => ...` arm.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ExportError {
    /// Filesystem error during write / copy / mkdir.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// YAML / JSON serialization failure.
    #[error("serialize: {0}")]
    Serialize(String),
    /// The exporter's configured target is invalid (e.g. empty path,
    /// non-existent parent dir on platforms that can't auto-create).
    #[error("invalid target: {0}")]
    InvalidTarget(String),
}

/// Sink that consumes a [`Scorecard`] + [`RunManifest`] and writes them
/// into an external format (MLflow, HuggingFace, ...).
///
/// Implementations must be **idempotent**: calling `export` twice with
/// the same scorecard + manifest must produce the same on-disk bytes.
/// This lets the harness re-export after a partial failure without
/// double-counting or duplicating records.
pub trait Exporter {
    /// Stable name for diagnostics + logging. Examples: `"mlflow"`,
    /// `"huggingface"`.
    fn name(&self) -> &'static str;

    /// Write the scorecard + manifest into the exporter's configured
    /// target.
    ///
    /// `artifacts_dir` is the harness's primary on-disk artefacts
    /// directory (typically `output.dir`); exporters that copy
    /// replays / trajectories pull them from here.
    fn export(
        &self,
        scorecard: &Scorecard,
        manifest: &RunManifest,
        artifacts_dir: &Path,
    ) -> Result<(), ExportError>;
}
