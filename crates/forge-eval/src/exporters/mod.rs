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
pub mod mlflow_payload;

// ─── Shared exporter constants ─────────────────────────────────────────────
// String literals consumed by BOTH the MLflow and HuggingFace exporters.
// Defined here so renaming requires a single edit and both exporters
// continue to agree on the on-disk contract. Tests in each exporter module
// pin these against the observed file names so a divergence is caught at
// test time.

/// Naming prefix for per-tier splits. The HF exporter emits `tier_1/`,
/// `tier_2/`, ... directories under `<export_root>/<run_id>/`; the MLflow
/// exporter uses the same prefix when keying per-tier metric series.
/// Format: `"{TIER_SPLIT_PREFIX}{tier}"`.
pub const TIER_SPLIT_PREFIX: &str = "tier_";

/// Filename of the on-disk RunManifest JSON artefact. Both exporters write
/// this file (MLflow under `artifacts/`, HF directly under the run dir).
pub const ARTIFACT_MANIFEST_JSON: &str = "manifest.json";

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporters::huggingface::{HF_DATA_SHARD_FILENAME, SPLIT_ALL};
    use crate::exporters::mlflow::{
        AGENT_PARAM_KEY_PREFIX, ARTIFACT_EPISODES_JSONL, ARTIFACT_SCORECARD_JSON,
        ARTIFACT_SCORECARD_MD, ARTIFACT_TIER_SUCCESS_RATES_HTML, DEFAULT_EXPERIMENT_ID,
        MLFLOW_META_FILE, MLFLOW_SUBDIR_ARTIFACTS, MLFLOW_SUBDIR_METRICS, MLFLOW_SUBDIR_PARAMS,
        MLFLOW_SUBDIR_TAGS,
    };

    /// Pin every shared + exporter-internal const to its on-disk-contract
    /// value. Existing test assertions in `mlflow.rs` and `huggingface.rs`
    /// still use the literal string (intentionally — they pin the *external*
    /// observable contract). This test pins the *internal* constant to that
    /// same string, so a rename of either the constant or the literal is
    /// caught as a deliberate, reviewable change.
    #[test]
    fn exporter_string_constants_are_stable_contract() {
        // Shared (this module).
        assert_eq!(TIER_SPLIT_PREFIX, "tier_");
        assert_eq!(ARTIFACT_MANIFEST_JSON, "manifest.json");

        // HuggingFace-internal.
        assert_eq!(SPLIT_ALL, "all");
        assert_eq!(HF_DATA_SHARD_FILENAME, "data-00000-of-00001.jsonl");

        // MLflow-internal.
        assert_eq!(DEFAULT_EXPERIMENT_ID, "0");
        assert_eq!(MLFLOW_META_FILE, "meta.yaml");
        assert_eq!(MLFLOW_SUBDIR_PARAMS, "params");
        assert_eq!(MLFLOW_SUBDIR_METRICS, "metrics");
        assert_eq!(MLFLOW_SUBDIR_TAGS, "tags");
        assert_eq!(MLFLOW_SUBDIR_ARTIFACTS, "artifacts");
        assert_eq!(ARTIFACT_SCORECARD_JSON, "scorecard.json");
        assert_eq!(ARTIFACT_SCORECARD_MD, "scorecard.md");
        assert_eq!(ARTIFACT_TIER_SUCCESS_RATES_HTML, "tier_success_rates.html");
        assert_eq!(ARTIFACT_EPISODES_JSONL, "episodes.jsonl");
        assert_eq!(AGENT_PARAM_KEY_PREFIX, "agent_param_");
    }
}
