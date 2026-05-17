// WIP-preserved (commit a91b3fa) — see exporters/huggingface.rs header for
// rationale on the module-level clippy allow.
#![allow(clippy::field_reassign_with_default)]

//! MLflow filesystem sink — consumes a transport-agnostic [`RunPayload`]
//! and writes the on-disk `mlruns/` layout that
//! `mlflow ui --backend-store-uri <dir>` reads natively.
//!
//! The legacy [`crate::exporters::mlflow::MlflowExporter`] now delegates to
//! [`MlflowFsSink`] so there is exactly one filesystem implementation
//! across the crate. The upcoming HTTP sink (Slice 2) consumes the same
//! `RunPayload` via a different serialiser; both sinks therefore stay
//! byte-equivalent on every observable contract.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tracing::{debug, instrument};

use super::mlflow::{
    DEFAULT_EXPERIMENT_ID, MLFLOW_META_FILE, MLFLOW_SUBDIR_ARTIFACTS, MLFLOW_SUBDIR_METRICS,
    MLFLOW_SUBDIR_PARAMS, MLFLOW_SUBDIR_TAGS,
};
use super::mlflow_payload::{
    build_run_payload, sanitize, validate_run_id, ArtifactRef, ArtifactSource, RunPayload,
};
use super::{ExportError, Exporter};
use crate::manifest::{RunManifest, MANIFEST_SOURCE_NAME};
use crate::scorecard::Scorecard;

// ─── Filesystem-only constants (mirrored from the legacy mlflow.rs) ────────
// Kept private to this module — they describe MLflow's on-disk serialisation
// and have no meaning for the HTTP sink.

/// MLflow source-type tag value for a non-Project (ad-hoc) run.
const SOURCE_TYPE_LOCAL: &str = "LOCAL";
/// Lifecycle stage written to every `meta.yaml`. `deleted` would hide
/// runs from the default UI view.
const LIFECYCLE_STAGE_ACTIVE: &str = "active";
/// MLflow encodes status as an int enum on disk: 1=RUNNING, 2=SCHEDULED,
/// 3=FINISHED, 4=FAILED, 5=KILLED. Writing the string "FINISHED" makes
/// MLflow's reader raise.
const RUN_STATUS_FINISHED: i32 = 3;
/// MLflow truncates params >500 chars on read; pre-truncate so we never
/// ship files the UI rejects.
const PARAM_MAX_LEN: usize = 500;

// ─── Sink ──────────────────────────────────────────────────────────────────

/// MLflow filesystem sink. Construct with [`MlflowFsSink::new`] (default
/// experiment id `"0"`) or [`MlflowFsSink::with_experiment_id`].
#[derive(Debug, Clone)]
pub struct MlflowFsSink {
    tracking_uri: PathBuf,
    experiment_id: String,
}

impl MlflowFsSink {
    /// Construct a sink rooted at `tracking_uri`. Pass the same path to
    /// `mlflow ui --backend-store-uri`. Experiment id defaults to
    /// [`DEFAULT_EXPERIMENT_ID`].
    pub fn new(tracking_uri: PathBuf) -> Self {
        Self {
            tracking_uri,
            experiment_id: DEFAULT_EXPERIMENT_ID.to_string(),
        }
    }

    /// Construct a sink with an explicit experiment id (must be a
    /// stringified integer to match MLflow's convention).
    pub fn with_experiment_id(tracking_uri: PathBuf, experiment_id: impl Into<String>) -> Self {
        Self {
            tracking_uri,
            experiment_id: experiment_id.into(),
        }
    }
}

impl Exporter for MlflowFsSink {
    fn name(&self) -> &'static str {
        "mlflow"
    }

    #[instrument(skip_all, fields(tracking_uri = %self.tracking_uri.display(), exp_id = %self.experiment_id))]
    fn export(
        &self,
        scorecard: &Scorecard,
        manifest: &RunManifest,
        artifacts_dir: &Path,
    ) -> Result<(), ExportError> {
        if self.tracking_uri.as_os_str().is_empty() {
            return Err(ExportError::InvalidTarget(
                "tracking_uri must not be empty".to_string(),
            ));
        }

        let exp_dir = self.tracking_uri.join(&self.experiment_id);
        fs::create_dir_all(&exp_dir)?;
        write_experiment_meta(&exp_dir, &self.experiment_id, &manifest.experiment_name)?;

        // Single payload for the whole suite. All children share the same
        // `start_ms` timestamp the metrics carry; per-run `end_ms` is captured
        // at meta.yaml-write time so a partial run is visibly incomplete.
        let start_ms = now_ms();
        let payload = build_run_payload(scorecard, manifest, artifacts_dir, start_ms)?;

        // Reject path-traversal-bearing run_ids before joining into the tracking
        // URI. Both parent + every child id is caller-influenced (derived from
        // manifest.run_id + child_run_id) so check each one.
        validate_run_id(&payload.run_id)?;
        for child in &payload.children {
            validate_run_id(&child.run_id)?;
        }

        let parent_dir = exp_dir.join(&payload.run_id);
        write_run_from_payload(
            &parent_dir,
            &payload,
            manifest,
            start_ms,
            &self.experiment_id,
        )?;

        for child in &payload.children {
            let child_dir = exp_dir.join(&child.run_id);
            write_run_from_payload(&child_dir, child, manifest, start_ms, &self.experiment_id)?;
        }

        Ok(())
    }
}

// ─── Payload → filesystem writer ───────────────────────────────────────────

fn write_run_from_payload(
    run_dir: &Path,
    payload: &RunPayload,
    manifest: &RunManifest,
    start_ms: u64,
    experiment_id: &str,
) -> Result<(), ExportError> {
    fs::create_dir_all(run_dir)?;
    let artifacts_subdir = run_dir.join(MLFLOW_SUBDIR_ARTIFACTS);
    fs::create_dir_all(&artifacts_subdir)?;

    // Idempotency: `write_metric` opens metric files in append mode (the
    // shape MLflow expects for log_metric semantics), so re-exporting the
    // SAME payload would double-count every line. Reset the metrics dir
    // before this export's writes start. Per-export this is a no-op (dir
    // is freshly created); on re-export it discards the prior write's
    // contents so the new write produces byte-identical output for the
    // same input. The trait's idempotency contract is satisfied.
    let metrics_dir = run_dir.join(MLFLOW_SUBDIR_METRICS);
    if metrics_dir.exists() {
        fs::remove_dir_all(&metrics_dir)?;
    }

    for param in &payload.params {
        write_param(run_dir, &param.key, &param.value)?;
    }
    for metric in &payload.metrics {
        write_metric(
            run_dir,
            &metric.key,
            metric.value,
            metric.timestamp_ms,
            metric.step,
        )?;
    }
    for tag in &payload.tags {
        write_tag(run_dir, &tag.key, &tag.value)?;
    }
    for artifact in &payload.artifact_refs {
        write_artifact_ref(&artifacts_subdir, artifact)?;
    }

    // meta.yaml LAST — this is what mlflow ui keys off; writing it last means
    // a partially-written run is visibly incomplete (no meta.yaml) rather
    // than silently corrupt.
    let end_ms = now_ms();
    write_run_meta(
        run_dir,
        &RunMetaArgs {
            run_id: payload.run_id.clone(),
            run_name: payload.run_name.clone(),
            experiment_id: experiment_id.to_string(),
            artifact_uri: artifacts_subdir.to_string_lossy().into_owned(),
            user_id: manifest.user.clone(),
            start_time: start_ms,
            end_time: end_ms,
            status: RUN_STATUS_FINISHED,
        },
    )?;

    Ok(())
}

fn write_artifact_ref(artifacts_subdir: &Path, artifact: &ArtifactRef) -> Result<(), ExportError> {
    let dest = artifacts_subdir.join(&artifact.rel_path);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    match &artifact.source {
        ArtifactSource::Inline(bytes) => {
            fs::write(&dest, bytes)?;
        }
        ArtifactSource::File(src) => {
            fs::copy(src, &dest)?;
        }
        ArtifactSource::Directory(src) => {
            copy_subdir_if_exists(src, &dest)?;
        }
    }
    Ok(())
}

// ─── Experiment + meta ─────────────────────────────────────────────────────

fn write_experiment_meta(dir: &Path, exp_id: &str, name: &str) -> Result<(), ExportError> {
    let meta = ExperimentMeta {
        artifact_location: dir.to_string_lossy().into_owned(),
        experiment_id: exp_id.to_string(),
        lifecycle_stage: LIFECYCLE_STAGE_ACTIVE.to_string(),
        name: name.to_string(),
    };
    write_yaml(&dir.join(MLFLOW_META_FILE), &meta)
}

#[derive(Serialize)]
struct ExperimentMeta {
    artifact_location: String,
    experiment_id: String,
    lifecycle_stage: String,
    name: String,
}

#[derive(Serialize)]
struct RunMeta {
    artifact_uri: String,
    end_time: u64,
    entry_point_name: String,
    experiment_id: String,
    lifecycle_stage: String,
    run_id: String,
    run_uuid: String,
    run_name: String,
    source_name: String,
    source_type: String,
    source_version: String,
    start_time: u64,
    status: i32,
    tags: Vec<String>,
    user_id: String,
}

struct RunMetaArgs {
    run_id: String,
    run_name: String,
    experiment_id: String,
    artifact_uri: String,
    user_id: String,
    start_time: u64,
    end_time: u64,
    status: i32,
}

fn write_run_meta(run_dir: &Path, args: &RunMetaArgs) -> Result<(), ExportError> {
    let meta = RunMeta {
        artifact_uri: args.artifact_uri.clone(),
        end_time: args.end_time,
        entry_point_name: String::new(),
        experiment_id: args.experiment_id.clone(),
        lifecycle_stage: LIFECYCLE_STAGE_ACTIVE.to_string(),
        run_id: args.run_id.clone(),
        run_uuid: args.run_id.clone(),
        run_name: args.run_name.clone(),
        source_name: MANIFEST_SOURCE_NAME.to_string(),
        source_type: SOURCE_TYPE_LOCAL.to_string(),
        source_version: env!("CARGO_PKG_VERSION").to_string(),
        start_time: args.start_time,
        status: args.status,
        tags: Vec::new(),
        user_id: args.user_id.clone(),
    };
    write_yaml(&run_dir.join(MLFLOW_META_FILE), &meta)
}

// ─── File primitives (params / metrics / tags / yaml / dir copy) ───────────

fn write_param(run_dir: &Path, name: &str, value: &str) -> Result<(), ExportError> {
    let dir = run_dir.join(MLFLOW_SUBDIR_PARAMS);
    fs::create_dir_all(&dir)?;
    let v = truncate_at_char_boundary(value, PARAM_MAX_LEN);
    fs::write(dir.join(sanitize(name)), v)?;
    Ok(())
}

/// UTF-8-safe truncation. Returns the prefix of `s` whose byte length is
/// `<= max_bytes` and that ends on a `char` boundary. Naive byte slicing
/// (`&s[..max_bytes]`) panics when `max_bytes` lands in the middle of a
/// multibyte char (common for agent metadata values that include
/// non-ASCII text). Walks `char_indices` instead so the slice always lies
/// on a valid boundary.
fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut last_ok = 0;
    for (i, _) in s.char_indices() {
        if i > max_bytes {
            break;
        }
        last_ok = i;
    }
    &s[..last_ok]
}

fn write_metric(
    run_dir: &Path,
    name: &str,
    value: f64,
    timestamp_ms: u64,
    step: u64,
) -> Result<(), ExportError> {
    let dir = run_dir.join(MLFLOW_SUBDIR_METRICS);
    fs::create_dir_all(&dir)?;
    let path = dir.join(sanitize(name));
    let mut file = File::options().create(true).append(true).open(&path)?;
    // MLflow's metric format: "<timestamp_ms> <value> <step>\n"
    writeln!(
        file,
        "{} {} {}",
        timestamp_ms,
        format_metric_value(value),
        step
    )?;
    Ok(())
}

fn write_tag(run_dir: &Path, name: &str, value: &str) -> Result<(), ExportError> {
    let dir = run_dir.join(MLFLOW_SUBDIR_TAGS);
    fs::create_dir_all(&dir)?;
    fs::write(dir.join(sanitize(name)), value)?;
    Ok(())
}

fn write_yaml<T: Serialize>(path: &Path, value: &T) -> Result<(), ExportError> {
    let yaml = serde_yaml::to_string(value).map_err(|e| ExportError::Serialize(e.to_string()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, yaml)?;
    Ok(())
}

fn copy_subdir_if_exists(src: &Path, dst: &Path) -> Result<(), ExportError> {
    if !src.exists() || !src.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_subdir_if_exists(&src_path, &dst_path)?;
        } else if ty.is_file() {
            fs::copy(&src_path, &dst_path)?;
        } else {
            debug!(
                src = %src_path.display(),
                "mlflow-fs sink: skipping non-file/non-dir entry"
            );
        }
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn format_metric_value(v: f64) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else if v.is_nan() {
        "NaN".to_string()
    } else if v.is_sign_positive() {
        "Infinity".to_string()
    } else {
        "-Infinity".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_metric_value_handles_special_floats() {
        assert_eq!(format_metric_value(1.5), "1.500000");
        assert_eq!(format_metric_value(0.0), "0.000000");
        assert_eq!(format_metric_value(-2.25), "-2.250000");
        assert_eq!(format_metric_value(f64::NAN), "NaN");
        assert_eq!(format_metric_value(f64::INFINITY), "Infinity");
        assert_eq!(format_metric_value(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn write_param_truncates_to_max_length() {
        let tmp = tempfile::tempdir().unwrap();
        let long = "a".repeat(PARAM_MAX_LEN + 100);
        write_param(tmp.path(), "k", &long).unwrap();
        let written = fs::read_to_string(tmp.path().join(MLFLOW_SUBDIR_PARAMS).join("k")).unwrap();
        assert_eq!(written.len(), PARAM_MAX_LEN);
    }

    #[test]
    fn write_metric_appends_one_line_per_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_metric(tmp.path(), "loss", 0.5, 100, 0).unwrap();
        write_metric(tmp.path(), "loss", 0.4, 200, 1).unwrap();
        let body = fs::read_to_string(tmp.path().join(MLFLOW_SUBDIR_METRICS).join("loss")).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "100 0.500000 0");
        assert_eq!(lines[1], "200 0.400000 1");
    }

    #[test]
    fn write_artifact_ref_inline_writes_bytes_at_rel_path() {
        let tmp = tempfile::tempdir().unwrap();
        let artifact = ArtifactRef {
            rel_path: "nested/file.txt".to_string(),
            source: ArtifactSource::Inline(b"hello".to_vec()),
        };
        write_artifact_ref(tmp.path(), &artifact).unwrap();
        let body = fs::read_to_string(tmp.path().join("nested/file.txt")).unwrap();
        assert_eq!(body, "hello");
    }

    #[test]
    fn write_artifact_ref_file_copies_source() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src.bin");
        fs::write(&src, b"copied").unwrap();
        let artifact = ArtifactRef {
            rel_path: "dst.bin".to_string(),
            source: ArtifactSource::File(src),
        };
        write_artifact_ref(tmp.path(), &artifact).unwrap();
        assert_eq!(fs::read(tmp.path().join("dst.bin")).unwrap(), b"copied");
    }

    #[test]
    fn write_artifact_ref_directory_copies_tree_recursively() {
        let tmp = tempfile::tempdir().unwrap();
        let src_root = tmp.path().join("src_tree");
        fs::create_dir_all(src_root.join("sub")).unwrap();
        fs::write(src_root.join("a.txt"), b"A").unwrap();
        fs::write(src_root.join("sub/b.txt"), b"B").unwrap();

        let artifact = ArtifactRef {
            rel_path: "tree".to_string(),
            source: ArtifactSource::Directory(src_root),
        };
        write_artifact_ref(tmp.path(), &artifact).unwrap();

        assert_eq!(fs::read(tmp.path().join("tree/a.txt")).unwrap(), b"A");
        assert_eq!(fs::read(tmp.path().join("tree/sub/b.txt")).unwrap(), b"B");
    }

    #[test]
    fn truncate_at_char_boundary_does_not_panic_on_multibyte() {
        // A 3-byte UTF-8 char ('π') spans byte indices 0..2 (in bytes).
        // Naive `&s[..1]` here would panic; the helper must hand back
        // either the empty prefix or the full char, never a half-char.
        let s = "πππ"; // 6 bytes, 3 chars
        assert_eq!(truncate_at_char_boundary(s, 0), "");
        assert_eq!(truncate_at_char_boundary(s, 1), "");
        assert_eq!(truncate_at_char_boundary(s, 2), "π");
        assert_eq!(truncate_at_char_boundary(s, 3), "π");
        assert_eq!(truncate_at_char_boundary(s, 4), "ππ");
        assert_eq!(truncate_at_char_boundary(s, 6), "πππ");
        assert_eq!(truncate_at_char_boundary(s, 999), "πππ");
        // ASCII falls through unchanged.
        assert_eq!(truncate_at_char_boundary("hello", 3), "hel");
    }

    #[test]
    fn write_param_does_not_panic_on_long_multibyte_value() {
        // 4-byte chars × 200 = 800 bytes, well past PARAM_MAX_LEN (500).
        // The naive `&value[..500]` slice would land inside a 4-byte
        // sequence and panic. The truncation helper must keep us on a
        // valid char boundary.
        let value: String = "🦀".repeat(200);
        let tmp = tempfile::tempdir().unwrap();
        write_param(tmp.path(), "k", &value).expect("write_param must not panic");
        let written = fs::read_to_string(tmp.path().join(MLFLOW_SUBDIR_PARAMS).join("k")).unwrap();
        // Result is at most PARAM_MAX_LEN bytes AND a valid UTF-8 string
        // ending on a char boundary. 🦀 is 4 bytes so 500 / 4 = 125
        // chars (500 bytes); 501 wouldn't fit so we stop at 124 chars
        // (496 bytes) — either is acceptable, both are < PARAM_MAX_LEN+1.
        assert!(written.len() <= PARAM_MAX_LEN);
        // Round-trip through `chars().count()` to confirm UTF-8 validity.
        let _char_count = written.chars().count();
    }

    #[test]
    fn write_metric_then_re_export_does_not_duplicate_lines() {
        // Pin the idempotency contract on the metrics dir: re-exporting
        // the same payload to the same run dir must produce one line
        // per sample, not two. The fix lives in
        // `write_run_from_payload`, which deletes the metrics dir at
        // the start of each export; this test exercises both passes.
        use super::super::mlflow_payload::{ArtifactRef, ArtifactSource, MetricSample, RunPayload};
        use crate::manifest::{RunManifest, MANIFEST_SOURCE_NAME, UNKNOWN};
        use chrono::Utc;

        let tmp = tempfile::tempdir().unwrap();
        let payload = RunPayload {
            run_id: "idem-run".to_string(),
            experiment_name: "exp".to_string(),
            run_name: "name".to_string(),
            params: vec![],
            tags: vec![],
            metrics: vec![MetricSample {
                key: "loss".to_string(),
                value: 0.5,
                timestamp_ms: 1,
                step: 0,
            }],
            children: vec![],
            artifact_refs: vec![] as Vec<ArtifactRef>,
        };
        let manifest = RunManifest {
            run_id: "idem-run".to_string(),
            experiment_name: "exp".to_string(),
            timestamp: Utc::now(),
            git_sha: UNKNOWN.to_string(),
            git_branch: UNKNOWN.to_string(),
            rustc_version: UNKNOWN.to_string(),
            user: UNKNOWN.to_string(),
            config_hash: "h".to_string(),
            scenario_file_hashes: vec![],
            source_name: MANIFEST_SOURCE_NAME.to_string(),
        };
        let run_dir = tmp.path().join("run");
        // Suppress unused-binding warning on the discarded ArtifactSource
        // case in the dead-code-eliminated path.
        let _ = ArtifactSource::Inline(vec![]);

        write_run_from_payload(&run_dir, &payload, &manifest, 0, "0").unwrap();
        let first = fs::read_to_string(run_dir.join(MLFLOW_SUBDIR_METRICS).join("loss")).unwrap();
        // Same payload, re-exported: must NOT accumulate.
        write_run_from_payload(&run_dir, &payload, &manifest, 0, "0").unwrap();
        let second = fs::read_to_string(run_dir.join(MLFLOW_SUBDIR_METRICS).join("loss")).unwrap();
        assert_eq!(
            first.lines().count(),
            1,
            "first export must produce exactly 1 metric line"
        );
        assert_eq!(
            second.lines().count(),
            1,
            "re-export must NOT duplicate the metric line (got {} lines)",
            second.lines().count()
        );
        assert_eq!(first, second, "byte-identical output across re-exports");
    }
}
