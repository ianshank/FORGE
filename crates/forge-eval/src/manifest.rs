// WIP-preserved test patterns (commit a91b3fa) trigger
// `field_reassign_with_default` and `single_element_loop` here. Allow at
// module scope to preserve the WIP author's intent; revisit in a dedicated
// cleanup commit.
#![allow(clippy::field_reassign_with_default)]

//! Reproducibility manifest captured alongside every [`Scorecard`].
//!
//! A [`RunManifest`] is the minimal record needed to re-execute an
//! evaluation run identically: git SHA + branch, rustc version, hashes of
//! the scenario TOML files, and a hash of the [`EvalConfig`] itself.
//!
//! Manifests are produced by [`RunManifest::capture`] and serialized to
//! JSON by both Phase B exporters (MLflow under `artifacts/manifest.json`
//! and HuggingFace at the root of the dataset directory).
//!
//! [`Scorecard`]: crate::scorecard::Scorecard
//! [`EvalConfig`]: crate::config::EvalConfig

use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{instrument, warn};
use uuid::Uuid;

use crate::config::EvalConfig;

/// Fallback string used wherever a detection probe (git, rustc, $USER)
/// fails. Kept as a module constant so callers can pattern-match on it.
pub const UNKNOWN: &str = "unknown";

/// Identifier of the source under which the manifest was captured.
/// Matches the value mirrored to MLflow's `mlflow.source.name` tag.
pub const MANIFEST_SOURCE_NAME: &str = "forge-eval";

/// Reproducibility metadata captured for one evaluation run.
///
/// Every field is sourced from the running environment at
/// [`RunManifest::capture`] time. Detection failures degrade to
/// [`UNKNOWN`] rather than panicking — a missing git binary or a
/// non-repo cwd must never fail an otherwise-successful eval run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunManifest {
    /// Stable run identifier — UUIDv4 hex when [`EvalConfig::run_id`]
    /// is `None`, otherwise the caller-supplied value verbatim.
    pub run_id: String,
    /// Experiment grouping — mirrored to MLflow's experiment name.
    pub experiment_name: String,
    /// Wall-clock capture time, UTC.
    pub timestamp: DateTime<Utc>,
    /// `git rev-parse HEAD` output (40-char hex), or [`UNKNOWN`].
    pub git_sha: String,
    /// `git rev-parse --abbrev-ref HEAD` output, or [`UNKNOWN`].
    pub git_branch: String,
    /// First line of `rustc --version`, or [`UNKNOWN`].
    pub rustc_version: String,
    /// User who launched the run (`$USER`/`$USERNAME`/`whoami`), or
    /// [`UNKNOWN`].
    pub user: String,
    /// SHA-256 over `serde_json::to_vec(&config)` — config-content
    /// fingerprint for cross-run equivalence checks.
    pub config_hash: String,
    /// `(path, sha256_hex)` for every scenario file the eval consumed.
    /// Empty for evaluations that don't load TOML scenarios.
    pub scenario_file_hashes: Vec<(PathBuf, String)>,
    /// Source identifier — always [`MANIFEST_SOURCE_NAME`]. Stored on
    /// the manifest so consumers don't have to import the constant.
    pub source_name: String,
}

impl RunManifest {
    /// Capture a manifest from the current environment + config.
    ///
    /// All detection probes are best-effort: anything that fails
    /// degrades to [`UNKNOWN`] and a `tracing::warn!` line. This call
    /// must never panic.
    #[instrument(skip_all, fields(scenario_file_count = scenario_files.len()))]
    pub fn capture(config: &EvalConfig, scenario_files: &[PathBuf]) -> Self {
        let run_id = config
            .run_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().simple().to_string());
        let experiment_name = config
            .experiment_name
            .clone()
            .unwrap_or_else(|| format!("{}-default", MANIFEST_SOURCE_NAME));
        let scenario_file_hashes: Vec<(PathBuf, String)> = scenario_files
            .iter()
            .map(|p| {
                let hash = hash_file(p).unwrap_or_else(|err| {
                    warn!(path = %p.display(), error = %err, "manifest: scenario file hash failed");
                    String::new()
                });
                (p.clone(), hash)
            })
            .collect();

        Self {
            run_id,
            experiment_name,
            timestamp: Utc::now(),
            git_sha: detect_git_field(&["rev-parse", "HEAD"]),
            git_branch: detect_git_field(&["rev-parse", "--abbrev-ref", "HEAD"]),
            rustc_version: detect_rustc_version(),
            user: detect_user(),
            config_hash: hash_config(config),
            scenario_file_hashes,
            source_name: MANIFEST_SOURCE_NAME.to_string(),
        }
    }

    /// Persist the manifest as pretty-printed JSON.
    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Short identifier suitable for filesystem-friendly run names.
    /// First 8 hex chars of [`Self::git_sha`], or `"unknown"` if the
    /// SHA was undetectable.
    pub fn short_git_sha(&self) -> &str {
        if self.git_sha == UNKNOWN {
            UNKNOWN
        } else {
            &self.git_sha[..self.git_sha.len().min(8)]
        }
    }
}

fn detect_git_field(args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| UNKNOWN.to_string())
}

fn detect_rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| UNKNOWN.to_string())
}

fn detect_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| UNKNOWN.to_string())
}

fn hash_config(config: &EvalConfig) -> String {
    match serde_json::to_vec(config) {
        Ok(json) => hex(&Sha256::digest(&json)),
        Err(err) => {
            warn!(error = %err, "manifest: config hash serialization failed");
            String::new()
        }
    }
}

fn hash_file(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(hex(&Sha256::digest(&bytes)))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        write!(&mut out, "{:02x}", b).expect("write to string never fails");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn capture_populates_required_fields_from_default_config() {
        let cfg = EvalConfig::default();
        let manifest = RunManifest::capture(&cfg, &[]);

        assert!(!manifest.run_id.is_empty(), "run_id must be generated");
        assert!(
            !manifest.experiment_name.is_empty(),
            "experiment_name must default"
        );
        assert_eq!(manifest.source_name, MANIFEST_SOURCE_NAME);
        assert!(manifest.timestamp.timestamp() > 0);
        assert_eq!(manifest.config_hash.len(), 64, "sha-256 hex is 64 chars");
        assert!(manifest.scenario_file_hashes.is_empty());
    }

    #[test]
    fn capture_uses_explicit_run_id_and_experiment_name_when_provided() {
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("explicit-run-001".to_string());
        cfg.experiment_name = Some("my-experiment".to_string());

        let manifest = RunManifest::capture(&cfg, &[]);

        assert_eq!(manifest.run_id, "explicit-run-001");
        assert_eq!(manifest.experiment_name, "my-experiment");
    }

    #[test]
    fn capture_hashes_scenario_files_with_sha256() {
        let tmp = TempDir::new().expect("tempdir");
        let scenario_path = tmp.path().join("scenario.toml");
        std::fs::write(&scenario_path, b"id = \"test\"\n").unwrap();

        // Compute the expected sha256 of the file contents by hand so we
        // catch any deviation from the standard digest.
        let expected = hex(&Sha256::digest(b"id = \"test\"\n"));

        let cfg = EvalConfig::default();
        let manifest = RunManifest::capture(&cfg, std::slice::from_ref(&scenario_path));

        assert_eq!(manifest.scenario_file_hashes.len(), 1);
        assert_eq!(manifest.scenario_file_hashes[0].0, scenario_path);
        assert_eq!(manifest.scenario_file_hashes[0].1, expected);
    }

    #[test]
    fn capture_handles_missing_scenario_file_gracefully() {
        let cfg = EvalConfig::default();
        let manifest = RunManifest::capture(
            &cfg,
            &[PathBuf::from("/definitely/does/not/exist.toml")],
        );

        // A missing file produces an empty-hash entry, not a panic.
        assert_eq!(manifest.scenario_file_hashes.len(), 1);
        assert_eq!(manifest.scenario_file_hashes[0].1, "");
    }

    #[test]
    fn config_hash_is_deterministic_across_captures() {
        let cfg = EvalConfig::default();
        let h1 = RunManifest::capture(&cfg, &[]).config_hash;
        let h2 = RunManifest::capture(&cfg, &[]).config_hash;
        assert_eq!(h1, h2);
    }

    #[test]
    fn config_hash_differs_when_a_field_changes() {
        let cfg_a = EvalConfig::default();
        let mut cfg_b = EvalConfig::default();
        cfg_b.episodes_per_scenario = 999;

        let h_a = RunManifest::capture(&cfg_a, &[]).config_hash;
        let h_b = RunManifest::capture(&cfg_b, &[]).config_hash;
        assert_ne!(h_a, h_b);
    }

    #[test]
    fn write_json_round_trips_through_disk() {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("subdir").join("manifest.json");
        let manifest = RunManifest::capture(&EvalConfig::default(), &[]);

        manifest.write_json(&path).expect("write");
        let bytes = std::fs::read(&path).expect("read");
        let parsed: RunManifest = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn short_git_sha_returns_unknown_when_sha_is_unknown() {
        let mut manifest = RunManifest::capture(&EvalConfig::default(), &[]);
        manifest.git_sha = UNKNOWN.to_string();
        assert_eq!(manifest.short_git_sha(), UNKNOWN);
    }

    #[test]
    fn short_git_sha_takes_first_eight_hex_chars() {
        let mut manifest = RunManifest::capture(&EvalConfig::default(), &[]);
        manifest.git_sha = "abcdef1234567890abcdef".to_string();
        assert_eq!(manifest.short_git_sha(), "abcdef12");
    }
}
