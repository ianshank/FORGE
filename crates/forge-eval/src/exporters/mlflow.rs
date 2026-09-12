//! MLflow filesystem-layout exporter.
//!
//! Writes the exact `mlruns/<experiment_id>/<run_id>/` tree that
//! `mlflow ui --backend-store-uri <tracking_uri>` reads natively.
//! No HTTP, no running tracking server required.
//!
//! ## Layout produced
//!
//! ```text
//! <tracking_uri>/
//!   <experiment_id>/                  # default "0"
//!     meta.yaml
//!     <parent_run_id>/                # whole-suite parent run
//!       meta.yaml
//!       params/<param>                # one-line text
//!       metrics/<metric>              # "<ts_ms> <value> <step>" per line
//!       tags/<tag>                    # one-line text incl. mlflow.* system tags
//!       artifacts/
//!         scorecard.json
//!         scorecard.md
//!         manifest.json
//!         tier_success_rates.html     # Plotly figure
//!         replays/                    # copied from <artifacts_dir>/replays/ if present
//!         trajectories/               # copied from <artifacts_dir>/trajectories/ if present
//!       tags/forge.eval.scenarios_digest   # combined sha256 of scenario file hashes
//!       tags/forge.eval.scenario_count     # how many scenario files contributed
//!     <child_run_id>/                 # one per scenario, mlflow.parentRunId set
//!       meta.yaml
//!       params/                       # scenario_id, tier
//!       metrics/                      # per-episode w/ step indexing
//!       tags/
//!         mlflow.parentRunId
//!         mlflow.runName
//! ```

use std::path::{Path, PathBuf};

use tracing::instrument;

use super::{ExportError, Exporter};
use crate::manifest::RunManifest;
use crate::scorecard::Scorecard;

// Re-export the shared helpers at their original `crate::exporters::mlflow::`
// paths so existing external consumers + in-module tests keep their imports
// stable. The canonical implementation now lives in `mlflow_payload`; the
// upcoming MlflowFsSink + MlflowHttpSink both consume it directly.
pub use super::mlflow_payload::{
    child_run_id, combined_scenario_digest, hex_short, render_tier_bar_chart_html,
    render_tier_bar_chart_html_with_url, resolve_plotly_js_url, sanitize, validate_run_id,
    ArtifactRef, ArtifactSource, MetricSample, ParamKv, RunPayload, TagKv, DEFAULT_PLOTLY_JS_URL,
    FORGE_PLOTLY_JS_URL_ENV, PLOTLY_JS_VERSION,
};

/// MLflow's default experiment id when no explicit experiment is created.
/// `mlflow ui` will list the run under "Default" with this id.
pub const DEFAULT_EXPERIMENT_ID: &str = "0";

// â”€â”€â”€ MLflow filesystem layout: directory + filename constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Names are part of the on-disk contract that `mlflow ui --backend-store-uri
// <dir>` reads. `pub(crate)` so both `mlflow_fs` (writer) and
// `mlflow_payload` (builder) consume the same source of truth.

/// Subdirectory under each run dir holding `<key>` files (one per param).
pub(crate) const MLFLOW_SUBDIR_PARAMS: &str = "params";
/// Subdirectory under each run dir holding `<key>` files (one per metric,
/// each storing `<ts_ms> <value> <step>` lines per sample).
pub(crate) const MLFLOW_SUBDIR_METRICS: &str = "metrics";
/// Subdirectory under each run dir holding `<key>` files (one per tag).
pub(crate) const MLFLOW_SUBDIR_TAGS: &str = "tags";
/// Subdirectory under each run dir holding artefact files (scorecard,
/// manifest, plots, copies of replays/trajectories).
pub(crate) const MLFLOW_SUBDIR_ARTIFACTS: &str = "artifacts";
/// Filename of the per-run + per-experiment YAML metadata that `mlflow ui`
/// reads to enumerate runs.
pub(crate) const MLFLOW_META_FILE: &str = "meta.yaml";

// â”€â”€â”€ Artefact filenames (under MLFLOW_SUBDIR_ARTIFACTS or run dir) â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Scorecard JSON artefact (parent run, MLflow only â€” HF uses the manifest).
pub(crate) const ARTIFACT_SCORECARD_JSON: &str = "scorecard.json";
/// Scorecard Markdown artefact (parent run).
pub(crate) const ARTIFACT_SCORECARD_MD: &str = "scorecard.md";
/// Plotly per-tier success-rate chart, self-contained HTML.
pub(crate) const ARTIFACT_TIER_SUCCESS_RATES_HTML: &str = "tier_success_rates.html";
/// JSONL of per-episode rows written under each child run's artefacts dir.
pub(crate) const ARTIFACT_EPISODES_JSONL: &str = "episodes.jsonl";

/// Prefix applied to every key derived from `AgentMetadata` before
/// it is written as an MLflow param. Keeps agent-provided keys from
/// colliding with intrinsic MLflow params.
pub(crate) const AGENT_PARAM_KEY_PREFIX: &str = "agent_param_";

/// MLflow tracking exporter writing the filesystem layout that
/// `mlflow ui --backend-store-uri <tracking_uri>` consumes.
#[derive(Debug, Clone)]
pub struct MlflowExporter {
    tracking_uri: PathBuf,
    experiment_id: String,
}

impl MlflowExporter {
    /// Construct an exporter rooted at `tracking_uri`. Pass the same
    /// path to `mlflow ui --backend-store-uri`. Experiment id defaults
    /// to `"0"` (the MLflow Default experiment).
    pub fn new(tracking_uri: PathBuf) -> Self {
        Self {
            tracking_uri,
            experiment_id: DEFAULT_EXPERIMENT_ID.to_string(),
        }
    }

    /// Construct an exporter with an explicit experiment id (must be a
    /// stringified integer to match MLflow's convention).
    pub fn with_experiment_id(tracking_uri: PathBuf, experiment_id: impl Into<String>) -> Self {
        Self {
            tracking_uri,
            experiment_id: experiment_id.into(),
        }
    }
}

impl Exporter for MlflowExporter {
    fn name(&self) -> &'static str {
        "mlflow"
    }

    /// Backwards-compatible entry point. Delegates to
    /// [`super::mlflow_fs::MlflowFsSink`], which consumes a
    /// [`super::mlflow_payload::RunPayload`] shared with the upcoming HTTP
    /// sink. The on-disk layout produced is byte-identical to the prior
    /// implementation; every existing assertion in this module's tests
    /// continues to hold.
    #[instrument(skip_all, fields(tracking_uri = %self.tracking_uri.display(), exp_id = %self.experiment_id))]
    fn export(
        &self,
        scorecard: &Scorecard,
        manifest: &RunManifest,
        artifacts_dir: &Path,
    ) -> Result<(), ExportError> {
        super::mlflow_fs::MlflowFsSink::with_experiment_id(
            self.tracking_uri.clone(),
            &self.experiment_id,
        )
        .export(scorecard, manifest, artifacts_dir)
    }
}

// All write-time helpers and YAML serde structs (write_experiment_meta,
// write_parent_run, write_child_run, write_param/metric/tag/run_meta,
// format_metric_value, copy_subdir_if_exists, now_ms, ExperimentMeta,
// RunMeta, RunMetaArgs, EpisodeLine, manifest_experiment_id, etc.) moved
// into `mlflow_fs` as part of the Slice 1.2 split. MlflowExporter::export
// above now delegates to MlflowFsSink; this module retains only the
// public API surface (constants + re-exports + the legacy exporter shim).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EvalConfig;
    use crate::scorecard::{EpisodeResult, ScenarioResult, Scorecard, SummaryStats, TierScore};
    use forge_types::agent_interface::AgentMetadata;
    use tempfile::TempDir;

    fn fixture_scorecard() -> Scorecard {
        Scorecard {
            agent_metadata: AgentMetadata::heuristic("NoopAgent"),
            timestamp: "2026-05-16T00:00:00Z".to_string(),
            overall_score: 0.42,
            tier_scores: vec![
                TierScore {
                    tier: 1,
                    success_rate: 0.8,
                    mean_reward: 1.2,
                    mean_steps_to_completion: 50.0,
                    episodes_evaluated: 10,
                    scenarios_count: 1,
                },
                TierScore {
                    tier: 2,
                    success_rate: 0.4,
                    mean_reward: 0.6,
                    mean_steps_to_completion: 120.0,
                    episodes_evaluated: 10,
                    scenarios_count: 1,
                },
            ],
            scenario_results: vec![
                ScenarioResult {
                    scenario_id: "patrol_basic".to_string(),
                    tier: 1,
                    episodes: vec![
                        EpisodeResult {
                            seed: 0,
                            total_reward: 1.0,
                            success: true,
                            steps: 45,
                            terminated: true,
                            truncated: false,
                            mean_decision_time_ms: 0.3,
                            ..EpisodeResult::default()
                        },
                        EpisodeResult {
                            seed: 1,
                            total_reward: 1.4,
                            success: true,
                            steps: 52,
                            terminated: true,
                            truncated: false,
                            mean_decision_time_ms: 0.4,
                            ..EpisodeResult::default()
                        },
                    ],
                    success_rate: 1.0,
                    mean_reward: 1.2,
                    mean_decision_time_ms: 0.35,
                },
                ScenarioResult {
                    scenario_id: "harvest_advanced".to_string(),
                    tier: 2,
                    episodes: vec![EpisodeResult {
                        seed: 100,
                        total_reward: 0.5,
                        success: false,
                        steps: 200,
                        terminated: false,
                        truncated: true,
                        mean_decision_time_ms: 1.1,
                        ..EpisodeResult::default()
                    }],
                    success_rate: 0.0,
                    mean_reward: 0.5,
                    mean_decision_time_ms: 1.1,
                },
            ],
            summary: SummaryStats {
                total_episodes: 3,
                total_steps: 297,
                wall_clock_seconds: 1.5,
                mean_decision_latency_ms: 0.6,
            },
        }
    }

    #[test]
    fn exporter_rejects_empty_tracking_uri() {
        let exporter = MlflowExporter::new(PathBuf::new());
        let manifest = RunManifest::capture(&EvalConfig::default(), &[]);
        let err = exporter
            .export(&fixture_scorecard(), &manifest, Path::new("."))
            .expect_err("empty tracking uri must error");
        assert!(matches!(err, ExportError::InvalidTarget(_)));
    }

    #[test]
    fn exporter_writes_experiment_and_parent_run_layout() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let cfg = EvalConfig {
            run_id: Some("test-run-001".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        let exporter = MlflowExporter::new(tmp.path().to_path_buf());
        exporter
            .export(&scorecard, &manifest, &tmp.path().join("artifacts"))
            .unwrap();

        let exp_dir = tmp.path().join(DEFAULT_EXPERIMENT_ID);
        assert!(exp_dir.join("meta.yaml").exists(), "experiment meta.yaml");

        let parent_dir = exp_dir.join("test-run-001");
        assert!(parent_dir.join("meta.yaml").exists(), "parent meta.yaml");
        assert!(parent_dir.join("artifacts/scorecard.json").exists());
        assert!(parent_dir.join("artifacts/scorecard.md").exists());
        assert!(parent_dir.join("artifacts/manifest.json").exists());
        assert!(parent_dir
            .join("artifacts/tier_success_rates.html")
            .exists());
        // Scenario digest is surfaced as a tag (MLflow's inputs/ dir
        // tree is version-fragile and not worth wiring this phase).
        assert!(parent_dir.join("tags/forge.eval.scenarios_digest").exists());
        assert!(parent_dir.join("tags/forge.eval.scenario_count").exists());

        // System tags must populate
        assert!(parent_dir.join("tags/mlflow.source.git.commit").exists());
        assert!(parent_dir.join("tags/mlflow.source.git.branch").exists());
        assert!(parent_dir.join("tags/mlflow.source.name").exists());
        assert!(parent_dir.join("tags/mlflow.runName").exists());
        assert!(parent_dir.join("tags/mlflow.user").exists());
        assert!(parent_dir.join("tags/forge.eval.rustc_version").exists());

        // meta.yaml must encode FINISHED + start/end times
        let meta = std::fs::read_to_string(parent_dir.join("meta.yaml")).unwrap();
        assert!(meta.contains("status: 3"), "meta.yaml: {}", meta);
        assert!(meta.contains("start_time:"));
        assert!(meta.contains("end_time:"));
        assert!(meta.contains("run_id: test-run-001"));
    }

    #[test]
    fn exporter_writes_one_child_run_per_scenario_with_parent_tag() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let cfg = EvalConfig {
            run_id: Some("test-parent".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        MlflowExporter::new(tmp.path().to_path_buf())
            .export(&scorecard, &manifest, &tmp.path().join("artifacts"))
            .unwrap();

        let exp_dir = tmp.path().join(DEFAULT_EXPERIMENT_ID);
        for scenario in &scorecard.scenario_results {
            let cid = child_run_id("test-parent", &scenario.scenario_id);
            let child_dir = exp_dir.join(&cid);
            assert!(child_dir.join("meta.yaml").exists(), "child {} meta", cid);
            let parent_tag =
                std::fs::read_to_string(child_dir.join("tags/mlflow.parentRunId")).unwrap();
            assert_eq!(parent_tag.trim(), "test-parent");
            let run_name = std::fs::read_to_string(child_dir.join("tags/mlflow.runName")).unwrap();
            assert_eq!(run_name.trim(), scenario.scenario_id);
        }
    }

    #[test]
    fn child_run_metrics_are_step_indexed_per_episode() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let cfg = EvalConfig {
            run_id: Some("step-test".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        MlflowExporter::new(tmp.path().to_path_buf())
            .export(&scorecard, &manifest, &tmp.path().join("artifacts"))
            .unwrap();

        let exp_dir = tmp.path().join(DEFAULT_EXPERIMENT_ID);
        let patrol = &scorecard.scenario_results[0]; // 2 episodes
        let cid = child_run_id("step-test", &patrol.scenario_id);
        let metric_path = exp_dir.join(&cid).join("metrics/episode_reward");
        let content = std::fs::read_to_string(&metric_path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 2, "one metric line per episode");
        // Each line: "<ts_ms> <value> <step>"
        let steps: Vec<u64> = lines
            .iter()
            .map(|l| l.split_whitespace().nth(2).unwrap().parse().unwrap())
            .collect();
        assert_eq!(steps, vec![0, 1]);
    }

    #[test]
    fn per_scenario_named_metrics_appear_on_parent_run() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let cfg = EvalConfig {
            run_id: Some("named-test".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        MlflowExporter::new(tmp.path().to_path_buf())
            .export(&scorecard, &manifest, &tmp.path().join("artifacts"))
            .unwrap();

        let parent_metrics = tmp
            .path()
            .join(DEFAULT_EXPERIMENT_ID)
            .join("named-test")
            .join("metrics");
        assert!(parent_metrics
            .join("scenario_patrol_basic_success_rate")
            .exists());
        assert!(parent_metrics
            .join("scenario_harvest_advanced_success_rate")
            .exists());
        assert!(parent_metrics.join("tier_1_success_rate").exists());
        assert!(parent_metrics.join("tier_2_success_rate").exists());
        assert!(parent_metrics.join("overall_score").exists());
    }

    #[test]
    fn plotly_artifact_is_self_contained_html() {
        let html = render_tier_bar_chart_html(
            &[TierScore {
                tier: 1,
                success_rate: 0.5,
                mean_reward: 0.5,
                mean_steps_to_completion: 0.0,
                episodes_evaluated: 0,
                scenarios_count: 0,
            }],
            "test-run",
        );
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Plotly.newPlot"));
        assert!(html.contains("cdn.plot.ly"));
        assert!(html.contains("test-run"));
    }

    #[test]
    fn artifact_subdirs_copy_replays_and_trajectories_when_present() {
        let tmp = TempDir::new().unwrap();
        // Pre-populate artifacts_dir with fake replays + trajectories.
        let artifacts = tmp.path().join("artifacts");
        std::fs::create_dir_all(artifacts.join("replays")).unwrap();
        std::fs::create_dir_all(artifacts.join("trajectories")).unwrap();
        std::fs::write(artifacts.join("replays/ep0.bin"), b"fake-replay").unwrap();
        std::fs::write(artifacts.join("trajectories/ep0.jsonl"), b"{}").unwrap();

        let scorecard = fixture_scorecard();
        let cfg = EvalConfig {
            run_id: Some("copy-test".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        MlflowExporter::new(tmp.path().join("mlruns"))
            .export(&scorecard, &manifest, &artifacts)
            .unwrap();

        let parent_artifacts = tmp
            .path()
            .join("mlruns")
            .join(DEFAULT_EXPERIMENT_ID)
            .join("copy-test")
            .join("artifacts");
        assert!(parent_artifacts.join("replays/ep0.bin").exists());
        assert!(parent_artifacts.join("trajectories/ep0.jsonl").exists());
    }

    #[test]
    fn scenarios_digest_tag_records_combined_sha256() {
        let tmp = TempDir::new().unwrap();
        let scenario_path = tmp.path().join("scn.toml");
        std::fs::write(&scenario_path, b"id = \"x\"\n").unwrap();
        let cfg = EvalConfig::default();
        let manifest = RunManifest::capture(&cfg, &[scenario_path]);

        MlflowExporter::new(tmp.path().join("mlruns"))
            .export(
                &fixture_scorecard(),
                &manifest,
                &tmp.path().join("artifacts"),
            )
            .unwrap();

        let run_dir = tmp
            .path()
            .join("mlruns")
            .join(DEFAULT_EXPERIMENT_ID)
            .join(&manifest.run_id);
        let digest_tag =
            std::fs::read_to_string(run_dir.join("tags/forge.eval.scenarios_digest")).unwrap();
        assert_eq!(digest_tag.len(), 32, "32-hex-char short digest");
        assert!(digest_tag.chars().all(|c| c.is_ascii_hexdigit()));
        let count_tag =
            std::fs::read_to_string(run_dir.join("tags/forge.eval.scenario_count")).unwrap();
        assert_eq!(count_tag.trim(), "1");
    }

    #[test]
    fn child_run_id_is_deterministic_and_hex_32() {
        let id_a = child_run_id("parent", "scenario_a");
        let id_b = child_run_id("parent", "scenario_a");
        let id_c = child_run_id("parent", "scenario_b");
        assert_eq!(id_a, id_b);
        assert_ne!(id_a, id_c);
        assert_eq!(id_a.len(), 32);
        assert!(id_a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn re_exporting_same_scorecard_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let cfg = EvalConfig {
            run_id: Some("idem".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        let exporter = MlflowExporter::new(tmp.path().to_path_buf());
        exporter
            .export(&scorecard, &manifest, &tmp.path().join("art"))
            .unwrap();
        // First run wrote 2 metric lines for episode_reward (step 0, 1).
        // Idempotency contract (post-refactor): re-exporting the same
        // scorecard must produce the SAME 2 lines, not double them. The
        // truncate-before-write fix in `write_run_from_payload` (Slice 2
        // Copilot-review followup) deletes the metrics dir at the start
        // of each export so this contract is enforced.
        exporter
            .export(&scorecard, &manifest, &tmp.path().join("art"))
            .unwrap();
        let exp_dir = tmp.path().join(DEFAULT_EXPERIMENT_ID);
        let patrol_cid = child_run_id("idem", "patrol_basic");
        let metric_path = exp_dir.join(&patrol_cid).join("metrics/episode_reward");
        let lines: Vec<String> = std::fs::read_to_string(&metric_path)
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect();
        // Post-fix: exactly 2 lines (one per episode) regardless of how
        // many times export() is invoked with the same scorecard. The
        // previous test assertion of 4 documented the broken behaviour
        // the Copilot review flagged; this updated assertion pins the
        // corrected contract.
        assert_eq!(
            lines.len(),
            2,
            "re-export must NOT duplicate metric lines (got {} lines)",
            lines.len()
        );
    }

    #[test]
    fn sanitize_replaces_disallowed_chars_with_underscore() {
        assert_eq!(sanitize("hello world"), "hello world");
        assert_eq!(sanitize("a/b.c-d_e"), "a/b.c-d_e");
        assert_eq!(sanitize("bad:chars?go!"), "bad_chars_go_");
    }

    // `format_metric_value_handles_special_floats` moved to
    // `mlflow_fs::tests` alongside the function it tests (Slice 1.2b).
}
