//! Shared helpers used by every MLflow sink (filesystem today, HTTP in
//! the upcoming Slice 2). Extracted from `mlflow.rs` so both sinks
//! consume one implementation rather than diverging copies.
//!
//! Scope: **pure helpers** only — small fragment-level utilities that
//! either format a string or derive a digest. The composite `RunPayload`
//! struct + `build_run_payload` that ties them into a sink-agnostic
//! intermediate representation arrives in Slice 1.2 alongside the
//! `MlflowFsSink` extraction; until then `mlflow.rs` re-exports each of
//! these so existing call sites stay byte-identical.
//!
//! All helpers are `pub`: when the HTTP sink lands in Slice 2 it consumes
//! the same `child_run_id`, the same `combined_scenario_digest`, the same
//! `render_tier_bar_chart_html`, etc. Drift between sinks is impossible
//! by construction.

use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{ExportError, ARTIFACT_MANIFEST_JSON, TIER_SPLIT_PREFIX};
use crate::manifest::{RunManifest, MANIFEST_SOURCE_NAME};
use crate::scorecard::{EpisodeResult, ScenarioResult, Scorecard, TierScore};

// ─── Transport-agnostic IR for an MLflow run ────────────────────────────────
//
// `RunPayload` is the single intermediate representation that every MLflow
// sink (filesystem today; HTTP in Slice 2) consumes. The builder
// [`build_run_payload`] produces a parent payload from a [`Scorecard`] +
// [`RunManifest`], with one `children[]` entry per scenario carrying its own
// per-episode metrics + per-scenario artefacts.
//
// **Artefact bytes are NOT in the payload.** [`ArtifactRef`] holds *refs*
// (paths into the harness's artefacts dir, or small inline bytes) so a
// 1000-episode run with gigabytes of replays + trajectories doesn't blow
// heap by buffering everything into the payload struct. Sinks stream from
// disk on demand.

/// A single param key/value, MLflow-friendly (string-typed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParamKv {
    /// Param key. Sanitised at sink-side (see [`super::mlflow_payload::sanitize`]).
    pub key: String,
    /// Param value. MLflow truncates values longer than 500 chars on read;
    /// sinks may pre-truncate to avoid writing files MLflow can't load.
    pub value: String,
}

/// A single tag key/value, MLflow-friendly (string-typed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TagKv {
    /// Tag key. `mlflow.*` keys get special UI treatment; `forge.eval.*`
    /// are custom Forge-namespaced.
    pub key: String,
    /// Tag value, free-form string.
    pub value: String,
}

/// A single metric sample. Multiple samples per `key` (different `step`)
/// render as a learning curve in the MLflow UI.
#[derive(Debug, Clone, Serialize)]
pub struct MetricSample {
    /// Metric name. Same key + multiple `step` values → time series.
    pub key: String,
    /// Numeric value. NaN/Infinity are serialised verbatim by the FsSink;
    /// the HTTP sink encodes them per the MLflow REST contract.
    pub value: f64,
    /// Wall-clock timestamp (milliseconds since epoch).
    pub timestamp_ms: u64,
    /// Sample step index, typically the episode index for per-episode
    /// metrics or `0` for run-level aggregates.
    pub step: u64,
}

/// How a sink should source the bytes of an artefact at `rel_path`.
///
/// `Inline` is for small artefacts built in memory (scorecard JSON/MD,
/// manifest JSON, the Plotly chart, the per-episode JSONL).
/// `File` and `Directory` are for things already on disk that the sink
/// reads on demand — replays/trajectories under the harness's
/// `artifacts_dir`. Never accumulate large file bytes into the payload
/// to keep peak heap predictable on long runs.
#[derive(Debug, Clone)]
pub enum ArtifactSource {
    /// Bytes already in memory (small artefacts only).
    Inline(Vec<u8>),
    /// Single file on disk; sink reads when it writes/uploads.
    File(PathBuf),
    /// Subdirectory on disk; sink walks it lazily file-by-file.
    Directory(PathBuf),
}

/// A logical artefact entry: where the bytes come from + the relative path
/// the sink should write/upload them under (relative to the per-run artefacts
/// root).
#[derive(Debug, Clone)]
pub struct ArtifactRef {
    /// Path relative to the run's artefacts root (no leading slash).
    pub rel_path: String,
    /// Where to read the bytes from when the sink commits the artefact.
    pub source: ArtifactSource,
}

/// Transport-agnostic payload for a single MLflow run, plus any nested
/// child runs. Both the filesystem and HTTP sinks build their on-disk /
/// REST writes from this struct.
#[derive(Debug, Clone)]
pub struct RunPayload {
    /// Stable run identifier. Parent payloads use [`RunManifest::run_id`];
    /// child payloads use [`child_run_id`] derived from `(parent, scenario)`.
    pub run_id: String,
    /// Experiment grouping name; mirrored from [`RunManifest::experiment_name`].
    pub experiment_name: String,
    /// Human-readable run name surfaced as the `mlflow.runName` tag (and the
    /// MLflow UI's run-table column).
    pub run_name: String,
    /// Param key/value pairs (string-typed, MLflow Params column).
    pub params: Vec<ParamKv>,
    /// Tag key/value pairs (MLflow Tags column + special `mlflow.*` UI hooks).
    pub tags: Vec<TagKv>,
    /// Step-indexed metric samples (run aggregates + per-episode curves).
    pub metrics: Vec<MetricSample>,
    /// Nested child runs (one per scenario for an eval suite). Empty for
    /// scenario-level payloads.
    pub children: Vec<RunPayload>,
    /// Artefact references — sinks read/upload bytes on demand to keep peak
    /// heap predictable on long runs.
    pub artifact_refs: Vec<ArtifactRef>,
}

// ─── Builder ────────────────────────────────────────────────────────────────

/// Convert a `bool` flag into the 0/1 float MLflow uses for boolean metrics.
/// Shared by every sink so the encoding stays consistent.
#[inline]
pub fn bool_metric(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// Build the parent run payload (with one nested child per scenario) for a
/// scorecard. `timestamp_ms` is the start-of-run wall clock millisecond
/// stamp every metric in the payload inherits (the FsSink writes a separate
/// `end_time` into meta.yaml; the HTTP sink calls `runs/update` post-export).
///
/// Returns `ExportError::Serialize` on a scorecard-JSON serialisation
/// failure; otherwise infallible.
#[allow(clippy::too_many_lines)]
pub fn build_run_payload(
    scorecard: &Scorecard,
    manifest: &RunManifest,
    artifacts_dir: &Path,
    timestamp_ms: u64,
) -> Result<RunPayload, ExportError> {
    let run_name = parent_run_name(manifest);

    // Params: agent metadata snapshot.
    let mut params = vec![
        ParamKv {
            key: "agent_type".to_string(),
            value: scorecard.agent_metadata.agent_type.clone(),
        },
        ParamKv {
            key: "model_name".to_string(),
            value: scorecard.agent_metadata.model_name.clone(),
        },
        ParamKv {
            key: "agent_version".to_string(),
            value: scorecard.agent_metadata.version.clone(),
        },
    ];
    for (k, v) in &scorecard.agent_metadata.parameters {
        params.push(ParamKv {
            // Caller will pass these through `sanitize()` at write-time; the
            // payload preserves the raw key + namespace prefix.
            key: format!("{}{}", crate::exporters::mlflow::AGENT_PARAM_KEY_PREFIX, k),
            value: v.clone(),
        });
    }

    // Metrics: parent-level summary + per-tier + per-scenario aggregates.
    let mut metrics = Vec::with_capacity(
        4 + scorecard.tier_scores.len() * 4 + scorecard.scenario_results.len() * 3,
    );
    push_metric(
        &mut metrics,
        "overall_score",
        scorecard.overall_score,
        timestamp_ms,
        0,
    );
    push_metric(
        &mut metrics,
        "total_episodes",
        scorecard.summary.total_episodes as f64,
        timestamp_ms,
        0,
    );
    push_metric(
        &mut metrics,
        "wall_clock_seconds",
        scorecard.summary.wall_clock_seconds,
        timestamp_ms,
        0,
    );
    push_metric(
        &mut metrics,
        "mean_decision_latency_ms",
        scorecard.summary.mean_decision_latency_ms,
        timestamp_ms,
        0,
    );
    for tier in &scorecard.tier_scores {
        let prefix = format!("{TIER_SPLIT_PREFIX}{}", tier.tier);
        push_metric(
            &mut metrics,
            &format!("{prefix}_success_rate"),
            tier.success_rate,
            timestamp_ms,
            0,
        );
        push_metric(
            &mut metrics,
            &format!("{prefix}_mean_reward"),
            tier.mean_reward,
            timestamp_ms,
            0,
        );
        push_metric(
            &mut metrics,
            &format!("{prefix}_mean_steps_to_completion"),
            tier.mean_steps_to_completion,
            timestamp_ms,
            0,
        );
        push_metric(
            &mut metrics,
            &format!("{prefix}_episodes_evaluated"),
            tier.episodes_evaluated as f64,
            timestamp_ms,
            0,
        );
    }
    for scenario in &scorecard.scenario_results {
        // Sanitisation of the scenario id happens at write time (sink-side);
        // payload preserves the original id so the HTTP sink can submit it
        // verbatim if the server permits.
        push_metric(
            &mut metrics,
            &format!("scenario_{}_success_rate", scenario.scenario_id),
            scenario.success_rate,
            timestamp_ms,
            0,
        );
        push_metric(
            &mut metrics,
            &format!("scenario_{}_mean_reward", scenario.scenario_id),
            scenario.mean_reward,
            timestamp_ms,
            0,
        );
        push_metric(
            &mut metrics,
            &format!("scenario_{}_mean_decision_time_ms", scenario.scenario_id),
            scenario.mean_decision_time_ms,
            timestamp_ms,
            0,
        );
    }

    // Tags: MLflow system tags + Forge custom tags. Source-version stamp
    // mirrors the original mlflow.rs write path.
    let mlflow_source_name = format!("{}@{}", MANIFEST_SOURCE_NAME, env!("CARGO_PKG_VERSION"));
    let scenarios_digest = combined_scenario_digest(manifest);
    let tags = vec![
        TagKv {
            key: "mlflow.source.git.commit".to_string(),
            value: manifest.git_sha.clone(),
        },
        TagKv {
            key: "mlflow.source.git.branch".to_string(),
            value: manifest.git_branch.clone(),
        },
        TagKv {
            key: "mlflow.source.name".to_string(),
            value: mlflow_source_name,
        },
        TagKv {
            key: "mlflow.source.type".to_string(),
            value: SOURCE_TYPE_LOCAL.to_string(),
        },
        TagKv {
            key: "mlflow.runName".to_string(),
            value: run_name.clone(),
        },
        TagKv {
            key: "mlflow.user".to_string(),
            value: manifest.user.clone(),
        },
        TagKv {
            key: "mlflow.note.content".to_string(),
            value: scorecard.to_markdown(),
        },
        TagKv {
            key: "forge.eval.rustc_version".to_string(),
            value: manifest.rustc_version.clone(),
        },
        TagKv {
            key: "forge.eval.config_hash".to_string(),
            value: manifest.config_hash.clone(),
        },
        TagKv {
            key: "forge.eval.scenarios_digest".to_string(),
            value: scenarios_digest,
        },
        TagKv {
            key: "forge.eval.scenario_count".to_string(),
            value: manifest.scenario_file_hashes.len().to_string(),
        },
        // Stable, client-side run id. The MLflow REST API assigns its own
        // server-side run id on create_run (the payload's run_id field is
        // ignored by the server), so this tag is the only way downstream
        // consumers (HuggingFace exporter writes under <hf_root>/<run_id>,
        // filesystem MLflow under <fs_root>/<run_id>) can correlate a
        // server-side MLflow run back to the manifest-anchored run id
        // shared across all exporters.
        TagKv {
            key: "forge.run_id".to_string(),
            value: manifest.run_id.clone(),
        },
    ];

    // Artefact refs — inline bytes for small derived files, dir refs for
    // optional replays/trajectories the harness writes under artifacts_dir.
    let scorecard_json = scorecard.to_json().map_err(ExportError::Serialize)?;
    let scorecard_md = scorecard.to_markdown();
    let manifest_json = serde_json::to_string_pretty(manifest)
        .map_err(|e| ExportError::Serialize(e.to_string()))?;
    let chart_html = render_tier_bar_chart_html(&scorecard.tier_scores, &run_name);

    let mut artifact_refs = vec![
        ArtifactRef {
            rel_path: super::mlflow::ARTIFACT_SCORECARD_JSON.to_string(),
            source: ArtifactSource::Inline(scorecard_json.into_bytes()),
        },
        ArtifactRef {
            rel_path: super::mlflow::ARTIFACT_SCORECARD_MD.to_string(),
            source: ArtifactSource::Inline(scorecard_md.into_bytes()),
        },
        ArtifactRef {
            rel_path: ARTIFACT_MANIFEST_JSON.to_string(),
            source: ArtifactSource::Inline(manifest_json.into_bytes()),
        },
        ArtifactRef {
            rel_path: super::mlflow::ARTIFACT_TIER_SUCCESS_RATES_HTML.to_string(),
            source: ArtifactSource::Inline(chart_html.into_bytes()),
        },
    ];
    // Optional on-disk subdirs — payload references them by path; the sink
    // skips silently if they don't exist (no error).
    push_optional_dir(&mut artifact_refs, artifacts_dir, "replays");
    push_optional_dir(&mut artifact_refs, artifacts_dir, "trajectories");

    // Build one child run per scenario.
    let children = scorecard
        .scenario_results
        .iter()
        .map(|scenario| build_child_payload(scenario, manifest, timestamp_ms))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(RunPayload {
        run_id: manifest.run_id.clone(),
        experiment_name: manifest.experiment_name.clone(),
        run_name,
        params,
        tags,
        metrics,
        children,
        artifact_refs,
    })
}

/// Build a single child-run payload from one scenario's results.
fn build_child_payload(
    scenario: &ScenarioResult,
    manifest: &RunManifest,
    timestamp_ms: u64,
) -> Result<RunPayload, ExportError> {
    let child_id = child_run_id(&manifest.run_id, &scenario.scenario_id);

    let params = vec![
        ParamKv {
            key: "scenario_id".to_string(),
            value: scenario.scenario_id.clone(),
        },
        ParamKv {
            key: "tier".to_string(),
            value: scenario.tier.to_string(),
        },
        ParamKv {
            key: "episode_count".to_string(),
            value: scenario.episodes.len().to_string(),
        },
    ];

    // Aggregate metrics + per-episode step-indexed metrics. Pre-size for
    // 3 aggregates + 6 per-episode samples × N episodes; keeps allocations
    // down on long runs.
    let mut metrics = Vec::with_capacity(3 + scenario.episodes.len() * 6);
    push_metric(
        &mut metrics,
        "scenario_success_rate",
        scenario.success_rate,
        timestamp_ms,
        0,
    );
    push_metric(
        &mut metrics,
        "scenario_mean_reward",
        scenario.mean_reward,
        timestamp_ms,
        0,
    );
    push_metric(
        &mut metrics,
        "scenario_mean_decision_time_ms",
        scenario.mean_decision_time_ms,
        timestamp_ms,
        0,
    );
    for (idx, ep) in scenario.episodes.iter().enumerate() {
        let step = idx as u64;
        push_metric(
            &mut metrics,
            "episode_reward",
            ep.total_reward,
            timestamp_ms,
            step,
        );
        push_metric(
            &mut metrics,
            "episode_steps",
            ep.steps as f64,
            timestamp_ms,
            step,
        );
        push_metric(
            &mut metrics,
            "episode_decision_time_ms",
            ep.mean_decision_time_ms,
            timestamp_ms,
            step,
        );
        push_metric(
            &mut metrics,
            "episode_success",
            bool_metric(ep.success),
            timestamp_ms,
            step,
        );
        push_metric(
            &mut metrics,
            "episode_terminated",
            bool_metric(ep.terminated),
            timestamp_ms,
            step,
        );
        push_metric(
            &mut metrics,
            "episode_truncated",
            bool_metric(ep.truncated),
            timestamp_ms,
            step,
        );
    }

    let tags = vec![
        TagKv {
            key: "mlflow.parentRunId".to_string(),
            value: manifest.run_id.clone(),
        },
        TagKv {
            key: "mlflow.runName".to_string(),
            value: scenario.scenario_id.clone(),
        },
        TagKv {
            key: "mlflow.source.git.commit".to_string(),
            value: manifest.git_sha.clone(),
        },
        TagKv {
            key: "mlflow.source.git.branch".to_string(),
            value: manifest.git_branch.clone(),
        },
        TagKv {
            key: "mlflow.source.type".to_string(),
            value: SOURCE_TYPE_LOCAL.to_string(),
        },
        TagKv {
            key: "mlflow.user".to_string(),
            value: manifest.user.clone(),
        },
        TagKv {
            key: "forge.eval.tier".to_string(),
            value: scenario.tier.to_string(),
        },
    ];

    // Episodes JSONL artefact, inline (typically small — one row per episode).
    let mut jsonl = String::new();
    for (idx, ep) in scenario.episodes.iter().enumerate() {
        let line = serde_json::to_string(&EpisodeLine::from(ep, idx as u32))
            .map_err(|e| ExportError::Serialize(e.to_string()))?;
        jsonl.push_str(&line);
        jsonl.push('\n');
    }
    let artifact_refs = vec![ArtifactRef {
        rel_path: super::mlflow::ARTIFACT_EPISODES_JSONL.to_string(),
        source: ArtifactSource::Inline(jsonl.into_bytes()),
    }];

    Ok(RunPayload {
        run_id: child_id,
        experiment_name: manifest.experiment_name.clone(),
        run_name: scenario.scenario_id.clone(),
        params,
        tags,
        metrics,
        children: Vec::new(),
        artifact_refs,
    })
}

/// MLflow `source.type` tag value for a non-Project (ad-hoc) run. Mirrors
/// the constant in `mlflow.rs` so the payload + filesystem writer agree.
const SOURCE_TYPE_LOCAL: &str = "LOCAL";

fn parent_run_name(manifest: &RunManifest) -> String {
    if manifest.experiment_name.is_empty() {
        format!(
            "eval-{}-{}",
            manifest.short_git_sha(),
            manifest.timestamp.timestamp()
        )
    } else {
        manifest.experiment_name.clone()
    }
}

#[inline]
fn push_metric(out: &mut Vec<MetricSample>, key: &str, value: f64, ts: u64, step: u64) {
    out.push(MetricSample {
        key: key.to_string(),
        value,
        timestamp_ms: ts,
        step,
    });
}

fn push_optional_dir(out: &mut Vec<ArtifactRef>, artifacts_dir: &Path, name: &str) {
    let dir = artifacts_dir.join(name);
    if dir.exists() {
        out.push(ArtifactRef {
            rel_path: name.to_string(),
            source: ArtifactSource::Directory(dir),
        });
    }
}

/// JSONL row layout for `episodes.jsonl` — kept here so both sinks emit
/// byte-identical bytes for the inline artefact.
#[derive(Serialize)]
struct EpisodeLine {
    episode_index: u32,
    seed: u64,
    total_reward: f64,
    success: bool,
    steps: u64,
    terminated: bool,
    truncated: bool,
    mean_decision_time_ms: f64,
}

impl EpisodeLine {
    fn from(ep: &EpisodeResult, idx: u32) -> Self {
        Self {
            episode_index: idx,
            seed: ep.seed,
            total_reward: ep.total_reward,
            success: ep.success,
            steps: ep.steps,
            terminated: ep.terminated,
            truncated: ep.truncated,
            mean_decision_time_ms: ep.mean_decision_time_ms,
        }
    }
}

/// Deterministic per-scenario child-run id derived from the parent run id
/// and the scenario id. Re-running export with the same `(parent, scenario)`
/// pair overwrites the same child run rather than creating a new one —
/// the idempotency guarantee both sinks rely on.
///
/// Output is a lower-case hex string of the first 16 bytes of the SHA-256
/// digest (32 hex chars total).
pub fn child_run_id(parent_run_id: &str, scenario_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(parent_run_id.as_bytes());
    hasher.update(b"::");
    hasher.update(scenario_id.as_bytes());
    let digest = hasher.finalize();
    hex_short(&digest, 16)
}

/// Combined sha256 over every scenario-file digest in the manifest.
/// Surfaced as the `forge.eval.scenarios_digest` tag in lieu of MLflow's
/// `inputs/` directory tree (which is version-fragile). Shared between
/// the filesystem and HTTP sinks so a re-run with identical scenarios
/// produces a stable digest regardless of transport.
pub fn combined_scenario_digest(manifest: &RunManifest) -> String {
    let mut hasher = Sha256::new();
    for (_, hash) in &manifest.scenario_file_hashes {
        hasher.update(hash.as_bytes());
    }
    hex_short(&hasher.finalize(), 16)
}

/// HTML-context escape: `<`, `>`, `&`, `"`, `'` rendered as entities.
/// Use for text that will be placed in HTML body / attributes (the
/// `<title>` element and the chart-title `format!` interpolation).
fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// JavaScript single-quoted string-literal escape. Escapes the chars
/// that would terminate the literal (`'`, `\`), the newline that would
/// produce a syntax error (`\n`, `\r`), and the `<` that could start
/// `</script>` and break out of the surrounding `<script>` block when
/// the page is parsed as HTML.
fn escape_js_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '<' => out.push_str("\\u003c"), // prevent `</script>` breakout
            _ => out.push(c),
        }
    }
    out
}

/// Pinned Plotly.js release for [`render_tier_bar_chart_html`]'s CDN
/// `<script>` tag. Pinned (rather than `-latest-`) so a regenerated report
/// renders identically regardless of when it's opened; bump deliberately,
/// not automatically. `3.7.0` is the newest release on the mature `3.x`
/// line as of this pin (the `4.0.0` major bump is only days old).
pub const PLOTLY_JS_VERSION: &str = "3.7.0";

/// Environment variable used to override the Plotly.js script URL
/// in air-gapped or offline environments.
pub const FORGE_PLOTLY_JS_URL_ENV: &str = "FORGE_PLOTLY_JS_URL";

/// Default CDN URL for Plotly.js.
pub const DEFAULT_PLOTLY_JS_URL: &str = "https://cdn.plot.ly/plotly-3.7.0.min.js";

/// Returns the configured Plotly.js script URL.
///
/// Priority:
/// 1. `FORGE_PLOTLY_JS_URL` environment variable if set and non-empty.
/// 2. Default CDN URL (`https://cdn.plot.ly/plotly-3.7.0.min.js`).
pub fn resolve_plotly_js_url() -> String {
    std::env::var(FORGE_PLOTLY_JS_URL_ENV)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js"))
}

/// Self-contained Plotly HTML for the per-tier success-rate + mean-reward
/// chart. Emitted as the `tier_success_rates.html` artefact under each
/// parent run's `artifacts/` directory.
///
/// `run_name` is interpolated into the chart title so a side-by-side view
/// of multiple runs in the MLflow UI is distinguishable at a glance. The
/// value comes from user-controlled `experiment_name`; we escape it for
/// both the surrounding HTML and the embedded JavaScript string contexts
/// so a name containing quotes / `</script>` cannot break the artefact
/// out of its container or execute arbitrary script when the file is
/// opened in MLflow's artefact viewer.
///
/// The Plotly library script URL is resolved via [`resolve_plotly_js_url`],
/// allowing air-gapped environments to override it via `FORGE_PLOTLY_JS_URL`.
pub fn render_tier_bar_chart_html(tier_scores: &[TierScore], run_name: &str) -> String {
    let plotly_url = resolve_plotly_js_url();
    render_tier_bar_chart_html_with_url(tier_scores, run_name, &plotly_url)
}

/// Variant of [`render_tier_bar_chart_html`] accepting an explicit Plotly script URL.
pub fn render_tier_bar_chart_html_with_url(
    tier_scores: &[TierScore],
    run_name: &str,
    plotly_url: &str,
) -> String {
    let tiers: Vec<u8> = tier_scores.iter().map(|t| t.tier).collect();
    let success: Vec<f64> = tier_scores.iter().map(|t| t.success_rate).collect();
    let reward: Vec<f64> = tier_scores.iter().map(|t| t.mean_reward).collect();
    let html_safe = escape_html(run_name);
    let js_safe = escape_js_string(run_name);
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FORGE eval tier success rates — {html_safe}</title>
<script src="{plotly_url}"></script>
</head>
<body>
<div id="chart" style="width:100%;height:480px;"></div>
<script>
Plotly.newPlot('chart', [
  {{x: {tiers:?}, y: {success:?}, type: 'bar', name: 'Success rate'}},
  {{x: {tiers:?}, y: {reward:?}, type: 'bar', name: 'Mean reward', yaxis: 'y2'}}
], {{
  title: 'Per-tier success rate + mean reward — {js_safe}',
  xaxis: {{title: 'Difficulty tier'}},
  yaxis: {{title: 'Success rate', range: [0, 1]}},
  yaxis2: {{title: 'Mean reward', overlaying: 'y', side: 'right'}},
  barmode: 'group'
}});
</script>
</body>
</html>
"#
    )
}

/// MLflow allows alphanumerics + `_ - . / ` (space) in param/metric/tag
/// keys; anything else is rewritten to `_` so the filesystem write never
/// fails on a key that came from agent metadata.
pub fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ' ') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Reject `run_id` values that are unsafe to use as a path component.
/// Both Phase B exporters (filesystem MLflow + HuggingFace) join the
/// manifest's `run_id` directly into an output path; without this check
/// a caller-controlled value like `"../escape"` or `"a/b"` could traverse
/// out of the exporter root and write where it shouldn't.
///
/// Rules:
/// - Non-empty after trim.
/// - No path separators (`/` or `\`).
/// - No `..` or `.` segments.
/// - No null bytes.
/// - No leading dot (prevents hidden files / `.git` collisions).
///
/// Returns the validated `&str` on success so call sites can chain it.
pub fn validate_run_id(run_id: &str) -> Result<&str, ExportError> {
    let trimmed = run_id.trim();
    if trimmed.is_empty() {
        return Err(ExportError::InvalidTarget(
            "run_id must not be empty or whitespace".to_string(),
        ));
    }
    if trimmed != run_id {
        return Err(ExportError::InvalidTarget(format!(
            "run_id must not have leading/trailing whitespace: {run_id:?}"
        )));
    }
    if run_id.contains('/') || run_id.contains('\\') {
        return Err(ExportError::InvalidTarget(format!(
            "run_id must not contain path separators: {run_id:?}"
        )));
    }
    if run_id == "." || run_id == ".." {
        return Err(ExportError::InvalidTarget(format!(
            "run_id must not be a `.` or `..` path segment: {run_id:?}"
        )));
    }
    if run_id.contains('\0') {
        return Err(ExportError::InvalidTarget(format!(
            "run_id must not contain null bytes: {run_id:?}"
        )));
    }
    if run_id.starts_with('.') {
        return Err(ExportError::InvalidTarget(format!(
            "run_id must not start with `.`: {run_id:?}"
        )));
    }
    Ok(run_id)
}

/// Lower-case hex encoding of the first `len` bytes of `bytes`. Centralised
/// so every digest representation across the eval exporters renders the
/// same way (`child_run_id`, `combined_scenario_digest`, future request-id
/// fingerprints in the HTTP sink, ...).
pub fn hex_short(bytes: &[u8], len: usize) -> String {
    let mut out = String::with_capacity(len * 2);
    for b in bytes.iter().take(len) {
        use std::fmt::Write;
        write!(&mut out, "{:02x}", b).expect("write to string");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{RunManifest, MANIFEST_SOURCE_NAME, UNKNOWN};
    use crate::scorecard::TierScore;
    use chrono::Utc;
    use std::path::PathBuf;

    /// Build a minimal RunManifest with `scenario_file_hashes` populated;
    /// all other fields are stable sentinels so tests focus on the helper
    /// under test rather than wall-clock noise.
    fn manifest_with_hashes(hashes: Vec<(PathBuf, String)>) -> RunManifest {
        RunManifest {
            run_id: "test-run".to_string(),
            experiment_name: "test-exp".to_string(),
            timestamp: Utc::now(),
            git_sha: UNKNOWN.to_string(),
            git_branch: UNKNOWN.to_string(),
            rustc_version: UNKNOWN.to_string(),
            user: UNKNOWN.to_string(),
            config_hash: "test-config-hash".to_string(),
            scenario_file_hashes: hashes,
            source_name: MANIFEST_SOURCE_NAME.to_string(),
        }
    }

    /// `child_run_id` MUST be deterministic AND a stable 32-char lower-hex
    /// string. Re-export sites depend on both properties for idempotency.
    #[test]
    fn child_run_id_is_deterministic_and_hex_32() {
        let a = child_run_id("parent-abc", "scenario-1");
        let b = child_run_id("parent-abc", "scenario-1");
        assert_eq!(a, b, "same inputs → same id");
        assert_eq!(a.len(), 32);
        assert!(a
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));

        let c = child_run_id("parent-abc", "scenario-2");
        assert_ne!(a, c, "different scenario → different id");
        let d = child_run_id("parent-XYZ", "scenario-1");
        assert_ne!(a, d, "different parent → different id");
    }

    /// `combined_scenario_digest` is deterministic + insensitive to the
    /// path component (only the hash matters).
    #[test]
    fn combined_scenario_digest_is_deterministic_and_path_independent() {
        let m1 = manifest_with_hashes(vec![
            (PathBuf::from("a.toml"), "deadbeef".to_string()),
            (PathBuf::from("b.toml"), "cafebabe".to_string()),
        ]);
        let m2 = manifest_with_hashes(vec![
            (PathBuf::from("other/path/a.toml"), "deadbeef".to_string()),
            (PathBuf::from("elsewhere/b.toml"), "cafebabe".to_string()),
        ]);
        assert_eq!(combined_scenario_digest(&m1), combined_scenario_digest(&m2));
        assert_eq!(combined_scenario_digest(&m1).len(), 32);
    }

    /// `combined_scenario_digest` IS sensitive to hash content + order.
    #[test]
    fn combined_scenario_digest_reflects_hash_set_changes() {
        let mut m = manifest_with_hashes(vec![
            (PathBuf::from("a"), "deadbeef".to_string()),
            (PathBuf::from("b"), "cafebabe".to_string()),
        ]);
        let baseline = combined_scenario_digest(&m);

        // Changing a hash value changes the digest.
        m.scenario_file_hashes[0].1 = "feedface".to_string();
        assert_ne!(baseline, combined_scenario_digest(&m));

        // Reordering changes the digest (hash order is observable).
        let m_reordered = manifest_with_hashes(vec![
            (PathBuf::from("a"), "cafebabe".to_string()),
            (PathBuf::from("b"), "deadbeef".to_string()),
        ]);
        let m_original = manifest_with_hashes(vec![
            (PathBuf::from("a"), "deadbeef".to_string()),
            (PathBuf::from("b"), "cafebabe".to_string()),
        ]);
        assert_ne!(
            combined_scenario_digest(&m_reordered),
            combined_scenario_digest(&m_original)
        );
    }

    #[test]
    fn render_tier_bar_chart_html_is_self_contained_html() {
        let tiers = vec![
            TierScore {
                tier: 1,
                success_rate: 0.75,
                mean_reward: 1.2,
                mean_steps_to_completion: 10.0,
                episodes_evaluated: 4,
                scenarios_count: 1,
            },
            TierScore {
                tier: 2,
                success_rate: 0.5,
                mean_reward: 0.8,
                mean_steps_to_completion: 12.0,
                episodes_evaluated: 4,
                scenarios_count: 1,
            },
        ];
        let html = render_tier_bar_chart_html(&tiers, "run-test");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains(&format!("plotly-{PLOTLY_JS_VERSION}.min.js")));
        assert!(html.contains("run-test"));
        // Tier values + success rates make it into the embedded JSON.
        assert!(html.contains("[1, 2]"));
        assert!(html.contains("0.75"));
        assert!(html.contains("0.5"));
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn resolve_plotly_js_url_defaults_to_pinned_cdn() {
        let _lock = ENV_LOCK.lock().unwrap();
        let orig = std::env::var(FORGE_PLOTLY_JS_URL_ENV).ok();
        std::env::remove_var(FORGE_PLOTLY_JS_URL_ENV);

        assert_eq!(
            resolve_plotly_js_url(),
            format!("https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js")
        );

        if let Some(val) = orig {
            std::env::set_var(FORGE_PLOTLY_JS_URL_ENV, val);
        }
    }

    #[test]
    fn resolve_plotly_js_url_respects_env_var() {
        let _lock = ENV_LOCK.lock().unwrap();
        let orig = std::env::var(FORGE_PLOTLY_JS_URL_ENV).ok();
        let custom_url = "https://internal-mirror.local/plotly-3.7.0.min.js";
        std::env::set_var(FORGE_PLOTLY_JS_URL_ENV, custom_url);

        assert_eq!(resolve_plotly_js_url(), custom_url);

        let html = render_tier_bar_chart_html(&[], "test-airgap");
        assert!(html.contains(custom_url));

        std::env::set_var(FORGE_PLOTLY_JS_URL_ENV, "   ");
        assert_eq!(
            resolve_plotly_js_url(),
            format!("https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js")
        );

        if let Some(val) = orig {
            std::env::set_var(FORGE_PLOTLY_JS_URL_ENV, val);
        } else {
            std::env::remove_var(FORGE_PLOTLY_JS_URL_ENV);
        }
    }

    #[test]
    fn render_tier_bar_chart_html_with_url_uses_supplied_script_url() {
        let custom_url = "/static/vendor/plotly.js";
        let html = render_tier_bar_chart_html_with_url(&[], "test-custom", custom_url);
        assert!(html.contains(&format!("<script src=\"{custom_url}\"></script>")));
    }

    #[test]
    fn sanitize_replaces_disallowed_chars_with_underscore() {
        // Allowed: alphanumerics + _ - . / space.
        assert_eq!(sanitize("a-b.c/d_e f9"), "a-b.c/d_e f9");
        // Disallowed → underscores.
        assert_eq!(sanitize("a!b@c#d$e%f^"), "a_b_c_d_e_f_");
        // Unicode alphanumerics pass through.
        assert_eq!(sanitize("π_α"), "π_α");
        // Empty input returns empty.
        assert_eq!(sanitize(""), "");
    }

    #[test]
    fn hex_short_truncates_to_requested_byte_count() {
        let bytes = [0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe];
        assert_eq!(hex_short(&bytes, 0), "");
        assert_eq!(hex_short(&bytes, 1), "de");
        assert_eq!(hex_short(&bytes, 3), "deadbe");
        assert_eq!(hex_short(&bytes, 6), "deadbeefcafe");
        // Asking for more bytes than available stops at the slice end.
        assert_eq!(hex_short(&bytes, 100), "deadbeefcafe");
    }

    // ─── build_run_payload contract tests ───────────────────────────────────

    use crate::scorecard::{EpisodeResult, ScenarioResult, Scorecard, SummaryStats};
    use forge_types::agent_interface::AgentMetadata;

    fn fixture_scorecard() -> Scorecard {
        let episodes = vec![
            EpisodeResult {
                seed: 1,
                total_reward: 0.75,
                success: true,
                steps: 10,
                terminated: true,
                truncated: false,
                mean_decision_time_ms: 1.5,
                ..EpisodeResult::default()
            },
            EpisodeResult {
                seed: 2,
                total_reward: 0.25,
                success: false,
                steps: 8,
                terminated: false,
                truncated: true,
                mean_decision_time_ms: 2.0,
                ..EpisodeResult::default()
            },
        ];
        let scenario = ScenarioResult::from_episodes("scenario_a".to_string(), 1, episodes);
        let tier_scores = vec![TierScore {
            tier: 1,
            success_rate: 0.5,
            mean_reward: 0.5,
            mean_steps_to_completion: 10.0,
            episodes_evaluated: 2,
            scenarios_count: 1,
        }];
        Scorecard {
            agent_metadata: AgentMetadata::heuristic("TestAgent"),
            timestamp: "1970-01-01T00:00:00Z".to_string(),
            overall_score: 0.5,
            tier_scores,
            scenario_results: vec![scenario],
            summary: SummaryStats {
                total_episodes: 2,
                total_steps: 18,
                wall_clock_seconds: 1.5,
                mean_decision_latency_ms: 1.75,
            },
        }
    }

    #[test]
    fn build_run_payload_includes_required_params_tags_metrics() {
        let scorecard = fixture_scorecard();
        let manifest = manifest_with_hashes(vec![]);
        let tmp = tempfile::tempdir().unwrap();

        let payload = build_run_payload(&scorecard, &manifest, tmp.path(), 12_345).unwrap();

        // Identity
        assert_eq!(payload.run_id, "test-run");
        assert_eq!(payload.experiment_name, "test-exp");
        assert!(!payload.run_name.is_empty());

        // Params: 3 fixed + 0 from empty agent metadata
        assert!(payload.params.iter().any(|p| p.key == "agent_type"));
        assert!(payload.params.iter().any(|p| p.key == "model_name"));
        assert!(payload.params.iter().any(|p| p.key == "agent_version"));

        // Tags: every required MLflow + Forge tag is present
        let tag_keys: Vec<&str> = payload.tags.iter().map(|t| t.key.as_str()).collect();
        for required in [
            "mlflow.source.git.commit",
            "mlflow.source.git.branch",
            "mlflow.source.name",
            "mlflow.source.type",
            "mlflow.runName",
            "mlflow.user",
            "mlflow.note.content",
            "forge.eval.rustc_version",
            "forge.eval.config_hash",
            "forge.eval.scenarios_digest",
            "forge.eval.scenario_count",
        ] {
            assert!(tag_keys.contains(&required), "missing tag {required}");
        }

        // Metrics: aggregate keys all present, every sample carries our ts
        let metric_keys: Vec<&str> = payload.metrics.iter().map(|m| m.key.as_str()).collect();
        for required in [
            "overall_score",
            "total_episodes",
            "wall_clock_seconds",
            "mean_decision_latency_ms",
        ] {
            assert!(metric_keys.contains(&required), "missing metric {required}");
        }
        assert!(payload.metrics.iter().all(|m| m.timestamp_ms == 12_345));
    }

    #[test]
    fn build_run_payload_emits_one_child_per_scenario_with_per_episode_metrics() {
        let scorecard = fixture_scorecard();
        let manifest = manifest_with_hashes(vec![]);
        let tmp = tempfile::tempdir().unwrap();

        let payload = build_run_payload(&scorecard, &manifest, tmp.path(), 1).unwrap();

        assert_eq!(payload.children.len(), 1);
        let child = &payload.children[0];

        assert_eq!(child.run_name, "scenario_a");
        // Deterministic child id derived from parent + scenario id
        assert_eq!(child.run_id, child_run_id("test-run", "scenario_a"));
        assert_eq!(child.children.len(), 0, "children have no grandchildren");

        // Child params
        assert!(child
            .params
            .iter()
            .any(|p| p.key == "scenario_id" && p.value == "scenario_a"));
        assert!(child
            .params
            .iter()
            .any(|p| p.key == "tier" && p.value == "1"));
        assert!(child
            .params
            .iter()
            .any(|p| p.key == "episode_count" && p.value == "2"));

        // Parent linkage tag
        assert!(child
            .tags
            .iter()
            .any(|t| t.key == "mlflow.parentRunId" && t.value == "test-run"));

        // Per-episode step-indexed metrics: 6 keys × 2 episodes
        let episode_metric_count = child
            .metrics
            .iter()
            .filter(|m| m.key.starts_with("episode_"))
            .count();
        assert_eq!(episode_metric_count, 12);

        // Episodes JSONL artefact is inline
        assert_eq!(child.artifact_refs.len(), 1);
        let ep_artifact = &child.artifact_refs[0];
        assert_eq!(
            ep_artifact.rel_path,
            super::super::mlflow::ARTIFACT_EPISODES_JSONL
        );
        match &ep_artifact.source {
            ArtifactSource::Inline(bytes) => {
                let text = std::str::from_utf8(bytes).unwrap();
                // Two rows, one per episode, NDJSON terminated.
                assert_eq!(text.lines().count(), 2);
                assert!(text.contains("\"seed\":1"));
                assert!(text.contains("\"seed\":2"));
            }
            other => panic!("expected Inline artefact, got {other:?}"),
        }
    }

    #[test]
    fn build_run_payload_includes_only_existing_optional_dir_refs() {
        let scorecard = fixture_scorecard();
        let manifest = manifest_with_hashes(vec![]);
        let tmp = tempfile::tempdir().unwrap();

        // No replays/ or trajectories/ yet → neither in payload.
        let payload = build_run_payload(&scorecard, &manifest, tmp.path(), 0).unwrap();
        let rel_paths: Vec<&str> = payload
            .artifact_refs
            .iter()
            .map(|a| a.rel_path.as_str())
            .collect();
        assert!(!rel_paths.contains(&"replays"));
        assert!(!rel_paths.contains(&"trajectories"));

        // Create replays/ → it shows up; trajectories/ stays absent.
        std::fs::create_dir(tmp.path().join("replays")).unwrap();
        let payload = build_run_payload(&scorecard, &manifest, tmp.path(), 0).unwrap();
        assert!(payload
            .artifact_refs
            .iter()
            .any(|a| a.rel_path == "replays" && matches!(a.source, ArtifactSource::Directory(_))));
        assert!(!payload
            .artifact_refs
            .iter()
            .any(|a| a.rel_path == "trajectories"));
    }

    #[test]
    fn build_run_payload_inline_artifacts_contain_expected_signatures() {
        let scorecard = fixture_scorecard();
        let manifest = manifest_with_hashes(vec![]);
        let tmp = tempfile::tempdir().unwrap();
        let payload = build_run_payload(&scorecard, &manifest, tmp.path(), 0).unwrap();

        // Locate each inline artefact by rel_path and probe a signature byte.
        let by_path: std::collections::HashMap<&str, &ArtifactSource> = payload
            .artifact_refs
            .iter()
            .map(|a| (a.rel_path.as_str(), &a.source))
            .collect();

        let scorecard_json = by_path
            .get(super::super::mlflow::ARTIFACT_SCORECARD_JSON)
            .expect("scorecard.json present");
        if let ArtifactSource::Inline(bytes) = scorecard_json {
            let text = std::str::from_utf8(bytes).unwrap();
            assert!(text.contains("overall_score"));
        }

        let scorecard_md = by_path
            .get(super::super::mlflow::ARTIFACT_SCORECARD_MD)
            .expect("scorecard.md present");
        if let ArtifactSource::Inline(bytes) = scorecard_md {
            assert!(std::str::from_utf8(bytes).unwrap().contains("Scorecard"));
        }

        let manifest_json = by_path
            .get(ARTIFACT_MANIFEST_JSON)
            .expect("manifest.json present");
        if let ArtifactSource::Inline(bytes) = manifest_json {
            assert!(std::str::from_utf8(bytes).unwrap().contains("test-run"));
        }

        let chart_html = by_path
            .get(super::super::mlflow::ARTIFACT_TIER_SUCCESS_RATES_HTML)
            .expect("tier chart present");
        if let ArtifactSource::Inline(bytes) = chart_html {
            assert!(std::str::from_utf8(bytes)
                .unwrap()
                .contains(&format!("plotly-{PLOTLY_JS_VERSION}.min.js")));
        }
    }

    #[test]
    fn bool_metric_maps_true_to_one_and_false_to_zero() {
        assert_eq!(bool_metric(true), 1.0);
        assert_eq!(bool_metric(false), 0.0);
    }

    // ─── Security regression tests (path traversal + escaping) ────────────

    #[test]
    fn validate_run_id_accepts_safe_values() {
        // The values build_run_payload typically produces — UUIDv4 hex
        // (32 chars), user-supplied identifiers without separators, and
        // child_run_id outputs.
        for id in [
            "a",
            "run-001",
            "abc123def456",
            "phase-b-smoke-001",
            "deadbeef".repeat(4).as_str(), // simulated child_run_id length
        ] {
            assert!(validate_run_id(id).is_ok(), "must accept safe id: {id:?}");
        }
    }

    #[test]
    fn validate_run_id_rejects_path_traversal_and_unsafe_chars() {
        // Each of these would let a caller escape the configured tracking
        // root or create unexpected files. Reject every shape.
        for bad in [
            "",
            "   ",
            "..",
            ".",
            "../escape",
            "..\\escape",
            "a/b",
            "a\\b",
            "/abs",
            "with\0null",
            ".hidden",
            " leading-space",
            "trailing-space ",
        ] {
            let err = validate_run_id(bad).unwrap_err();
            assert!(
                matches!(err, ExportError::InvalidTarget(_)),
                "must reject unsafe id {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn escape_html_neutralises_html_specials() {
        assert_eq!(
            escape_html("<script>alert('x')</script>"),
            "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"
        );
        assert_eq!(escape_html("a & b"), "a &amp; b");
        assert_eq!(escape_html("\"quoted\""), "&quot;quoted&quot;");
        // Unicode + plain text passes through unchanged.
        assert_eq!(escape_html("hello π"), "hello π");
        assert_eq!(escape_html(""), "");
    }

    #[test]
    fn escape_js_string_prevents_quote_break_and_script_breakout() {
        // `'` would terminate the surrounding string literal.
        assert_eq!(escape_js_string("it's"), "it\\'s");
        // `\` must be escaped first to avoid double-substitution surprises.
        assert_eq!(escape_js_string("a\\b"), "a\\\\b");
        // Newlines turn into `\n` so the literal stays single-line.
        assert_eq!(escape_js_string("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_js_string("a\rb"), "a\\rb");
        // `<` is escaped as `<` so `</script>` cannot terminate the
        // surrounding <script> block when the HTML is parsed. `>` is left
        // alone (only the opening `<` of `</script>` is dangerous).
        assert_eq!(
            escape_js_string("</script><img src=x onerror=alert(1)>"),
            "\\u003c/script>\\u003cimg src=x onerror=alert(1)>"
        );
    }

    #[test]
    fn render_tier_bar_chart_html_escapes_malicious_run_name() {
        // Construct a run_name that would break out of both the HTML
        // <title> and the JS string literal if not escaped.
        let payload_name = "evil\"</script><img src=x onerror=alert('XSS')>";
        let html = render_tier_bar_chart_html(&[], payload_name);
        // Raw payload must NOT appear in the output.
        assert!(
            !html.contains("</script><img"),
            "malicious payload leaked verbatim: {html}"
        );
        // The template emits exactly TWO legitimate `</script>` tags:
        // one for the Plotly CDN `<script src="..."></script>`, one for
        // the inline chart `<script>...</script>`. If a third appears,
        // that's the breakout the test is guarding against.
        let script_blocks = html.matches("</script>").count();
        assert_eq!(
            script_blocks, 2,
            "expected exactly two legitimate </script> tags from the template, got {script_blocks}: {html}"
        );
        // The escaped HTML <title> must contain &quot; instead of `"`.
        assert!(
            html.contains("&quot;"),
            "expected HTML-escaped quotes in <title>, got {html}"
        );
        // The escaped JS literal must contain `<` for the user-supplied
        // `<` inside the chart-title JS string (replaces what would have
        // been a `</script>` breakout).
        assert!(
            html.contains(r"</script"),
            "expected JS-escaped \\u003c for `<` in chart title JS literal, got {html}"
        );
    }
}
