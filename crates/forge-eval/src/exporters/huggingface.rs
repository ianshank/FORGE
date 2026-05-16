//! HuggingFace `datasets`-compatible exporter.
//!
//! Writes a `DatasetDict` directory under `<export_root>/<run_id>/` that
//! `datasets.load_from_disk(...)` opens natively and `huggingface-cli
//! upload` can push to the Hub.
//!
//! ## Layout produced
//!
//! ```text
//! <export_root>/<run_id>/
//!   dataset_dict.json                # {"splits": ["all", "tier_1", ...]}
//!   manifest.json                    # RunManifest, programmatic access
//!   README.md                        # Hub-ready dataset card (YAML frontmatter)
//!   all/
//!     dataset_info.json              # explicit HF features schema
//!     state.json                     # data file linkage
//!     data-00000-of-00001.jsonl      # one record per episode
//!   tier_1/ ... tier_N/              # one subdir per tier present
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tracing::instrument;

use super::{ExportError, Exporter};
use crate::manifest::RunManifest;
use crate::scorecard::Scorecard;

/// Composite split that mirrors every record from every tier. Always
/// emitted, even when no tier-specific splits exist.
pub const SPLIT_ALL: &str = "all";

/// HuggingFace `datasets` exporter writing a `DatasetDict` directory
/// under `<export_root>/<run_id>/`.
#[derive(Debug, Clone)]
pub struct HuggingFaceExporter {
    export_root: PathBuf,
}

impl HuggingFaceExporter {
    /// Construct an exporter rooted at `export_root`. The exporter will
    /// create the `<run_id>` subdirectory underneath it.
    pub fn new(export_root: PathBuf) -> Self {
        Self { export_root }
    }
}

impl Exporter for HuggingFaceExporter {
    fn name(&self) -> &'static str {
        "huggingface"
    }

    #[instrument(skip_all, fields(export_root = %self.export_root.display()))]
    fn export(
        &self,
        scorecard: &Scorecard,
        manifest: &RunManifest,
        _artifacts_dir: &Path,
    ) -> Result<(), ExportError> {
        if self.export_root.as_os_str().is_empty() {
            return Err(ExportError::InvalidTarget(
                "huggingface export_root must not be empty".to_string(),
            ));
        }

        let run_dir = self.export_root.join(&manifest.run_id);
        fs::create_dir_all(&run_dir)?;

        // Collect records (one per episode, denormalised).
        let records = collect_records(scorecard, manifest);

        // Determine which tier splits to emit (only tiers with episodes).
        let tiers_present: Vec<u8> = {
            let mut s: Vec<u8> = records.iter().map(|r| r.tier).collect();
            s.sort_unstable();
            s.dedup();
            s
        };

        // Always emit `all` + one per tier present.
        let mut split_names: Vec<String> = vec![SPLIT_ALL.to_string()];
        for tier in &tiers_present {
            split_names.push(format!("tier_{}", tier));
        }

        // Write `all` split.
        write_split(&run_dir, SPLIT_ALL, &records)?;
        // Write per-tier splits.
        for tier in &tiers_present {
            let split_name = format!("tier_{}", tier);
            let filtered: Vec<EpisodeRecord> = records
                .iter()
                .filter(|r| r.tier == *tier)
                .cloned()
                .collect();
            write_split(&run_dir, &split_name, &filtered)?;
        }

        // Root-level dataset_dict.json marker.
        let dict_marker = DatasetDictMarker {
            splits: split_names.clone(),
        };
        write_json(&run_dir.join("dataset_dict.json"), &dict_marker)?;

        // Manifest for programmatic access (mirrors what's embedded in README).
        manifest.write_json(&run_dir.join("manifest.json"))?;

        // README dataset card with YAML frontmatter for HF Hub.
        let card = render_dataset_card(scorecard, manifest, &split_names, &records);
        fs::write(run_dir.join("README.md"), card)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Record + per-split writers
// ---------------------------------------------------------------------------

/// Flattened per-episode record (one JSONL line). Denormalised — every
/// row carries the run-level + scenario-level context needed to load and
/// filter without joins.
#[derive(Debug, Clone, Serialize)]
struct EpisodeRecord {
    run_id: String,
    scenario_id: String,
    tier: u8,
    episode_index: u32,
    seed: u64,
    total_reward: f64,
    success: bool,
    steps: u64,
    terminated: bool,
    truncated: bool,
    mean_decision_time_ms: f64,
    git_sha: String,
    timestamp: String,
}

fn collect_records(scorecard: &Scorecard, manifest: &RunManifest) -> Vec<EpisodeRecord> {
    let mut records = Vec::new();
    for scenario in &scorecard.scenario_results {
        for (idx, ep) in scenario.episodes.iter().enumerate() {
            records.push(EpisodeRecord {
                run_id: manifest.run_id.clone(),
                scenario_id: scenario.scenario_id.clone(),
                tier: scenario.tier,
                episode_index: idx as u32,
                seed: ep.seed,
                total_reward: ep.total_reward,
                success: ep.success,
                steps: ep.steps,
                terminated: ep.terminated,
                truncated: ep.truncated,
                mean_decision_time_ms: ep.mean_decision_time_ms,
                git_sha: manifest.git_sha.clone(),
                timestamp: manifest.timestamp.to_rfc3339(),
            });
        }
    }
    records
}

fn write_split(
    run_dir: &Path,
    split_name: &str,
    records: &[EpisodeRecord],
) -> Result<(), ExportError> {
    let split_dir = run_dir.join(split_name);
    fs::create_dir_all(&split_dir)?;

    // JSONL data shard.
    let data_filename = "data-00000-of-00001.jsonl";
    let data_path = split_dir.join(data_filename);
    let mut file = fs::File::create(&data_path)?;
    for record in records {
        let line = serde_json::to_string(record)
            .map_err(|e| ExportError::Serialize(e.to_string()))?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
    }

    // Per-split fingerprint = sha256(jsonl bytes).
    let jsonl_bytes = fs::read(&data_path)?;
    let fingerprint = hex(&Sha256::digest(&jsonl_bytes));

    // state.json — HF datasets' Dataset.save_to_disk-compatible state.
    let state = DatasetState {
        data_files: vec![DataFile {
            filename: data_filename.to_string(),
        }],
        fingerprint,
        format_columns: None,
        format_kwargs: BTreeMap::new(),
        format_type: None,
        output_all_columns: false,
        split: split_name.to_string(),
    };
    write_json(&split_dir.join("state.json"), &state)?;

    // dataset_info.json — explicit HF Features schema, NOT auto-inferred.
    let info = DatasetInfo {
        description: "FORGE evaluation results — one record per episode.".to_string(),
        citation: String::new(),
        homepage: "https://github.com/ianshank/FORGE".to_string(),
        license: "apache-2.0".to_string(),
        features: episode_features_schema(),
        splits: BTreeMap::from([(
            split_name.to_string(),
            SplitInfo {
                name: split_name.to_string(),
                num_examples: records.len() as u64,
            },
        )]),
        version: DatasetVersion {
            version_str: "phase-b".to_string(),
            description: String::new(),
            major: 0,
            minor: 1,
            patch: 0,
        },
    };
    write_json(&split_dir.join("dataset_info.json"), &info)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Explicit HF features schema (must NOT rely on auto-inference)
// ---------------------------------------------------------------------------

fn episode_features_schema() -> BTreeMap<String, FeatureSpec> {
    let mut m = BTreeMap::new();
    m.insert("run_id".to_string(), value("string"));
    m.insert("scenario_id".to_string(), value("string"));
    m.insert("tier".to_string(), value("int32"));
    m.insert("episode_index".to_string(), value("int32"));
    m.insert("seed".to_string(), value("uint64"));
    m.insert("total_reward".to_string(), value("float64"));
    m.insert("success".to_string(), value("bool"));
    m.insert("steps".to_string(), value("int64"));
    m.insert("terminated".to_string(), value("bool"));
    m.insert("truncated".to_string(), value("bool"));
    m.insert("mean_decision_time_ms".to_string(), value("float64"));
    m.insert("git_sha".to_string(), value("string"));
    m.insert("timestamp".to_string(), value("string"));
    m
}

fn value(dtype: &str) -> FeatureSpec {
    FeatureSpec {
        dtype: dtype.to_string(),
        ty: "Value".to_string(),
    }
}

#[derive(Debug, Serialize)]
struct FeatureSpec {
    dtype: String,
    #[serde(rename = "_type")]
    ty: String,
}

// ---------------------------------------------------------------------------
// dataset_dict.json / dataset_info.json / state.json envelopes
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct DatasetDictMarker {
    splits: Vec<String>,
}

#[derive(Serialize)]
struct DatasetInfo {
    description: String,
    citation: String,
    homepage: String,
    license: String,
    features: BTreeMap<String, FeatureSpec>,
    splits: BTreeMap<String, SplitInfo>,
    version: DatasetVersion,
}

#[derive(Serialize)]
struct SplitInfo {
    name: String,
    num_examples: u64,
}

#[derive(Serialize)]
struct DatasetVersion {
    version_str: String,
    description: String,
    major: u32,
    minor: u32,
    patch: u32,
}

#[derive(Serialize)]
struct DatasetState {
    #[serde(rename = "_data_files")]
    data_files: Vec<DataFile>,
    #[serde(rename = "_fingerprint")]
    fingerprint: String,
    #[serde(rename = "_format_columns")]
    format_columns: Option<Vec<String>>,
    #[serde(rename = "_format_kwargs")]
    format_kwargs: BTreeMap<String, String>,
    #[serde(rename = "_format_type")]
    format_type: Option<String>,
    #[serde(rename = "_output_all_columns")]
    output_all_columns: bool,
    #[serde(rename = "_split")]
    split: String,
}

#[derive(Serialize)]
struct DataFile {
    filename: String,
}

// ---------------------------------------------------------------------------
// README.md dataset card with HF Hub YAML frontmatter
// ---------------------------------------------------------------------------

fn render_dataset_card(
    scorecard: &Scorecard,
    manifest: &RunManifest,
    split_names: &[String],
    records: &[EpisodeRecord],
) -> String {
    let pretty_name = format!("FORGE Evaluation — {}", manifest.experiment_name);
    let size_category = size_category_for(records.len());

    // configs[].data_files[] — one entry per split.
    let mut configs_block = String::from("configs:\n  - config_name: default\n    data_files:\n");
    for split in split_names {
        configs_block.push_str(&format!(
            "      - split: {}\n        path: \"{}/data-*.jsonl\"\n",
            split, split
        ));
    }

    // Per-split episode counts table.
    let mut splits_table = String::from("| Split | Episodes |\n|---|---|\n");
    for split in split_names {
        let count = if split == SPLIT_ALL {
            records.len()
        } else if let Some(tier_str) = split.strip_prefix("tier_") {
            tier_str
                .parse::<u8>()
                .ok()
                .map(|tier| records.iter().filter(|r| r.tier == tier).count())
                .unwrap_or(0)
        } else {
            0
        };
        splits_table.push_str(&format!("| {} | {} |\n", split, count));
    }

    format!(
        r#"---
language:
  - en
license: apache-2.0
pretty_name: "{pretty_name}"
size_categories:
  - "{size_category}"
task_categories:
  - reinforcement-learning
tags:
  - forge
  - simulation
  - evaluation
  - benchmark
{configs_block}---

# FORGE Evaluation Run `{run_id}`

Generated by `forge-eval` Phase B exporter on {timestamp}.

## Reproducibility

- **Git SHA**: `{git_sha}`
- **Branch**: `{git_branch}`
- **Rustc**: `{rustc_version}`
- **Config hash**: `{config_hash}`
- **Scenario file hashes**: see `manifest.json`.

## Overall

- Agent type: `{agent_type}`
- Model name: `{model_name}`
- Overall score: **{overall_score:.4}** (weighted)
- Total episodes: {total_episodes}
- Wall clock: {wall_clock_s:.2}s

## Splits

{splits_table}
## Schema

See `all/dataset_info.json` for the full HF Features schema. The schema is
explicit (not auto-inferred), so loading via `datasets.load_from_disk` is
type-stable across runs.

## Reproduce

```bash
git checkout {git_sha}
cargo run -p forge-eval --bin run_suite -- --config <path/to/eval.toml>
```
"#,
        pretty_name = pretty_name,
        size_category = size_category,
        configs_block = configs_block,
        run_id = manifest.run_id,
        timestamp = manifest.timestamp.to_rfc3339(),
        git_sha = manifest.git_sha,
        git_branch = manifest.git_branch,
        rustc_version = manifest.rustc_version,
        config_hash = manifest.config_hash,
        agent_type = scorecard.agent_metadata.agent_type,
        model_name = scorecard.agent_metadata.model_name,
        overall_score = scorecard.overall_score,
        total_episodes = scorecard.summary.total_episodes,
        wall_clock_s = scorecard.summary.wall_clock_seconds,
        splits_table = splits_table,
    )
}

/// HF Hub `size_categories` are one of `n<1K`, `1K<n<10K`, `10K<n<100K`,
/// `100K<n<1M`, `1M<n<10M`, `10M<n<100M`, `100M<n<1B`, `n>1B`.
fn size_category_for(n: usize) -> &'static str {
    match n {
        0..=999 => "n<1K",
        1_000..=9_999 => "1K<n<10K",
        10_000..=99_999 => "10K<n<100K",
        100_000..=999_999 => "100K<n<1M",
        1_000_000..=9_999_999 => "1M<n<10M",
        10_000_000..=99_999_999 => "10M<n<100M",
        100_000_000..=999_999_999 => "100M<n<1B",
        _ => "n>1B",
    }
}

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), ExportError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| ExportError::Serialize(e.to_string()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        write!(&mut out, "{:02x}", b).expect("write to string");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EvalConfig;
    use crate::scorecard::{
        EpisodeResult, ScenarioResult, Scorecard, SummaryStats, TierScore,
    };
    use forge_types::agent_interface::AgentMetadata;
    use tempfile::TempDir;

    fn fixture_scorecard() -> Scorecard {
        Scorecard {
            agent_metadata: AgentMetadata::heuristic("NoopAgent"),
            timestamp: "2026-05-16T00:00:00Z".to_string(),
            overall_score: 0.5,
            tier_scores: vec![
                TierScore {
                    tier: 1,
                    success_rate: 1.0,
                    mean_reward: 1.0,
                    mean_steps_to_completion: 50.0,
                    episodes_evaluated: 2,
                    scenarios_count: 1,
                },
                TierScore {
                    tier: 3,
                    success_rate: 0.0,
                    mean_reward: 0.0,
                    mean_steps_to_completion: 0.0,
                    episodes_evaluated: 1,
                    scenarios_count: 1,
                },
            ],
            scenario_results: vec![
                ScenarioResult {
                    scenario_id: "patrol".to_string(),
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
                        },
                        EpisodeResult {
                            seed: 1,
                            total_reward: 1.4,
                            success: true,
                            steps: 52,
                            terminated: true,
                            truncated: false,
                            mean_decision_time_ms: 0.4,
                        },
                    ],
                    success_rate: 1.0,
                    mean_reward: 1.2,
                    mean_decision_time_ms: 0.35,
                },
                ScenarioResult {
                    scenario_id: "harvest".to_string(),
                    tier: 3,
                    episodes: vec![EpisodeResult {
                        seed: 100,
                        total_reward: 0.5,
                        success: false,
                        steps: 200,
                        terminated: false,
                        truncated: true,
                        mean_decision_time_ms: 1.1,
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
    fn exporter_rejects_empty_export_root() {
        let exporter = HuggingFaceExporter::new(PathBuf::new());
        let manifest = RunManifest::capture(&EvalConfig::default(), &[]);
        let err = exporter
            .export(&fixture_scorecard(), &manifest, Path::new("."))
            .expect_err("empty export_root must error");
        assert!(matches!(err, ExportError::InvalidTarget(_)));
    }

    #[test]
    fn exporter_writes_datasetdict_with_all_and_per_tier_splits() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("hf-test-001".to_string());
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&scorecard, &manifest, &tmp.path().join("artifacts"))
            .unwrap();

        let run_dir = tmp.path().join("hf-test-001");
        assert!(run_dir.join("dataset_dict.json").exists());
        assert!(run_dir.join("manifest.json").exists());
        assert!(run_dir.join("README.md").exists());

        // `all` split always present.
        let all_dir = run_dir.join("all");
        assert!(all_dir.join("dataset_info.json").exists());
        assert!(all_dir.join("state.json").exists());
        assert!(all_dir.join("data-00000-of-00001.jsonl").exists());

        // Per-tier splits: only tiers with episodes (1 and 3 in fixture).
        assert!(run_dir.join("tier_1/dataset_info.json").exists());
        assert!(run_dir.join("tier_3/dataset_info.json").exists());
        assert!(!run_dir.join("tier_2").exists(), "no episodes at tier 2");
    }

    #[test]
    fn per_tier_split_counts_sum_to_all_split() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("counts".to_string());
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&scorecard, &manifest, &tmp.path().join("art"))
            .unwrap();

        let run_dir = tmp.path().join("counts");
        let count_lines =
            |split: &str| -> usize {
                std::fs::read_to_string(run_dir.join(split).join("data-00000-of-00001.jsonl"))
                    .unwrap()
                    .lines()
                    .filter(|l| !l.is_empty())
                    .count()
            };

        let all = count_lines("all");
        let t1 = count_lines("tier_1");
        let t3 = count_lines("tier_3");
        assert_eq!(all, t1 + t3);
        assert_eq!(all, 3); // 2 patrol + 1 harvest
        assert_eq!(t1, 2);
        assert_eq!(t3, 1);
    }

    #[test]
    fn dataset_info_features_match_jsonl_keys() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("schema".to_string());
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&fixture_scorecard(), &manifest, &tmp.path().join("art"))
            .unwrap();

        let run_dir = tmp.path().join("schema");
        let info: serde_json::Value = serde_json::from_slice(
            &std::fs::read(run_dir.join("all/dataset_info.json")).unwrap(),
        )
        .unwrap();
        let feature_keys: Vec<&str> = info["features"]
            .as_object()
            .unwrap()
            .keys()
            .map(|s| s.as_str())
            .collect();

        let first_line = std::fs::read_to_string(run_dir.join("all/data-00000-of-00001.jsonl"))
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .to_string();
        let row: serde_json::Value = serde_json::from_str(&first_line).unwrap();
        let row_keys: Vec<&str> = row.as_object().unwrap().keys().map(|s| s.as_str()).collect();

        let mut fk: Vec<&str> = feature_keys.clone();
        let mut rk: Vec<&str> = row_keys.clone();
        fk.sort();
        rk.sort();
        assert_eq!(fk, rk, "features schema must match JSONL keys");
    }

    #[test]
    fn dataset_info_uses_explicit_value_types_not_inferred() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("typed".to_string());
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&fixture_scorecard(), &manifest, &tmp.path().join("art"))
            .unwrap();

        let info: serde_json::Value = serde_json::from_slice(
            &std::fs::read(tmp.path().join("typed/all/dataset_info.json")).unwrap(),
        )
        .unwrap();

        // Spot-check a few critical types.
        assert_eq!(info["features"]["tier"]["dtype"], "int32");
        assert_eq!(info["features"]["tier"]["_type"], "Value");
        assert_eq!(info["features"]["success"]["dtype"], "bool");
        assert_eq!(info["features"]["total_reward"]["dtype"], "float64");
        assert_eq!(info["features"]["seed"]["dtype"], "uint64");
    }

    #[test]
    fn readme_starts_with_yaml_frontmatter_and_required_keys() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("card".to_string());
        cfg.experiment_name = Some("phase-b-smoke".to_string());
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&fixture_scorecard(), &manifest, &tmp.path().join("art"))
            .unwrap();

        let readme = std::fs::read_to_string(tmp.path().join("card/README.md")).unwrap();
        assert!(readme.starts_with("---\n"), "frontmatter must lead");
        // Required HF Hub keys.
        for key in [
            "license:",
            "pretty_name:",
            "size_categories:",
            "task_categories:",
            "tags:",
            "configs:",
            "config_name: default",
            "split: all",
            "split: tier_1",
            "split: tier_3",
        ] {
            assert!(readme.contains(key), "missing key: {}", key);
        }

        // Frontmatter must parse as valid YAML.
        let after_first = &readme[4..]; // skip "---\n"
        let end = after_first.find("\n---").expect("frontmatter must close");
        let frontmatter = &after_first[..end];
        let parsed: serde_yaml::Value = serde_yaml::from_str(frontmatter).expect("valid yaml");
        assert_eq!(parsed["license"], serde_yaml::Value::String("apache-2.0".into()));
    }

    #[test]
    fn size_category_for_handles_each_bucket() {
        assert_eq!(size_category_for(0), "n<1K");
        assert_eq!(size_category_for(999), "n<1K");
        assert_eq!(size_category_for(1_000), "1K<n<10K");
        assert_eq!(size_category_for(9_999), "1K<n<10K");
        assert_eq!(size_category_for(10_000), "10K<n<100K");
        assert_eq!(size_category_for(100_000_000), "100M<n<1B");
        assert_eq!(size_category_for(1_500_000_000), "n>1B");
    }

    #[test]
    fn dataset_dict_json_lists_every_emitted_split() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("marker".to_string());
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&fixture_scorecard(), &manifest, &tmp.path().join("art"))
            .unwrap();

        let marker: serde_json::Value = serde_json::from_slice(
            &std::fs::read(tmp.path().join("marker/dataset_dict.json")).unwrap(),
        )
        .unwrap();
        let splits: Vec<&str> = marker["splits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(splits, vec!["all", "tier_1", "tier_3"]);
    }

    #[test]
    fn handles_empty_scorecard_with_only_all_split() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("empty".to_string());
        let manifest = RunManifest::capture(&cfg, &[]);

        let empty = Scorecard {
            agent_metadata: AgentMetadata::heuristic("Empty"),
            timestamp: "2026-05-16T00:00:00Z".to_string(),
            overall_score: 0.0,
            tier_scores: vec![],
            scenario_results: vec![],
            summary: SummaryStats::default(),
        };

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&empty, &manifest, &tmp.path().join("art"))
            .unwrap();

        let run_dir = tmp.path().join("empty");
        assert!(run_dir.join("all/data-00000-of-00001.jsonl").exists());
        let content =
            std::fs::read_to_string(run_dir.join("all/data-00000-of-00001.jsonl")).unwrap();
        assert!(content.is_empty(), "no records => empty JSONL");
        // dataset_dict.json should list only "all".
        let marker: serde_json::Value = serde_json::from_slice(
            &std::fs::read(run_dir.join("dataset_dict.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(marker["splits"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn jsonl_records_are_denormalised_with_run_and_scenario_context() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = EvalConfig::default();
        cfg.run_id = Some("denorm".to_string());
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&fixture_scorecard(), &manifest, &tmp.path().join("art"))
            .unwrap();

        let line = std::fs::read_to_string(tmp.path().join("denorm/all/data-00000-of-00001.jsonl"))
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .to_string();
        let row: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(row["run_id"], "denorm");
        assert_eq!(row["scenario_id"], "patrol");
        assert_eq!(row["tier"], 1);
        assert!(row["git_sha"].is_string());
        assert!(row["timestamp"].is_string());
    }
}
