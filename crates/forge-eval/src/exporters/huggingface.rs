//! HuggingFace `datasets`-compatible exporter.
//!
//! Writes a Hugging-Face-friendly JSONL export under
//! `<export_root>/<run_id>/` that
//! `datasets.load_dataset("json", data_files=...)` opens natively and
//! `huggingface-cli upload` can push to the Hub. This is intentionally
//! NOT a `Dataset.save_to_disk()` Arrow DatasetDict — no
//! `dataset_info.json` / `state.json` / `dataset_dict.json` is emitted
//! (those would force consumers to ship `pyarrow`).
//!
//! ## Layout produced
//!
//! ```text
//! <export_root>/<run_id>/
//!   manifest.json                    # RunManifest, programmatic access
//!   README.md                        # Hub-ready dataset card with YAML
//!                                    # frontmatter (license, configs.data_files
//!                                    # pointing to each split's JSONL)
//!   all/
//!     data-00000-of-00001.jsonl      # one record per episode
//!   tier_1/ ... tier_N/              # one subdir per tier present
//! ```
//!
//! ## Loading
//!
//! ```python
//! from datasets import load_dataset
//! ds = load_dataset(<run_dir>)  # discovers splits via README's configs YAML
//! ```
//!
//! We do NOT emit `dataset_info.json` / `state.json` / `dataset_dict.json`
//! (those signal `Dataset.save_to_disk` Arrow IPC format and force
//! consumers to ship `pyarrow`). The `load_dataset` workflow reads our
//! JSONL natively and the dataset card's `configs.data_files` declares
//! the per-split paths — same Hub-uploadability, no Arrow dependency.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tracing::instrument;

use super::mlflow_payload::validate_run_id;
use super::{ExportError, Exporter, ARTIFACT_MANIFEST_JSON, TIER_SPLIT_PREFIX};
use crate::manifest::RunManifest;
use crate::scorecard::Scorecard;

/// Composite split that mirrors every record from every tier. Always
/// emitted, even when no tier-specific splits exist.
pub const SPLIT_ALL: &str = "all";

/// HuggingFace `datasets` single-shard data filename. The
/// `data-NNNNN-of-MMMMM.jsonl` naming convention is what
/// `datasets.load_dataset("json", data_files=...)` discovers via the
/// `configs[].data_files` glob in the dataset card. Single-shard for now;
/// if the harness later writes multi-shard splits this constant becomes a
/// formatter and the test below catches the contract change.
pub(crate) const HF_DATA_SHARD_FILENAME: &str = "data-00000-of-00001.jsonl";

/// Glob pattern used inside the dataset card's `configs[].data_files[].path`
/// entries to tell `datasets.load_dataset` which JSONL shards belong to a
/// given split. Must stay in sync with [`HF_DATA_SHARD_FILENAME`].
pub(crate) const HF_DATA_GLOB: &str = "data-*.jsonl";

// Static fields baked into every dataset card's YAML frontmatter. Kept as
// module-level constants (not literals inside `render_dataset_card`) so
// the contract is explicit + greppable + adjustable without touching the
// rendering code.
const FRONTMATTER_LANGUAGE: &str = "en";
const FRONTMATTER_LICENSE: &str = "apache-2.0";
const FRONTMATTER_TASK_CATEGORY: &str = "reinforcement-learning";
const FRONTMATTER_TAGS: &[&str] = &["forge", "simulation", "evaluation", "benchmark"];
const FRONTMATTER_CONFIG_DEFAULT: &str = "default";

/// Typed mirror of the HF Hub dataset-card YAML frontmatter. Serialising
/// through `serde_yaml` (rather than hand-rolling the YAML string)
/// guarantees that any colon, quote, or other reserved character in
/// caller-controlled fields like `pretty_name` is properly escaped and
/// the resulting card stays parseable.
#[derive(Debug, Serialize)]
struct DatasetCardFrontmatter {
    language: Vec<String>,
    license: String,
    pretty_name: String,
    size_categories: Vec<String>,
    task_categories: Vec<String>,
    tags: Vec<String>,
    configs: Vec<DatasetCardConfig>,
}

#[derive(Debug, Serialize)]
struct DatasetCardConfig {
    config_name: String,
    data_files: Vec<DatasetCardDataFile>,
}

#[derive(Debug, Serialize)]
struct DatasetCardDataFile {
    split: String,
    path: String,
}

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

        // Reject path-traversal-bearing run_ids before joining into the
        // export root. `manifest.run_id` is caller-influenced (config or
        // UUID); a value like `"../escape"` would otherwise write outside
        // the configured export tree.
        validate_run_id(&manifest.run_id)?;
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
            split_names.push(format!("{TIER_SPLIT_PREFIX}{tier}"));
        }

        // Write `all` split.
        write_split(&run_dir, SPLIT_ALL, &records)?;
        // Write per-tier splits.
        for tier in &tiers_present {
            let split_name = format!("{TIER_SPLIT_PREFIX}{tier}");
            let filtered: Vec<EpisodeRecord> = records
                .iter()
                .filter(|r| r.tier == *tier)
                .cloned()
                .collect();
            write_split(&run_dir, &split_name, &filtered)?;
        }

        // Manifest for programmatic access (mirrors what's embedded in README).
        manifest.write_json(&run_dir.join(ARTIFACT_MANIFEST_JSON))?;

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

    // JSONL data shard — consumer loads via load_dataset(<run_dir>),
    // which reads JSONL natively (no Arrow IPC dependency).
    let data_path = split_dir.join(HF_DATA_SHARD_FILENAME);
    let mut file = fs::File::create(&data_path)?;
    for record in records {
        let line =
            serde_json::to_string(record).map_err(|e| ExportError::Serialize(e.to_string()))?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
    }
    Ok(())
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

    // YAML frontmatter — serialize via `serde_yaml` so any special chars
    // in user-controlled strings (pretty_name embeds manifest.experiment_name)
    // are properly escaped. Hand-rolled string interpolation here would
    // happily emit invalid YAML for a name containing a colon or a quote.
    let frontmatter = DatasetCardFrontmatter {
        language: vec![FRONTMATTER_LANGUAGE.to_string()],
        license: FRONTMATTER_LICENSE.to_string(),
        pretty_name: pretty_name.clone(),
        size_categories: vec![size_category.to_string()],
        task_categories: vec![FRONTMATTER_TASK_CATEGORY.to_string()],
        tags: FRONTMATTER_TAGS.iter().map(|s| (*s).to_string()).collect(),
        configs: vec![DatasetCardConfig {
            config_name: FRONTMATTER_CONFIG_DEFAULT.to_string(),
            data_files: split_names
                .iter()
                .map(|split| DatasetCardDataFile {
                    split: split.clone(),
                    path: format!("{split}/{HF_DATA_GLOB}"),
                })
                .collect(),
        }],
    };
    let frontmatter_yaml = serde_yaml::to_string(&frontmatter)
        .unwrap_or_else(|e| format!("# frontmatter serialization failed: {e}\n"));

    // Per-split episode counts table.
    let mut splits_table = String::from("| Split | Episodes |\n|---|---|\n");
    for split in split_names {
        let count = if split == SPLIT_ALL {
            records.len()
        } else if let Some(tier_str) = split.strip_prefix(TIER_SPLIT_PREFIX) {
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
{frontmatter_yaml}---

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

Each JSONL row has these fields with stable types: `run_id` (string),
`scenario_id` (string), `tier` (int), `episode_index` (int), `seed` (int),
`total_reward` (float), `success` (bool), `steps` (int), `terminated`
(bool), `truncated` (bool), `mean_decision_time_ms` (float),
`git_sha` (string), `timestamp` (ISO-8601 string). Load via
`load_dataset("<dir>")` and `datasets` infers these from the JSONL.

## Reproduce

```bash
git checkout {git_sha}
cargo run -p forge-eval --bin run_suite -- --config <path/to/eval.toml>
```
"#,
        frontmatter_yaml = frontmatter_yaml,
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
    fn exporter_rejects_empty_export_root() {
        let exporter = HuggingFaceExporter::new(PathBuf::new());
        let manifest = RunManifest::capture(&EvalConfig::default(), &[]);
        let err = exporter
            .export(&fixture_scorecard(), &manifest, Path::new("."))
            .expect_err("empty export_root must error");
        assert!(matches!(err, ExportError::InvalidTarget(_)));
    }

    #[test]
    fn exporter_writes_per_tier_jsonl_splits_plus_card_and_manifest() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let cfg = EvalConfig {
            run_id: Some("hf-test-001".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&scorecard, &manifest, &tmp.path().join("artifacts"))
            .unwrap();

        let run_dir = tmp.path().join("hf-test-001");
        assert!(run_dir.join("manifest.json").exists());
        assert!(run_dir.join("README.md").exists());
        assert!(run_dir.join("all/data-00000-of-00001.jsonl").exists());

        // Per-tier splits: only tiers with episodes (1 and 3 in fixture).
        assert!(run_dir.join("tier_1/data-00000-of-00001.jsonl").exists());
        assert!(run_dir.join("tier_3/data-00000-of-00001.jsonl").exists());
        assert!(!run_dir.join("tier_2").exists(), "no episodes at tier 2");
    }

    #[test]
    fn per_tier_split_counts_sum_to_all_split() {
        let tmp = TempDir::new().unwrap();
        let scorecard = fixture_scorecard();
        let cfg = EvalConfig {
            run_id: Some("counts".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&scorecard, &manifest, &tmp.path().join("art"))
            .unwrap();

        let run_dir = tmp.path().join("counts");
        let count_lines = |split: &str| -> usize {
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
    fn jsonl_record_keys_are_the_documented_schema() {
        // The HF consumer relies on JSONL key stability — load_dataset
        // infers Features from these. If a contributor renames a field,
        // every downstream Hub dataset breaks. Lock the surface here.
        let tmp = TempDir::new().unwrap();
        let cfg = EvalConfig {
            run_id: Some("schema".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&fixture_scorecard(), &manifest, &tmp.path().join("art"))
            .unwrap();

        let first_line =
            std::fs::read_to_string(tmp.path().join("schema/all/data-00000-of-00001.jsonl"))
                .unwrap()
                .lines()
                .next()
                .unwrap()
                .to_string();
        let row: serde_json::Value = serde_json::from_str(&first_line).unwrap();
        let mut row_keys: Vec<&str> = row
            .as_object()
            .unwrap()
            .keys()
            .map(|s| s.as_str())
            .collect();
        row_keys.sort();
        let expected = vec![
            "episode_index",
            "git_sha",
            "mean_decision_time_ms",
            "run_id",
            "scenario_id",
            "seed",
            "steps",
            "success",
            "terminated",
            "tier",
            "timestamp",
            "total_reward",
            "truncated",
        ];
        assert_eq!(row_keys, expected, "JSONL schema drift");
    }

    #[test]
    fn readme_starts_with_yaml_frontmatter_and_required_keys() {
        let tmp = TempDir::new().unwrap();
        let cfg = EvalConfig {
            run_id: Some("card".to_string()),
            experiment_name: Some("phase-b-smoke".to_string()),
            ..Default::default()
        };
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
        assert_eq!(
            parsed["license"],
            serde_yaml::Value::String("apache-2.0".into())
        );
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
    fn card_configs_block_lists_every_split() {
        let tmp = TempDir::new().unwrap();
        let cfg = EvalConfig {
            run_id: Some("marker".to_string()),
            ..Default::default()
        };
        let manifest = RunManifest::capture(&cfg, &[]);

        HuggingFaceExporter::new(tmp.path().to_path_buf())
            .export(&fixture_scorecard(), &manifest, &tmp.path().join("art"))
            .unwrap();

        let readme = std::fs::read_to_string(tmp.path().join("marker/README.md")).unwrap();
        let end = readme[4..].find("\n---").unwrap();
        let frontmatter = &readme[4..4 + end];
        let parsed: serde_yaml::Value = serde_yaml::from_str(frontmatter).unwrap();
        let splits: Vec<String> = parsed["configs"][0]["data_files"]
            .as_sequence()
            .unwrap()
            .iter()
            .map(|e| e["split"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(splits, vec!["all", "tier_1", "tier_3"]);
    }

    #[test]
    fn handles_empty_scorecard_with_only_all_split() {
        let tmp = TempDir::new().unwrap();
        let cfg = EvalConfig {
            run_id: Some("empty".to_string()),
            ..Default::default()
        };
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
        // Card configs block should list only "all".
        let readme = std::fs::read_to_string(run_dir.join("README.md")).unwrap();
        let end = readme[4..].find("\n---").unwrap();
        let parsed: serde_yaml::Value = serde_yaml::from_str(&readme[4..4 + end]).unwrap();
        let splits = parsed["configs"][0]["data_files"].as_sequence().unwrap();
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0]["split"], serde_yaml::Value::String("all".into()));
    }

    #[test]
    fn jsonl_records_are_denormalised_with_run_and_scenario_context() {
        let tmp = TempDir::new().unwrap();
        let cfg = EvalConfig {
            run_id: Some("denorm".to_string()),
            ..Default::default()
        };
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
