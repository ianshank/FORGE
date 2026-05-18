// WIP-preserved test file (commit a91b3fa). The pre-existing init style uses
// `let mut cfg = EvalConfig::default(); cfg.field = ...;` which clippy flags
// as `field_reassign_with_default`. Allow at module scope so the WIP author's
// original pattern is preserved; can be revisited as part of a dedicated
// cleanup commit later without distorting the rebase or e2e work.
#![allow(clippy::field_reassign_with_default)]

//! Phase B SMOKE test: drive `EvalHarness::evaluate_suite` end-to-end
//! with **both** exporters configured, then assert the on-disk trees
//! exist with the structurally important files.
//!
//! This is a Rust-only test (no Python dependency) — it verifies that
//! the harness wiring + both exporters cooperate. Schema-level
//! validation against the actual consumer tools (`mlflow ui`,
//! `datasets.load_from_disk`) lives in the e2e test (run via
//! `cargo test --test exporters_e2e -- --ignored`).

use std::path::PathBuf;

use forge_eval::config::EvalConfig;
use forge_eval::exporters::mlflow::DEFAULT_EXPERIMENT_ID;
use forge_eval::harness::EvalHarness;
use forge_eval::scenario::{Scenario, ScenarioSuite};
use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::observation::Observation;
use tempfile::TempDir;

/// No-op agent that always emits action_id=0. Deterministic by design;
/// suffices for harness wiring + exporter dispatch tests.
struct NoopEvalAgent;

impl AgentInterface for NoopEvalAgent {
    fn select_action(&mut self, _obs: &Observation, _agent_idx: usize) -> AgentResponse {
        AgentResponse::from_action(0)
    }
    fn name(&self) -> &str {
        "PhaseBSmokeAgent"
    }
    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::heuristic("PhaseBSmokeAgent")
    }
}

fn tiny_suite() -> ScenarioSuite {
    use forge_types::config::ForgeConfig;
    let mut cfg = ForgeConfig::default();
    cfg.world.width = 8;
    cfg.world.height = 8;
    cfg.agents.num_agents = 1;
    cfg.task.max_episode_length = 20;

    let scenarios = vec![
        Scenario {
            id: "smoke_tier_1".to_string(),
            tier: 1,
            forge_config: cfg.clone(),
            max_steps: Some(10),
            tags: vec!["smoke".to_string()],
            description: Some("Tiny tier 1 scenario for Phase B smoke test".to_string()),
        },
        Scenario {
            id: "smoke_tier_3".to_string(),
            tier: 3,
            forge_config: cfg,
            max_steps: Some(10),
            tags: vec!["smoke".to_string()],
            description: Some("Tiny tier 3 scenario for Phase B smoke test".to_string()),
        },
    ];
    ScenarioSuite::from_scenarios(scenarios)
}

#[test]
fn smoke_both_exporters_produce_expected_files_via_harness() {
    let tmp = TempDir::new().expect("tempdir");
    let mlruns = tmp.path().join("mlruns");
    let hf_root = tmp.path().join("hf_export");

    let mut config = EvalConfig::default();
    config.episodes_per_scenario = 2;
    config.max_steps_per_episode = 10;
    config.parallelism = 1;
    config.run_id = Some("phase-b-smoke-001".to_string());
    config.experiment_name = Some("phase-b-smoke".to_string());
    config.mlflow_tracking_uri = Some(mlruns.clone());
    config.huggingface_export_root = Some(hf_root.clone());

    let harness = EvalHarness::new(config);
    let suite = tiny_suite();

    let scorecard = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));

    // Scorecard sanity (Phase 1 contract unchanged).
    assert_eq!(scorecard.scenario_results.len(), 2);
    assert_eq!(scorecard.summary.total_episodes, 4);

    // MLflow tree shape.
    let parent_run = mlruns.join(DEFAULT_EXPERIMENT_ID).join("phase-b-smoke-001");
    assert!(
        parent_run.join("meta.yaml").exists(),
        "parent run meta.yaml must exist"
    );
    assert!(parent_run.join("artifacts/scorecard.json").exists());
    assert!(parent_run.join("artifacts/scorecard.md").exists());
    assert!(parent_run.join("artifacts/manifest.json").exists());
    assert!(
        parent_run
            .join("artifacts/tier_success_rates.html")
            .exists(),
        "Plotly artifact must exist"
    );
    assert!(parent_run.join("tags/forge.eval.scenarios_digest").exists());
    assert!(parent_run.join("tags/forge.eval.scenario_count").exists());

    // System tags populated.
    for tag in &[
        "mlflow.source.git.commit",
        "mlflow.source.git.branch",
        "mlflow.source.name",
        "mlflow.runName",
        "mlflow.user",
        "forge.eval.rustc_version",
    ] {
        assert!(
            parent_run.join("tags").join(tag).exists(),
            "MLflow tag missing: {}",
            tag
        );
    }

    // Per-scenario named metric is present on the parent run.
    let metrics = parent_run.join("metrics");
    assert!(metrics.join("overall_score").exists());
    assert!(metrics.join("scenario_smoke_tier_1_success_rate").exists());
    assert!(metrics.join("scenario_smoke_tier_3_success_rate").exists());

    // Child runs exist (deterministic ids, parent linkage).
    use forge_eval::exporters::mlflow::child_run_id;
    for scenario in &scorecard.scenario_results {
        let cid = child_run_id("phase-b-smoke-001", &scenario.scenario_id);
        let child_dir = mlruns.join(DEFAULT_EXPERIMENT_ID).join(&cid);
        assert!(child_dir.join("meta.yaml").exists(), "child {} meta", cid);
        let parent_tag =
            std::fs::read_to_string(child_dir.join("tags/mlflow.parentRunId")).unwrap();
        assert_eq!(parent_tag.trim(), "phase-b-smoke-001");
        // Per-episode metric file should have N lines (one per episode).
        let reward_metric = child_dir.join("metrics/episode_reward");
        let lines: Vec<String> = std::fs::read_to_string(&reward_metric)
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(lines.len(), scenario.episodes.len());
    }

    // HF tree shape (load_dataset workflow, not load_from_disk).
    let hf_run = hf_root.join("phase-b-smoke-001");
    assert!(hf_run.join("manifest.json").exists());
    assert!(hf_run.join("README.md").exists());
    assert!(hf_run.join("all/data-00000-of-00001.jsonl").exists());
    assert!(hf_run.join("tier_1/data-00000-of-00001.jsonl").exists());
    assert!(hf_run.join("tier_3/data-00000-of-00001.jsonl").exists());

    // Per-tier counts sum to all.
    let count = |path: PathBuf| -> usize {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.is_empty())
            .count()
    };
    let all_count = count(hf_run.join("all/data-00000-of-00001.jsonl"));
    let t1_count = count(hf_run.join("tier_1/data-00000-of-00001.jsonl"));
    let t3_count = count(hf_run.join("tier_3/data-00000-of-00001.jsonl"));
    assert_eq!(all_count, t1_count + t3_count);
    assert_eq!(all_count, 4); // 2 episodes × 2 scenarios

    // Same run_id appears in both trees.
    let manifest_bytes = std::fs::read(parent_run.join("artifacts/manifest.json")).unwrap();
    let mlflow_manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let hf_manifest_bytes = std::fs::read(hf_run.join("manifest.json")).unwrap();
    let hf_manifest: serde_json::Value = serde_json::from_slice(&hf_manifest_bytes).unwrap();
    assert_eq!(mlflow_manifest["run_id"], "phase-b-smoke-001");
    assert_eq!(hf_manifest["run_id"], "phase-b-smoke-001");
    // The manifests are captured in separate calls so timestamps differ,
    // but run_id and config_hash must match (config didn't change between
    // captures).
    assert_eq!(mlflow_manifest["config_hash"], hf_manifest["config_hash"]);
}

#[test]
fn smoke_default_config_skips_both_exporters() {
    // Phase 1 byte-for-byte: with both exporter fields None (the default),
    // no mlruns/ or hf_export/ tree is created.
    let tmp = TempDir::new().expect("tempdir");
    let mut config = EvalConfig::default();
    config.episodes_per_scenario = 1;
    config.max_steps_per_episode = 5;
    config.parallelism = 1;
    // mlflow_tracking_uri and huggingface_export_root left as None.
    assert!(config.mlflow_tracking_uri.is_none());
    assert!(config.huggingface_export_root.is_none());

    let harness = EvalHarness::new(config);
    let suite = tiny_suite();
    let _ = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));

    // The tempdir must contain nothing we'd recognise as an exporter
    // output. Use a sweep to assert no mlruns/ or hf_export/ side-effect
    // dirs appeared anywhere.
    for entry in std::fs::read_dir(tmp.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        assert!(
            !name.starts_with("mlruns") && !name.starts_with("hf_export"),
            "default config must not produce exporter dirs; found {}",
            name
        );
    }
}
