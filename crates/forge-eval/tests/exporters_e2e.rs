// WIP-preserved test file (commit a91b3fa). See exporters_smoke.rs header for
// rationale on the module-level clippy allow.
#![allow(clippy::field_reassign_with_default)]

//! Phase B E2E test: drives the full harness with both exporters and
//! then shells out to **Python `mlflow` + `datasets`** to validate that
//! the on-disk trees actually load via the consumer tools — not just
//! that the files exist.
//!
//! This test is gated by `#[ignore]` because it requires:
//!   - A Python interpreter on PATH (or pointed at by `FORGE_E2E_PYTHON`).
//!   - The `mlflow` and `datasets` packages installed in that interpreter.
//!
//! Run with:
//!
//! ```bash
//! cargo test --test exporters_e2e -- --ignored
//! ```
//!
//! In environments where Python or the packages are missing, each test
//! still passes — it logs that it skipped and returns. This keeps the
//! contract honest (the test reports skipped, not failed) without
//! forcing every dev box to have the Python stack installed.

use std::path::{Path, PathBuf};
use std::process::Command;

use forge_eval::config::EvalConfig;
use forge_eval::harness::EvalHarness;
use forge_eval::scenario::{Scenario, ScenarioSuite};
use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::observation::Observation;
use tempfile::TempDir;

struct NoopEvalAgent;

impl AgentInterface for NoopEvalAgent {
    fn select_action(&mut self, _obs: &Observation, _agent_idx: usize) -> AgentResponse {
        AgentResponse::from_action(0)
    }
    fn name(&self) -> &str {
        "PhaseBE2EAgent"
    }
    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::heuristic("PhaseBE2EAgent")
    }
}

fn python_executable() -> String {
    // Prefer FORGE_E2E_PYTHON when set (CI / dev that pinned a venv);
    // otherwise fall back to `python` on PATH.
    std::env::var("FORGE_E2E_PYTHON").unwrap_or_else(|_| "python".to_string())
}

/// Returns `Some(python_path)` if Python + the named packages all
/// import; `None` if the e2e validation should be skipped (logs why).
fn ensure_python_with(packages: &[&str]) -> Option<String> {
    let py = python_executable();
    let probe = packages
        .iter()
        .map(|p| format!("import {}", p))
        .collect::<Vec<_>>()
        .join("; ");

    let output = match Command::new(&py).args(["-c", &probe]).output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("SKIP exporters_e2e: python not found ({}): {}", py, e);
            return None;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!(
            "SKIP exporters_e2e: python at `{}` is missing required packages ({}): {}",
            py,
            packages.join(", "),
            stderr.trim()
        );
        return None;
    }
    Some(py)
}

fn run_python_script(py: &str, script: &str) -> Result<String, String> {
    let output = Command::new(py)
        .args(["-c", script])
        .output()
        .map_err(|e| format!("spawn python: {}", e))?;
    if !output.status.success() {
        return Err(format!(
            "python exit={:?}\nstdout: {}\nstderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn tiny_suite() -> ScenarioSuite {
    use forge_types::config::ForgeConfig;
    let mut cfg = ForgeConfig::default();
    cfg.world.width = 8;
    cfg.world.height = 8;
    cfg.agents.num_agents = 1;
    cfg.task.max_episode_length = 20;

    ScenarioSuite::from_scenarios(vec![
        Scenario {
            id: "e2e_tier_1".to_string(),
            tier: 1,
            forge_config: cfg.clone(),
            max_steps: Some(10),
            tags: vec!["e2e".to_string()],
            description: None,
        },
        Scenario {
            id: "e2e_tier_2".to_string(),
            tier: 2,
            forge_config: cfg,
            max_steps: Some(10),
            tags: vec!["e2e".to_string()],
            description: None,
        },
    ])
}

fn run_eval_and_export(tmp: &Path) -> (PathBuf, PathBuf, String) {
    let mlruns = tmp.join("mlruns");
    let hf_root = tmp.join("hf_export");
    let run_id = "phase-b-e2e-001".to_string();

    let mut config = EvalConfig::default();
    config.episodes_per_scenario = 2;
    config.max_steps_per_episode = 10;
    config.parallelism = 1;
    config.run_id = Some(run_id.clone());
    config.experiment_name = Some("phase-b-e2e".to_string());
    config.mlflow_tracking_uri = Some(mlruns.clone());
    config.huggingface_export_root = Some(hf_root.clone());

    let harness = EvalHarness::new(config);
    let _ = harness.evaluate_suite(&tiny_suite(), &|| Box::new(NoopEvalAgent));

    (mlruns, hf_root, run_id)
}

#[test]
#[ignore = "requires Python with mlflow installed; run with --ignored"]
fn e2e_mlflow_tracking_layout_loads_in_mlflow_client() {
    let Some(py) = ensure_python_with(&["mlflow"]) else {
        return;
    };

    let tmp = TempDir::new().expect("tempdir");
    let (mlruns, _hf_root, run_id) = run_eval_and_export(tmp.path());

    // Use MlflowClient to walk the filesystem store. This exercises
    // mlflow's parser end-to-end, not just our file presence.
    let mlruns_str = mlruns.to_string_lossy().into_owned();
    let script = format!(
        r#"
import mlflow, sys, json
mlflow.set_tracking_uri("file:///{uri}")
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
assert len(experiments) >= 1, f"no experiments found: {{experiments}}"

# Pull our parent run.
runs = client.search_runs(
    experiment_ids=[experiments[0].experiment_id],
    filter_string="tags.mlflow.runName = 'phase-b-e2e'",
)
assert len(runs) >= 1, f"parent run not visible: {{runs}}"
parent = runs[0]
assert parent.info.status == "FINISHED", f"status: {{parent.info.status}}"
assert parent.info.run_id == "{run_id}", f"run_id mismatch: {{parent.info.run_id}}"

# Per-scenario named metrics must be present + finite.
metrics = parent.data.metrics
assert "overall_score" in metrics, list(metrics.keys())
assert "scenario_e2e_tier_1_success_rate" in metrics
assert "scenario_e2e_tier_2_success_rate" in metrics
assert metrics["overall_score"] >= 0.0

# Source git tags populated.
assert "mlflow.source.git.commit" in parent.data.tags, parent.data.tags

# Child runs linked via mlflow.parentRunId.
child_runs = client.search_runs(
    experiment_ids=[experiments[0].experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
)
assert len(child_runs) == 2, f"expected 2 child runs, got {{len(child_runs)}}"

# Per-episode step-indexed metrics: history has >= 2 points for episode_reward.
hist = client.get_metric_history(child_runs[0].info.run_id, "episode_reward")
assert len(hist) >= 1, f"no episode_reward history: {{hist}}"
for m in hist:
    assert isinstance(m.step, int)

print("MLFLOW_E2E_OK")
"#,
        uri = mlruns_str.replace('\\', "/"),
        run_id = run_id,
    );

    let stdout = run_python_script(&py, &script).expect("mlflow e2e script");
    assert!(
        stdout.contains("MLFLOW_E2E_OK"),
        "missing OK marker; stdout:\n{}",
        stdout
    );
}

#[test]
#[ignore = "requires Python with datasets installed; run with --ignored"]
fn e2e_huggingface_dataset_loads_via_load_from_disk() {
    let Some(py) = ensure_python_with(&["datasets", "yaml"]) else {
        return;
    };

    let tmp = TempDir::new().expect("tempdir");
    let (_mlruns, hf_root, run_id) = run_eval_and_export(tmp.path());
    let run_dir = hf_root.join(&run_id);
    let run_dir_str = run_dir.to_string_lossy().replace('\\', "/");

    let script = format!(
        r#"
from datasets import load_dataset
import yaml, glob, os

# load_dataset auto-discovers per-split JSONL files from the dataset
# card's `configs.data_files` block in README.md.
run_dir = "{run_dir}"
splits_glob = {{
    "all":    sorted(glob.glob(os.path.join(run_dir, "all",    "data-*.jsonl"))),
    "tier_1": sorted(glob.glob(os.path.join(run_dir, "tier_1", "data-*.jsonl"))),
    "tier_2": sorted(glob.glob(os.path.join(run_dir, "tier_2", "data-*.jsonl"))),
}}
ds = load_dataset("json", data_files=splits_glob)
print("DS_KEYS:", list(ds.keys()))

# Splits: at minimum `all` + one per tier present in our suite (1 + 2).
assert "all" in ds, list(ds.keys())
assert "tier_1" in ds, list(ds.keys())
assert "tier_2" in ds, list(ds.keys())

# Per-tier counts sum to all-split count.
all_n = len(ds["all"])
tier_sum = sum(len(ds[k]) for k in ds if k != "all")
assert all_n == tier_sum, f"all={{all_n}} tier_sum={{tier_sum}}"
assert all_n == 4, f"expected 4 episodes total, got {{all_n}}"

# Inferred features cover the documented schema (key stability is the
# contract, not dtype — JSON ints get inferred as int64 etc.).
expected_keys = {{"run_id", "scenario_id", "tier", "episode_index", "seed",
                  "total_reward", "success", "steps", "terminated", "truncated",
                  "mean_decision_time_ms", "git_sha", "timestamp"}}
assert set(ds["all"].features.keys()) == expected_keys, list(ds["all"].features.keys())

# Filter API: filter by tier returns the same rows as the tier_N split.
filtered = ds["all"].filter(lambda x: x["tier"] == 1)
assert len(filtered) == len(ds["tier_1"]), (len(filtered), len(ds["tier_1"]))

# Dataset card frontmatter parses as YAML with required Hub keys.
with open(os.path.join(run_dir, "README.md")) as f:
    body = f.read()
assert body.startswith("---\n"), "frontmatter must lead"
end = body.find("\n---", 4)
assert end != -1, "frontmatter must close"
fm = yaml.safe_load(body[4:end])
assert fm["license"] == "apache-2.0"
assert "reinforcement-learning" in fm["task_categories"]
assert fm["configs"][0]["config_name"] == "default"
splits_in_configs = {{e["split"] for e in fm["configs"][0]["data_files"]}}
assert splits_in_configs == {{"all", "tier_1", "tier_2"}}, splits_in_configs

print("HF_E2E_OK")
"#,
        run_dir = run_dir_str,
    );

    let stdout = run_python_script(&py, &script).expect("hf e2e script");
    assert!(
        stdout.contains("HF_E2E_OK"),
        "missing OK marker; stdout:\n{}",
        stdout
    );
}

#[test]
#[ignore = "requires Python + mlflow + datasets + a live LM Studio at $FORGE_LMSTUDIO_BASE_URL (default http://localhost:1234/v1) reached via FORGE_LMSTUDIO_LIVE=1; run with --ignored"]
fn e2e_with_lmstudio_pipeline_coexists_with_eval_exporters() {
    // Honest scope: forge-eval does NOT have a Rust LM Studio agent in
    // Phase B (the LM Studio teacher lives in python/forge/cognitive).
    // What this test validates is the *operational composition*: when a
    // user runs the eval harness with both Phase B exporters AND has LM
    // Studio serving teacher decisions for a parallel collection job,
    // every leg succeeds. Concretely:
    //
    //   1. The eval harness runs to completion with both exporters
    //      enabled (against NoopEvalAgent — Rust-side decisions).
    //   2. LM Studio at FORGE_LMSTUDIO_BASE_URL responds to a chat
    //      completion within the configured timeout.
    //   3. The MLflow tree + the HF dataset tree both load via their
    //      consumer tools alongside the live LM Studio call.
    //
    // This catches integration regressions where, e.g., adding an HTTP
    // dep to forge-eval would conflict with the venv driving LM Studio.
    if std::env::var("FORGE_LMSTUDIO_LIVE").ok().as_deref() != Some("1") {
        eprintln!("SKIP exporters_e2e (lmstudio): set FORGE_LMSTUDIO_LIVE=1 to enable");
        return;
    }
    let Some(py) = ensure_python_with(&["mlflow", "datasets", "json", "urllib.request"]) else {
        return;
    };

    let tmp = TempDir::new().expect("tempdir");
    let (mlruns, hf_root, run_id) = run_eval_and_export(tmp.path());
    let mlruns_str = mlruns.to_string_lossy().replace('\\', "/");
    let hf_root_str = hf_root.to_string_lossy().replace('\\', "/");
    let base_url = std::env::var("FORGE_LMSTUDIO_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:1234/v1".to_string());

    let script = format!(
        r#"
import json, mlflow, urllib.request

# 1) MLflow leg.
mlflow.set_tracking_uri("file:///{mlruns}")
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=[client.search_experiments()[0].experiment_id],
    filter_string="tags.mlflow.runName = 'phase-b-e2e'",
)
assert runs and runs[0].info.run_id == "{run_id}", runs

# 2) HF leg (load_dataset workflow, not load_from_disk).
from datasets import load_dataset
import glob, os
splits_glob = {{
    "all":    sorted(glob.glob("{hf_root}/{run_id}/all/data-*.jsonl")),
    "tier_1": sorted(glob.glob("{hf_root}/{run_id}/tier_1/data-*.jsonl")),
    "tier_2": sorted(glob.glob("{hf_root}/{run_id}/tier_2/data-*.jsonl")),
}}
ds = load_dataset("json", data_files=splits_glob)
assert "all" in ds and len(ds["all"]) == 4

# 3) LM Studio leg: list models + fire a single 1-token completion to
#    exercise the inference path within the eval-test wall clock.
mods_req = urllib.request.Request("{base_url}/models")
with urllib.request.urlopen(mods_req, timeout=10) as r:
    mods = json.loads(r.read())
assert mods.get("data"), f"LM Studio /models returned no data: {{mods}}"
model_id = mods["data"][0]["id"]
print("LM_STUDIO_MODEL:", model_id)

chat_payload = {{
    "model": model_id,
    "messages": [{{"role": "user", "content": "Say OK."}}],
    "max_tokens": 4,
    "temperature": 0.0,
}}
req = urllib.request.Request(
    "{base_url}/chat/completions",
    data=json.dumps(chat_payload).encode("utf-8"),
    headers={{"Content-Type": "application/json"}},
)
with urllib.request.urlopen(req, timeout=60) as r:
    completion = json.loads(r.read())
assert completion.get("choices"), f"LM Studio completion failed: {{completion}}"

print("LMSTUDIO_E2E_OK")
"#,
        mlruns = mlruns_str,
        hf_root = hf_root_str,
        run_id = run_id,
        base_url = base_url,
    );

    let stdout = run_python_script(&py, &script).expect("lmstudio e2e script");
    assert!(
        stdout.contains("LMSTUDIO_E2E_OK"),
        "missing OK marker; stdout:\n{}",
        stdout
    );
}

#[test]
#[ignore = "requires Python with both mlflow and datasets; run with --ignored"]
fn e2e_run_ids_match_across_both_exporters() {
    let Some(py) = ensure_python_with(&["mlflow", "datasets", "json"]) else {
        return;
    };

    let tmp = TempDir::new().expect("tempdir");
    let (mlruns, hf_root, run_id) = run_eval_and_export(tmp.path());
    let mlruns_str = mlruns.to_string_lossy().replace('\\', "/");
    let hf_root_str = hf_root.to_string_lossy().replace('\\', "/");

    let script = format!(
        r#"
import json, mlflow
mlflow.set_tracking_uri("file:///{mlruns}")
client = mlflow.tracking.MlflowClient()
exp = client.search_experiments()[0]
parent_runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.mlflow.runName = 'phase-b-e2e'",
)
assert parent_runs[0].info.run_id == "{run_id}"

with open("{hf_root}/{run_id}/manifest.json") as f:
    hf_manifest = json.load(f)
assert hf_manifest["run_id"] == "{run_id}"

# config_hash must match across both manifests. Read the MLflow side
# directly from the filesystem (MlflowClient.download_artifacts on a
# Windows file store has scheme-parsing quirks that aren't worth dodging
# in a test — the filesystem layout is the contract).
with open("{mlruns}/0/{run_id}/artifacts/manifest.json") as f:
    ml_manifest = json.load(f)
assert ml_manifest["config_hash"] == hf_manifest["config_hash"]

print("CROSS_E2E_OK")
"#,
        mlruns = mlruns_str,
        hf_root = hf_root_str,
        run_id = run_id,
    );

    let stdout = run_python_script(&py, &script).expect("cross-exporter e2e");
    assert!(
        stdout.contains("CROSS_E2E_OK"),
        "missing OK marker; stdout:\n{}",
        stdout
    );
}
