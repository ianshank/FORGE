//! Phase 1 integration tests: end-to-end suite + on-disk persistence.
//!
//! These tests exercise the full pipeline a real user would touch:
//! 1. Author scenario TOML files on disk.
//! 2. Load them through [`ScenarioSuite::load_dir`].
//! 3. Run a harness with persistence enabled.
//! 4. Verify on-disk artefacts (replays, trajectories, metadata, scorecard).
//!
//! Backward compatibility check: the legacy single-scenario `evaluate()`
//! entry point is exercised alongside the new `evaluate_suite()` path.

use std::path::PathBuf;

use forge_eval::config::EvalConfig;
use forge_eval::harness::EvalHarness;
use forge_eval::output::{OutputConfig, ScorecardFormat};
use forge_eval::scenario::{Scenario, ScenarioSuite};
use forge_eval::scorecard::Scorecard;
use forge_replay::compact::CompactReplay;
use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::observation::Observation;
use tempfile::tempdir;

/// Pure heuristic agent for tests — always returns the configured action.
struct ConstAgent {
    action: u32,
    name: String,
}

impl ConstAgent {
    fn new(action: u32, name: &str) -> Self {
        Self {
            action,
            name: name.to_string(),
        }
    }
}

impl AgentInterface for ConstAgent {
    fn select_action(&mut self, _obs: &Observation, _agent_idx: usize) -> AgentResponse {
        AgentResponse::from_action(self.action)
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::heuristic(&self.name)
    }
}

fn base_eval_config(episodes: u32) -> EvalConfig {
    let mut cfg = EvalConfig {
        episodes_per_scenario: episodes,
        max_steps_per_episode: 12,
        base_seed: 1234,
        ..EvalConfig::default()
    };
    cfg.base_forge_config.world.width = 16;
    cfg.base_forge_config.world.height = 16;
    cfg.base_forge_config.agents.num_agents = 1;
    cfg.base_forge_config.agents.comm_vocab_size = 0;
    cfg.base_forge_config.task.max_episode_length = 12;
    cfg
}

fn write_scenario(dir: &std::path::Path, file_name: &str, scenario: &Scenario) {
    let p = dir.join(file_name);
    std::fs::write(&p, toml::to_string(scenario).unwrap()).unwrap();
}

#[test]
fn loads_suite_from_toml_dir_and_runs_evaluation() {
    let dir = tempdir().unwrap();
    let cfg = base_eval_config(2);

    write_scenario(
        dir.path(),
        "01_easy.toml",
        &Scenario::new("easy", 1, cfg.base_forge_config.clone()),
    );
    write_scenario(
        dir.path(),
        "02_hard.toml",
        &Scenario::new("hard", 3, cfg.base_forge_config.clone()),
    );

    let suite = ScenarioSuite::load_dir(dir.path()).unwrap();
    assert_eq!(suite.scenarios.len(), 2);

    let harness = EvalHarness::new(cfg);
    let card = harness.evaluate_suite(&suite, &|| Box::new(ConstAgent::new(0, "const0")));
    assert_eq!(card.scenario_results.len(), 2);
    assert_eq!(card.summary.total_episodes, 4);
    assert!(card.tier_scores.iter().any(|t| t.tier == 1));
    assert!(card.tier_scores.iter().any(|t| t.tier == 3));
}

#[test]
fn persistence_artefacts_round_trip_from_disk() {
    let out_dir = tempdir().unwrap();
    let mut cfg = base_eval_config(1);
    cfg.output = OutputConfig {
        enabled: true,
        dir: out_dir.path().to_path_buf(),
        write_replays: true,
        write_trajectories: true,
        write_scorecard: true,
        scorecard_format: ScorecardFormat::Both,
        ..OutputConfig::default()
    };

    let suite = ScenarioSuite::from_scenarios(vec![Scenario::new(
        "round_trip",
        2,
        cfg.base_forge_config.clone(),
    )]);

    let harness = EvalHarness::new(cfg.clone());
    let card = harness.evaluate_suite(&suite, &|| Box::new(ConstAgent::new(0, "noop")));

    // Scorecard files exist and round-trip.
    let json_path = cfg.output.scorecard_path("json");
    let md_path = cfg.output.scorecard_path("md");
    assert!(json_path.exists());
    assert!(md_path.exists());
    let json_blob = std::fs::read_to_string(&json_path).unwrap();
    let restored: Scorecard = Scorecard::from_json(&json_blob).unwrap();
    assert_eq!(restored.summary.total_episodes, card.summary.total_episodes);
    assert_eq!(restored.scenario_results.len(), 1);

    let md_blob = std::fs::read_to_string(&md_path).unwrap();
    assert!(md_blob.contains("FORGE Evaluation Scorecard"));

    // Replay round-trips through CompactReplay::from_bytes.
    let replay_path = cfg.output.replay_path("round_trip", cfg.base_seed);
    let bytes = std::fs::read(&replay_path).unwrap();
    let replay = CompactReplay::from_bytes(&bytes).unwrap();
    assert_eq!(replay.seed, cfg.base_seed);
    assert_eq!(replay.metadata.agent_names, vec!["noop".to_string()]);
    assert_eq!(replay.metadata.scenario_id.as_deref(), Some("round_trip"));

    // Trajectory metadata sidecar exists with the right shape.
    let meta_path = cfg
        .output
        .trajectory_metadata_path("round_trip", cfg.base_seed);
    let meta_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta_json["seed"], cfg.base_seed);
    assert_eq!(meta_json["scenario_id"], "round_trip");
}

#[test]
fn legacy_evaluate_remains_backward_compatible() {
    let cfg = base_eval_config(2);
    let harness = EvalHarness::new(cfg);
    let card = harness.evaluate(&|| Box::new(ConstAgent::new(0, "const")));
    assert_eq!(card.scenario_results.len(), 1);
    assert_eq!(card.scenario_results[0].scenario_id, "default");
    assert_eq!(card.tier_scores[0].tier, 1);
    assert_eq!(card.summary.total_episodes, 2);
}

#[test]
fn tier_filter_skips_excluded_scenarios() {
    let mut cfg = base_eval_config(1);
    cfg.tiers = vec![2]; // only tier-2 scenarios should run
    let suite = ScenarioSuite::from_scenarios(vec![
        Scenario::new("a", 1, cfg.base_forge_config.clone()),
        Scenario::new("b", 2, cfg.base_forge_config.clone()),
        Scenario::new("c", 6, cfg.base_forge_config.clone()),
    ]);
    let harness = EvalHarness::new(cfg);
    let card = harness.evaluate_suite(&suite, &|| Box::new(ConstAgent::new(0, "const")));
    assert_eq!(card.scenario_results.len(), 1);
    assert_eq!(card.scenario_results[0].scenario_id, "b");
}

#[test]
fn invalid_suite_dir_yields_validation_error() {
    let dir = tempdir().unwrap();
    let mut bad = Scenario::new("bad", 1, base_eval_config(1).base_forge_config);
    bad.id = String::new();
    write_scenario(dir.path(), "bad.toml", &bad);
    let err = ScenarioSuite::load_dir(dir.path()).unwrap_err();
    // Error path must surface to the caller.
    assert!(format!("{err}").contains("scenario at"));
}

#[test]
fn empty_output_dir_is_created_when_create_dir_set() {
    let tmp = tempdir().unwrap();
    let nested: PathBuf = tmp.path().join("never").join("existed");
    let mut cfg = base_eval_config(1);
    cfg.output = OutputConfig {
        enabled: true,
        dir: nested.clone(),
        create_dir: true,
        write_replays: true,
        write_trajectories: false,
        write_scorecard: true,
        scorecard_format: ScorecardFormat::Json,
        ..OutputConfig::default()
    };
    let suite = ScenarioSuite::from_scenarios(vec![Scenario::new(
        "deep",
        1,
        cfg.base_forge_config.clone(),
    )]);
    let harness = EvalHarness::new(cfg.clone());
    let _ = harness.evaluate_suite(&suite, &|| Box::new(ConstAgent::new(0, "const")));
    assert!(nested.exists());
    assert!(cfg.output.scorecard_path("json").exists());
}
