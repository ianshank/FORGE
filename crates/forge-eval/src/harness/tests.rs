use super::*;
use forge_types::agent_interface::{AgentMetadata, AgentResponse};
use forge_types::observation::Observation;

/// Agent that always returns Noop.
struct NoopEvalAgent;

impl AgentInterface for NoopEvalAgent {
    fn select_action(&mut self, _obs: &Observation, _agent_idx: usize) -> AgentResponse {
        AgentResponse::from_action(0)
    }
    fn name(&self) -> &str {
        "NoopEvalAgent"
    }
    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::heuristic("NoopEvalAgent")
    }
}

fn make_eval_config() -> EvalConfig {
    let mut config = EvalConfig {
        episodes_per_scenario: 3,
        max_steps_per_episode: 20,
        base_seed: 42,
        ..EvalConfig::default()
    };
    config.base_forge_config.world.width = 16;
    config.base_forge_config.world.height = 16;
    config.base_forge_config.agents.num_agents = 1;
    config.base_forge_config.agents.comm_vocab_size = 0;
    config.base_forge_config.task.max_episode_length = 20;
    config
}

#[test]
fn test_eval_harness_creation() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config.clone());
    assert_eq!(harness.config().episodes_per_scenario, 3);
    assert_eq!(harness.config().base_seed, 42);
}

#[test]
fn test_eval_harness_evaluate_noop() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    assert_eq!(scorecard.agent_metadata.model_name, "NoopEvalAgent");
    assert_eq!(scorecard.summary.total_episodes, 3);
    assert!(scorecard.summary.total_steps > 0);
    assert!(!scorecard.tier_scores.is_empty());
    assert!(!scorecard.timestamp.is_empty());
}

#[test]
fn test_eval_harness_determinism() {
    let config = make_eval_config();

    let harness1 = EvalHarness::new(config.clone());
    let scorecard1 = harness1.evaluate(&|| Box::new(NoopEvalAgent));

    let harness2 = EvalHarness::new(config);
    let scorecard2 = harness2.evaluate(&|| Box::new(NoopEvalAgent));

    // Results should be deterministic (same seeds, same agent)
    assert_eq!(
        scorecard1.summary.total_steps,
        scorecard2.summary.total_steps
    );
    assert_eq!(
        scorecard1.tier_scores[0].success_rate,
        scorecard2.tier_scores[0].success_rate
    );
}

#[test]
fn test_eval_harness_scorecard_json() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    let json = scorecard.to_json().unwrap();
    let deser = Scorecard::from_json(&json).unwrap();
    assert_eq!(
        deser.summary.total_episodes,
        scorecard.summary.total_episodes
    );
}

#[test]
fn test_eval_harness_scorecard_markdown() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    let md = scorecard.to_markdown();
    assert!(md.contains("FORGE Evaluation Scorecard"));
    assert!(md.contains("NoopEvalAgent"));
}

#[test]
fn test_eval_with_replays() {
    let mut config = make_eval_config();
    config.record_replays = true;
    config.episodes_per_scenario = 1;

    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    // Should still produce valid results
    assert_eq!(scorecard.summary.total_episodes, 1);
}

#[test]
fn test_overall_score_in_scorecard() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    assert!(scorecard.overall_score >= 0.0);
    assert!(scorecard.overall_score <= 1.0);
}

#[test]
fn test_single_episode() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);

    let forge_config = &harness.config().base_forge_config.clone();
    let result = harness.run_single_episode(42, forge_config, &|| Box::new(NoopEvalAgent));

    assert_eq!(result.seed, 42);
    assert!(result.steps > 0);
    assert!(result.truncated || result.terminated);
}

#[test]
fn test_eval_harness_config_accessor() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config.clone());
    assert_eq!(
        harness.config().max_steps_per_episode,
        config.max_steps_per_episode
    );
    assert_eq!(harness.config().base_seed, config.base_seed);
}

#[test]
fn test_eval_single_episode_determinism() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config.clone());
    let forge_config = &harness.config().base_forge_config.clone();

    let r1 = harness.run_single_episode(99, forge_config, &|| Box::new(NoopEvalAgent));
    let r2 = harness.run_single_episode(99, forge_config, &|| Box::new(NoopEvalAgent));

    assert_eq!(r1.steps, r2.steps, "same seed should produce same steps");
    assert!(
        (r1.total_reward - r2.total_reward).abs() < f64::EPSILON,
        "same seed should produce same reward"
    );
    assert_eq!(r1.success, r2.success);
}

#[test]
fn test_eval_different_seeds_differ() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config.clone());
    let forge_config = &harness.config().base_forge_config.clone();

    let r1 = harness.run_single_episode(0, forge_config, &|| Box::new(NoopEvalAgent));
    let r2 = harness.run_single_episode(12345, forge_config, &|| Box::new(NoopEvalAgent));

    // With different seeds, results are likely different (not guaranteed but very probable).
    // At minimum, the seed field should differ.
    assert_ne!(r1.seed, r2.seed);
}

#[test]
fn test_eval_episode_result_fields_populated() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config.clone());
    let forge_config = &harness.config().base_forge_config.clone();

    let result = harness.run_single_episode(42, forge_config, &|| Box::new(NoopEvalAgent));
    assert!(result.mean_decision_time_ms >= 0.0);
    assert!(result.total_reward.is_finite());
}

#[test]
fn test_eval_scorecard_scenario_results_nonempty() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    assert!(!scorecard.scenario_results.is_empty());
    assert_eq!(scorecard.scenario_results[0].scenario_id, "default");
    assert_eq!(scorecard.scenario_results[0].tier, 1);
    assert_eq!(scorecard.scenario_results[0].episodes.len(), 3);
}

#[test]
fn test_eval_scorecard_tier_scores_sorted() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    for window in scorecard.tier_scores.windows(2) {
        assert!(
            window[0].tier <= window[1].tier,
            "tier scores should be sorted ascending"
        );
    }
}

#[test]
fn test_eval_scorecard_wall_clock_positive() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    assert!(
        scorecard.summary.wall_clock_seconds > 0.0,
        "wall clock time should be positive"
    );
}

#[test]
fn test_eval_single_episode_config() {
    let mut config = make_eval_config();
    config.episodes_per_scenario = 1;
    let harness = EvalHarness::new(config);
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

    assert_eq!(scorecard.summary.total_episodes, 1);
}

#[test]
fn test_eval_harness_base_seed_wrapping() {
    let mut config = make_eval_config();
    config.base_seed = u64::MAX - 1;
    config.episodes_per_scenario = 5;
    let harness = EvalHarness::new(config);
    // Should not panic — seeds wrap using wrapping_add.
    let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));
    assert_eq!(scorecard.summary.total_episodes, 5);
}

// ─── Phase 1: suite + persistence path ─────────────────────────────────

use crate::output::{OutputConfig, ScorecardFormat};
use crate::scenario::{Scenario, ScenarioSuite};
use tempfile::tempdir;

fn make_scenario(id: &str, tier: u8, base: &EvalConfig) -> Scenario {
    let mut s = Scenario::new(id, tier, base.base_forge_config.clone());
    s.max_steps = Some(base.max_steps_per_episode);
    s
}

#[test]
fn test_evaluate_suite_multi_scenario_multi_tier() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config.clone());
    let suite = ScenarioSuite::from_scenarios(vec![
        make_scenario("nav_easy", 1, &config),
        make_scenario("nav_med", 2, &config),
        make_scenario("nav_hard", 3, &config),
    ]);
    let card = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));

    assert_eq!(card.scenario_results.len(), 3);
    assert_eq!(card.tier_scores.len(), 3);
    assert_eq!(card.summary.total_episodes, 9);
    for window in card.tier_scores.windows(2) {
        assert!(window[0].tier < window[1].tier);
    }
}

#[test]
fn test_evaluate_suite_respects_tier_filter() {
    let mut config = make_eval_config();
    config.tiers = vec![1, 3];
    let harness = EvalHarness::new(config.clone());
    let suite = ScenarioSuite::from_scenarios(vec![
        make_scenario("a", 1, &config),
        make_scenario("b", 2, &config),
        make_scenario("c", 3, &config),
    ]);
    let card = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));
    let ids: Vec<&str> = card
        .scenario_results
        .iter()
        .map(|r| r.scenario_id.as_str())
        .collect();
    assert_eq!(ids, vec!["a", "c"]);
}

#[test]
fn test_evaluate_suite_scenario_max_steps_override() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config.clone());
    let mut short = make_scenario("short", 1, &config);
    short.max_steps = Some(3);
    let suite = ScenarioSuite::from_scenarios(vec![short]);
    let card = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));
    for ep in &card.scenario_results[0].episodes {
        assert!(ep.steps <= 3, "scenario max_steps must cap episode length");
    }
}

#[test]
fn test_persistence_writes_replay_and_trajectory_files() {
    let dir = tempdir().unwrap();
    let mut config = make_eval_config();
    config.episodes_per_scenario = 1;
    config.output = OutputConfig {
        enabled: true,
        dir: dir.path().to_path_buf(),
        scorecard_format: ScorecardFormat::Both,
        ..OutputConfig::default()
    };

    let harness = EvalHarness::new(config.clone());
    let suite = ScenarioSuite::from_scenarios(vec![make_scenario("artefacts", 1, &config)]);
    let _ = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));

    let seed = config.base_seed;
    assert!(
        config.output.replay_path("artefacts", seed).exists(),
        "replay missing"
    );
    assert!(
        config.output.trajectory_path("artefacts", seed).exists(),
        "trajectory missing"
    );
    assert!(
        config
            .output
            .trajectory_metadata_path("artefacts", seed)
            .exists(),
        "trajectory metadata missing"
    );
    assert!(
        config.output.scorecard_path("json").exists(),
        "scorecard.json missing"
    );
    assert!(
        config.output.scorecard_path("md").exists(),
        "scorecard.md missing"
    );
}

#[test]
fn test_persistence_json_only_format() {
    let dir = tempdir().unwrap();
    let mut config = make_eval_config();
    config.episodes_per_scenario = 1;
    config.output = OutputConfig {
        enabled: true,
        dir: dir.path().to_path_buf(),
        scorecard_format: ScorecardFormat::Json,
        write_replays: false,
        write_trajectories: false,
        ..OutputConfig::default()
    };
    let harness = EvalHarness::new(config.clone());
    let _ = harness.evaluate(&|| Box::new(NoopEvalAgent));
    assert!(config.output.scorecard_path("json").exists());
    assert!(!config.output.scorecard_path("md").exists());
}

#[test]
fn test_persistence_disabled_writes_nothing() {
    let dir = tempdir().unwrap();
    let mut config = make_eval_config();
    config.episodes_per_scenario = 1;
    config.output = OutputConfig {
        enabled: false, // disabled
        dir: dir.path().to_path_buf(),
        ..OutputConfig::default()
    };
    let harness = EvalHarness::new(config);
    let _ = harness.evaluate(&|| Box::new(NoopEvalAgent));
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[test]
fn test_persistence_replay_is_deterministic_bytes_for_same_seed() {
    let dir = tempdir().unwrap();
    let mut config = make_eval_config();
    config.episodes_per_scenario = 1;
    config.output = OutputConfig {
        enabled: true,
        dir: dir.path().join("run1"),
        write_trajectories: false,
        write_scorecard: false,
        ..OutputConfig::default()
    };

    let scenario = make_scenario("det", 1, &config);
    let suite = ScenarioSuite::from_scenarios(vec![scenario.clone()]);

    let h1 = EvalHarness::new(config.clone());
    let _ = h1.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));

    let mut cfg2 = config.clone();
    cfg2.output.dir = dir.path().join("run2");
    let h2 = EvalHarness::new(cfg2.clone());
    let _ = h2.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));

    let bytes1 = std::fs::read(config.output.replay_path("det", config.base_seed)).unwrap();
    let bytes2 = std::fs::read(cfg2.output.replay_path("det", config.base_seed)).unwrap();

    // Strip the timestamp field by re-deserializing and comparing actions+seed+config.
    let r1 = forge_replay::compact::CompactReplay::from_bytes(&bytes1).unwrap();
    let r2 = forge_replay::compact::CompactReplay::from_bytes(&bytes2).unwrap();
    assert_eq!(r1.seed, r2.seed);
    assert_eq!(r1.actions, r2.actions);
    assert_eq!(r1.config_hash, r2.config_hash);
}

#[test]
fn test_with_output_builder_method() {
    let dir = tempdir().unwrap();
    let cfg = make_eval_config();
    let output = OutputConfig {
        enabled: true,
        dir: dir.path().to_path_buf(),
        ..OutputConfig::default()
    };
    let harness = EvalHarness::new(cfg).with_output(output);
    assert!(harness.output_enabled());
}

#[test]
fn test_output_enabled_false_when_invalid() {
    let cfg = make_eval_config();
    let output = OutputConfig {
        enabled: true,
        dir: std::path::PathBuf::new(), // invalid
        ..OutputConfig::default()
    };
    let harness = EvalHarness::new(cfg).with_output(output);
    assert!(!harness.output_enabled());
}

#[test]
fn test_suite_overall_score_weighted_across_tiers() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config.clone());
    let suite = ScenarioSuite::from_scenarios(vec![
        make_scenario("t1", 1, &config),
        make_scenario("t6", 6, &config),
    ]);
    let card = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));
    assert!(card.overall_score >= 0.0 && card.overall_score <= 1.0);
    // Both tiers present; weighted sum uses tier as weight.
    let tier1 = card.tier_scores.iter().find(|t| t.tier == 1).unwrap();
    let tier6 = card.tier_scores.iter().find(|t| t.tier == 6).unwrap();
    let expected = (1.0 * tier1.success_rate + 6.0 * tier6.success_rate) / (1.0 + 6.0);
    assert!((card.overall_score - expected).abs() < 1e-9);
}

#[test]
fn test_evaluate_suite_total_episodes_match_per_scenario() {
    let mut config = make_eval_config();
    config.episodes_per_scenario = 2;
    let harness = EvalHarness::new(config.clone());
    let suite = ScenarioSuite::from_scenarios(vec![
        make_scenario("a", 1, &config),
        make_scenario("b", 2, &config),
        make_scenario("c", 3, &config),
    ]);
    let card = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));
    assert_eq!(card.summary.total_episodes, 6);
}

#[test]
fn test_evaluate_backward_compat_still_works() {
    let config = make_eval_config();
    let harness = EvalHarness::new(config);
    let card = harness.evaluate(&|| Box::new(NoopEvalAgent));
    assert_eq!(card.scenario_results.len(), 1);
    // Use the public constant so a future rename surfaces here, not as
    // a silent string-drift in the consumer.
    assert_eq!(
        card.scenario_results[0].scenario_id,
        DEFAULT_EVALUATE_SCENARIO_ID
    );
    assert_eq!(card.tier_scores[0].tier, 1);
}

#[test]
fn test_harness_constants_are_stable_contract() {
    // Pin the public-constant values so renaming them is an explicit,
    // reviewable change rather than a silent contract break for any
    // external assertion that references them.
    assert_eq!(DEFAULT_EVALUATE_SCENARIO_ID, "default");
    assert_eq!(UNCONTROLLED_AGENT_NAME_PREFIX, "uncontrolled");
    assert_eq!(NOOP_SENTINEL_ACTION_ID, 0);
    assert_eq!(EMPTY_AGGREGATE_VALUE, 0.0);
}

#[test]
fn test_build_phase_b_exporters_empty_when_unconfigured() {
    let cfg = make_eval_config();
    let harness = EvalHarness::new(cfg);
    let sinks = harness.build_phase_b_exporters();
    assert!(
        sinks.is_empty(),
        "no Phase B targets configured → no exporters built"
    );
}

#[test]
fn test_build_phase_b_exporters_includes_each_configured_target() {
    let tmp = tempdir().unwrap();
    let mut cfg = make_eval_config();
    cfg.mlflow_tracking_uri = Some(tmp.path().join("mlruns"));
    cfg.huggingface_export_root = Some(tmp.path().join("hf"));
    let harness = EvalHarness::new(cfg);
    let sinks = harness.build_phase_b_exporters();
    // Two configured → two sinks. Order is mlflow-first today; not part
    // of the public contract, so only assert the count + names.
    assert_eq!(sinks.len(), 2);
    let names: Vec<&str> = sinks.iter().map(|(s, _)| s.name()).collect();
    assert!(names.contains(&"mlflow"));
    assert!(names.contains(&"huggingface"));
}

// ──────────────────────────────────────────────────────────────────
// Review-feedback fills.
// ──────────────────────────────────────────────────────────────────

/// `EvalConfig.parallelism` must actually constrain the rayon pool used
/// for a scenario. We verify by counting how many distinct
/// `rayon::current_num_threads()` values are observed across episodes.
/// (review thread r3252771987)
#[test]
fn test_parallelism_field_constrains_rayon_pool() {
    use std::sync::Arc;
    use std::sync::Mutex;

    struct ProbeAgent {
        observed: Arc<Mutex<Vec<usize>>>,
    }
    impl AgentInterface for ProbeAgent {
        fn select_action(
            &mut self,
            _obs: &forge_types::observation::Observation,
            _agent_idx: usize,
        ) -> forge_types::agent_interface::AgentResponse {
            self.observed
                .lock()
                .unwrap()
                .push(rayon::current_num_threads());
            forge_types::agent_interface::AgentResponse::from_action(0)
        }
        fn name(&self) -> &str {
            "ProbeAgent"
        }
        fn metadata(&self) -> AgentMetadata {
            AgentMetadata::heuristic("ProbeAgent")
        }
    }

    let observed: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));

    let mut config = make_eval_config();
    config.parallelism = 2;
    config.episodes_per_scenario = 4;
    let harness = EvalHarness::new(config);
    let observed_clone = observed.clone();
    let card = harness.evaluate(&move || {
        Box::new(ProbeAgent {
            observed: observed_clone.clone(),
        }) as Box<dyn AgentInterface>
    });

    assert_eq!(card.summary.total_episodes, 4);
    let widths = observed.lock().unwrap().clone();
    assert!(!widths.is_empty(), "agent must have been polled");
    // Every observation of `current_num_threads()` from inside the
    // scoped pool must report exactly 2.
    for w in &widths {
        assert_eq!(*w, 2, "expected rayon pool width 2, saw {w}");
    }
}

/// Multi-agent trajectory rows must stay rectangular: each step's
/// `actions/reasoning/confidences/decision_times_ms` length matches
/// `observations.len()` even when only agent 0 is controlled.
/// (review thread r3252813062)
#[test]
fn test_trajectory_recording_is_rectangular_for_multi_agent() {
    use forge_replay::trajectory::TrajectoryBuilder;
    // We can't easily inspect the harness-internal builder without
    // persistence, so go through the on-disk path with a 2-agent world
    // and read back the JSONL.
    let dir = tempfile::tempdir().unwrap();
    let mut config = make_eval_config();
    config.base_forge_config.agents.num_agents = 2;
    config.episodes_per_scenario = 1;
    config.max_steps_per_episode = 3;
    config.base_forge_config.task.max_episode_length = 3;
    config.output = OutputConfig {
        enabled: true,
        dir: dir.path().to_path_buf(),
        write_replays: true,
        write_trajectories: true,
        write_scorecard: false,
        ..OutputConfig::default()
    };

    let harness = EvalHarness::new(config.clone());
    let suite = ScenarioSuite::from_scenarios(vec![Scenario::new(
        "multi",
        1,
        config.base_forge_config.clone(),
    )]);
    let _ = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));

    let traj_path = config.output.trajectory_path("multi", config.base_seed);
    let blob = std::fs::read_to_string(&traj_path).unwrap();
    for (line_idx, line) in blob.lines().enumerate() {
        let step: forge_replay::trajectory::TrajectoryStep = serde_json::from_str(line).unwrap();
        let obs_len = step.observations.len();
        assert!(obs_len > 0, "line {line_idx}: empty observations");
        assert_eq!(
            step.actions.len(),
            obs_len,
            "line {line_idx}: actions vs observations mismatch"
        );
        assert_eq!(step.reasoning.len(), obs_len);
        assert_eq!(step.confidences.len(), obs_len);
        assert_eq!(step.decision_times_ms.len(), obs_len);
        assert_eq!(step.rewards.len(), obs_len);
    }

    // Metadata vectors must also align with num_agents.
    let meta_path = config
        .output
        .trajectory_metadata_path("multi", config.base_seed);
    let meta: forge_replay::trajectory::TrajectoryMetadata =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta.agent_names.len(), 2);
    assert_eq!(meta.agent_metadata.len(), 2);
    assert_eq!(meta.final_rewards.len(), 2);

    // Builder sanity (unrelated, just confirms the imported type still works).
    let _ = TrajectoryBuilder::new();
}

/// Replay metadata must also size `agent_names` to the agent count.
#[test]
fn test_replay_metadata_sized_to_num_agents() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = make_eval_config();
    config.base_forge_config.agents.num_agents = 3;
    config.episodes_per_scenario = 1;
    config.max_steps_per_episode = 2;
    config.base_forge_config.task.max_episode_length = 2;
    config.output = OutputConfig {
        enabled: true,
        dir: dir.path().to_path_buf(),
        write_replays: true,
        write_trajectories: false,
        write_scorecard: false,
        ..OutputConfig::default()
    };
    let harness = EvalHarness::new(config.clone());
    let suite = ScenarioSuite::from_scenarios(vec![Scenario::new(
        "replay_multi",
        1,
        config.base_forge_config.clone(),
    )]);
    let _ = harness.evaluate_suite(&suite, &|| Box::new(NoopEvalAgent));

    let path = config.output.replay_path("replay_multi", config.base_seed);
    let replay =
        forge_replay::compact::CompactReplay::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(replay.metadata.agent_names.len(), 3);
    assert_eq!(replay.metadata.agent_metadata.len(), 3);
    assert_eq!(replay.metadata.final_rewards.len(), 3);
}
