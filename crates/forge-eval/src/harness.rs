//! Evaluation harness: runs agents against FORGE scenarios.
//!
//! The [`EvalHarness`] is the primary entry point for evaluating agents.
//! It creates environments, runs episodes (optionally in parallel),
//! and aggregates results into a [`Scorecard`].

use std::time::Instant;

use forge_core::WorldState;
use forge_replay::compact::CompactReplay;
use forge_types::agent_interface::AgentInterface;
use forge_types::config::ForgeConfig;
use forge_types::Action;
use rayon::prelude::*;
use tracing::{debug, info, instrument, warn};

use crate::config::EvalConfig;
use crate::scorecard::{EpisodeResult, ScenarioResult, Scorecard, SummaryStats, TierScore};

/// The evaluation harness.
///
/// Runs any [`AgentInterface`] implementation against FORGE scenarios
/// and produces a standardized [`Scorecard`].
pub struct EvalHarness {
    config: EvalConfig,
}

impl EvalHarness {
    /// Creates a new evaluation harness.
    pub fn new(config: EvalConfig) -> Self {
        Self { config }
    }

    /// Returns the evaluation configuration.
    pub fn config(&self) -> &EvalConfig {
        &self.config
    }

    /// Runs a full evaluation using the base config as a single scenario.
    ///
    /// The `agent_factory` creates a fresh agent per episode for safe
    /// parallel execution (agents may have internal mutable state).
    #[instrument(skip_all)]
    pub fn evaluate<F>(&self, agent_factory: &F) -> Scorecard
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let wall_start = Instant::now();
        let config = &self.config.base_forge_config;
        let tier = 1_u8; // Default tier for base config

        info!(
            episodes = self.config.episodes_per_scenario,
            max_steps = self.config.max_steps_per_episode,
            "Starting evaluation"
        );

        let scenario_result = self.evaluate_scenario("default", tier, config, agent_factory);

        let agent_meta = agent_factory().metadata();

        let tier_scores = self.aggregate_tiers(&[&scenario_result]);
        let overall_score = Scorecard::compute_overall_score(&tier_scores);

        let wall_seconds = wall_start.elapsed().as_secs_f64();

        Scorecard {
            agent_metadata: agent_meta,
            timestamp: chrono::Utc::now().to_rfc3339(),
            overall_score,
            tier_scores,
            scenario_results: vec![scenario_result.clone()],
            summary: SummaryStats {
                total_episodes: scenario_result.episodes.len() as u32,
                total_steps: scenario_result.episodes.iter().map(|e| e.steps).sum(),
                wall_clock_seconds: wall_seconds,
                mean_decision_latency_ms: scenario_result.mean_decision_time_ms,
            },
        }
    }

    /// Runs evaluation for a single scenario (set of episodes with the same config).
    #[instrument(skip(self, agent_factory))]
    fn evaluate_scenario<F>(
        &self,
        scenario_id: &str,
        tier: u8,
        forge_config: &ForgeConfig,
        agent_factory: &F,
    ) -> ScenarioResult
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let episodes: Vec<EpisodeResult> = (0..self.config.episodes_per_scenario)
            .into_par_iter()
            .map(|episode_idx| {
                let seed = self.config.base_seed.wrapping_add(episode_idx as u64);
                self.run_single_episode(seed, forge_config, agent_factory)
            })
            .collect();

        debug!(
            scenario_id,
            episodes = episodes.len(),
            "Scenario evaluation complete"
        );

        ScenarioResult::from_episodes(scenario_id.to_string(), tier, episodes)
    }

    /// Runs a single episode and returns the result.
    fn run_single_episode<F>(
        &self,
        seed: u64,
        forge_config: &ForgeConfig,
        agent_factory: &F,
    ) -> EpisodeResult
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let mut config = forge_config.clone();
        config.world.seed = seed;

        let world = match WorldState::new(config.clone()) {
            Ok(w) => w,
            Err(e) => {
                warn!(seed, error = %e, "Failed to create WorldState for episode");
                return EpisodeResult {
                    seed,
                    total_reward: 0.0,
                    success: false,
                    steps: 0,
                    terminated: false,
                    truncated: false,
                    mean_decision_time_ms: 0.0,
                };
            }
        };

        let mut agent = agent_factory();
        agent.reset();

        let comm_vocab = config.agents.comm_vocab_size;
        let drone_enabled = config.drone.enabled;
        let max_steps = self.config.max_steps_per_episode;

        let mut current_world = world;
        let initial_result = current_world.reset(Some(seed));
        let mut current_obs = initial_result.observations;

        let mut total_reward = 0.0_f64;
        let mut total_decision_time_ms = 0_u64;
        let mut step_count = 0_u64;
        let mut replay_builder = if self.config.record_replays {
            Some(CompactReplay::builder(config, seed))
        } else {
            None
        };

        for _ in 0..max_steps {
            if current_world.terminated || current_world.truncated {
                break;
            }

            if current_obs.is_empty() {
                break;
            }

            // Get agent response
            let response = agent.select_action(&current_obs[0], 0);
            total_decision_time_ms += response.decision_time_ms;

            // Convert to FORGE action
            let action = Action::from_discrete(response.action_id, comm_vocab, drone_enabled)
                .unwrap_or(Action::Noop);

            // Build full action vector (pad with Noop for other agents)
            let num_agents = current_world.agents.len();
            let mut actions = vec![Action::Noop; num_agents];
            actions[0] = action;

            // Record replay
            if let Some(ref mut builder) = replay_builder {
                let action_ids: Vec<u32> = actions
                    .iter()
                    .map(|a| a.to_discrete_full(comm_vocab))
                    .collect();
                builder.record_tick(action_ids);
            }

            let step_result = current_world.step(&actions);
            let reward = step_result.rewards.first().copied().unwrap_or(0.0);
            total_reward += reward as f64;
            step_count += 1;
            current_obs = step_result.observations;
        }

        let mean_decision = if step_count > 0 {
            total_decision_time_ms as f64 / step_count as f64
        } else {
            0.0
        };

        // Determine success: episode terminated naturally (not truncated)
        // and agent is still alive
        let success = current_world.terminated && !current_world.truncated;

        EpisodeResult {
            seed,
            total_reward,
            success,
            steps: step_count,
            terminated: current_world.terminated,
            truncated: current_world.truncated,
            mean_decision_time_ms: mean_decision,
        }
    }

    /// Aggregates scenario results into per-tier scores.
    fn aggregate_tiers(&self, scenario_results: &[&ScenarioResult]) -> Vec<TierScore> {
        let mut tier_map: std::collections::HashMap<u8, Vec<&ScenarioResult>> =
            std::collections::HashMap::new();

        for result in scenario_results {
            tier_map.entry(result.tier).or_default().push(result);
        }

        let mut tier_scores: Vec<TierScore> = tier_map
            .into_iter()
            .map(|(tier, results)| {
                let total_episodes: u32 = results.iter().map(|r| r.episodes.len() as u32).sum();
                let total_successes: u32 = results
                    .iter()
                    .flat_map(|r| r.episodes.iter())
                    .filter(|e| e.success)
                    .count() as u32;
                let success_rate = if total_episodes > 0 {
                    total_successes as f64 / total_episodes as f64
                } else {
                    0.0
                };

                let all_episodes: Vec<&EpisodeResult> =
                    results.iter().flat_map(|r| r.episodes.iter()).collect();

                let mean_reward = if !all_episodes.is_empty() {
                    all_episodes.iter().map(|e| e.total_reward).sum::<f64>()
                        / all_episodes.len() as f64
                } else {
                    0.0
                };

                let successful_episodes: Vec<&&EpisodeResult> =
                    all_episodes.iter().filter(|e| e.success).collect();
                let mean_steps = if !successful_episodes.is_empty() {
                    successful_episodes
                        .iter()
                        .map(|e| e.steps as f64)
                        .sum::<f64>()
                        / successful_episodes.len() as f64
                } else {
                    0.0
                };

                TierScore {
                    tier,
                    success_rate,
                    mean_reward,
                    mean_steps_to_completion: mean_steps,
                    episodes_evaluated: total_episodes,
                    scenarios_count: results.len() as u32,
                }
            })
            .collect();

        tier_scores.sort_by_key(|t| t.tier);
        tier_scores
    }
}

#[cfg(test)]
mod tests {
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
}
