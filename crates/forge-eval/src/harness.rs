//! Evaluation harness: runs agents against FORGE scenarios.
//!
//! The [`EvalHarness`] is the primary entry point for evaluating agents.
//! It resolves scenarios, runs episodes (optionally in parallel),
//! and aggregates results into a [`Scorecard`].

use std::time::Instant;

use forge_agent::baselines::Agent;
use forge_core::WorldState;
use forge_replay::compact::CompactReplay;
use forge_replay::trajectory::{Trajectory, TrajectoryBuilder};
use forge_scenario::registry::ScenarioRegistry;
use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::config::ForgeConfig;
use forge_types::Action;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
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

/// Optional artifacts captured for a single evaluated episode.
#[derive(Debug, Clone, Default)]
pub struct EpisodeArtifacts {
    /// Compact replay recorded for the episode, if enabled.
    pub replay: Option<CompactReplay>,
    /// Full trajectory recorded for the episode, if enabled.
    pub trajectory: Option<Trajectory>,
}

/// Detailed result for a single evaluated episode.
#[derive(Debug, Clone)]
pub struct EpisodeEvaluation {
    /// Scorecard-facing episode result.
    pub result: EpisodeResult,
    /// Optional recorded artifacts for the episode.
    pub artifacts: EpisodeArtifacts,
}

impl EpisodeEvaluation {
    fn from_result(result: EpisodeResult) -> Self {
        Self {
            result,
            artifacts: EpisodeArtifacts::default(),
        }
    }
}

/// Detailed results for a single evaluated scenario.
#[derive(Debug, Clone)]
pub struct ScenarioEvaluation {
    /// Aggregated scorecard-facing scenario summary.
    pub scenario_result: ScenarioResult,
    /// Per-episode detailed evaluations for the scenario.
    pub episode_runs: Vec<EpisodeEvaluation>,
}

/// Detailed output for a full evaluation run.
#[derive(Debug, Clone)]
pub struct EvalRunResult {
    /// Aggregated scorecard output.
    pub scorecard: Scorecard,
    /// Detailed per-scenario results including optional artifacts.
    pub scenario_runs: Vec<ScenarioEvaluation>,
}

#[derive(Debug, Clone)]
struct ResolvedScenario {
    scenario_id: String,
    tier: u8,
    forge_config: ForgeConfig,
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

    /// Runs a full evaluation using either the base config as a single scenario
    /// or the configured scenario registry.
    ///
    /// The `agent_factory` creates a fresh agent per episode for safe
    /// parallel execution (agents may have internal mutable state).
    #[instrument(skip_all)]
    pub fn evaluate<F>(&self, agent_factory: &F) -> Scorecard
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        self.evaluate_with_artifacts(agent_factory).scorecard
    }

    /// Runs a full evaluation and returns optional per-episode artifacts.
    #[instrument(skip_all)]
    pub fn evaluate_with_artifacts<F>(&self, agent_factory: &F) -> EvalRunResult
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let wall_start = Instant::now();
        let agent_metadata = agent_factory().metadata();
        let scenarios = match self.resolve_scenarios() {
            Ok(scenarios) => scenarios,
            Err(error) => {
                warn!(error = %error, "Failed to resolve evaluation scenarios");
                return Self::build_run_result(agent_metadata, vec![], wall_start);
            }
        };

        info!(
            scenarios = scenarios.len(),
            episodes = self.config.episodes_per_scenario,
            max_steps = self.config.max_steps_per_episode,
            "Starting evaluation"
        );

        let scenario_runs = scenarios
            .iter()
            .map(|scenario| self.evaluate_scenario(scenario, agent_factory))
            .collect();

        Self::build_run_result(agent_metadata, scenario_runs, wall_start)
    }

    /// Runs a full evaluation using clone-free privileged agents.
    #[instrument(skip_all)]
    pub fn evaluate_privileged<F>(&self, agent_factory: &F) -> Scorecard
    where
        F: Fn() -> Box<dyn Agent> + Send + Sync,
    {
        self.evaluate_privileged_with_artifacts(agent_factory).scorecard
    }

    /// Runs a full privileged evaluation and returns optional per-episode artifacts.
    #[instrument(skip_all)]
    pub fn evaluate_privileged_with_artifacts<F>(&self, agent_factory: &F) -> EvalRunResult
    where
        F: Fn() -> Box<dyn Agent> + Send + Sync,
    {
        let wall_start = Instant::now();
        let agent_metadata = AgentMetadata::heuristic(agent_factory().name());
        let scenarios = match self.resolve_scenarios() {
            Ok(scenarios) => scenarios,
            Err(error) => {
                warn!(error = %error, "Failed to resolve evaluation scenarios");
                return Self::build_run_result(agent_metadata, vec![], wall_start);
            }
        };

        info!(
            scenarios = scenarios.len(),
            episodes = self.config.episodes_per_scenario,
            max_steps = self.config.max_steps_per_episode,
            "Starting privileged evaluation"
        );

        let scenario_runs = scenarios
            .iter()
            .map(|scenario| self.evaluate_privileged_scenario(scenario, agent_factory))
            .collect();

        Self::build_run_result(agent_metadata, scenario_runs, wall_start)
    }

    fn build_run_result(
        agent_metadata: AgentMetadata,
        scenario_runs: Vec<ScenarioEvaluation>,
        wall_start: Instant,
    ) -> EvalRunResult {
        let scenario_result_refs: Vec<&ScenarioResult> =
            scenario_runs.iter().map(|run| &run.scenario_result).collect();
        let tier_scores = Self::aggregate_tiers(&scenario_result_refs);
        let overall_score = Scorecard::compute_overall_score(&tier_scores);

        let scenario_results: Vec<ScenarioResult> = scenario_runs
            .iter()
            .map(|run| run.scenario_result.clone())
            .collect();

        let total_episodes = scenario_results.iter().map(|r| r.episodes.len() as u32).sum();
        let total_steps = scenario_results
            .iter()
            .flat_map(|r| r.episodes.iter())
            .map(|e| e.steps)
            .sum();
        let mean_decision_latency_ms = if scenario_results.is_empty() {
            0.0
        } else {
            scenario_results
                .iter()
                .map(|r| r.mean_decision_time_ms)
                .sum::<f64>()
                / scenario_results.len() as f64
        };

        let wall_seconds = wall_start.elapsed().as_secs_f64();

        EvalRunResult {
            scorecard: Scorecard {
                agent_metadata,
                timestamp: chrono::Utc::now().to_rfc3339(),
                overall_score,
                tier_scores,
                scenario_results,
                summary: SummaryStats {
                    total_episodes,
                    total_steps,
                    wall_clock_seconds: wall_seconds,
                    mean_decision_latency_ms,
                },
            },
            scenario_runs,
        }
    }

    /// Runs evaluation for a single scenario (set of episodes with the same config).
    #[instrument(skip(self, agent_factory))]
    fn evaluate_scenario<F>(
        &self,
        scenario: &ResolvedScenario,
        agent_factory: &F,
    ) -> ScenarioEvaluation
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        self.evaluate_scenario_with_runner(scenario, &|scenario_id, seed, forge_config| {
            self.run_single_episode(scenario_id, seed, forge_config, agent_factory)
        })
    }

    #[instrument(skip(self, agent_factory))]
    fn evaluate_privileged_scenario<F>(
        &self,
        scenario: &ResolvedScenario,
        agent_factory: &F,
    ) -> ScenarioEvaluation
    where
        F: Fn() -> Box<dyn Agent> + Send + Sync,
    {
        self.evaluate_scenario_with_runner(scenario, &|scenario_id, seed, forge_config| {
            self.run_single_episode_privileged(scenario_id, seed, forge_config, agent_factory)
        })
    }

    fn evaluate_scenario_with_runner<R>(
        &self,
        scenario: &ResolvedScenario,
        run_episode: &R,
    ) -> ScenarioEvaluation
    where
        R: Fn(&str, u64, &ForgeConfig) -> EpisodeEvaluation + Send + Sync,
    {
        let scenario_id = scenario.scenario_id.clone();
        let tier = scenario.tier;
        let forge_config = &scenario.forge_config;

        let collect_episode_runs = || {
            (0..self.config.episodes_per_scenario)
                .into_par_iter()
                .map(|episode_idx| {
                    let seed = self.config.base_seed.wrapping_add(episode_idx as u64);
                    run_episode(&scenario_id, seed, forge_config)
                })
                .collect::<Vec<_>>()
        };

        let episode_runs = if self.config.parallelism > 0 {
            match ThreadPoolBuilder::new()
                .num_threads(self.config.parallelism as usize)
                .build()
            {
                Ok(pool) => pool.install(collect_episode_runs),
                Err(error) => {
                    warn!(
                        parallelism = self.config.parallelism,
                        error = %error,
                        "Failed to build scoped rayon thread pool; falling back to default"
                    );
                    collect_episode_runs()
                }
            }
        } else {
            collect_episode_runs()
        };

        let episodes: Vec<EpisodeResult> =
            episode_runs.iter().map(|run| run.result.clone()).collect();

        debug!(
            scenario_id = %scenario_id,
            episodes = episodes.len(),
            "Scenario evaluation complete"
        );

        ScenarioEvaluation {
            scenario_result: ScenarioResult::from_episodes(scenario_id, tier, episodes),
            episode_runs,
        }
    }

    /// Runs a single episode and returns the result.
    fn run_single_episode<F>(
        &self,
        scenario_id: &str,
        seed: u64,
        forge_config: &ForgeConfig,
        agent_factory: &F,
    ) -> EpisodeEvaluation
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let mut config = forge_config.clone();
        config.world.seed = seed;

        let world = match WorldState::new(config.clone()) {
            Ok(w) => w,
            Err(e) => {
                warn!(seed, error = %e, "Failed to create WorldState for episode");
                return EpisodeEvaluation::from_result(EpisodeResult {
                    seed,
                    total_reward: 0.0,
                    success: false,
                    steps: 0,
                    terminated: false,
                    truncated: false,
                    mean_decision_time_ms: 0.0,
                });
            }
        };

        let mut agent = agent_factory();
        agent.reset();

        let agent_name = agent.name().to_string();
        let agent_metadata = agent.metadata();

        let comm_vocab = config.agents.comm_vocab_size;
        let drone_enabled = config.drone.enabled;
        let max_steps = self.config.max_steps_per_episode;

        let mut current_world = world;
        let initial_result = current_world.reset(Some(seed));
        let mut current_obs = initial_result.observations;
        let num_agents = current_world.agents.len();

        let mut total_reward = 0.0_f64;
        let mut total_rewards = vec![0.0_f32; num_agents];
        let mut total_decision_time_ms = 0_u64;
        let mut step_count = 0_u64;
        let mut replay_builder = if self.config.record_replays {
            Some(
                CompactReplay::builder(config.clone(), seed)
                    .agent_names(vec![agent_name.clone()])
                    .agent_metadata(vec![agent_metadata.clone()])
                    .scenario_id(scenario_id.to_string()),
            )
        } else {
            None
        };
        let mut trajectory_builder = if self.config.record_trajectories {
            Some(
                TrajectoryBuilder::new()
                    .seed(seed)
                    .agent_names(vec![agent_name])
                    .agent_metadata(vec![agent_metadata])
                    .scenario_id(scenario_id.to_string()),
            )
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

            let tick = current_world.tick;
            let observations_before_step = current_obs.clone();

            // Get agent response
            agent.update_context(&current_world);
            let response = agent.select_action(&current_obs[0], 0);
            total_decision_time_ms += response.decision_time_ms;

            // Convert to FORGE action
            let action = Action::from_discrete(response.action_id, comm_vocab, drone_enabled)
                .unwrap_or(Action::Noop);

            // Build full action vector (pad with Noop for other agents)
            let mut actions = vec![Action::Noop; num_agents];
            actions[0] = action;
            let action_ids: Vec<u32> = actions
                .iter()
                .map(|candidate| candidate.to_discrete_full(comm_vocab))
                .collect();

            let mut responses = vec![AgentResponse::from_action(Action::Noop.to_discrete_full(comm_vocab)); num_agents];
            let mut executed_response = response;
            executed_response.action_id = action_ids[0];
            responses[0] = executed_response;

            // Record replay
            if let Some(ref mut builder) = replay_builder {
                builder.record_tick(action_ids);
            }

            let step_result = current_world.step(&actions);
            if let Some(ref mut builder) = trajectory_builder {
                builder.record_step(
                    tick,
                    observations_before_step,
                    &responses,
                    step_result.rewards.clone(),
                    step_result.terminated,
                    step_result.truncated,
                );
            }

            for (agent_idx, reward) in step_result.rewards.iter().copied().enumerate() {
                if agent_idx < total_rewards.len() {
                    total_rewards[agent_idx] += reward;
                }
            }

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

        let success = episode_succeeded(&current_world);

        let replay = replay_builder
            .map(|builder| builder.final_rewards(total_rewards.clone()).build());
        let trajectory = trajectory_builder.map(|builder| builder.build(total_rewards));

        EpisodeEvaluation {
            result: EpisodeResult {
                seed,
                total_reward,
                success,
                steps: step_count,
                terminated: current_world.terminated,
                truncated: current_world.truncated,
                mean_decision_time_ms: mean_decision,
            },
            artifacts: EpisodeArtifacts { replay, trajectory },
        }
    }

    fn run_single_episode_privileged<F>(
        &self,
        scenario_id: &str,
        seed: u64,
        forge_config: &ForgeConfig,
        agent_factory: &F,
    ) -> EpisodeEvaluation
    where
        F: Fn() -> Box<dyn Agent> + Send + Sync,
    {
        let mut config = forge_config.clone();
        config.world.seed = seed;

        let world = match WorldState::new(config.clone()) {
            Ok(w) => w,
            Err(e) => {
                warn!(seed, error = %e, "Failed to create WorldState for privileged episode");
                return EpisodeEvaluation::from_result(EpisodeResult {
                    seed,
                    total_reward: 0.0,
                    success: false,
                    steps: 0,
                    terminated: false,
                    truncated: false,
                    mean_decision_time_ms: 0.0,
                });
            }
        };

        let mut agent = agent_factory();
        let agent_name = agent.name().to_string();
        let agent_metadata = AgentMetadata::heuristic(&agent_name);

        let comm_vocab = config.agents.comm_vocab_size;
        let drone_enabled = config.drone.enabled;
        let max_steps = self.config.max_steps_per_episode;

        let mut current_world = world;
        let initial_result = current_world.reset(Some(seed));
        let mut current_obs = initial_result.observations;
        let num_agents = current_world.agents.len();

        let mut total_reward = 0.0_f64;
        let mut total_rewards = vec![0.0_f32; num_agents];
        let mut total_decision_time_ms = 0_u64;
        let mut step_count = 0_u64;
        let mut replay_builder = if self.config.record_replays {
            Some(
                CompactReplay::builder(config.clone(), seed)
                    .agent_names(vec![agent_name.clone()])
                    .agent_metadata(vec![agent_metadata.clone()])
                    .scenario_id(scenario_id.to_string()),
            )
        } else {
            None
        };
        let mut trajectory_builder = if self.config.record_trajectories {
            Some(
                TrajectoryBuilder::new()
                    .seed(seed)
                    .agent_names(vec![agent_name])
                    .agent_metadata(vec![agent_metadata])
                    .scenario_id(scenario_id.to_string()),
            )
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

            let tick = current_world.tick;
            let observations_before_step = current_obs.clone();

            let started = Instant::now();
            let action = agent.select_action(&current_world, 0);
            let action_id = action.to_discrete_full(comm_vocab);
            let response = AgentResponse::with_timing(action_id, started);
            total_decision_time_ms += response.decision_time_ms;

            let executed_action =
                Action::from_discrete(action_id, comm_vocab, drone_enabled).unwrap_or(Action::Noop);

            let mut actions = vec![Action::Noop; num_agents];
            actions[0] = executed_action;
            let action_ids: Vec<u32> = actions
                .iter()
                .map(|candidate| candidate.to_discrete_full(comm_vocab))
                .collect();

            let mut responses = vec![
                AgentResponse::from_action(Action::Noop.to_discrete_full(comm_vocab));
                num_agents
            ];
            responses[0] = response;

            if let Some(ref mut builder) = replay_builder {
                builder.record_tick(action_ids);
            }

            let step_result = current_world.step(&actions);
            if let Some(ref mut builder) = trajectory_builder {
                builder.record_step(
                    tick,
                    observations_before_step,
                    &responses,
                    step_result.rewards.clone(),
                    step_result.terminated,
                    step_result.truncated,
                );
            }

            for (agent_idx, reward) in step_result.rewards.iter().copied().enumerate() {
                if agent_idx < total_rewards.len() {
                    total_rewards[agent_idx] += reward;
                }
            }

            total_reward += step_result.rewards.first().copied().unwrap_or(0.0) as f64;
            step_count += 1;
            current_obs = step_result.observations;
        }

        let mean_decision = if step_count > 0 {
            total_decision_time_ms as f64 / step_count as f64
        } else {
            0.0
        };

        let success = episode_succeeded(&current_world);
        let replay = replay_builder
            .map(|builder| builder.final_rewards(total_rewards.clone()).build());
        let trajectory = trajectory_builder.map(|builder| builder.build(total_rewards));

        EpisodeEvaluation {
            result: EpisodeResult {
                seed,
                total_reward,
                success,
                steps: step_count,
                terminated: current_world.terminated,
                truncated: current_world.truncated,
                mean_decision_time_ms: mean_decision,
            },
            artifacts: EpisodeArtifacts { replay, trajectory },
        }
    }

    /// Aggregates scenario results into per-tier scores.
    fn aggregate_tiers(scenario_results: &[&ScenarioResult]) -> Vec<TierScore> {
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

    fn resolve_scenarios(&self) -> Result<Vec<ResolvedScenario>, String> {
        let mut scenarios = if let Some(path) = &self.config.scenario_registry_path {
            let registry = ScenarioRegistry::from_directory(path)
                .map_err(|error| format!("failed to load scenario registry: {error}"))?;
            let selected = registry.by_tiers(&self.config.tiers);

            if selected.is_empty() {
                warn!(
                    path = %path.display(),
                    requested_tiers = ?self.config.tiers,
                    "Scenario registry produced no matching scenarios"
                );
            }

            selected
                .into_iter()
                .map(|scenario| ResolvedScenario {
                    scenario_id: scenario.scenario.id.clone(),
                    tier: scenario.scenario.difficulty_tier,
                    forge_config: scenario.effective_config().clone(),
                })
                .collect::<Vec<_>>()
        } else {
            if !self.config.tiers.is_empty() {
                warn!(
                    requested_tiers = ?self.config.tiers,
                    "EvalConfig.tiers is ignored when scenario_registry_path is not set"
                );
            }

            vec![ResolvedScenario {
                scenario_id: "default".to_string(),
                tier: 1,
                forge_config: self.config.base_forge_config.clone(),
            }]
        };

        scenarios.sort_by(|left, right| {
            left.tier
                .cmp(&right.tier)
                .then_with(|| left.scenario_id.cmp(&right.scenario_id))
        });

        Ok(scenarios)
    }
}

fn episode_succeeded(world: &WorldState) -> bool {
    !world.tasks.is_empty() && world.tasks.iter().all(|task| task.completed && !task.failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_agent::baselines::NoopAgent;
    use forge_types::agent_interface::{AgentMetadata, AgentResponse};
    use forge_types::observation::Observation;
    use forge_types::task::{ActiveTask, Predicate, TaskComposition, TaskDefinition, TaskTier};
    use std::path::PathBuf;

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

    fn eval_registry_fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/eval_registry")
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

    fn make_task(completed: bool, failed: bool) -> ActiveTask {
        ActiveTask {
            definition: TaskDefinition {
                id: 1,
                description: "Test task".into(),
                goal: TaskComposition::Atom(Predicate::TimeElapsed(0)),
                tier: TaskTier::new(1),
                estimated_steps: 1,
                reward: 1.0,
                dense_reward_weights: vec![1.0],
            },
            progress: vec![if completed { 1.0 } else { 0.0 }],
            sequence_index: 0,
            completed,
            failed,
        }
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
    fn test_eval_harness_parallelism_path() {
        let mut config = make_eval_config();
        config.parallelism = 1;

        let harness = EvalHarness::new(config);
        let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

        assert_eq!(scorecard.summary.total_episodes, 3);
    }

    #[test]
    fn test_eval_harness_with_artifacts_records_replay_and_trajectory() {
        let mut config = make_eval_config();
        config.record_replays = true;
        config.record_trajectories = true;
        config.episodes_per_scenario = 1;

        let harness = EvalHarness::new(config);
        let run = harness.evaluate_with_artifacts(&|| Box::new(NoopEvalAgent));

        assert_eq!(run.scorecard.summary.total_episodes, 1);
        assert_eq!(run.scenario_runs.len(), 1);
        assert_eq!(run.scenario_runs[0].episode_runs.len(), 1);

        let episode = &run.scenario_runs[0].episode_runs[0];
        assert!(episode.artifacts.replay.is_some());
        assert!(episode.artifacts.trajectory.is_some());

        let replay = episode.artifacts.replay.as_ref().unwrap();
        assert_eq!(replay.metadata.scenario_id.as_deref(), Some("default"));
        assert_eq!(replay.metadata.final_rewards.len(), 1);

        let trajectory = episode.artifacts.trajectory.as_ref().unwrap();
        assert_eq!(trajectory.metadata.scenario_id.as_deref(), Some("default"));
        assert_eq!(trajectory.metadata.final_rewards.len(), 1);
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
    fn test_eval_harness_ignores_reserved_tiers_for_single_scenario() {
        let mut config = make_eval_config();
        config.tiers = vec![6];

        let harness = EvalHarness::new(config);
        let scorecard = harness.evaluate(&|| Box::new(NoopEvalAgent));

        assert_eq!(scorecard.summary.total_episodes, 3);
        assert_eq!(scorecard.scenario_results[0].tier, 1);
    }

    #[test]
    fn test_eval_harness_resolves_registry_scenarios_with_tier_filtering() {
        let mut config = make_eval_config();
        config.scenario_registry_path = Some(eval_registry_fixture_path());
        config.tiers = vec![2];

        let harness = EvalHarness::new(config);
        let scenarios = harness.resolve_scenarios().unwrap();

        let scenario_ids: Vec<&str> = scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.as_str())
            .collect();

        assert_eq!(scenario_ids, vec!["escort_coordination", "rescue_chain"]);
        assert_eq!(scenarios[0].forge_config.world.width, 20);
        assert_eq!(scenarios[1].forge_config.world.width, 24);
        assert!(scenarios.iter().all(|scenario| scenario.tier == 2));
    }

    #[test]
    fn test_eval_harness_evaluate_registry_suite() {
        let mut config = make_eval_config();
        config.scenario_registry_path = Some(eval_registry_fixture_path());
        config.episodes_per_scenario = 1;

        let harness = EvalHarness::new(config);
        let run = harness.evaluate_with_artifacts(&|| Box::new(NoopEvalAgent));

        let scenario_ids: Vec<&str> = run
            .scorecard
            .scenario_results
            .iter()
            .map(|scenario| scenario.scenario_id.as_str())
            .collect();

        assert_eq!(scenario_ids, vec!["basic_patrol", "escort_coordination", "rescue_chain"]);
        assert_eq!(run.scorecard.summary.total_episodes, 3);
        assert_eq!(run.scenario_runs.len(), 3);
        assert_eq!(run.scorecard.tier_scores.len(), 2);
        assert_eq!(run.scorecard.tier_scores[0].tier, 1);
        assert_eq!(run.scorecard.tier_scores[0].episodes_evaluated, 1);
        assert_eq!(run.scorecard.tier_scores[1].tier, 2);
        assert_eq!(run.scorecard.tier_scores[1].episodes_evaluated, 2);
    }

    #[test]
    fn test_eval_harness_privileged_registry_mode_records_artifacts() {
        let mut config = make_eval_config();
        config.scenario_registry_path = Some(eval_registry_fixture_path());
        config.tiers = vec![1];
        config.episodes_per_scenario = 1;
        config.record_replays = true;
        config.record_trajectories = true;

        let harness = EvalHarness::new(config);
        let run = harness.evaluate_privileged_with_artifacts(&|| Box::new(NoopAgent));

        assert_eq!(run.scorecard.agent_metadata.model_name, "NoopAgent");
        assert_eq!(run.scorecard.summary.total_episodes, 1);
        assert_eq!(run.scenario_runs.len(), 1);
        assert_eq!(run.scorecard.scenario_results[0].scenario_id, "basic_patrol");

        let episode = &run.scenario_runs[0].episode_runs[0];
        assert!(episode.artifacts.replay.is_some());
        assert!(episode.artifacts.trajectory.is_some());
        assert_eq!(
            episode
                .artifacts
                .replay
                .as_ref()
                .and_then(|replay| replay.metadata.scenario_id.as_deref()),
            Some("basic_patrol")
        );
        assert_eq!(
            episode
                .artifacts
                .trajectory
                .as_ref()
                .and_then(|trajectory| trajectory.metadata.scenario_id.as_deref()),
            Some("basic_patrol")
        );
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
        let result = harness.run_single_episode("default", 42, forge_config, &|| {
            Box::new(NoopEvalAgent)
        });

        assert_eq!(result.result.seed, 42);
        assert!(result.result.steps > 0);
        assert!(result.result.truncated || result.result.terminated);
    }

    #[test]
    fn test_episode_succeeded_requires_tasks() {
        let world = WorldState::new(make_eval_config().base_forge_config).unwrap();
        assert!(!episode_succeeded(&world));
    }

    #[test]
    fn test_episode_succeeded_when_all_tasks_completed() {
        let mut world = WorldState::new(make_eval_config().base_forge_config).unwrap();
        world.tasks = vec![make_task(true, false), make_task(true, false)];

        assert!(episode_succeeded(&world));
    }

    #[test]
    fn test_episode_succeeded_false_for_incomplete_or_failed_tasks() {
        let mut world = WorldState::new(make_eval_config().base_forge_config).unwrap();
        world.tasks = vec![make_task(true, false), make_task(false, false)];
        assert!(!episode_succeeded(&world));

        world.tasks = vec![make_task(true, false), make_task(true, true)];
        assert!(!episode_succeeded(&world));
    }
}
