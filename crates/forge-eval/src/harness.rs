//! Evaluation harness: runs agents against FORGE scenarios.
//!
//! The [`EvalHarness`] is the primary entry point for evaluating agents.
//! It creates environments, runs episodes (optionally in parallel),
//! and aggregates results into a [`Scorecard`].

use std::time::Instant;

use forge_core::WorldState;
use forge_replay::compact::CompactReplay;
use forge_replay::trajectory::TrajectoryBuilder;
use forge_types::agent_interface::{AgentInterface, AgentMetadata};
use forge_types::config::ForgeConfig;
use forge_types::Action;
use rayon::prelude::*;
use tracing::{debug, info, instrument, warn};

use crate::config::EvalConfig;
use crate::exporters::{huggingface::HuggingFaceExporter, mlflow::MlflowExporter, Exporter};
use crate::manifest::RunManifest;
use crate::output::{self, OutputConfig, ScorecardFormat};
use crate::scenario::{Scenario, ScenarioSuite, DEFAULT_TIER};
use crate::scorecard::{EpisodeResult, ScenarioResult, Scorecard, SummaryStats, TierScore};

// ─── Defaults (no hardcoded literals in the body) ─────────────────────────

/// Scenario id used by the backward-compatible [`EvalHarness::evaluate`]
/// entry point when wrapping the base config into a single-scenario suite.
/// Exposed `pub` so external assertions can reference the same constant
/// instead of duplicating the string literal.
pub const DEFAULT_EVALUATE_SCENARIO_ID: &str = "default";

/// Prefix used for synthetic agent names emitted for un-controlled agents
/// when persisting replay / trajectory metadata. The full name is
/// `"{UNCONTROLLED_AGENT_NAME_PREFIX}_{idx}"` for each index `>= 1`.
pub const UNCONTROLLED_AGENT_NAME_PREFIX: &str = "uncontrolled";

/// Discrete action id used as a sentinel response for un-controlled agents
/// in the trajectory recorder. Mirrors [`Action::Noop`] in the action-space
/// projection used by the harness.
pub const NOOP_SENTINEL_ACTION_ID: u32 = 0;

/// Value returned by aggregate accessors (mean reward, mean steps, mean
/// decision latency) when the input set is empty. Centralised so a future
/// policy change (e.g. `f64::NAN` instead of `0.0`) is a one-line edit.
pub const EMPTY_AGGREGATE_VALUE: f64 = 0.0;

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

    /// Builder method: overrides the output configuration on the contained
    /// [`EvalConfig`].
    ///
    /// Useful for callers that want to opt into on-disk persistence without
    /// constructing a brand-new [`EvalConfig`].
    pub fn with_output(mut self, output: OutputConfig) -> Self {
        self.config.output = output;
        self
    }

    /// Returns true if the configured output is enabled and valid.
    pub fn output_enabled(&self) -> bool {
        self.config.output.enabled && self.config.output.is_valid()
    }

    /// Runs a full evaluation using the base config as a single scenario.
    ///
    /// Backward-compatible entry point: wraps the base config into a single
    /// tier-`DEFAULT_TIER` scenario named `"default"` and delegates to
    /// [`evaluate_suite`](Self::evaluate_suite).
    ///
    /// The `agent_factory` creates a fresh agent per episode for safe
    /// parallel execution (agents may have internal mutable state).
    #[instrument(skip_all)]
    pub fn evaluate<F>(&self, agent_factory: &F) -> Scorecard
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let scenario = Scenario::new(
            DEFAULT_EVALUATE_SCENARIO_ID,
            DEFAULT_TIER,
            self.config.base_forge_config.clone(),
        );
        let suite = ScenarioSuite::from_scenarios(vec![scenario]);
        self.evaluate_suite(&suite, agent_factory)
    }

    /// Runs a full evaluation across every scenario in a suite, optionally
    /// filtered by [`EvalConfig::tiers`].
    #[instrument(skip_all, fields(scenarios = suite.scenarios.len()))]
    pub fn evaluate_suite<F>(&self, suite: &ScenarioSuite, agent_factory: &F) -> Scorecard
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let wall_start = Instant::now();
        let agent_meta = agent_factory().metadata();

        let active: Vec<&Scenario> = suite.filter_tiers(&self.config.tiers);
        info!(
            scenarios = active.len(),
            episodes_each = self.config.episodes_per_scenario,
            max_steps = self.config.max_steps_per_episode,
            tier_filter = ?self.config.tiers,
            "Starting suite evaluation"
        );

        let scenario_results: Vec<ScenarioResult> = active
            .iter()
            .map(|s| self.evaluate_scenario(s, &agent_meta, agent_factory))
            .collect();

        let total_episodes: u32 = scenario_results
            .iter()
            .map(|r| r.episodes.len() as u32)
            .sum();
        let total_steps: u64 = scenario_results
            .iter()
            .flat_map(|r| r.episodes.iter())
            .map(|e| e.steps)
            .sum();
        let mean_decision_latency_ms = if total_episodes > 0 {
            scenario_results
                .iter()
                .map(|r| r.mean_decision_time_ms * (r.episodes.len() as f64))
                .sum::<f64>()
                / total_episodes as f64
        } else {
            EMPTY_AGGREGATE_VALUE
        };

        let tier_scores = self.aggregate_tiers(&scenario_results.iter().collect::<Vec<_>>());
        let overall_score = Scorecard::compute_overall_score(&tier_scores);
        let wall_seconds = wall_start.elapsed().as_secs_f64();

        let scorecard = Scorecard {
            agent_metadata: agent_meta,
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
        };

        if self.output_enabled() && self.config.output.write_scorecard {
            if let Err(e) = self.persist_scorecard(&scorecard) {
                warn!(error = %e, "Failed to persist scorecard");
            }
        }

        // Phase B: optional MLflow + HuggingFace exporters. Both are
        // best-effort: a failed export logs warn! and does not affect the
        // returned scorecard. Disabled by default (Option<>::None).
        self.dispatch_phase_b_exporters(&scorecard);

        scorecard
    }

    /// Build the list of Phase B exporters enabled by the current config.
    ///
    /// Returns each exporter paired with a short `target` string used only
    /// for diagnostics. Adding a new exporter is a single `.push(...)` here;
    /// the dispatch loop in [`dispatch_phase_b_exporters`](Self::dispatch_phase_b_exporters)
    /// doesn't change.
    fn build_phase_b_exporters(&self) -> Vec<(Box<dyn Exporter>, String)> {
        let mut sinks: Vec<(Box<dyn Exporter>, String)> = Vec::new();

        // MLflow HTTP sink takes precedence over the filesystem sink when
        // configured — the long-run pipeline (Slice 3+) targets a real
        // tracking server, but the filesystem sink stays available for
        // dev iteration without a server. Only one is dispatched per run
        // to avoid double-recording the same scorecard.
        #[cfg(feature = "http-mlflow")]
        {
            if let Some(uri) = &self.config.mlflow_http_tracking_uri {
                match crate::exporters::mlflow_http::MlflowHttpSink::from_config(&self.config) {
                    Ok(sink) => {
                        sinks.push((Box::new(sink), uri.clone()));
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            tracking_uri = %uri,
                            "mlflow_http_tracking_uri is set but the HTTP sink could not be built; \
                             falling back to filesystem sink if configured"
                        );
                    }
                }
            }
        }

        // Filesystem sink fires unless the HTTP sink already took its slot.
        // This keeps existing `mlflow_tracking_uri`-only configs working
        // byte-identically post-rebase.
        let http_active = sinks.iter().any(|(s, _)| s.name() == "mlflow-http");
        if !http_active {
            if let Some(uri) = &self.config.mlflow_tracking_uri {
                sinks.push((
                    Box::new(MlflowExporter::new(uri.clone())),
                    uri.display().to_string(),
                ));
            }
        }

        if let Some(root) = &self.config.huggingface_export_root {
            sinks.push((
                Box::new(HuggingFaceExporter::new(root.clone())),
                root.display().to_string(),
            ));
        }
        sinks
    }

    /// Dispatch each Phase B exporter whose target is configured.
    /// Failures are logged but never propagated — an evaluation that
    /// produced a valid scorecard must not be reported as failed just
    /// because a downstream export hit an I/O error.
    fn dispatch_phase_b_exporters(&self, scorecard: &Scorecard) {
        let sinks = self.build_phase_b_exporters();
        if sinks.is_empty() {
            return;
        }

        // Manifest is captured once and shared by every exporter so they
        // agree on run_id, timestamp, git_sha, and config_hash.
        let manifest = RunManifest::capture(&self.config, &[]);
        // No-on-disk-artefacts case: pass output.dir anyway so exporters
        // can no-op subdir copies (replays/, trajectories/) cleanly.
        let artifacts_dir = self.config.output.dir.clone();

        for (sink, target) in &sinks {
            match sink.export(scorecard, &manifest, &artifacts_dir) {
                Ok(()) => info!(
                    exporter = sink.name(),
                    target = %target,
                    run_id = %manifest.run_id,
                    "Phase B exporter completed"
                ),
                Err(e) => warn!(
                    exporter = sink.name(),
                    target = %target,
                    error = %e,
                    "Phase B exporter failed (eval result unaffected)"
                ),
            }
        }
    }

    /// Runs evaluation for a single scenario (set of episodes with the same config).
    ///
    /// Episodes run in parallel. If [`EvalConfig::parallelism`] is non-zero,
    /// a dedicated [`rayon::ThreadPool`] of that size is used; otherwise
    /// rayon's global pool is used (one worker per logical core).
    #[instrument(skip(self, agent_factory), fields(scenario_id = %scenario.id, tier = scenario.tier))]
    fn evaluate_scenario<F>(
        &self,
        scenario: &Scenario,
        agent_meta: &AgentMetadata,
        agent_factory: &F,
    ) -> ScenarioResult
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let max_steps = scenario
            .max_steps
            .unwrap_or(self.config.max_steps_per_episode);

        let run_episodes = || -> Vec<EpisodeResult> {
            (0..self.config.episodes_per_scenario)
                .into_par_iter()
                .map(|episode_idx| {
                    let seed = self.config.base_seed.wrapping_add(episode_idx as u64);
                    self.run_single_episode_persisted(
                        &scenario.id,
                        seed,
                        max_steps,
                        &scenario.forge_config,
                        agent_meta,
                        agent_factory,
                    )
                })
                .collect()
        };

        let episodes: Vec<EpisodeResult> = if self.config.parallelism == 0 {
            run_episodes()
        } else {
            match rayon::ThreadPoolBuilder::new()
                .num_threads(self.config.parallelism as usize)
                .build()
            {
                Ok(pool) => pool.install(run_episodes),
                Err(e) => {
                    warn!(
                        parallelism = self.config.parallelism,
                        error = %e,
                        "Failed to build local rayon pool, falling back to global"
                    );
                    run_episodes()
                }
            }
        };

        debug!(
            scenario_id = %scenario.id,
            parallelism = self.config.parallelism,
            episodes = episodes.len(),
            "Scenario evaluation complete"
        );

        ScenarioResult::from_episodes(scenario.id.clone(), scenario.tier, episodes)
    }

    /// Backward-compatible wrapper used by existing tests.
    ///
    /// Delegates to [`run_single_episode_persisted`](Self::run_single_episode_persisted)
    /// with a synthetic scenario id of `"default"` and the harness-level
    /// `max_steps_per_episode`. Persistence still respects
    /// [`OutputConfig::enabled`].
    #[cfg(test)]
    fn run_single_episode<F>(
        &self,
        seed: u64,
        forge_config: &ForgeConfig,
        agent_factory: &F,
    ) -> EpisodeResult
    where
        F: Fn() -> Box<dyn AgentInterface> + Send + Sync,
    {
        let agent_meta = agent_factory().metadata();
        self.run_single_episode_persisted(
            DEFAULT_EVALUATE_SCENARIO_ID,
            seed,
            self.config.max_steps_per_episode,
            forge_config,
            &agent_meta,
            agent_factory,
        )
    }

    /// Runs a single episode, optionally persisting replay/trajectory artefacts.
    #[allow(clippy::too_many_arguments)]
    fn run_single_episode_persisted<F>(
        &self,
        scenario_id: &str,
        seed: u64,
        max_steps: u64,
        forge_config: &ForgeConfig,
        agent_meta: &AgentMetadata,
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
                    ..EpisodeResult::default()
                };
            }
        };

        let mut agent = agent_factory();
        agent.reset();
        let agent_name = agent.name().to_string();

        let comm_vocab = config.agents.comm_vocab_size;
        let drone_enabled = config.drone.enabled;

        let mut current_world = world;
        let initial_result = current_world.reset(Some(seed));
        let mut current_obs = initial_result.observations;

        let mut total_reward = 0.0_f64;
        let mut total_decision_time_ms = 0_u64;
        let mut step_count = 0_u64;

        let output = &self.config.output;
        let want_replay = self.config.record_replays || (output.enabled && output.write_replays);
        let want_trajectory =
            self.config.record_trajectories || (output.enabled && output.write_trajectories);

        let mut replay_builder = if want_replay {
            Some(CompactReplay::builder(config.clone(), seed))
        } else {
            None
        };
        let mut trajectory_builder = if want_trajectory {
            Some(TrajectoryBuilder::new())
        } else {
            None
        };

        let mut per_agent_rewards: Vec<f32> = Vec::new();
        let mut battery_nonnegative = true;
        let mut altitude_within_cap = true;
        let max_altitude = config.drone.max_altitude;

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

            let agri_enabled = current_world.config.agri.enabled && drone_enabled;
            let hex_enabled =
                current_world.config.world.grid_type == forge_types::config::GridType::Hex;

            // Convert to FORGE action using the active action-space layout.
            let action = Action::from_discrete_full(
                response.action_id,
                comm_vocab,
                drone_enabled,
                agri_enabled,
                hex_enabled,
            )
            .unwrap_or(Action::Noop);

            // Build full action vector (pad with Noop for other agents)
            let num_agents = current_world.agents.len();
            let mut actions = vec![Action::Noop; num_agents];
            actions[0] = action;

            // Snapshot observation set for trajectory recording before stepping.
            // `current_obs` is overwritten at end-of-iteration, so it's safe to
            // move (`mem::take`) rather than clone in the hot loop.
            let pre_step_obs = if trajectory_builder.is_some() {
                Some(std::mem::take(&mut current_obs))
            } else {
                None
            };

            // Record replay
            if let Some(ref mut builder) = replay_builder {
                let action_ids: Vec<u32> = actions
                    .iter()
                    .map(|a| {
                        a.to_discrete_configured(
                            comm_vocab,
                            drone_enabled,
                            agri_enabled,
                            hex_enabled,
                        )
                    })
                    .collect();
                builder.record_tick(action_ids);
            }

            let mut step_result = current_world.step(&actions);
            let reward = step_result.rewards.first().copied().unwrap_or(0.0);
            total_reward += reward as f64;
            step_count += 1;

            if per_agent_rewards.len() < step_result.rewards.len() {
                per_agent_rewards.resize(step_result.rewards.len(), 0.0);
            }
            for (idx, r) in step_result.rewards.iter().enumerate() {
                per_agent_rewards[idx] += *r;
            }

            if let (Some(builder), Some(pre_obs)) = (trajectory_builder.as_mut(), pre_step_obs) {
                // Trajectory rows must stay rectangular: build a `responses`
                // vec sized to the real agent count, with the controlled
                // agent at index 0 and a Noop sentinel for the rest. This
                // keeps `actions/reasoning/confidences/decision_times_ms`
                // aligned with `observations/rewards` for downstream readers.
                let mut responses = Vec::with_capacity(num_agents);
                responses.push(response);
                for _ in 1..num_agents {
                    responses.push(forge_types::agent_interface::AgentResponse::from_action(
                        NOOP_SENTINEL_ACTION_ID,
                    ));
                }
                builder.record_step(
                    step_count.saturating_sub(1),
                    pre_obs,
                    &responses,
                    std::mem::take(&mut step_result.rewards),
                    current_world.terminated,
                    current_world.truncated,
                );
            }

            current_obs = step_result.observations;

            for agent in &current_world.agents {
                if agent.battery < 0 {
                    battery_nonnegative = false;
                }
                if agent.altitude > max_altitude {
                    altitude_within_cap = false;
                }
            }
        }

        let mean_decision = if step_count > 0 {
            total_decision_time_ms as f64 / step_count as f64
        } else {
            EMPTY_AGGREGATE_VALUE
        };

        let controlled_alive = current_world
            .agents
            .first()
            .is_some_and(|agent| agent.alive);
        let tasks_complete = current_world.tasks.iter().all(|task| task.completed);
        let success = controlled_alive && tasks_complete && !current_world.truncated;

        // Persist artefacts. Errors are logged but never block the episode result.
        if output.enabled && output.is_valid() {
            // Build per-agent name + metadata vectors sized to the real
            // agent count. The controlled agent's name and metadata go at
            // index 0; sentinel placeholders fill the rest. This keeps
            // `agent_names.len()` consistent with `final_rewards.len()` and
            // with the per-tick action vectors recorded in the replay.
            let num_agents = current_world.agents.len().max(1);
            let mut agent_names_vec = Vec::with_capacity(num_agents);
            let mut agent_metadata_vec = Vec::with_capacity(num_agents);
            agent_names_vec.push(agent_name.clone());
            agent_metadata_vec.push(agent_meta.clone());
            for idx in 1..num_agents {
                agent_names_vec.push(format!("{UNCONTROLLED_AGENT_NAME_PREFIX}_{idx}"));
                agent_metadata_vec.push(AgentMetadata::default());
            }
            // Pad final_rewards too, so length matches agent_names.
            if per_agent_rewards.len() < num_agents {
                per_agent_rewards.resize(num_agents, 0.0);
            }

            if output.write_replays {
                if let Some(builder) = replay_builder.take() {
                    let replay = builder
                        .agent_names(agent_names_vec.clone())
                        .agent_metadata(agent_metadata_vec.clone())
                        .scenario_id(scenario_id.to_string())
                        .final_rewards(per_agent_rewards.clone())
                        .build();
                    if let Err(e) = output::write_replay(output, scenario_id, seed, &replay) {
                        warn!(error = %e, "Failed to write replay artefact");
                    }
                }
            }
            if output.write_trajectories {
                if let Some(builder) = trajectory_builder.take() {
                    let traj = builder
                        .seed(seed)
                        .agent_names(agent_names_vec)
                        .agent_metadata(agent_metadata_vec)
                        .scenario_id(scenario_id.to_string())
                        .build(per_agent_rewards.clone());
                    if let Err(e) = output::write_trajectory(output, scenario_id, seed, &traj) {
                        warn!(error = %e, "Failed to write trajectory artefact");
                    }
                }
            }
        }

        EpisodeResult {
            seed,
            total_reward,
            success,
            steps: step_count,
            terminated: current_world.terminated,
            truncated: current_world.truncated,
            mean_decision_time_ms: mean_decision,
            battery_nonnegative,
            altitude_within_cap,
        }
    }

    /// Writes the aggregate scorecard in the configured format(s).
    #[instrument(skip_all)]
    fn persist_scorecard(&self, scorecard: &Scorecard) -> Result<(), String> {
        let cfg = &self.config.output;
        if !cfg.write_scorecard {
            return Ok(());
        }
        let format = cfg.scorecard_format;
        if matches!(format, ScorecardFormat::Json | ScorecardFormat::Both) {
            let path = cfg.scorecard_path("json");
            cfg.ensure_parent(&path)?;
            let json = scorecard.to_json()?;
            std::fs::write(&path, json)
                .map_err(|e| format!("failed to write scorecard JSON {}: {e}", path.display()))?;
            debug!(path = %path.display(), "Wrote scorecard JSON");
        }
        if matches!(format, ScorecardFormat::Markdown | ScorecardFormat::Both) {
            let path = cfg.scorecard_path("md");
            cfg.ensure_parent(&path)?;
            std::fs::write(&path, scorecard.to_markdown())
                .map_err(|e| format!("failed to write scorecard MD {}: {e}", path.display()))?;
            debug!(path = %path.display(), "Wrote scorecard Markdown");
        }
        Ok(())
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
                    EMPTY_AGGREGATE_VALUE
                };

                let all_episodes: Vec<&EpisodeResult> =
                    results.iter().flat_map(|r| r.episodes.iter()).collect();

                let mean_reward = if !all_episodes.is_empty() {
                    all_episodes.iter().map(|e| e.total_reward).sum::<f64>()
                        / all_episodes.len() as f64
                } else {
                    EMPTY_AGGREGATE_VALUE
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
                    EMPTY_AGGREGATE_VALUE
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
#[path = "harness/tests.rs"]
mod tests;
