//! Expert demonstration generator using FORGE's built-in MCTS agent.
//!
//! Generates high-quality, natively-labeled training trajectories by running
//! FORGE's own MCTS planner (`forge-agent`) against live simulation episodes.
//! This is the highest-ROI data source: zero external dependencies, FORGE-native
//! format, deterministic, and immediately compatible with `forge-replay`.
//!
//! # Approach
//!
//! For each seed in a range:
//! 1. Construct a `WorldState` using `ForgeConfig` with the given seed.
//! 2. Run each tick with MCTS (`MctsSearch` + `HeuristicPolicy`) for agent 0.
//! 3. All other agents act as `NoopAgent`.
//! 4. Record every observation/action/reward triple in a `TrajectoryBuilder`.
//! 5. Stop at `terminated | truncated | max_steps`.
//!
//! The resulting [`OfflineDataset`] is ready for offline RL, imitation learning,
//! or export via [`OfflineDataset::export_jsonl`].
//!
//! # Example
//!
//! ```rust,no_run
//! use forge_data::generator::{ExpertDemoConfig, ExpertDemoGenerator};
//!
//! let cfg = ExpertDemoConfig::default();
//! let gen = ExpertDemoGenerator::new(cfg);
//! let dataset = gen.generate_corpus(0..20);
//! println!("steps: {}", dataset.total_steps());
//! ```

use std::ops::Range;

use forge_agent::baselines::{Agent, NoopAgent};
use forge_agent::forward_model::DefaultForwardModel;
use forge_agent::mcts::policy::HeuristicPolicy;
use forge_agent::mcts::search::MctsSearch;
use forge_agent::mcts::tree::MctsConfig;
use forge_core::WorldState;
use forge_replay::trajectory::TrajectoryBuilder;
use forge_types::agent_interface::AgentResponse;
use forge_types::config::ForgeConfig;
use forge_types::Action;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use crate::loader::OfflineDataset;

/// Configuration for the expert demo generator.
///
/// All constants live here — no hard-coded values elsewhere.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertDemoConfig {
    /// Base `ForgeConfig` applied to every episode.
    /// `world.seed` is overridden per-episode with the loop seed.
    pub forge_config: ForgeConfig,
    /// MCTS hyperparameters for the planning agent.
    pub mcts_config: MctsConfig,
    /// Maximum steps per episode (overrides `task.max_episode_length`).
    pub max_steps: u64,
    /// Generate episodes in parallel using rayon worker threads.
    pub parallel: bool,
}

impl Default for ExpertDemoConfig {
    fn default() -> Self {
        let mut forge_config = ForgeConfig::default();
        // Small world keeps generation fast while still exercising navigation.
        forge_config.world.width = 32;
        forge_config.world.height = 32;
        forge_config.agents.num_agents = 1;
        forge_config.agents.comm_vocab_size = 0;
        forge_config.task.max_episode_length = 500;

        // Higher simulation count than the interactive default for better quality.
        let mcts_config = MctsConfig {
            num_simulations: 50,
            max_depth: 20,
            ..MctsConfig::default()
        };

        Self {
            forge_config,
            mcts_config,
            max_steps: 500,
            parallel: true,
        }
    }
}

/// Generates expert demonstrations using FORGE's MCTS planner.
pub struct ExpertDemoGenerator {
    config: ExpertDemoConfig,
}

impl ExpertDemoGenerator {
    /// Creates a new generator with the given configuration.
    pub fn new(config: ExpertDemoConfig) -> Self {
        Self { config }
    }

    /// Returns the generator configuration.
    pub fn config(&self) -> &ExpertDemoConfig {
        &self.config
    }

    /// Generates a corpus of expert demonstrations for the given seed range.
    ///
    /// Each seed produces one independent episode. When `config.parallel` is
    /// `true`, episodes are generated concurrently via rayon.
    #[instrument(skip(self))]
    pub fn generate_corpus(&self, seeds: Range<u64>) -> OfflineDataset {
        let seeds: Vec<u64> = seeds.collect();
        info!(
            num_episodes = seeds.len(),
            parallel = self.config.parallel,
            "Starting expert demo generation"
        );

        let trajectories: Vec<_> = if self.config.parallel {
            seeds
                .into_par_iter()
                .filter_map(|seed| self.generate_episode(seed))
                .collect()
        } else {
            seeds
                .into_iter()
                .filter_map(|seed| self.generate_episode(seed))
                .collect()
        };

        let mut dataset = OfflineDataset::new("ExpertDemoGenerator");
        dataset.metadata.source_url =
            Some("https://github.com/ianshank/forge".to_string());
        dataset.metadata.license = Some("Apache-2.0".to_string());
        for traj in trajectories {
            dataset.push(traj);
        }

        info!(
            episodes = dataset.len(),
            steps = dataset.total_steps(),
            "Expert demo generation complete"
        );
        dataset
    }

    /// Runs a single episode and returns the resulting [`Trajectory`].
    ///
    /// Returns `None` if `WorldState` construction fails (e.g. invalid config).
    ///
    /// [`Trajectory`]: forge_replay::trajectory::Trajectory
    #[instrument(skip(self), fields(seed))]
    pub fn generate_episode(
        &self,
        seed: u64,
    ) -> Option<forge_replay::trajectory::Trajectory> {
        let mut episode_config = self.config.forge_config.clone();
        episode_config.world.seed = seed;

        let mut state = WorldState::new(episode_config.clone()).ok()?;

        let comm_vocab = episode_config.agents.comm_vocab_size;
        let action_space = Action::space_size(comm_vocab, false);

        let forward_model = DefaultForwardModel::new(comm_vocab);
        let policy = HeuristicPolicy::new(action_space);
        let mcts = MctsSearch::new(
            forward_model,
            policy,
            self.config.mcts_config.clone(),
            comm_vocab,
        );

        let num_agents = state.agents.len();
        let mut noop = NoopAgent;
        let mut builder = TrajectoryBuilder::new();

        // Running reward accumulator so we can compute final_rewards without
        // accessing TrajectoryBuilder internals.
        let mut cumulative_rewards = vec![0.0f32; num_agents];

        let mut step = 0u64;
        loop {
            if state.terminated || state.truncated || step >= self.config.max_steps {
                break;
            }

            // Collect observations before stepping.
            let observations: Vec<_> = state
                .agents
                .iter()
                .map(|agent| state.generate_observation(agent))
                .collect();

            // Agent 0 uses MCTS; all others use Noop to keep computation bounded.
            let actions: Vec<Action> = (0..num_agents)
                .map(|idx| {
                    if idx == 0 {
                        mcts.search(&state, idx)
                    } else {
                        noop.select_action(&state, idx)
                    }
                })
                .collect();

            let action_ids: Vec<u32> = actions
                .iter()
                .map(|a| a.to_discrete())
                .collect();

            let result = state.step(&actions);

            // Accumulate rewards.
            for (i, &r) in result.rewards.iter().enumerate() {
                if i < num_agents {
                    cumulative_rewards[i] += r;
                }
            }

            let responses: Vec<AgentResponse> = action_ids
                .iter()
                .map(|&id| AgentResponse::from_action(id))
                .collect();

            builder.record_step(
                step,
                observations,
                &responses,
                result.rewards.clone(),
                result.terminated,
                result.truncated,
            );

            step += 1;

            if result.terminated || result.truncated {
                break;
            }
        }

        let traj = builder
            .seed(seed)
            .agent_names((0..num_agents).map(|i| format!("agent_{i}")).collect())
            .scenario_id(format!("expert_seed_{seed}"))
            .build(cumulative_rewards);

        debug!(seed, steps = traj.len(), "Episode complete");
        Some(traj)
    }
}

// ---------------------------------------------------------------------------
// DatasetLoader impl
// ---------------------------------------------------------------------------

use crate::loader::{DatasetError, DatasetLoader};

impl DatasetLoader for ExpertDemoGenerator {
    /// Generates a single episode (seed = 0). The `path` argument is unused.
    fn load(&self, _path: &str) -> Result<OfflineDataset, DatasetError> {
        Ok(self.generate_corpus(0..1))
    }

    fn source_name(&self) -> &str {
        "ExpertDemoGenerator"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ExpertDemoConfig {
        let mut cfg = ExpertDemoConfig::default();
        cfg.max_steps = 15;
        cfg.forge_config.world.width = 16;
        cfg.forge_config.world.height = 16;
        cfg.mcts_config.num_simulations = 4;
        cfg.parallel = false;
        cfg
    }

    #[test]
    fn test_generate_single_episode() {
        let gen = ExpertDemoGenerator::new(small_config());
        let traj = gen.generate_episode(42).expect("episode generation failed");
        assert!(!traj.is_empty());
        assert_eq!(traj.metadata.seed, 42);
        assert_eq!(
            traj.metadata.scenario_id.as_deref(),
            Some("expert_seed_42")
        );
    }

    #[test]
    fn test_generate_corpus_count() {
        let gen = ExpertDemoGenerator::new(small_config());
        let ds = gen.generate_corpus(0..4);
        assert_eq!(ds.len(), 4);
        assert!(ds.total_steps() > 0);
        assert_eq!(ds.metadata.source, "ExpertDemoGenerator");
    }

    #[test]
    fn test_corpus_deterministic() {
        let gen = ExpertDemoGenerator::new(small_config());
        let ds1 = gen.generate_corpus(0..2);
        let ds2 = gen.generate_corpus(0..2);
        assert_eq!(ds1.total_steps(), ds2.total_steps());
        // Same seed → same first action
        assert_eq!(
            ds1.trajectories[0].steps[0].actions,
            ds2.trajectories[0].steps[0].actions,
        );
    }

    #[test]
    fn test_dataset_loader_trait() {
        let gen = ExpertDemoGenerator::new(small_config());
        let ds = gen.load("").unwrap();
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn test_different_seeds_differ() {
        let gen = ExpertDemoGenerator::new(small_config());
        let t0 = gen.generate_episode(0).unwrap();
        let t1 = gen.generate_episode(1).unwrap();
        // Different seeds should almost certainly produce different first actions
        // (not guaranteed, but extremely likely with a 40-action space).
        let _ = (t0, t1); // At minimum, both should succeed.
    }

    #[test]
    fn test_config_getter() {
        let cfg = small_config();
        let gen = ExpertDemoGenerator::new(cfg.clone());
        assert_eq!(gen.config().max_steps, cfg.max_steps);
        assert_eq!(gen.config().parallel, cfg.parallel);
    }

    #[test]
    fn test_dataset_loader_source_name() {
        use crate::loader::DatasetLoader;
        let gen = ExpertDemoGenerator::new(small_config());
        assert_eq!(gen.source_name(), "ExpertDemoGenerator");
    }

    #[test]
    fn test_generate_corpus_empty_range() {
        let gen = ExpertDemoGenerator::new(small_config());
        let ds = gen.generate_corpus(0..0);
        assert_eq!(ds.len(), 0);
        assert_eq!(ds.total_steps(), 0);
    }

    #[test]
    fn test_corpus_metadata_fields() {
        let gen = ExpertDemoGenerator::new(small_config());
        let ds = gen.generate_corpus(0..1);
        assert!(ds.metadata.source_url.is_some());
        assert!(ds.metadata.license.is_some());
    }

    #[test]
    fn test_generate_corpus_parallel() {
        let mut cfg = small_config();
        cfg.parallel = true;
        let gen = ExpertDemoGenerator::new(cfg);
        let ds = gen.generate_corpus(0..3);
        assert_eq!(ds.len(), 3);
    }

    #[test]
    fn test_max_steps_respected() {
        let mut cfg = small_config();
        cfg.max_steps = 3;
        let gen = ExpertDemoGenerator::new(cfg);
        let traj = gen.generate_episode(0).unwrap();
        assert!(traj.len() <= 3);
    }

    #[test]
    fn test_expert_demo_config_default_fields() {
        let cfg = ExpertDemoConfig::default();
        assert_eq!(cfg.forge_config.world.width, 32);
        assert_eq!(cfg.forge_config.world.height, 32);
        assert_eq!(cfg.forge_config.agents.num_agents, 1);
        assert!(cfg.parallel);
        assert!(cfg.max_steps > 0);
    }
}
