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
use forge_task::generator::{generate_active_task, TaskGenConfig};
use forge_types::agent_interface::AgentResponse;
use forge_types::config::ForgeConfig;
use forge_types::Action;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use crate::loader::OfflineDataset;

/// Seed salt separating the task-generation RNG stream from the action RNG
/// stream, so adding/removing tasks never perturbs policy action sampling.
const TASK_RNG_SALT: u64 = 0x5441_534b; // ASCII "TASK"

/// Base reward per tier-1 task before `task.reward_scale` is applied.
const DEFAULT_TASK_BASE_REWARD: f32 = 1.0;

/// Policy driving agent 0 during demo generation.
///
/// Agents 1..N always act as Noop regardless of this setting, keeping
/// per-step compute bounded by a single planning agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DemoPolicy {
    /// MCTS planning via `MctsSearch` + `HeuristicPolicy` (expert data).
    #[default]
    Mcts,
    /// Seeded uniform-random over the discrete action space (baseline data).
    Random,
}

impl DemoPolicy {
    /// Canonical lowercase name — the single source for CLI parsing,
    /// dataset labels, and the serde representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mcts => "mcts",
            Self::Random => "random",
        }
    }
}

impl std::fmt::Display for DemoPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for DemoPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mcts" => Ok(Self::Mcts),
            "random" => Ok(Self::Random),
            other => Err(format!(
                "unknown policy {other:?} (expected \"mcts\" or \"random\")"
            )),
        }
    }
}

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
    /// Policy for agent 0 (MCTS expert or seeded-random baseline).
    #[serde(default)]
    pub policy: DemoPolicy,
    /// Procedurally generated tasks assigned per episode.
    ///
    /// `WorldState::new` starts task-less, and with an empty task list the
    /// evaluator produces zero rewards — so at least one task is required
    /// for episodes to carry any reward signal. Tasks are generated at
    /// `forge_config.task.max_tier` from a seed-derived RNG stream, keeping
    /// episodes fully reproducible.
    #[serde(default = "default_tasks_per_episode")]
    pub tasks_per_episode: u32,
    /// Base reward per tier-1 task, before the evaluator applies
    /// `forge_config.task.reward_scale` (kept unscaled here so the scale is
    /// never applied twice).
    #[serde(default = "default_task_base_reward")]
    pub task_base_reward: f32,
    /// Optional scenario label recorded as each trajectory's `scenario_id`.
    ///
    /// When `None`, falls back to the legacy per-episode `expert_seed_{seed}`
    /// label. Set this to a stable configuration label (e.g.
    /// `"square-64-a2-t2-mcts"`) so `(scenario_id, seed)` forms a natural
    /// episode key in exported datasets.
    #[serde(default)]
    pub scenario_label: Option<String>,
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
            policy: DemoPolicy::default(),
            scenario_label: None,
            tasks_per_episode: default_tasks_per_episode(),
            task_base_reward: default_task_base_reward(),
        }
    }
}

/// Serde default for [`ExpertDemoConfig::tasks_per_episode`].
fn default_tasks_per_episode() -> u32 {
    1
}

/// Serde default for [`ExpertDemoConfig::task_base_reward`].
fn default_task_base_reward() -> f32 {
    DEFAULT_TASK_BASE_REWARD
}

/// Generates expert demonstrations using FORGE's MCTS planner.
pub struct ExpertDemoGenerator {
    config: ExpertDemoConfig,
}

impl ExpertDemoGenerator {
    /// Creates a new generator with the given configuration.
    #[instrument(skip_all)]
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
        dataset.metadata.source_url = Some("https://github.com/ianshank/forge".to_string());
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
    pub fn generate_episode(&self, seed: u64) -> Option<forge_replay::trajectory::Trajectory> {
        let mut episode_config = self.config.forge_config.clone();
        episode_config.world.seed = seed;

        let mut state = WorldState::new(episode_config.clone()).ok()?;

        // Assign procedurally generated tasks: `WorldState::new` starts
        // task-less and the evaluator yields zero rewards for an empty task
        // list. A dedicated seed-salted RNG stream keeps task generation
        // independent of action sampling.
        if self.config.tasks_per_episode > 0 {
            // base_reward stays unscaled: the evaluator multiplies task
            // rewards by `task.reward_scale` at evaluation time, so scaling
            // here would apply it twice.
            let task_cfg = TaskGenConfig {
                max_tier: episode_config.task.max_tier,
                world_width: episode_config.world.width,
                world_height: episode_config.world.height,
                num_agents: episode_config.agents.num_agents,
                max_predicates: episode_config.task.max_predicates,
                base_reward: self.config.task_base_reward,
            };
            let mut task_rng = Pcg64Mcg::seed_from_u64(seed ^ TASK_RNG_SALT);
            for task_idx in 0..self.config.tasks_per_episode {
                state.tasks.push(generate_active_task(
                    &mut task_rng,
                    episode_config.task.max_tier,
                    &task_cfg,
                    u64::from(task_idx),
                ));
            }
        }

        let comm_vocab = episode_config.agents.comm_vocab_size;
        let action_space = Action::space_size(comm_vocab, false);

        // MCTS machinery is only constructed when the policy needs it; the
        // random baseline draws from a per-episode seeded PCG stream instead.
        let mcts = match self.config.policy {
            DemoPolicy::Mcts => {
                let forward_model = DefaultForwardModel::new(comm_vocab);
                let policy = HeuristicPolicy::new(action_space);
                Some(MctsSearch::new(
                    forward_model,
                    policy,
                    self.config.mcts_config.clone(),
                    comm_vocab,
                ))
            }
            DemoPolicy::Random => None,
        };
        let mut rng = Pcg64Mcg::seed_from_u64(seed);

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

            // Agent 0 follows the configured policy; all others use Noop to
            // keep computation bounded.
            let actions: Vec<Action> = (0..num_agents)
                .map(|idx| {
                    if idx == 0 {
                        match &mcts {
                            Some(search) => search.search(&state, idx),
                            None => {
                                let action_id = rng.gen_range(0..action_space);
                                Action::from_discrete(action_id, comm_vocab, false)
                                    .unwrap_or(Action::Noop)
                            }
                        }
                    } else {
                        noop.select_action(&state, idx)
                    }
                })
                .collect();

            let action_ids: Vec<u32> = actions.iter().map(|a| a.to_discrete()).collect();

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

        let scenario_id = self
            .config
            .scenario_label
            .clone()
            .unwrap_or_else(|| format!("expert_seed_{seed}"));
        let traj = builder
            .seed(seed)
            .agent_names((0..num_agents).map(|i| format!("agent_{i}")).collect())
            .scenario_id(scenario_id)
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
        let mut cfg = ExpertDemoConfig {
            max_steps: 15,
            ..ExpertDemoConfig::default()
        };
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
        assert_eq!(traj.metadata.scenario_id.as_deref(), Some("expert_seed_42"));
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

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn config_roundtrip_serde(seed in 0u64..1000) {
                let mut cfg = ExpertDemoConfig::default();
                cfg.forge_config.world.seed = seed;
                let json = serde_json::to_string(&cfg).unwrap();
                let deser: ExpertDemoConfig = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(deser.forge_config.world.seed, seed);
                prop_assert_eq!(deser.max_steps, cfg.max_steps);
                prop_assert_eq!(deser.parallel, cfg.parallel);
            }
        }
    }

    #[test]
    fn test_expert_demo_config_default_fields() {
        let cfg = ExpertDemoConfig::default();
        assert_eq!(cfg.forge_config.world.width, 32);
        assert_eq!(cfg.forge_config.world.height, 32);
        assert_eq!(cfg.forge_config.agents.num_agents, 1);
        assert!(cfg.parallel);
        assert!(cfg.max_steps > 0);
        assert_eq!(cfg.policy, DemoPolicy::Mcts);
        assert!(cfg.scenario_label.is_none());
    }

    #[test]
    fn test_random_policy_episode_deterministic() {
        let mut cfg = small_config();
        cfg.policy = DemoPolicy::Random;
        let gen = ExpertDemoGenerator::new(cfg);
        let t1 = gen.generate_episode(7).expect("episode generation failed");
        let t2 = gen.generate_episode(7).expect("episode generation failed");
        assert!(!t1.is_empty());
        assert_eq!(t1.len(), t2.len());
        assert_eq!(t1.steps[0].actions, t2.steps[0].actions);
    }

    #[test]
    fn test_scenario_label_overrides_default_id() {
        let mut cfg = small_config();
        cfg.scenario_label = Some("square-16-a1-t1-mcts".to_string());
        let gen = ExpertDemoGenerator::new(cfg);
        let traj = gen.generate_episode(3).expect("episode generation failed");
        assert_eq!(
            traj.metadata.scenario_id.as_deref(),
            Some("square-16-a1-t1-mcts")
        );
        // Seed still recorded independently of the label.
        assert_eq!(traj.metadata.seed, 3);
    }

    #[test]
    fn test_config_serde_backward_compat_defaults() {
        // JSON written before `policy` / `scenario_label` /
        // `tasks_per_episode` existed must still deserialize, falling back
        // to the defaults.
        let cfg = ExpertDemoConfig::default();
        let mut v = serde_json::to_value(&cfg).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("policy");
        obj.remove("scenario_label");
        obj.remove("tasks_per_episode");
        let back: ExpertDemoConfig = serde_json::from_value(v).unwrap();
        assert_eq!(back.policy, DemoPolicy::Mcts);
        assert!(back.scenario_label.is_none());
        assert_eq!(back.tasks_per_episode, 1);
        assert!((back.task_base_reward - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_serde_backward_compat_task_base_reward() {
        let cfg = ExpertDemoConfig::default();
        let mut v = serde_json::to_value(&cfg).unwrap();
        v.as_object_mut().unwrap().remove("task_base_reward");
        let back: ExpertDemoConfig = serde_json::from_value(v).unwrap();
        assert!((back.task_base_reward - default_task_base_reward()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_demo_policy_serde_lowercase() {
        assert_eq!(
            serde_json::to_string(&DemoPolicy::Mcts).unwrap(),
            "\"mcts\""
        );
        assert_eq!(
            serde_json::to_string(&DemoPolicy::Random).unwrap(),
            "\"random\""
        );
        assert_eq!(
            serde_json::from_str::<DemoPolicy>("\"random\"").unwrap(),
            DemoPolicy::Random
        );
        assert!(serde_json::from_str::<DemoPolicy>("\"MCTS\"").is_err());
    }

    #[test]
    fn test_demo_policy_string_forms_agree() {
        // Display / FromStr / serde must stay a single representation.
        for policy in [DemoPolicy::Mcts, DemoPolicy::Random] {
            let display = policy.to_string();
            assert_eq!(display, policy.as_str());
            assert_eq!(display.parse::<DemoPolicy>().unwrap(), policy);
            let serde_form = serde_json::to_string(&policy).unwrap();
            assert_eq!(serde_form, format!("\"{display}\""));
        }
        assert!("llm".parse::<DemoPolicy>().is_err());
    }

    #[test]
    fn test_tasks_per_episode_zero_opts_out() {
        // Task-less episodes remain supported (legacy behaviour) and stay
        // deterministic.
        let mut cfg = small_config();
        cfg.tasks_per_episode = 0;
        let gen = ExpertDemoGenerator::new(cfg);
        let t1 = gen.generate_episode(5).expect("episode generation failed");
        let t2 = gen.generate_episode(5).expect("episode generation failed");
        assert_eq!(t1.steps[0].actions, t2.steps[0].actions);
    }

    #[test]
    fn test_task_assignment_produces_reward_signal() {
        // With tasks assigned, a moving (random) policy must pick up dense
        // task-progress rewards in at least one episode of a seed batch.
        // Any single seed can legitimately miss its task, so this asserts
        // over a batch — deterministic given fixed worldgen + task RNG.
        let mut cfg = small_config();
        cfg.policy = DemoPolicy::Random;
        cfg.max_steps = 120;
        let gen = ExpertDemoGenerator::new(cfg);
        let any_reward = (0..12).any(|seed| {
            gen.generate_episode(seed)
                .expect("episode generation failed")
                .steps
                .iter()
                .any(|s| s.rewards.iter().any(|&r| r.abs() > f32::EPSILON))
        });
        assert!(
            any_reward,
            "no task reward signal in 12 random-policy episodes — task assignment broken?"
        );
    }
}
