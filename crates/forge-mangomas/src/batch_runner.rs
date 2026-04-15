//! High-throughput headless episode runner for MangoMAS training data collection.
//!
//! Uses rayon for parallel episode collection across multiple FORGE
//! environments. Each environment runs independently with a unique seed
//! derived from the base seed + environment index.

use std::sync::Mutex;

use forge_core::WorldState;
use forge_types::action::Action;
use forge_types::observation::Observation;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use tracing::{debug, instrument, warn};

use crate::config::BatchRunnerConfig;
use crate::error::{MangoMasError, MangoMasResult};

/// A single transition: (observation, action, reward, next_observation, done).
#[derive(Debug, Clone)]
pub struct Transition {
    /// Observation before the action.
    pub observation: Observation,
    /// Action taken (discrete ID).
    pub action_id: u32,
    /// Reward received.
    pub reward: f32,
    /// Whether the episode terminated after this step.
    pub done: bool,
}

/// A complete episode of transitions.
#[derive(Debug, Clone)]
pub struct Episode {
    /// Sequence of transitions in this episode.
    pub transitions: Vec<Transition>,
    /// Total reward accumulated.
    pub total_reward: f32,
    /// Episode length in steps.
    pub length: u64,
    /// Seed used for this episode's environment.
    pub seed: u64,
}

/// A batch of collected episodes.
#[derive(Debug, Clone)]
pub struct EpisodeBatch {
    /// All episodes in the batch.
    pub episodes: Vec<Episode>,
    /// Total transitions across all episodes.
    pub total_transitions: u64,
}

/// Action selection strategy for batch collection.
pub trait ActionPolicy: Send + Sync {
    /// Select an action given the current observation for a specific agent.
    fn select_action(&self, obs: &Observation, agent_idx: usize) -> u32;
}

/// Random action policy for baseline data collection.
///
/// Uses a seeded `Pcg64Mcg` RNG to ensure deterministic action selection.
/// Two policies created with the same seed and action space size will
/// produce identical action sequences.
pub struct RandomActionPolicy {
    /// Size of the discrete action space.
    action_space_size: u32,
    /// Deterministic RNG guarded by a mutex for `Send + Sync` compatibility.
    rng: Mutex<Pcg64Mcg>,
}

impl RandomActionPolicy {
    /// Creates a new random policy with a default seed of 0.
    #[instrument(skip_all)]
    pub fn new(action_space_size: u32) -> Self {
        Self::new_with_seed(action_space_size, 0)
    }

    /// Creates a new random policy seeded for deterministic reproducibility.
    #[instrument(skip_all)]
    pub fn new_with_seed(action_space_size: u32, seed: u64) -> Self {
        Self {
            action_space_size,
            rng: Mutex::new(Pcg64Mcg::seed_from_u64(seed)),
        }
    }
}

impl ActionPolicy for RandomActionPolicy {
    fn select_action(&self, _obs: &Observation, _agent_idx: usize) -> u32 {
        use rand::Rng;
        let mut rng = self
            .rng
            .lock()
            .expect("RandomActionPolicy RNG lock poisoned");
        rng.gen_range(0..self.action_space_size)
    }
}

/// Headless batch runner for high-throughput episode collection.
///
/// Runs multiple FORGE environments in parallel via rayon, collecting
/// transition data for training MangoMAS agents.
pub struct BatchRunner {
    config: BatchRunnerConfig,
}

impl BatchRunner {
    /// Creates a new batch runner.
    #[instrument(skip_all)]
    pub fn new(config: BatchRunnerConfig) -> Self {
        Self { config }
    }

    /// Returns the batch runner configuration.
    pub fn config(&self) -> &BatchRunnerConfig {
        &self.config
    }

    /// Collects a batch of episodes using the given action policy.
    ///
    /// Runs `num_episodes` episodes across parallel environments,
    /// each with a unique seed for deterministic reproducibility.
    #[instrument(skip(self, policy))]
    pub fn collect_episodes(
        &self,
        num_episodes: u32,
        policy: &(dyn ActionPolicy + Sync),
    ) -> MangoMasResult<EpisodeBatch> {
        let episodes: Vec<MangoMasResult<Episode>> = (0..num_episodes)
            .into_par_iter()
            .map(|episode_idx| {
                let seed = self.config.seed.wrapping_add(episode_idx as u64);
                self.run_single_episode(seed, policy)
            })
            .collect();

        let mut collected = Vec::with_capacity(num_episodes as usize);
        let mut total_transitions = 0u64;
        for result in episodes {
            let episode = result?;
            total_transitions += episode.length;
            collected.push(episode);
        }

        debug!(
            num_episodes = collected.len(),
            total_transitions, "batch collection complete"
        );

        Ok(EpisodeBatch {
            episodes: collected,
            total_transitions,
        })
    }

    /// Runs a single episode and returns the collected transitions.
    fn run_single_episode(&self, seed: u64, policy: &dyn ActionPolicy) -> MangoMasResult<Episode> {
        let mut forge_config = self.config.forge_config.clone();
        forge_config.world.seed = seed;

        let mut world = WorldState::new(forge_config)
            .map_err(|e| MangoMasError::BatchRunner(format!("world creation failed: {}", e)))?;

        let comm_vocab = world.config.agents.comm_vocab_size;
        let drone_enabled = world.config.drone.enabled;

        let agri_enabled = world.config.agri.enabled && drone_enabled;
        let hex_enabled = world.config.world.grid_type == forge_types::config::GridType::Hex;

        let mut transitions = Vec::with_capacity(self.config.max_episode_steps as usize);
        let mut total_reward = 0.0f32;

        // Get initial observations via reset
        let initial_result = world.reset(Some(seed));
        let mut current_obs = initial_result.observations;

        for _step in 0..self.config.max_episode_steps {
            if current_obs.is_empty() {
                break;
            }

            // Select actions for all agents
            let actions: Vec<Action> = current_obs
                .iter()
                .enumerate()
                .map(|(idx, o)| {
                    let action_id = policy.select_action(o, idx);
                    Action::from_discrete_full(
                        action_id,
                        comm_vocab,
                        drone_enabled,
                        agri_enabled,
                        hex_enabled,
                    )
                    .unwrap_or_else(|| {
                        warn!(action_id, "Invalid action ID, falling back to Noop");
                        Action::Noop
                    })
                })
                .collect();

            let action_ids: Vec<u32> = actions
                .iter()
                .map(|a: &Action| {
                    a.to_discrete_configured(comm_vocab, drone_enabled, agri_enabled, hex_enabled)
                })
                .collect();

            let step_result = world.step(&actions);

            // Record transition for agent 0 (primary agent)
            let reward = step_result.rewards.first().copied().unwrap_or(0.0);
            let done = step_result.terminated || step_result.truncated;

            let obs_for_transition = current_obs.into_iter().next().unwrap_or_else(|| {
                debug!("Empty observation from step, using zero-state fallback");
                forge_types::observation::Observation {
                    grid_view: vec![],
                    view_width: 0,
                    view_height: 0,
                    inventory: forge_types::observation::InventoryObservation { slots: vec![] },
                    health: 0.0,
                    stamina: 0.0,
                    position: (0, 0),
                    messages: vec![],
                    day_phase: 0,
                    task_progress: vec![],
                    altitude: 0,
                    battery: 1.0,
                    morphology: 0,
                    heading: 0,
                    crop_scan_results: vec![],
                    soil_readings: vec![],
                    disease_detections: 0,
                    report_ready: false,
                }
            });

            transitions.push(Transition {
                observation: obs_for_transition,
                action_id: action_ids.first().copied().unwrap_or(0),
                reward,
                done,
            });

            total_reward += reward;

            if done {
                break;
            }

            current_obs = step_result.observations;
        }

        let length = transitions.len() as u64;
        Ok(Episode {
            transitions,
            total_reward,
            length,
            seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BatchRunnerConfig;

    fn test_config() -> BatchRunnerConfig {
        let mut config = BatchRunnerConfig {
            max_episode_steps: 10,
            num_envs: 2,
            ..BatchRunnerConfig::default()
        };
        config.forge_config.world.width = 8;
        config.forge_config.world.height = 8;
        config.forge_config.agents.num_agents = 1;
        config.forge_config.agents.default_vision_radius = 2;
        config
    }

    #[test]
    fn test_batch_runner_creation() {
        let runner = BatchRunner::new(test_config());
        assert_eq!(runner.config().max_episode_steps, 10);
    }

    #[test]
    fn test_collect_single_episode() {
        let runner = BatchRunner::new(test_config());
        let action_space = Action::space_size(
            runner.config().forge_config.agents.comm_vocab_size,
            runner.config().forge_config.drone.enabled,
        );
        let policy = RandomActionPolicy::new_with_seed(action_space, 42);
        let batch = runner.collect_episodes(1, &policy).unwrap();
        assert_eq!(batch.episodes.len(), 1);
        assert!(batch.episodes[0].length > 0);
        assert!(batch.total_transitions > 0);
    }

    #[test]
    fn test_collect_multiple_episodes() {
        let runner = BatchRunner::new(test_config());
        let action_space = Action::space_size(
            runner.config().forge_config.agents.comm_vocab_size,
            runner.config().forge_config.drone.enabled,
        );
        let policy = RandomActionPolicy::new_with_seed(action_space, 42);
        let batch = runner.collect_episodes(4, &policy).unwrap();
        assert_eq!(batch.episodes.len(), 4);
    }

    #[test]
    fn test_deterministic_episodes() {
        let mut config = test_config();
        config.seed = 42;
        let runner = BatchRunner::new(config);

        // Use noop policy for determinism
        struct NoopPolicy;
        impl ActionPolicy for NoopPolicy {
            fn select_action(&self, _obs: &Observation, _agent_idx: usize) -> u32 {
                0 // Always noop
            }
        }

        let batch1 = runner.collect_episodes(2, &NoopPolicy).unwrap();
        let batch2 = runner.collect_episodes(2, &NoopPolicy).unwrap();

        assert_eq!(batch1.episodes[0].length, batch2.episodes[0].length);
        assert_eq!(
            batch1.episodes[0].total_reward,
            batch2.episodes[0].total_reward
        );
    }

    #[test]
    fn test_batch_runner_max_steps() {
        let mut config = test_config();
        config.max_episode_steps = 5;
        let runner = BatchRunner::new(config);
        let policy = RandomActionPolicy::new_with_seed(42, 0);
        let batch = runner.collect_episodes(1, &policy).unwrap();
        assert_eq!(batch.episodes.len(), 1);
        assert!(batch.episodes[0].length <= 5);
    }

    #[test]
    fn test_episode_respects_max_steps() {
        let mut config = test_config();
        config.max_episode_steps = 5;
        let runner = BatchRunner::new(config);

        struct NoopPolicy;
        impl ActionPolicy for NoopPolicy {
            fn select_action(&self, _obs: &Observation, _agent_idx: usize) -> u32 {
                0
            }
        }

        let batch = runner.collect_episodes(1, &NoopPolicy).unwrap();
        assert!(batch.episodes[0].length <= 5);
    }

    #[test]
    fn test_random_policy_determinism() {
        let seed = 12345u64;
        let action_space = 10u32;

        let policy_a = RandomActionPolicy::new_with_seed(action_space, seed);
        let policy_b = RandomActionPolicy::new_with_seed(action_space, seed);

        let dummy_obs = Observation {
            grid_view: vec![],
            view_width: 0,
            view_height: 0,
            inventory: forge_types::observation::InventoryObservation { slots: vec![] },
            health: 1.0,
            stamina: 1.0,
            position: (0, 0),
            messages: vec![],
            day_phase: 0,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        };

        // Same seed must produce identical action sequences
        for i in 0..100 {
            let a = policy_a.select_action(&dummy_obs, 0);
            let b = policy_b.select_action(&dummy_obs, 0);
            assert_eq!(a, b, "Actions diverged at step {i}");
            assert!(a < action_space, "Action {a} out of range");
        }
    }

    #[test]
    fn test_random_policy_default_seed_is_deterministic() {
        let policy_a = RandomActionPolicy::new(10);
        let policy_b = RandomActionPolicy::new(10);

        let dummy_obs = Observation {
            grid_view: vec![],
            view_width: 0,
            view_height: 0,
            inventory: forge_types::observation::InventoryObservation { slots: vec![] },
            health: 1.0,
            stamina: 1.0,
            position: (0, 0),
            messages: vec![],
            day_phase: 0,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        };

        for i in 0..50 {
            let a = policy_a.select_action(&dummy_obs, 0);
            let b = policy_b.select_action(&dummy_obs, 0);
            assert_eq!(a, b, "Default-seed actions diverged at step {i}");
        }
    }
}
