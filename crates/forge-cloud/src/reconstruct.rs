//! Reconstructs full [`Trajectory`] objects from [`CompactReplay`] data.
//!
//! Because FORGE simulations are deterministic (same seed + config + actions
//! = identical state), a compact replay stores only actions and seeds.
//! This module replays those actions through the simulation engine and
//! captures observations, rewards, and termination signals to build
//! full trajectories suitable for offline RL training.

use forge_data::loader::OfflineDataset;
use forge_replay::compact::CompactReplay;
use forge_replay::trajectory::{Trajectory, TrajectoryBuilder};
use forge_types::error::{ForgeError, ForgeResult};
use tracing::{debug, info, instrument, warn};

/// Reconstructs full [`Trajectory`] objects from compact deterministic replays.
///
/// Uses [`CompactReplay::replay()`] to step through the simulation, collecting
/// observations, rewards, and termination signals at each tick.
pub struct TrajectoryReconstructor;

impl TrajectoryReconstructor {
    /// Reconstructs a [`Trajectory`] from a [`CompactReplay`].
    ///
    /// Steps through the replay tick-by-tick using the deterministic
    /// simulation engine, collecting the full observation-action-reward
    /// tuple at each step.
    ///
    /// # Errors
    ///
    /// Returns an error if the replay cannot be replayed (e.g., invalid
    /// config hash or world creation failure).
    #[instrument(skip_all, fields(seed = replay.seed, ticks = replay.metadata.total_ticks))]
    pub fn reconstruct(replay: &CompactReplay) -> ForgeResult<Trajectory> {
        let mut iter = replay.replay().ok_or_else(|| {
            warn!(seed = replay.seed, "Failed to create replay iterator");
            ForgeError::Cloud(forge_types::error::CloudError::Storage(
                "failed to create replay iterator: config hash mismatch or world creation failure"
                    .to_string(),
            ))
        })?;

        let mut builder = TrajectoryBuilder::new();

        for (tick, result) in (0_u64..).zip(iter.by_ref()) {
            let observations = result.observations;
            let rewards = result.rewards;
            let terminated = result.terminated;
            let truncated = result.truncated;
            let action_tick = replay.actions.get(tick as usize);

            // Build placeholder AgentResponse-like data: we only have
            // action IDs from the compact replay, no reasoning traces.
            let num_agents = observations.len();
            let mut actions: Vec<u32> = action_tick.cloned().unwrap_or_else(|| vec![0; num_agents]);
            actions.resize(num_agents, 0); // pad with Noop
            actions.truncate(num_agents);
            let responses: Vec<forge_types::agent_interface::AgentResponse> = actions
                .iter()
                .map(|&a| forge_types::agent_interface::AgentResponse::from_action(a))
                .collect();

            builder.record_step(
                tick,
                observations,
                &responses,
                rewards,
                terminated,
                truncated,
            );
        }

        // Build with metadata carried over from the replay
        let builder = builder
            .seed(replay.seed)
            .agent_names(replay.metadata.agent_names.clone())
            .agent_metadata(replay.metadata.agent_metadata.clone());

        let builder = if let Some(ref scenario_id) = replay.metadata.scenario_id {
            builder.scenario_id(scenario_id.clone())
        } else {
            builder
        };

        let trajectory = builder.build(replay.metadata.final_rewards.clone());

        info!(
            seed = replay.seed,
            steps = trajectory.len(),
            "Reconstructed trajectory"
        );

        Ok(trajectory)
    }

    /// Reconstructs multiple replays into an [`OfflineDataset`].
    ///
    /// Each replay is individually reconstructed into a [`Trajectory`] and
    /// collected into a single dataset. Replays that fail to reconstruct
    /// are logged as warnings and skipped.
    ///
    /// # Errors
    ///
    /// Returns an error only if no replays can be reconstructed at all and
    /// the input slice is non-empty.
    #[instrument(skip_all, fields(count = replays.len()))]
    pub fn reconstruct_batch(replays: &[CompactReplay]) -> ForgeResult<OfflineDataset> {
        let mut dataset = OfflineDataset::new("compact_replay_reconstruction");
        let mut failures = 0u64;

        for (i, replay) in replays.iter().enumerate() {
            match Self::reconstruct(replay) {
                Ok(traj) => {
                    dataset.push(traj);
                }
                Err(e) => {
                    warn!(
                        index = i,
                        seed = replay.seed,
                        error = %e,
                        "Failed to reconstruct replay, skipping"
                    );
                    failures += 1;
                }
            }
        }

        debug!(
            total = replays.len(),
            succeeded = dataset.len(),
            failures,
            "Batch reconstruction complete"
        );

        if dataset.is_empty() && !replays.is_empty() {
            return Err(ForgeError::Cloud(forge_types::error::CloudError::Storage(
                "all replays failed to reconstruct".to_string(),
            )));
        }

        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;

    fn test_config() -> ForgeConfig {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 100;
        config
    }

    fn test_replay(seed: u64, ticks: usize) -> CompactReplay {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, seed);
        for _ in 0..ticks {
            builder.record_tick(vec![0]); // Noop actions
        }
        builder
            .agent_names(vec!["TestAgent".to_string()])
            .final_rewards(vec![0.0])
            .build()
    }

    #[test]
    fn test_reconstruct_produces_valid_trajectory() {
        let replay = test_replay(42, 5);
        let trajectory = TrajectoryReconstructor::reconstruct(&replay).unwrap();

        assert_eq!(trajectory.len(), 5);
        assert_eq!(trajectory.metadata.total_steps, 5);
        // Each step should have observations for our 1 agent
        for step in &trajectory.steps {
            assert_eq!(step.observations.len(), 1);
            assert_eq!(step.rewards.len(), 1);
        }
    }

    #[test]
    fn test_reconstruct_preserves_seed() {
        let replay = test_replay(12345, 3);
        let trajectory = TrajectoryReconstructor::reconstruct(&replay).unwrap();

        assert_eq!(trajectory.metadata.seed, 12345);
    }

    #[test]
    fn test_reconstruct_preserves_metadata() {
        let config = test_config();
        let builder = CompactReplay::builder(config, 42);
        let replay = builder
            .agent_names(vec!["Alpha".to_string()])
            .final_rewards(vec![5.0])
            .scenario_id("test_scenario".to_string())
            .build();

        let trajectory = TrajectoryReconstructor::reconstruct(&replay).unwrap();
        assert_eq!(trajectory.metadata.agent_names, vec!["Alpha"]);
        assert_eq!(trajectory.metadata.final_rewards, vec![5.0]);
        assert_eq!(
            trajectory.metadata.scenario_id.as_deref(),
            Some("test_scenario")
        );
    }

    #[test]
    fn test_reconstruct_batch_produces_dataset() {
        let replays = vec![test_replay(1, 3), test_replay(2, 4), test_replay(3, 5)];

        let dataset = TrajectoryReconstructor::reconstruct_batch(&replays).unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.trajectories[0].len(), 3);
        assert_eq!(dataset.trajectories[1].len(), 4);
        assert_eq!(dataset.trajectories[2].len(), 5);
    }

    #[test]
    fn test_reconstruct_with_invalid_replay() {
        let config = test_config();
        let mut replay = CompactReplay::builder(config, 42).build();
        // Corrupt the config hash to force replay failure
        replay.config_hash = 0;

        let result = TrajectoryReconstructor::reconstruct(&replay);
        assert!(result.is_err());
    }

    #[test]
    fn test_reconstruct_batch_empty_input() {
        let dataset = TrajectoryReconstructor::reconstruct_batch(&[]).unwrap();
        assert!(dataset.is_empty());
    }

    #[test]
    fn test_reconstruct_empty_replay() {
        let replay = test_replay(42, 0);
        let trajectory = TrajectoryReconstructor::reconstruct(&replay).unwrap();
        assert!(trajectory.is_empty());
        assert_eq!(trajectory.metadata.seed, 42);
    }
}
