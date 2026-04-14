//! Edge replay loader for the FORGE data pipeline.
//!
//! Reads [`CompactReplay`] files from a directory (produced by edge device
//! telemetry upload), reconstructs full [`Trajectory`] objects using
//! deterministic replay, and packages them as an [`OfflineDataset`] for
//! offline RL training.
//!
//! This is Phase 4 of the FORGE cloud-edge pipeline: edge devices record
//! lightweight compact replays (seed + actions only), upload them to cloud
//! storage, and this loader expands them into full observation-action-reward
//! trajectories for training.
//!
//! # Usage
//!
//! ```rust,no_run
//! use forge_data::edge_replay::EdgeReplayLoader;
//! use forge_data::loader::DatasetLoader;
//!
//! let loader = EdgeReplayLoader::new();
//! let dataset = loader.load("data/edge_uploads/").unwrap();
//! println!("{} trajectories reconstructed", dataset.len());
//! ```
//!
//! [`CompactReplay`]: forge_replay::compact::CompactReplay
//! [`Trajectory`]: forge_replay::trajectory::Trajectory
//! [`OfflineDataset`]: crate::loader::OfflineDataset

use forge_replay::compact::CompactReplay;
use forge_replay::trajectory::{Trajectory, TrajectoryBuilder};
use forge_types::agent_interface::AgentResponse;
use tracing::{debug, info, instrument, warn};

use crate::loader::{DatasetError, DatasetLoader, OfflineDataset};

/// Loader that reconstructs full trajectories from edge-uploaded
/// [`CompactReplay`] files.
///
/// Scans a directory for `*.bin` files (bincode-encoded compact replays),
/// deterministically replays each episode to reconstruct observations and
/// rewards, and returns them as an [`OfflineDataset`].
///
/// [`CompactReplay`]: forge_replay::compact::CompactReplay
/// [`OfflineDataset`]: crate::loader::OfflineDataset
#[derive(Debug, Clone)]
pub struct EdgeReplayLoader;

impl EdgeReplayLoader {
    /// Creates a new edge replay loader.
    #[instrument]
    pub fn new() -> Self {
        Self
    }

    /// Reconstructs a single [`Trajectory`] from a [`CompactReplay`].
    ///
    /// Uses [`CompactReplay::replay()`] to step through the simulation
    /// deterministically, extracting observations and rewards at each tick.
    ///
    /// [`Trajectory`]: forge_replay::trajectory::Trajectory
    /// [`CompactReplay`]: forge_replay::compact::CompactReplay
    /// [`CompactReplay::replay()`]: forge_replay::compact::CompactReplay::replay
    #[instrument(skip_all, fields(seed = replay.seed, ticks = replay.metadata.total_ticks))]
    fn reconstruct_trajectory(&self, replay: &CompactReplay) -> Result<Trajectory, DatasetError> {
        let mut replay_iter = replay.replay().ok_or_else(|| {
            DatasetError::Io(
                "failed to start replay: config hash mismatch or world creation error".to_string(),
            )
        })?;

        let mut builder = TrajectoryBuilder::new();
        let mut tick: u64 = 0;

        for step_result in replay_iter.by_ref() {
            // Build AgentResponse for each agent from the recorded action IDs.
            let action_ids = replay
                .actions
                .get(tick as usize)
                .cloned()
                .unwrap_or_default();

            // Pad action_ids to match observation count (engine pads with Noop=0)
            let num_agents = step_result.observations.len();
            let mut action_ids = action_ids; // make mutable
            action_ids.resize(num_agents, 0); // pad with Noop
            action_ids.truncate(num_agents); // truncate excess

            let responses: Vec<AgentResponse> = action_ids
                .iter()
                .map(|&id| AgentResponse::from_action(id))
                .collect();

            builder.record_step(
                tick,
                step_result.observations,
                &responses,
                step_result.rewards,
                step_result.terminated,
                step_result.truncated,
            );

            tick += 1;
        }

        debug!(steps = tick, seed = replay.seed, "Reconstructed trajectory");

        let final_rewards = replay.metadata.final_rewards.clone();
        let trajectory = builder
            .seed(replay.seed)
            .agent_names(replay.metadata.agent_names.clone())
            .agent_metadata(replay.metadata.agent_metadata.clone())
            .scenario_id(
                replay
                    .metadata
                    .scenario_id
                    .clone()
                    .unwrap_or_else(|| format!("edge_replay_{}", replay.seed)),
            )
            .build(final_rewards);

        Ok(trajectory)
    }
}

impl Default for EdgeReplayLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetLoader for EdgeReplayLoader {
    /// Loads all `*.bin` compact replay files from the given directory path.
    ///
    /// Each file is deserialized as a [`CompactReplay`], deterministically
    /// replayed to reconstruct the full trajectory, and added to the
    /// returned [`OfflineDataset`].
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::Io`] if the directory cannot be read, or if
    /// any individual replay file fails to load. Files that fail
    /// deserialization are logged as warnings and skipped.
    ///
    /// [`CompactReplay`]: forge_replay::compact::CompactReplay
    /// [`OfflineDataset`]: crate::loader::OfflineDataset
    /// [`DatasetError::Io`]: crate::loader::DatasetError::Io
    #[instrument(skip(self), fields(path))]
    fn load(&self, path: &str) -> Result<OfflineDataset, DatasetError> {
        let dir = std::path::Path::new(path);
        if !dir.is_dir() {
            return Err(DatasetError::Io(format!("path is not a directory: {path}")));
        }

        let mut dataset = OfflineDataset::new(self.source_name());
        dataset.metadata.source_url = None;
        dataset.metadata.license = None;

        let entries = std::fs::read_dir(dir).map_err(DatasetError::from)?;

        let mut bin_files: Vec<std::path::PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("bin"))
            .collect();

        // Sort for deterministic ordering.
        bin_files.sort();

        info!(
            file_count = bin_files.len(),
            directory = %path,
            "Scanning edge replay directory"
        );

        for file_path in &bin_files {
            let bytes = match std::fs::read(file_path) {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        path = %file_path.display(),
                        error = %e,
                        "Failed to read replay file, skipping"
                    );
                    continue;
                }
            };

            let replay = match CompactReplay::from_bytes(&bytes) {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        path = %file_path.display(),
                        error = %e,
                        "Failed to deserialize replay file, skipping"
                    );
                    continue;
                }
            };

            match self.reconstruct_trajectory(&replay) {
                Ok(trajectory) => {
                    debug!(
                        path = %file_path.display(),
                        steps = trajectory.len(),
                        "Reconstructed trajectory from edge replay"
                    );
                    dataset.push(trajectory);
                }
                Err(e) => {
                    warn!(
                        path = %file_path.display(),
                        error = %e,
                        "Failed to reconstruct trajectory, skipping"
                    );
                }
            }
        }

        info!(
            trajectories = dataset.len(),
            total_steps = dataset.total_steps(),
            "Loaded edge replay dataset"
        );

        Ok(dataset)
    }

    fn source_name(&self) -> &str {
        "EdgeReplayLoader"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;

    /// Creates a test [`ForgeConfig`] with a small 16x16 world and 1 agent.
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

    /// Builds a test [`CompactReplay`] with a few Noop ticks.
    fn make_test_replay(seed: u64, num_ticks: usize) -> CompactReplay {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, seed);
        for _ in 0..num_ticks {
            builder.record_tick(vec![0]); // Noop
        }
        builder
            .agent_names(vec!["test_agent".to_string()])
            .final_rewards(vec![0.0])
            .build()
    }

    #[test]
    fn test_reconstruct_from_compact_replay() {
        let replay = make_test_replay(42, 5);
        let loader = EdgeReplayLoader::new();
        let trajectory = loader.reconstruct_trajectory(&replay).unwrap();

        assert_eq!(trajectory.len(), 5);
        assert_eq!(trajectory.metadata.seed, 42);
        assert_eq!(trajectory.metadata.agent_names, vec!["test_agent"]);
    }

    #[test]
    fn test_reconstruct_preserves_step_count() {
        let replay = make_test_replay(99, 10);
        let loader = EdgeReplayLoader::new();
        let trajectory = loader.reconstruct_trajectory(&replay).unwrap();

        assert_eq!(trajectory.len(), 10);
        assert_eq!(trajectory.metadata.total_steps, 10);
    }

    #[test]
    fn test_reconstruct_observations_populated() {
        let replay = make_test_replay(42, 3);
        let loader = EdgeReplayLoader::new();
        let trajectory = loader.reconstruct_trajectory(&replay).unwrap();

        for step in &trajectory.steps {
            // Each step should have one observation per agent.
            assert_eq!(step.observations.len(), 1);
            // Action should be Noop (0).
            assert_eq!(step.actions, vec![0]);
        }
    }

    #[test]
    fn test_reconstruct_empty_replay() {
        let replay = make_test_replay(42, 0);
        let loader = EdgeReplayLoader::new();
        let trajectory = loader.reconstruct_trajectory(&replay).unwrap();

        assert!(trajectory.is_empty());
        assert_eq!(trajectory.len(), 0);
    }

    #[test]
    fn test_load_from_directory() {
        let dir = tempfile::tempdir().unwrap();

        // Write two replay files.
        for seed in [1u64, 2] {
            let replay = make_test_replay(seed, 3);
            let bytes = replay.to_bytes().unwrap();
            let path = dir.path().join(format!("replay_{seed}.bin"));
            std::fs::write(&path, bytes).unwrap();
        }

        let loader = EdgeReplayLoader::new();
        let dataset = loader.load(dir.path().to_str().unwrap()).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.metadata.source, "EdgeReplayLoader");
        // Each replay has 3 steps, so total should be 6.
        assert_eq!(dataset.total_steps(), 6);
    }

    #[test]
    fn test_load_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let loader = EdgeReplayLoader::new();
        let dataset = loader.load(dir.path().to_str().unwrap()).unwrap();

        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
        assert_eq!(dataset.total_steps(), 0);
    }

    #[test]
    fn test_load_skips_non_bin_files() {
        let dir = tempfile::tempdir().unwrap();

        // Write a valid replay as .bin
        let replay = make_test_replay(1, 3);
        let bytes = replay.to_bytes().unwrap();
        std::fs::write(dir.path().join("valid.bin"), bytes).unwrap();

        // Write junk with non-.bin extension
        std::fs::write(dir.path().join("readme.txt"), b"hello").unwrap();
        std::fs::write(dir.path().join("data.json"), b"{}").unwrap();

        let loader = EdgeReplayLoader::new();
        let dataset = loader.load(dir.path().to_str().unwrap()).unwrap();

        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_load_skips_corrupt_files() {
        let dir = tempfile::tempdir().unwrap();

        // Write a valid replay
        let replay = make_test_replay(1, 3);
        let bytes = replay.to_bytes().unwrap();
        std::fs::write(dir.path().join("good.bin"), bytes).unwrap();

        // Write corrupt data as .bin
        std::fs::write(dir.path().join("bad.bin"), b"not a valid replay").unwrap();

        let loader = EdgeReplayLoader::new();
        let dataset = loader.load(dir.path().to_str().unwrap()).unwrap();

        // Only the valid replay should be loaded.
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_load_nonexistent_path_returns_error() {
        let loader = EdgeReplayLoader::new();
        let result = loader.load("/nonexistent/path/to/replays");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_file_not_directory_returns_error() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let loader = EdgeReplayLoader::new();
        let result = loader.load(f.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_source_name() {
        let loader = EdgeReplayLoader::new();
        assert_eq!(loader.source_name(), "EdgeReplayLoader");
    }

    #[test]
    fn test_default_impl() {
        let loader = EdgeReplayLoader::default();
        assert_eq!(loader.source_name(), "EdgeReplayLoader");
    }

    #[test]
    fn test_deterministic_reconstruction() {
        // Reconstruct the same replay twice and verify identical trajectories.
        let replay = make_test_replay(42, 5);
        let loader = EdgeReplayLoader::new();

        let traj_a = loader.reconstruct_trajectory(&replay).unwrap();
        let traj_b = loader.reconstruct_trajectory(&replay).unwrap();

        assert_eq!(traj_a.len(), traj_b.len());
        for (a, b) in traj_a.steps.iter().zip(traj_b.steps.iter()) {
            assert_eq!(a.observations.len(), b.observations.len());
            assert_eq!(a.actions, b.actions);
            assert_eq!(a.rewards, b.rewards);
            assert_eq!(a.terminated, b.terminated);
            assert_eq!(a.truncated, b.truncated);
            // Verify position matches between runs.
            for (obs_a, obs_b) in a.observations.iter().zip(b.observations.iter()) {
                assert_eq!(obs_a.position, obs_b.position);
                assert_eq!(obs_a.health, obs_b.health);
            }
        }
    }

    #[test]
    fn test_reconstruct_with_movement_actions() {
        let config = test_config();
        let mut builder = CompactReplay::builder(config, 42);
        // Noop, Move Up, Move Down, Move Left, Move Right
        builder.record_tick(vec![0]);
        builder.record_tick(vec![1]);
        builder.record_tick(vec![2]);
        builder.record_tick(vec![3]);
        builder.record_tick(vec![4]);
        let replay = builder
            .agent_names(vec!["mover".to_string()])
            .final_rewards(vec![0.0])
            .build();

        let loader = EdgeReplayLoader::new();
        let trajectory = loader.reconstruct_trajectory(&replay).unwrap();

        assert_eq!(trajectory.len(), 5);
        // Verify the actions are correctly mapped.
        assert_eq!(trajectory.steps[0].actions, vec![0]);
        assert_eq!(trajectory.steps[1].actions, vec![1]);
        assert_eq!(trajectory.steps[2].actions, vec![2]);
        assert_eq!(trajectory.steps[3].actions, vec![3]);
        assert_eq!(trajectory.steps[4].actions, vec![4]);
    }

    #[test]
    fn test_load_deterministic_ordering() {
        let dir = tempfile::tempdir().unwrap();

        // Write replays with different seeds in reverse filename order.
        for (i, seed) in [3u64, 1, 2].iter().enumerate() {
            let replay = make_test_replay(*seed, 2);
            let bytes = replay.to_bytes().unwrap();
            let path = dir.path().join(format!("replay_{i:03}.bin"));
            std::fs::write(&path, bytes).unwrap();
        }

        let loader = EdgeReplayLoader::new();
        let dataset = loader.load(dir.path().to_str().unwrap()).unwrap();

        assert_eq!(dataset.len(), 3);
        // Files are sorted by name, so replay_000 (seed=3) comes first.
        assert_eq!(dataset.trajectories[0].metadata.seed, 3);
        assert_eq!(dataset.trajectories[1].metadata.seed, 1);
        assert_eq!(dataset.trajectories[2].metadata.seed, 2);
    }

    #[test]
    fn test_reconstruct_metadata_propagation() {
        let config = test_config();
        let replay = CompactReplay::builder(config, 77)
            .agent_names(vec!["agent_alpha".to_string()])
            .final_rewards(vec![42.0])
            .scenario_id("patrol_scenario_7".to_string())
            .build();

        let loader = EdgeReplayLoader::new();
        let trajectory = loader.reconstruct_trajectory(&replay).unwrap();

        assert_eq!(trajectory.metadata.seed, 77);
        assert_eq!(trajectory.metadata.agent_names, vec!["agent_alpha"]);
        assert_eq!(trajectory.metadata.final_rewards, vec![42.0]);
        assert_eq!(
            trajectory.metadata.scenario_id.as_deref(),
            Some("patrol_scenario_7")
        );
    }

    #[test]
    fn test_reconstruct_scenario_id_fallback() {
        let config = test_config();
        // No scenario_id set.
        let replay = CompactReplay::builder(config, 55).build();

        let loader = EdgeReplayLoader::new();
        let trajectory = loader.reconstruct_trajectory(&replay).unwrap();

        assert_eq!(
            trajectory.metadata.scenario_id.as_deref(),
            Some("edge_replay_55")
        );
    }

    #[test]
    fn test_multi_agent_reconstruction() {
        let mut config = test_config();
        config.agents.num_agents = 2;

        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0, 1]); // Agent 0: Noop, Agent 1: Move Up
        builder.record_tick(vec![1, 0]); // Agent 0: Move Up, Agent 1: Noop
        let replay = builder
            .agent_names(vec!["agent_a".to_string(), "agent_b".to_string()])
            .final_rewards(vec![1.0, 0.5])
            .build();

        let loader = EdgeReplayLoader::new();
        let trajectory = loader.reconstruct_trajectory(&replay).unwrap();

        assert_eq!(trajectory.len(), 2);
        assert_eq!(trajectory.steps[0].actions, vec![0, 1]);
        assert_eq!(trajectory.steps[1].actions, vec![1, 0]);
        assert_eq!(trajectory.steps[0].observations.len(), 2);
    }
}
