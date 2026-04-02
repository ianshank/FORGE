//! Minari-format JSONL dataset reader.
//!
//! [Minari](https://minari.farama.org/) is the Farama Foundation's standard for
//! offline RL datasets (successor to D4RL). It stores episodes in HDF5 files,
//! but the `forge-replay` export pipeline already produces compatible JSONL
//! (one JSON object per step), so this reader targets that interchange format.
//!
//! # Expected file format
//!
//! A `.jsonl` file where each line is a JSON object with at minimum:
//!
//! ```json
//! {
//!   "observations": { "position": [x, y], "health": 0.9, "stamina": 0.8 },
//!   "actions": [3],
//!   "rewards": [0.5],
//!   "terminated": false,
//!   "truncated": false
//! }
//! ```
//!
//! Observations that cannot be reconstructed fully (e.g. missing `grid_view`)
//! are filled with safe defaults so the loader is lenient about partial exports.
//!
//! # Supported datasets
//!
//! | Dataset | URL | License |
//! |---------|-----|---------|
//! | D4RL PointMaze (umaze / medium / large) | <https://minari.farama.org/> | Apache 2.0 |
//! | D4RL Adroit Hand | <https://minari.farama.org/> | Apache 2.0 |
//!
//! # Usage
//!
//! ```rust,no_run
//! use forge_data::minari::MinariLoader;
//! use forge_data::loader::DatasetLoader;
//!
//! let loader = MinariLoader::default();
//! let dataset = loader.load("path/to/episodes.jsonl").unwrap();
//! println!("{} trajectories loaded", dataset.len());
//! ```

use std::io::{BufRead, BufReader};

use forge_replay::trajectory::TrajectoryBuilder;
use forge_types::agent_interface::AgentResponse;
use forge_types::observation::Observation;
use serde::Deserialize;
use tracing::{debug, instrument, warn};

use crate::loader::{DatasetError, DatasetLoader, OfflineDataset};

/// Loader for Minari / D4RL JSONL episode files.
///
/// Converts external observation/action/reward triples into FORGE
/// [`Trajectory`] objects using a best-effort field mapping.
#[derive(Debug, Clone, Default)]
pub struct MinariLoader {
    /// Maximum number of steps to read (0 = unlimited).
    pub max_steps_per_episode: u64,
    /// Maximum number of episodes to load (0 = unlimited).
    pub max_episodes: usize,
}

impl MinariLoader {
    /// Creates a loader with optional limits.
    #[instrument]
    pub fn new(max_steps_per_episode: u64, max_episodes: usize) -> Self {
        Self {
            max_steps_per_episode,
            max_episodes,
        }
    }
}

// ---------------------------------------------------------------------------
// Wire format — permissive serde structs
// ---------------------------------------------------------------------------

/// A single step as serialised in a Minari-compatible JSONL file.
#[derive(Debug, Deserialize)]
struct MinariStep {
    /// Observation dict (partial is fine — missing fields get defaults).
    #[serde(default)]
    observations: MinariObs,
    /// Per-agent action IDs.
    #[serde(default)]
    actions: Vec<u32>,
    /// Per-agent rewards.
    #[serde(default)]
    rewards: Vec<f32>,
    #[serde(default)]
    terminated: bool,
    #[serde(default)]
    truncated: bool,
    /// Optional tick counter; derived from line index if absent.
    #[serde(default)]
    tick: Option<u64>,
}

/// Partial observation fields from Minari.
#[derive(Debug, Default, Deserialize)]
struct MinariObs {
    #[serde(default)]
    position: Option<[u16; 2]>,
    #[serde(default)]
    health: Option<f32>,
    #[serde(default)]
    stamina: Option<f32>,
    #[serde(default)]
    day_phase: Option<u8>,
    #[serde(default)]
    task_progress: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn minari_obs_to_forge(obs: &MinariObs, view_size: u16) -> Observation {
    crate::build_default_obs(
        obs.position.map(|p| (p[0], p[1])).unwrap_or((0, 0)),
        obs.health.unwrap_or(1.0).clamp(0.0, 1.0),
        obs.stamina.unwrap_or(1.0).clamp(0.0, 1.0),
        view_size,
        obs.day_phase.unwrap_or(1),
        obs.task_progress.clone().unwrap_or_default(),
    )
}

// ---------------------------------------------------------------------------
// DatasetLoader impl
// ---------------------------------------------------------------------------

impl DatasetLoader for MinariLoader {
    /// Loads episodes from a JSONL file at `path`.
    ///
    /// Each contiguous block of steps between episode boundaries
    /// (`terminated = true` or `truncated = true`) becomes one `Trajectory`.
    #[instrument(skip(self), fields(path))]
    fn load(&self, path: &str) -> Result<OfflineDataset, DatasetError> {
        let file = std::fs::File::open(path).map_err(DatasetError::from)?;
        let reader = BufReader::new(file);

        let mut dataset = OfflineDataset::new("Minari");
        dataset.metadata.source_url =
            Some("https://minari.farama.org/".to_string());
        dataset.metadata.license = Some("Apache-2.0".to_string());

        let mut builder = TrajectoryBuilder::new();
        let mut episode_seed: u64 = 0;
        let mut cumulative_rewards: Vec<f32> = Vec::new();
        let mut step_in_episode: u64 = 0;

        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(DatasetError::from)?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let raw: MinariStep = serde_json::from_str(line).map_err(|e| {
                DatasetError::Deserialize(format!("line {}: {e}", line_no + 1))
            })?;

            let tick = raw.tick.unwrap_or(step_in_episode);
            let obs = minari_obs_to_forge(&raw.observations, crate::DEFAULT_VIEW_SIZE);
            let num_agents = raw.actions.len().max(1);

            // Grow reward accumulator if this episode introduced more agents.
            if cumulative_rewards.len() < num_agents {
                cumulative_rewards.resize(num_agents, 0.0);
            }
            for (i, &r) in raw.rewards.iter().enumerate() {
                cumulative_rewards[i] += r;
            }

            let responses: Vec<AgentResponse> = raw
                .actions
                .iter()
                .map(|&id| AgentResponse::from_action(id))
                .collect();

            let padded_rewards = {
                let mut r = raw.rewards.clone();
                r.resize(num_agents, 0.0);
                r
            };

            builder.record_step(
                tick,
                vec![obs],
                &responses,
                padded_rewards,
                raw.terminated,
                raw.truncated,
            );

            step_in_episode += 1;

            let episode_done = raw.terminated
                || raw.truncated
                || (self.max_steps_per_episode > 0
                    && step_in_episode >= self.max_steps_per_episode);

            if episode_done {
                let final_rewards = std::mem::take(&mut cumulative_rewards);
                let traj = std::mem::replace(&mut builder, TrajectoryBuilder::new())
                    .seed(episode_seed)
                    .scenario_id(format!("minari_ep_{episode_seed}"))
                    .build(final_rewards);

                debug!(
                    episode = episode_seed,
                    steps = traj.len(),
                    "Loaded Minari episode"
                );
                dataset.push(traj);

                episode_seed += 1;
                step_in_episode = 0;

                if self.max_episodes > 0 && dataset.len() >= self.max_episodes {
                    break;
                }
            }
        }

        Ok(dataset)
    }

    fn source_name(&self) -> &str {
        "Minari"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn write_jsonl(lines: &[&str]) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        f
    }

    #[test]
    fn test_load_single_episode() {
        let step_a = r#"{"observations":{"position":[3,4],"health":0.9},"actions":[1],"rewards":[0.5],"terminated":false,"truncated":false}"#;
        let step_b = r#"{"observations":{"position":[4,4],"health":0.8},"actions":[2],"rewards":[1.0],"terminated":true,"truncated":false}"#;
        let f = write_jsonl(&[step_a, step_b]);

        let loader = MinariLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();

        assert_eq!(ds.len(), 1);
        assert_eq!(ds.trajectories[0].len(), 2);
        assert_eq!(ds.metadata.source, "Minari");
    }

    #[test]
    fn test_load_multiple_episodes() {
        let ep1_step1 = r#"{"actions":[0],"rewards":[0.0],"terminated":false,"truncated":false}"#;
        let ep1_step2 = r#"{"actions":[1],"rewards":[1.0],"terminated":true,"truncated":false}"#;
        let ep2_step1 = r#"{"actions":[2],"rewards":[0.5],"terminated":true,"truncated":false}"#;
        let f = write_jsonl(&[ep1_step1, ep1_step2, ep2_step1]);

        let loader = MinariLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();

        assert_eq!(ds.len(), 2);
        assert_eq!(ds.trajectories[0].len(), 2);
        assert_eq!(ds.trajectories[1].len(), 1);
    }

    #[test]
    fn test_max_episodes_limit() {
        let lines: Vec<String> = (0..6)
            .map(|_| {
                r#"{"actions":[0],"rewards":[0.0],"terminated":true,"truncated":false}"#
                    .to_string()
            })
            .collect();
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let f = write_jsonl(&refs);

        let loader = MinariLoader::new(0, 3);
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 3);
    }

    #[test]
    fn test_max_steps_per_episode() {
        let lines: Vec<String> = (0..10)
            .map(|_| {
                r#"{"actions":[1],"rewards":[0.1],"terminated":false,"truncated":false}"#
                    .to_string()
            })
            .collect();
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let f = write_jsonl(&refs);

        let loader = MinariLoader::new(3, 0);
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();

        // 10 steps / 3 per episode = 3 full episodes + remainder
        assert!(ds.len() >= 3);
        for traj in &ds.trajectories {
            assert!(traj.len() <= 3);
        }
    }

    #[test]
    fn test_malformed_line_returns_error() {
        let f = write_jsonl(&["this is not json"]);
        let loader = MinariLoader::default();
        assert!(loader.load(f.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn test_observations_mapped_correctly() {
        let step = r#"{"observations":{"position":[7,12],"health":0.75,"stamina":0.5,"day_phase":2},"actions":[3],"rewards":[0.2],"terminated":true,"truncated":false}"#;
        let f = write_jsonl(&[step]);

        let loader = MinariLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        let obs = &ds.trajectories[0].steps[0].observations[0];

        assert_eq!(obs.position, (7, 12));
        assert!((obs.health - 0.75).abs() < 1e-5);
        assert!((obs.stamina - 0.5).abs() < 1e-5);
        assert_eq!(obs.day_phase, 2);
    }

    #[test]
    fn test_nonexistent_file_returns_error() {
        let loader = MinariLoader::default();
        assert!(loader.load("/nonexistent/path/file.jsonl").is_err());
    }

    #[test]
    fn test_source_name() {
        let loader = MinariLoader::default();
        assert_eq!(loader.source_name(), "Minari");
    }

    #[test]
    fn test_loader_new_fields() {
        let loader = MinariLoader::new(50, 5);
        assert_eq!(loader.max_steps_per_episode, 50);
        assert_eq!(loader.max_episodes, 5);
    }

    #[test]
    fn test_load_empty_file() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let loader = MinariLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn test_blank_lines_skipped() {
        let step = r#"{"actions":[1],"rewards":[1.0],"terminated":true}"#;
        let f = write_jsonl(&["", step, "", ""]);
        let loader = MinariLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn test_truncated_episode_boundary() {
        let step_a = r#"{"actions":[0],"rewards":[0.0],"terminated":false,"truncated":false}"#;
        let step_b = r#"{"actions":[1],"rewards":[0.5],"terminated":false,"truncated":true}"#;
        let f = write_jsonl(&[step_a, step_b]);

        let loader = MinariLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.trajectories[0].len(), 2);
    }

    #[test]
    fn test_metadata_source_name_set() {
        let step = r#"{"actions":[0],"rewards":[0.0],"terminated":true}"#;
        let f = write_jsonl(&[step]);
        let loader = MinariLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.metadata.source, "Minari");
        assert!(ds.metadata.source_url.is_some());
    }
}
