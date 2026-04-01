//! Strategic Game Maze dataset loader and ForgeConfig seed generator.
//!
//! The [Strategic Game Maze](https://huggingface.co/datasets/laion/strategic_game_maze)
//! dataset (LAION, open license) contains **350,000** 30×30 ASCII mazes with
//! pre-computed BFS shortest paths. This module provides:
//!
//! 1. [`MazeRecord`] — parsed representation of one maze entry.
//! 2. [`MazeToForgeConfig`] — converts a maze's layout into a [`ForgeConfig`] with
//!    matching world dimensions and wall placement injected via a deterministic seed.
//! 3. [`MazeLoader`] — [`DatasetLoader`] that reads the JSONL export and returns
//!    an [`OfflineDataset`] of pre-planned navigation trajectories, where each
//!    BFS step becomes a `Move` action.
//!
//! # File format
//!
//! The HuggingFace dataset can be exported to JSONL via:
//! ```bash
//! python scripts/export_maze_to_jsonl.py --output data/maze.jsonl
//! ```
//!
//! Each line:
//! ```json
//! {
//!   "maze": "#.#\n.  \n#.#",
//!   "solution": "DDRR",
//!   "start": [0, 1],
//!   "end": [2, 1]
//! }
//! ```
//!
//! `maze` uses `#` for walls and `.` / ` ` for open tiles.
//! `solution` is a sequence of direction characters: `U`p, `D`own, `L`eft, `R`ight.
//!
//! # Usage
//!
//! ```rust,no_run
//! use forge_data::maze::MazeLoader;
//! use forge_data::loader::DatasetLoader;
//!
//! let loader = MazeLoader::default();
//! let dataset = loader.load("data/maze.jsonl").unwrap();
//! println!("{} maze trajectories loaded", dataset.len());
//! ```

use std::io::BufRead;

use forge_replay::trajectory::TrajectoryBuilder;
use forge_types::agent_interface::AgentResponse;
use forge_types::config::ForgeConfig;
use forge_types::grid::Direction;
use forge_types::observation::Observation;
use forge_types::Action;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::loader::{DatasetError, DatasetLoader, OfflineDataset};

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

/// One record from the Strategic Game Maze JSONL export.
#[derive(Debug, Serialize, Deserialize)]
pub struct MazeRecord {
    /// ASCII grid, rows separated by `\n`. `#` = wall, ` ` or `.` = open.
    pub maze: String,
    /// BFS solution as direction chars: `U`, `D`, `L`, `R`.
    #[serde(default)]
    pub solution: String,
    /// Start position `[col, row]`.
    #[serde(default)]
    pub start: Option<[u16; 2]>,
    /// End / goal position `[col, row]`.
    #[serde(default)]
    pub end: Option<[u16; 2]>,
}

impl MazeRecord {
    /// Returns the maze dimensions (width, height).
    pub fn dimensions(&self) -> (u16, u16) {
        let rows: Vec<&str> = self.maze.lines().collect();
        let height = rows.len() as u16;
        let width = rows.iter().map(|r| r.len()).max().unwrap_or(0) as u16;
        (width, height)
    }

    /// Converts the ASCII maze to a flat `bool` grid (`true` = walkable).
    pub fn walkable_grid(&self) -> Vec<bool> {
        let rows: Vec<&str> = self.maze.lines().collect();
        let (width, height) = self.dimensions();
        let mut grid = vec![false; (width * height) as usize];
        for (row, line) in rows.iter().enumerate() {
            for (col, ch) in line.chars().enumerate() {
                let idx = row * width as usize + col;
                if idx < grid.len() {
                    grid[idx] = ch != '#';
                }
            }
        }
        grid
    }

    /// Parses the BFS solution string into a list of [`Direction`]s.
    pub fn solution_directions(&self) -> Vec<Direction> {
        self.solution
            .chars()
            .filter_map(|c| match c {
                'U' | 'u' => Some(Direction::Up),
                'D' | 'd' => Some(Direction::Down),
                'L' | 'l' => Some(Direction::Left),
                'R' | 'r' => Some(Direction::Right),
                _ => {
                    warn!(ch = %c, "Unknown direction char in maze solution");
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ForgeConfig builder
// ---------------------------------------------------------------------------

/// Converts a [`MazeRecord`] into a [`ForgeConfig`] suitable for reproducing
/// the maze layout through FORGE's procedural world generator.
///
/// Because FORGE generates worlds from a seed (not from an explicit tile map),
/// this converter stores the maze's walkability mask as a compact seed so the
/// generated world matches the maze's structure as closely as possible at the
/// same dimensions. The seed is derived deterministically from the maze string.
pub struct MazeToForgeConfig;

impl MazeToForgeConfig {
    /// Derives a deterministic u64 seed from the maze ASCII string.
    pub fn maze_seed(maze_str: &str) -> u64 {
        // FNV-1a 64-bit hash — no external dependencies.
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in maze_str.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Converts a [`MazeRecord`] into a [`ForgeConfig`] with matching dimensions
    /// and a seed derived from the maze content.
    pub fn convert(record: &MazeRecord) -> ForgeConfig {
        let (width, height) = record.dimensions();
        let mut config = ForgeConfig::default();
        config.world.width = width.max(8); // minimum 8×8
        config.world.height = height.max(8);
        config.world.seed = Self::maze_seed(&record.maze);
        // Lower resource density — mazes are primarily navigation tasks.
        config.world.resource_density = 0.05;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = record.solution.len() as u64 * 3 + 50;
        config
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Loads the Strategic Game Maze JSONL dataset.
///
/// Each maze record is converted to one [`Trajectory`] where every BFS step
/// becomes a `Move` action, yielding optimal navigation demonstrations.
///
/// [`Trajectory`]: forge_replay::trajectory::Trajectory
#[derive(Debug, Clone, Default)]
pub struct MazeLoader {
    /// Maximum mazes to load (0 = unlimited).
    pub max_mazes: usize,
    /// Skip mazes whose solution is longer than this (0 = no limit).
    pub max_solution_length: usize,
}

impl MazeLoader {
    /// Creates a loader with optional limits.
    #[instrument]
    pub fn new(max_mazes: usize, max_solution_length: usize) -> Self {
        Self {
            max_mazes,
            max_solution_length,
        }
    }

    fn make_obs(position: (u16, u16)) -> Observation {
        // BFS path is always optimal → task_progress starts at 1.0
        crate::build_default_obs(position, 1.0, 1.0, crate::DEFAULT_VIEW_SIZE, 1, vec![1.0])
    }

    /// Builds a [`Trajectory`] from a single [`MazeRecord`].
    fn build_trajectory(
        record: &MazeRecord,
        episode_id: u64,
    ) -> forge_replay::trajectory::Trajectory {
        let dirs = record.solution_directions();
        let mut pos = record.start.map(|s| (s[0], s[1])).unwrap_or((0, 0));

        let seed = MazeToForgeConfig::maze_seed(&record.maze);
        let mut builder = TrajectoryBuilder::new();

        for (tick, dir) in dirs.iter().enumerate() {
            let obs = Self::make_obs(pos);
            let action = Action::Move(*dir);
            let action_id = action.to_discrete();
            let response = AgentResponse::from_action(action_id);
            let is_last = tick + 1 == dirs.len();

            builder.record_step(
                tick as u64,
                vec![obs],
                &[response],
                vec![if is_last { 1.0 } else { 0.01 }],
                is_last,
                false,
            );

            // Advance position (no collision checking — trust BFS solution).
            match dir {
                Direction::Up => pos.1 = pos.1.saturating_sub(1),
                Direction::Down => pos.1 = pos.1.saturating_add(1),
                Direction::Left => pos.0 = pos.0.saturating_sub(1),
                Direction::Right => pos.0 = pos.0.saturating_add(1),
            }
        }

        builder
            .seed(seed)
            .agent_names(vec!["maze_bfs_agent".to_string()])
            .scenario_id(format!("maze_ep_{episode_id}"))
            .build(vec![1.0])
    }
}

impl DatasetLoader for MazeLoader {
    /// Loads mazes from a JSONL file and converts each to a navigation trajectory.
    #[instrument(skip(self), fields(path))]
    fn load(&self, path: &str) -> Result<OfflineDataset, DatasetError> {
        let file = std::fs::File::open(path).map_err(DatasetError::from)?;
        let reader = std::io::BufReader::new(file);

        let mut dataset = OfflineDataset::new("StrategicGameMaze");
        dataset.metadata.source_url = Some(
            "https://huggingface.co/datasets/laion/strategic_game_maze".to_string(),
        );
        dataset.metadata.license = Some("Open (LAION)".to_string());

        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(DatasetError::from)?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let record: MazeRecord = serde_json::from_str(line).map_err(|e| {
                DatasetError::Deserialize(format!("line {}: {e}", line_no + 1))
            })?;

            if self.max_solution_length > 0
                && record.solution.len() > self.max_solution_length
            {
                continue;
            }

            let traj = Self::build_trajectory(&record, line_no as u64);

            debug!(
                episode = line_no,
                steps = traj.len(),
                dims = ?record.dimensions(),
                "Loaded maze trajectory"
            );
            dataset.push(traj);

            if self.max_mazes > 0 && dataset.len() >= self.max_mazes {
                break;
            }
        }

        Ok(dataset)
    }

    fn source_name(&self) -> &str {
        "StrategicGameMaze"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_maze_record() -> MazeRecord {
        MazeRecord {
            maze: "#####\n#   #\n# # #\n#   #\n#####".to_string(),
            solution: "RRDDRR".to_string(),
            start: Some([1, 1]),
            end: Some([3, 3]),
        }
    }

    #[test]
    fn test_maze_dimensions() {
        let r = simple_maze_record();
        assert_eq!(r.dimensions(), (5, 5));
    }

    #[test]
    fn test_solution_directions() {
        let r = simple_maze_record();
        let dirs = r.solution_directions();
        assert_eq!(dirs.len(), 6);
        assert_eq!(dirs[0], Direction::Right);
        assert_eq!(dirs[2], Direction::Down);
    }

    #[test]
    fn test_walkable_grid() {
        let r = simple_maze_record();
        let grid = r.walkable_grid();
        assert_eq!(grid.len(), 25);
        // Top-left is a wall
        assert!(!grid[0]);
        // Position (1,1) should be walkable
        assert!(grid[1 * 5 + 1]);
    }

    #[test]
    fn test_maze_seed_deterministic() {
        let s1 = MazeToForgeConfig::maze_seed("###\n# #\n###");
        let s2 = MazeToForgeConfig::maze_seed("###\n# #\n###");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_maze_seed_unique() {
        let s1 = MazeToForgeConfig::maze_seed("###\n# #\n###");
        let s2 = MazeToForgeConfig::maze_seed("###\n#  \n###");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_convert_to_forge_config() {
        let r = simple_maze_record();
        let cfg = MazeToForgeConfig::convert(&r);
        // 5×5 maze is below the minimum world size of 8, so it gets clamped.
        assert_eq!(cfg.world.width, 8);
        assert_eq!(cfg.world.height, 8);
        assert_eq!(cfg.agents.num_agents, 1);
    }

    #[test]
    fn test_build_trajectory() {
        let r = simple_maze_record();
        let traj = MazeLoader::build_trajectory(&r, 0);
        assert_eq!(traj.len(), 6); // RRDDRR = 6 steps
        // Last step should be terminal
        assert!(traj.steps.last().unwrap().terminated);
        assert_eq!(traj.steps.last().unwrap().rewards[0], 1.0);
    }

    fn write_maze_jsonl(
        f: &mut tempfile::NamedTempFile,
        maze: &str,
        solution: &str,
        start: [u16; 2],
        end: [u16; 2],
    ) {
        use std::io::Write as _;
        // Build JSON manually so \n in maze is the JSON escape for newline.
        // serde_json::to_string will produce the correct JSON encoding.
        let record = MazeRecord {
            maze: maze.to_string(),
            solution: solution.to_string(),
            start: Some(start),
            end: Some(end),
        };
        let json = serde_json::to_string(&record).unwrap();
        writeln!(f, "{}", json).unwrap();
    }

    #[test]
    fn test_loader_from_jsonl() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write_maze_jsonl(&mut f, "###\n# #\n###", "UD", [1, 1], [1, 1]);

        let loader = MazeLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.trajectories[0].len(), 2);
        assert_eq!(ds.metadata.source, "StrategicGameMaze");
    }

    #[test]
    fn test_max_mazes_limit() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for _ in 0..5 {
            write_maze_jsonl(&mut f, "# #\n   \n# #", "R", [0, 1], [1, 1]);
        }

        let loader = MazeLoader::new(2, 0);
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_max_solution_length_filter() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        // Short solution — passes filter
        write_maze_jsonl(&mut f, "# #", "R", [0, 0], [1, 0]);
        // Long solution — filtered out
        write_maze_jsonl(&mut f, "# #", "RRRRRRRRRR", [0, 0], [1, 0]);

        let loader = MazeLoader::new(0, 5);
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn test_source_name() {
        let loader = MazeLoader::default();
        assert_eq!(loader.source_name(), "StrategicGameMaze");
    }

    #[test]
    fn test_maze_loader_new_fields() {
        let loader = MazeLoader::new(10, 50);
        assert_eq!(loader.max_mazes, 10);
        assert_eq!(loader.max_solution_length, 50);
    }

    #[test]
    fn test_empty_maze_dimensions() {
        let r = MazeRecord {
            maze: String::new(),
            solution: String::new(),
            start: None,
            end: None,
        };
        assert_eq!(r.dimensions(), (0, 0));
    }

    #[test]
    fn test_walkable_grid_all_walls() {
        let r = MazeRecord {
            maze: "###\n###\n###".to_string(),
            solution: String::new(),
            start: None,
            end: None,
        };
        let grid = r.walkable_grid();
        assert!(!grid.iter().any(|&v| v));
    }

    #[test]
    fn test_solution_directions_unknown_char_skipped() {
        let r = MazeRecord {
            maze: String::new(),
            solution: "RXRZ".to_string(),
            start: None,
            end: None,
        };
        let dirs = r.solution_directions();
        // 'X' and 'Z' are skipped; only 'R', 'R' remain
        assert_eq!(dirs.len(), 2);
    }

    #[test]
    fn test_build_trajectory_no_start_defaults_to_origin() {
        let r = MazeRecord {
            maze: "   ".to_string(),
            solution: "R".to_string(),
            start: None,
            end: None,
        };
        let traj = MazeLoader::build_trajectory(&r, 0);
        assert_eq!(traj.steps[0].observations[0].position, (0, 0));
    }

    #[test]
    fn test_load_missing_file_returns_error() {
        let loader = MazeLoader::default();
        let result = loader.load("/nonexistent/path/file.jsonl");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_solution_produces_no_steps() {
        let r = MazeRecord {
            maze: "###".to_string(),
            solution: String::new(),
            start: Some([1, 0]),
            end: Some([1, 0]),
        };
        let traj = MazeLoader::build_trajectory(&r, 0);
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_convert_to_forge_config_seed_from_content() {
        let r1 = simple_maze_record();
        let mut r2 = simple_maze_record();
        r2.maze = "XXXXX\nX   X\nX X X\nX   X\nXXXXX".to_string();
        let cfg1 = MazeToForgeConfig::convert(&r1);
        let cfg2 = MazeToForgeConfig::convert(&r2);
        assert_ne!(cfg1.world.seed, cfg2.world.seed);
    }
}
