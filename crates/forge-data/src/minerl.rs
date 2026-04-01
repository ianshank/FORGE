//! MineRL dataset action adapter.
//!
//! [MineRL](https://minerl.io/) provides 60M+ labelled state–action pairs from
//! human Minecraft gameplay, including resource gathering and multi-step crafting
//! — the two mechanics that map most directly onto FORGE.
//!
//! This module does **not** download or parse MineRL's raw HDF5 files (which
//! require the `minerl` Python package). Instead it provides:
//!
//! 1. [`MinerlAction`] — a Rust representation of MineRL's action dict that can
//!    be deserialised from the JSONL exports produced by
//!    `scripts/export_minerl_to_jsonl.py` (see project root).
//!
//! 2. [`MinerlActionMapper`] — maps `MinerlAction` values onto FORGE's discrete
//!    [`Action`] enum so MineRL episodes can be used as imitation-learning seeds.
//!
//! 3. [`MinerlLoader`] — a [`DatasetLoader`] that reads the exported JSONL files
//!    and produces [`OfflineDataset`] trajectories.
//!
//! # MineRL → FORGE action mapping rationale
//!
//! | MineRL action | FORGE equivalent | Notes |
//! |---------------|-----------------|-------|
//! | `forward=1` | `Move(Up)` | North |
//! | `back=1` | `Move(Down)` | |
//! | `left=1` | `Move(Left)` | |
//! | `right=1` | `Move(Right)` | |
//! | `attack=1` | `Use(0)` (Sword slot) | |
//! | `use=1` | `Interact` | Open chest / door |
//! | `pickup=1` | `PickUp` | |
//! | `craft=<item>` | `Craft(<recipe_idx>)` | See [`CRAFT_MAP`] |
//! | `no_op` | `Noop` | |
//!
//! Actions without a FORGE equivalent (jump, sprint, look) are silently mapped
//! to `Noop` and a warning is emitted.
//!
//! # Datasets
//!
//! | Dataset | URL | License |
//! |---------|-----|---------|
//! | MineRL v1.0 | <https://zenodo.org/records/12659939> | MIT-like |
//! | BASALT benchmark | <https://github.com/minerllabs/basalt-benchmark> | Open |
//!
//! # Usage
//!
//! ```rust,no_run
//! use forge_data::minerl::MinerlLoader;
//! use forge_data::loader::DatasetLoader;
//!
//! let loader = MinerlLoader::default();
//! let dataset = loader.load("path/to/minerl_export.jsonl").unwrap();
//! ```

use forge_replay::trajectory::TrajectoryBuilder;
use forge_types::agent_interface::AgentResponse;
use forge_types::grid::Direction;
use forge_types::observation::Observation;
use forge_types::Action;
use serde::Deserialize;
use tracing::{instrument, warn};

use crate::loader::{DatasetError, DatasetLoader, OfflineDataset};

// ---------------------------------------------------------------------------
// Craft item → FORGE recipe index mapping
// ---------------------------------------------------------------------------

/// Maps MineRL craft target item names to FORGE recipe indices.
///
/// FORGE default recipe book (indices 0–8):
/// 0=Axe, 1=Pickaxe, 2=Plank, 3=Bridge, 4=Sword,
/// 5=Rope, 6=Brick, 7=Torch, 8=Shield
///
/// **Order matters**: longer / more-specific patterns must appear before shorter
/// ones to avoid substring collisions (e.g. "pickaxe" must precede "axe").
pub const CRAFT_MAP: &[(&str, u16)] = &[
    // Pickaxe entries — must come before "axe" since "pickaxe" contains "axe"
    ("wooden_pickaxe", 1),
    ("stone_pickaxe", 1),
    ("iron_pickaxe", 1),
    ("pickaxe", 1),
    // Axe entries
    ("wooden_axe", 0),
    ("stone_axe", 0),
    ("iron_axe", 0),
    ("axe", 0),
    // Plank / bridge
    ("log_to_planks", 2),
    ("planks", 2),
    ("plank", 2),
    ("bridge", 3),
    // Sword — before "sword" bare so variants match first
    ("wooden_sword", 4),
    ("stone_sword", 4),
    ("iron_sword", 4),
    ("sword", 4),
    // Other tools
    ("string", 5), // closest to rope (fiber)
    ("rope", 5),
    ("brick", 6),
    ("torch", 7),
    ("shield", 8),
];

/// Looks up a MineRL craft target in [`CRAFT_MAP`].
pub fn craft_name_to_recipe(name: &str) -> Option<u16> {
    let lower = name.to_lowercase();
    CRAFT_MAP
        .iter()
        .find(|(k, _)| lower.contains(k))
        .map(|(_, v)| *v)
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

/// MineRL action as exported by the Python helper script.
///
/// All fields are optional; absent fields default to inactive / 0.
#[derive(Debug, Default, Deserialize)]
pub struct MinerlAction {
    /// Forward movement (0 or 1).
    #[serde(default)]
    pub forward: u8,
    /// Backward movement.
    #[serde(default)]
    pub back: u8,
    /// Strafe left.
    #[serde(default)]
    pub left: u8,
    /// Strafe right.
    #[serde(default)]
    pub right: u8,
    /// Attack / use primary item.
    #[serde(default)]
    pub attack: u8,
    /// Context-sensitive use (open chest, lever, etc.).
    #[serde(rename = "use", default)]
    pub use_: u8,
    /// Pick up nearby item.
    #[serde(default)]
    pub pickup: u8,
    /// Craft target item name (e.g. `"wooden_axe"`).
    #[serde(default)]
    pub craft: Option<String>,
    /// Equip slot index (0–8).
    #[serde(default)]
    pub equip: Option<u8>,
    /// No-op flag.
    #[serde(default)]
    pub no_op: bool,
}

/// One step of a MineRL episode as exported to JSONL.
#[derive(Debug, Deserialize)]
pub struct MinerlStep {
    /// Agent action.
    pub action: MinerlAction,
    /// Scalar reward.
    /// Scalar reward for this step.
    #[serde(default)]
    pub reward: f32,
    /// Whether the episode ended (goal reached / death).
    #[serde(default)]
    pub terminated: bool,
    /// Whether the episode was cut short (time limit).
    #[serde(default)]
    pub truncated: bool,
    /// Optional partial observation metadata.
    #[serde(default)]
    pub obs: Option<MinerlObs>,
}

/// Partial observation exported alongside a MineRL step.
#[derive(Debug, Default, Deserialize)]
pub struct MinerlObs {
    /// Agent health in [0.0, 20.0] (Minecraft scale).
    #[serde(default)]
    pub health: Option<f32>,
    /// Agent position in 3D world-space `[x, y, z]`.
    #[serde(default)]
    pub position: Option<[f32; 3]>,
}

// ---------------------------------------------------------------------------
// Action mapper
// ---------------------------------------------------------------------------

/// Maps [`MinerlAction`] values onto FORGE [`Action`] variants.
///
/// Priority order (first match wins):
/// 1. Craft
/// 2. Attack → Use(0)
/// 3. Use → Interact
/// 4. Pickup → PickUp
/// 5. Movement (forward > back > left > right)
/// 6. Noop
#[derive(Debug, Clone, Default)]
pub struct MinerlActionMapper;

impl MinerlActionMapper {
    /// Converts a [`MinerlAction`] to a FORGE [`Action`].
    ///
    /// Returns [`Action::Noop`] for actions that cannot be mapped and emits a
    /// `tracing::warn` with the unmapped action name.
    pub fn map(&self, action: &MinerlAction) -> Action {
        if action.no_op {
            return Action::Noop;
        }
        if let Some(ref craft_name) = action.craft {
            if let Some(recipe_idx) = craft_name_to_recipe(craft_name) {
                return Action::Craft(recipe_idx);
            } else {
                warn!(craft = %craft_name, "MineRL craft action has no FORGE equivalent; mapping to Noop");
                return Action::Noop;
            }
        }
        if action.attack != 0 {
            return Action::Use(0); // slot 0 = weapon
        }
        if action.use_ != 0 {
            return Action::Interact;
        }
        if action.pickup != 0 {
            return Action::PickUp;
        }
        if action.forward != 0 {
            return Action::Move(Direction::Up);
        }
        if action.back != 0 {
            return Action::Move(Direction::Down);
        }
        if action.left != 0 {
            return Action::Move(Direction::Left);
        }
        if action.right != 0 {
            return Action::Move(Direction::Right);
        }
        Action::Noop
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Loads MineRL JSONL episode exports into FORGE [`OfflineDataset`].
#[derive(Debug, Clone, Default)]
pub struct MinerlLoader {
    /// Maximum steps per episode (0 = unlimited).
    pub max_steps_per_episode: u64,
    /// Maximum episodes to load (0 = unlimited).
    pub max_episodes: usize,
}

impl MinerlLoader {
    /// Creates a loader with optional limits.
    #[instrument]
    pub fn new(max_steps_per_episode: u64, max_episodes: usize) -> Self {
        Self {
            max_steps_per_episode,
            max_episodes,
        }
    }

    fn make_obs(step: &MinerlStep) -> Observation {
        let health = step
            .obs
            .as_ref()
            .and_then(|o| o.health)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let position = step
            .obs
            .as_ref()
            .and_then(|o| o.position)
            .map(|p| (p[0] as u16, p[2] as u16)) // MineRL is (x, y, z); map x,z → FORGE (x,y)
            .unwrap_or((0, 0));

        crate::build_default_obs(
            position,
            health,
            1.0, // MineRL has no stamina concept
            crate::DEFAULT_VIEW_SIZE,
            1,
            vec![],
        )
    }
}

impl DatasetLoader for MinerlLoader {
    /// Loads a MineRL JSONL export from `path`.
    ///
    /// Lines where `terminated` or `truncated` is `true` end an episode;
    /// a new [`Trajectory`] starts on the next line.
    ///
    /// [`Trajectory`]: forge_replay::trajectory::Trajectory
    #[instrument(skip(self), fields(path))]
    fn load(&self, path: &str) -> Result<OfflineDataset, DatasetError> {
        use std::io::BufRead;
        let file = std::fs::File::open(path).map_err(DatasetError::from)?;
        let reader = std::io::BufReader::new(file);
        let mapper = MinerlActionMapper;

        let mut dataset = OfflineDataset::new("MineRL");
        dataset.metadata.source_url =
            Some("https://zenodo.org/records/12659939".to_string());
        dataset.metadata.license = Some("MIT-like (MineRL)".to_string());

        let mut builder = TrajectoryBuilder::new();
        let mut episode: u64 = 0;
        let mut cumulative_reward = 0.0f32;
        let mut step_in_ep: u64 = 0;

        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(DatasetError::from)?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let raw: MinerlStep = serde_json::from_str(line).map_err(|e| {
                DatasetError::Deserialize(format!("line {}: {e}", line_no + 1))
            })?;

            let forge_action = mapper.map(&raw.action);
            let action_id = forge_action.to_discrete();
            cumulative_reward += raw.reward;

            let obs = Self::make_obs(&raw);
            let response = AgentResponse::from_action(action_id);

            builder.record_step(
                step_in_ep,
                vec![obs],
                &[response],
                vec![raw.reward],
                raw.terminated,
                raw.truncated,
            );
            step_in_ep += 1;

            let episode_done = raw.terminated
                || raw.truncated
                || (self.max_steps_per_episode > 0
                    && step_in_ep >= self.max_steps_per_episode);

            if episode_done {
                let traj = std::mem::replace(&mut builder, TrajectoryBuilder::new())
                    .seed(episode)
                    .agent_names(vec!["minerl_agent".to_string()])
                    .scenario_id(format!("minerl_ep_{episode}"))
                    .build(vec![cumulative_reward]);

                dataset.push(traj);
                episode += 1;
                cumulative_reward = 0.0;
                step_in_ep = 0;

                if self.max_episodes > 0 && dataset.len() >= self.max_episodes {
                    break;
                }
            }
        }

        Ok(dataset)
    }

    fn source_name(&self) -> &str {
        "MineRL"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mapper() -> MinerlActionMapper {
        MinerlActionMapper
    }

    #[test]
    fn test_map_forward() {
        let a = MinerlAction { forward: 1, ..Default::default() };
        assert_eq!(mapper().map(&a), Action::Move(Direction::Up));
    }

    #[test]
    fn test_map_back() {
        let a = MinerlAction { back: 1, ..Default::default() };
        assert_eq!(mapper().map(&a), Action::Move(Direction::Down));
    }

    #[test]
    fn test_map_attack() {
        let a = MinerlAction { attack: 1, ..Default::default() };
        assert_eq!(mapper().map(&a), Action::Use(0));
    }

    #[test]
    fn test_map_use() {
        let a = MinerlAction { use_: 1, ..Default::default() };
        assert_eq!(mapper().map(&a), Action::Interact);
    }

    #[test]
    fn test_map_pickup() {
        let a = MinerlAction { pickup: 1, ..Default::default() };
        assert_eq!(mapper().map(&a), Action::PickUp);
    }

    #[test]
    fn test_map_craft_axe() {
        let a = MinerlAction {
            craft: Some("wooden_axe".to_string()),
            ..Default::default()
        };
        assert_eq!(mapper().map(&a), Action::Craft(0));
    }

    #[test]
    fn test_map_craft_pickaxe() {
        let a = MinerlAction {
            craft: Some("stone_pickaxe".to_string()),
            ..Default::default()
        };
        assert_eq!(mapper().map(&a), Action::Craft(1));
    }

    #[test]
    fn test_map_craft_unknown_noop() {
        let a = MinerlAction {
            craft: Some("enchanting_table".to_string()),
            ..Default::default()
        };
        assert_eq!(mapper().map(&a), Action::Noop);
    }

    #[test]
    fn test_map_noop_flag() {
        let a = MinerlAction { forward: 1, no_op: true, ..Default::default() };
        assert_eq!(mapper().map(&a), Action::Noop);
    }

    #[test]
    fn test_craft_map_completeness() {
        // Each entry in CRAFT_MAP should map to a valid recipe index (0–8)
        for (name, idx) in CRAFT_MAP {
            assert!(*idx <= 8, "Recipe index {idx} out of range for '{name}'");
        }
    }

    #[test]
    fn test_load_jsonl() {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            f,
            r#"{{"action":{{"forward":1}},"reward":0.5,"terminated":false,"truncated":false}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"action":{{"pickup":1}},"reward":1.0,"terminated":true,"truncated":false}}"#
        )
        .unwrap();

        let loader = MinerlLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.trajectories[0].len(), 2);
        assert_eq!(ds.metadata.source, "MineRL");
    }

    #[test]
    fn test_action_roundtrip_through_discrete() {
        // Verify that mapped actions can be converted to discrete IDs
        let mapper = MinerlActionMapper;
        let actions = [
            MinerlAction { forward: 1, ..Default::default() },
            MinerlAction { back: 1, ..Default::default() },
            MinerlAction { attack: 1, ..Default::default() },
            MinerlAction { craft: Some("axe".to_string()), ..Default::default() },
        ];
        for a in &actions {
            let forge = mapper.map(a);
            let _id = forge.to_discrete(); // must not panic
        }
    }

    #[test]
    fn test_source_name() {
        let loader = MinerlLoader::default();
        assert_eq!(loader.source_name(), "MineRL");
    }

    #[test]
    fn test_loader_new_limits() {
        let loader = MinerlLoader::new(100, 5);
        assert_eq!(loader.max_steps_per_episode, 100);
        assert_eq!(loader.max_episodes, 5);
    }

    #[test]
    fn test_load_empty_file() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let loader = MinerlLoader::default();
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn test_load_missing_file_returns_error() {
        let loader = MinerlLoader::default();
        let result = loader.load("/nonexistent/path/file.jsonl");
        assert!(result.is_err());
    }

    #[test]
    fn test_map_left() {
        let a = MinerlAction { left: 1, ..Default::default() };
        assert_eq!(mapper().map(&a), Action::Move(Direction::Left));
    }

    #[test]
    fn test_map_right() {
        let a = MinerlAction { right: 1, ..Default::default() };
        assert_eq!(mapper().map(&a), Action::Move(Direction::Right));
    }

    #[test]
    fn test_all_craft_map_entries_via_mapper() {
        let m = MinerlActionMapper;
        for (name, expected_idx) in CRAFT_MAP {
            let a = MinerlAction {
                craft: Some(name.to_string()),
                ..Default::default()
            };
            assert_eq!(m.map(&a), Action::Craft(*expected_idx),
                "CRAFT_MAP entry '{}' → expected Craft({})", name, expected_idx);
        }
    }

    #[test]
    fn test_max_episodes_limit() {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for _ in 0..3 {
            writeln!(f, r#"{{"action":{{"forward":1}},"reward":0.0,"terminated":false}}"#).unwrap();
            writeln!(f, r#"{{"action":{{}},"reward":1.0,"terminated":true}}"#).unwrap();
        }
        // new(max_steps_per_episode, max_episodes): limit to 2 episodes
        let loader = MinerlLoader::new(0, 2);
        let ds = loader.load(f.path().to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 2);
    }
}
