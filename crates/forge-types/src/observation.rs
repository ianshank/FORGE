//! Observation types for agent perception of the world.
//!
//! Observations are structured tensors designed for efficient GPU batching.
//! The default mode is symbolic (compact numerical arrays), with optional
//! pixel rendering for vision-based research.

use serde::{Deserialize, Serialize};

use crate::entity::CommToken;
use crate::grid::TerrainType;

/// A single tile as observed by an agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TileObservation {
    /// Terrain type (encoded as u8).
    pub terrain: u8,
    /// Whether an agent is present.
    pub has_agent: bool,
    /// Whether an object is present.
    pub has_object: bool,
    /// Whether a resource is present.
    pub has_resource: bool,
    /// Elevation value.
    pub elevation: u8,
    /// Object type if present (encoded as u8), or 255 for none.
    pub object_type: u8,
    /// Resource type if present (encoded as u8), or 255 for none.
    pub resource_type: u8,
}

impl Default for TileObservation {
    fn default() -> Self {
        Self {
            terrain: TerrainType::Ground as u8,
            elevation: 0,
            has_agent: false,
            has_object: false,
            has_resource: false,
            object_type: 255,
            resource_type: 255,
        }
    }
}

/// Inventory observation — what the agent carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryObservation {
    /// Per-slot: (item_type as u8, count). Empty slots use (255, 0).
    pub slots: Vec<(u8, u16)>,
}

/// Complete observation for a single agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Ego-centric grid view (partial, based on vision radius).
    /// Flattened: size = (2*vision_radius+1)^2 tiles.
    pub grid_view: Vec<TileObservation>,
    /// Width of the observation grid (2*vision_radius+1).
    pub view_width: u16,
    /// Height of the observation grid (2*vision_radius+1).
    pub view_height: u16,
    /// Agent's inventory state.
    pub inventory: InventoryObservation,
    /// Current health (normalized 0.0-1.0).
    pub health: f32,
    /// Current stamina (normalized 0.0-1.0).
    pub stamina: f32,
    /// Agent's absolute position.
    pub position: (u16, u16),
    /// Recent communication messages received.
    pub messages: Vec<CommToken>,
    /// Current day phase (0=dawn, 1=day, 2=dusk, 3=night).
    pub day_phase: u8,
    /// Per-task-predicate completion progress (0.0-1.0 each).
    pub task_progress: Vec<f32>,
}

impl Observation {
    /// Returns the total number of elements when flattened to a 1D array.
    /// Useful for defining observation space dimensions.
    pub fn flat_size(
        vision_radius: u8,
        carry_capacity: u8,
        comm_buffer_size: u8,
        num_predicates: u16,
    ) -> usize {
        let view_side = 2 * vision_radius as usize + 1;
        let grid_elements = view_side * view_side * 7; // 7 features per tile
        let inventory_elements = carry_capacity as usize * 2; // (type, count) per slot
        let scalar_elements = 4; // health, stamina, position.x, position.y
        let comm_elements = comm_buffer_size as usize;
        let day_elements = 1;
        let task_elements = num_predicates as usize;

        grid_elements
            + inventory_elements
            + scalar_elements
            + comm_elements
            + day_elements
            + task_elements
    }
}

/// The result of a single simulation step for one agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Per-agent observations.
    pub observations: Vec<Observation>,
    /// Per-agent rewards.
    pub rewards: Vec<f32>,
    /// Whether the episode has terminated (natural end: goal reached or agent died).
    pub terminated: bool,
    /// Whether the episode was truncated (artificial end: max steps reached).
    pub truncated: bool,
    /// Additional diagnostic information.
    pub info: StepInfo,
}

/// Diagnostic information from a simulation step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepInfo {
    /// Current simulation tick.
    pub tick: u64,
    /// Per-agent: whether each agent is alive.
    pub agents_alive: Vec<bool>,
    /// Per-agent: tasks completed this step.
    pub tasks_completed: Vec<Vec<u16>>,
    /// Total resources remaining in the world.
    pub total_resources: u32,
    /// Current day phase.
    pub day_phase: u8,
}

/// Description of the observation space for API consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSpace {
    /// Shape of the flattened observation vector.
    pub flat_shape: Vec<usize>,
    /// Low bound for each element.
    pub low: f32,
    /// High bound for each element.
    pub high: f32,
    /// Grid view shape: (height, width, features).
    pub grid_shape: (usize, usize, usize),
    /// Number of inventory slots.
    pub inventory_size: usize,
    /// Communication buffer size.
    pub comm_buffer_size: usize,
}

/// Description of the action space for API consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpace {
    /// Total number of discrete actions.
    pub n: u32,
    /// Human-readable action names indexed by action ID.
    pub action_names: Vec<String>,
}

impl ActionSpace {
    /// Creates an action space with the given communication vocabulary size.
    pub fn new(comm_vocab_size: u16) -> Self {
        let n = crate::action::Action::space_size(comm_vocab_size);
        let mut names = vec![
            "Noop".to_string(),
            "Move Up".to_string(),
            "Move Down".to_string(),
            "Move Left".to_string(),
            "Move Right".to_string(),
            "Pick Up".to_string(),
        ];
        for i in 0..10 {
            names.push(format!("Drop Slot {}", i));
        }
        for i in 0..10 {
            names.push(format!("Use Slot {}", i));
        }
        names.push("Craft".to_string());
        names.push("Push Up".to_string());
        names.push("Push Down".to_string());
        names.push("Push Left".to_string());
        names.push("Push Right".to_string());
        names.push("Interact".to_string());
        for i in 0..comm_vocab_size {
            names.push(format!("Communicate {}", i));
        }
        Self {
            n,
            action_names: names,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_flat_size() {
        let size = Observation::flat_size(5, 10, 8, 4);
        let view_side = 11; // 2*5+1
        let expected = view_side * view_side * 7 + 10 * 2 + 4 + 8 + 1 + 4;
        assert_eq!(size, expected);
    }

    #[test]
    fn test_action_space_creation() {
        let space = ActionSpace::new(16);
        assert_eq!(space.n, 48); // 32 base + 16 comm
        assert_eq!(space.action_names.len(), 48);
        assert_eq!(space.action_names[0], "Noop");
        assert_eq!(space.action_names[1], "Move Up");
    }

    #[test]
    fn test_tile_observation_default() {
        let tile = TileObservation::default();
        assert_eq!(tile.terrain, 0); // Ground
        assert!(!tile.has_agent);
        assert_eq!(tile.object_type, 255);
    }
}
