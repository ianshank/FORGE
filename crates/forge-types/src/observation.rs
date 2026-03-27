//! Observation types for agent perception of the world.
//!
//! Observations are structured tensors designed for efficient GPU batching.
//! The default mode is symbolic (compact numerical arrays), with optional
//! pixel rendering for vision-based research.

use serde::{Deserialize, Serialize};

use crate::constants;
use crate::entity::CommToken;
use crate::grid::TerrainType;

/// Default battery observation value for serde (fully charged).
fn default_battery_obs() -> f32 {
    1.0
}

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
    /// Object type if present (encoded as u8), or `OBS_NO_OBJECT` sentinel for none.
    pub object_type: u8,
    /// Resource type if present (encoded as u8), or `OBS_NO_RESOURCE` sentinel for none.
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
            object_type: constants::OBS_NO_OBJECT,
            resource_type: constants::OBS_NO_RESOURCE,
        }
    }
}

/// Inventory observation — what the agent carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryObservation {
    /// Per-slot: (item_type as u8, count). Empty slots use (`OBS_EMPTY_SLOT_ITEM`, 0).
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
    /// Agent's current altitude (0 = ground level).
    #[serde(default)]
    pub altitude: u8,
    /// Agent's battery level, normalized 0.0-1.0. Defaults to 1.0 when drone disabled.
    #[serde(default = "default_battery_obs")]
    pub battery: f32,
    /// Agent's morphology type (0=Ground, 1=GroundVehicle, 2=Aerial).
    #[serde(default)]
    pub morphology: u8,
    /// Agent's current heading direction (0=Up, 1=Down, 2=Left, 3=Right).
    #[serde(default)]
    pub heading: u8,
}

impl Observation {
    /// Returns the total number of elements when flattened to a 1D array.
    /// Useful for defining observation space dimensions.
    pub fn flat_size(
        vision_radius: u8,
        carry_capacity: u8,
        comm_buffer_size: u8,
        num_predicates: u16,
        drone_enabled: bool,
    ) -> usize {
        let view_side = 2 * vision_radius as usize + 1;
        let grid_elements = view_side * view_side * constants::OBS_FEATURES_PER_TILE;
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
            + if drone_enabled {
                constants::OBS_DRONE_FIELDS_COUNT
            } else {
                0
            }
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
    pub fn new(comm_vocab_size: u16, drone_actions_enabled: bool) -> Self {
        let n = crate::action::Action::space_size(comm_vocab_size, drone_actions_enabled);
        let mut names = vec![
            "Noop".to_string(),
            "Move Up".to_string(),
            "Move Down".to_string(),
            "Move Left".to_string(),
            "Move Right".to_string(),
            "Pick Up".to_string(),
        ];
        for i in 0..constants::ACTION_DROP_SLOTS {
            names.push(format!("Drop Slot {}", i));
        }
        for i in 0..constants::ACTION_USE_SLOTS {
            names.push(format!("Use Slot {}", i));
        }
        for i in 0..constants::ACTION_CRAFT_SLOTS {
            names.push(format!("Craft Recipe {}", i));
        }
        names.push("Push Up".to_string());
        names.push("Push Down".to_string());
        names.push("Push Left".to_string());
        names.push("Push Right".to_string());
        names.push("Interact".to_string());
        for i in 0..comm_vocab_size {
            names.push(format!("Communicate {}", i));
        }
        if drone_actions_enabled {
            names.push("Ascend".to_string());
            names.push("Descend".to_string());
            names.push("Hover".to_string());
            names.push("TakeOff".to_string());
            names.push("Land".to_string());
            names.push("Scan Up".to_string());
            names.push("Scan Down".to_string());
            names.push("Scan Left".to_string());
            names.push("Scan Right".to_string());
            for i in 0..10 {
                names.push(format!("DropPayload Slot {}", i));
            }
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
        let size = Observation::flat_size(5, 10, 8, 4, false);
        let view_side = 11; // 2*5+1
        let expected =
            view_side * view_side * constants::OBS_FEATURES_PER_TILE + 10 * 2 + 4 + 8 + 1 + 4;
        assert_eq!(size, expected);
    }

    #[test]
    fn test_action_space_creation() {
        let comm_vocab: u16 = 16;
        let expected_n = crate::action::Action::space_size(comm_vocab, false);
        let space = ActionSpace::new(comm_vocab, false);
        assert_eq!(space.n, expected_n);
        assert_eq!(space.action_names.len(), expected_n as usize);
        assert_eq!(space.action_names[0], "Noop");
        assert_eq!(space.action_names[1], "Move Up");
    }

    #[test]
    fn test_tile_observation_default() {
        let tile = TileObservation::default();
        assert_eq!(tile.terrain, 0); // Ground
        assert!(!tile.has_agent);
        assert_eq!(tile.object_type, constants::OBS_NO_OBJECT);
    }

    #[test]
    fn test_step_result_construction() {
        let obs = Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation {
                slots: vec![(constants::OBS_EMPTY_SLOT_ITEM, 0)],
            },
            health: 1.0,
            stamina: 0.8,
            position: (5, 10),
            messages: vec![],
            day_phase: 1,
            task_progress: vec![0.5],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
        };

        let step = StepResult {
            observations: vec![obs],
            rewards: vec![1.0],
            terminated: false,
            truncated: false,
            info: StepInfo::default(),
        };

        assert_eq!(step.observations.len(), 1);
        assert_eq!(step.rewards.len(), 1);
        assert!(!step.terminated);
        assert!(!step.truncated);
        assert_eq!(step.rewards[0], 1.0);
        assert_eq!(step.observations[0].position, (5, 10));
        assert_eq!(step.observations[0].health, 1.0);
        assert_eq!(step.observations[0].stamina, 0.8);
    }

    #[test]
    fn test_step_info_defaults() {
        let info = StepInfo::default();
        assert_eq!(info.tick, 0);
        assert!(info.agents_alive.is_empty());
        assert!(info.tasks_completed.is_empty());
        assert_eq!(info.total_resources, 0);
        assert_eq!(info.day_phase, 0);
    }

    #[test]
    fn test_observation_drone_fields_default() {
        let obs = Observation {
            grid_view: vec![],
            view_width: 0,
            view_height: 0,
            inventory: InventoryObservation { slots: vec![] },
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
        };
        assert_eq!(obs.altitude, 0);
        assert_eq!(obs.battery, 1.0);
        assert_eq!(obs.morphology, 0);
    }

    #[test]
    fn test_inventory_observation() {
        let inv_obs = InventoryObservation {
            slots: vec![(0, 5), (1, 3), (constants::OBS_EMPTY_SLOT_ITEM, 0)],
        };
        assert_eq!(inv_obs.slots.len(), 3);
        // First slot: item type 0 (Wood) with count 5.
        assert_eq!(inv_obs.slots[0], (0, 5));
        // Second slot: item type 1 (Stone) with count 3.
        assert_eq!(inv_obs.slots[1], (1, 3));
        // Third slot: empty (sentinel, count 0).
        assert_eq!(inv_obs.slots[2], (constants::OBS_EMPTY_SLOT_ITEM, 0));
    }

    #[test]
    fn test_observation_flat_size_with_drones() {
        let size_no_drone = Observation::flat_size(5, 10, 8, 4, false);
        let size_drone = Observation::flat_size(5, 10, 8, 4, true);
        assert_eq!(
            size_drone - size_no_drone,
            constants::OBS_DRONE_FIELDS_COUNT,
            "drone adds extra features (altitude, battery, morphology, heading)"
        );
    }

    #[test]
    fn test_observation_serde_with_drone_defaults() {
        // Deserialize an observation without drone fields (backwards compat)
        let json = r#"{
            "grid_view": [],
            "view_width": 0,
            "view_height": 0,
            "inventory": {"slots": []},
            "health": 1.0,
            "stamina": 1.0,
            "position": [0, 0],
            "messages": [],
            "day_phase": 0,
            "task_progress": []
        }"#;
        let obs: Observation = serde_json::from_str(json).unwrap();
        assert_eq!(obs.altitude, 0, "altitude default should be 0");
        assert_eq!(obs.battery, 1.0, "battery default should be 1.0");
        assert_eq!(obs.morphology, 0, "morphology default should be 0 (Ground)");
        assert_eq!(obs.heading, 0, "heading default should be 0 (Up)");
    }

    #[test]
    fn test_action_space_names_match_count() {
        let space = ActionSpace::new(0, false);
        assert_eq!(
            space.action_names.len() as u32,
            space.n,
            "action names count must match space size"
        );

        let space16 = ActionSpace::new(16, false);
        assert_eq!(space16.action_names.len() as u32, space16.n);
    }

    #[test]
    fn test_observation_space_construction() {
        let obs_space = ObservationSpace {
            flat_shape: vec![100],
            low: 0.0,
            high: 1.0,
            grid_shape: (11, 11, constants::OBS_FEATURES_PER_TILE),
            inventory_size: 10,
            comm_buffer_size: 8,
        };
        assert_eq!(obs_space.flat_shape[0], 100);
        assert_eq!(obs_space.grid_shape.2, constants::OBS_FEATURES_PER_TILE);
    }
}
