//! Observation types for agent perception of the world.
//!
//! Observations are structured tensors designed for efficient GPU batching.
//! The default mode is symbolic (compact numerical arrays), with optional
//! pixel rendering for vision-based research.

use serde::{Deserialize, Serialize};

use crate::agriculture::{CropScanResult, SoilReading};
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InventoryObservation {
    /// Per-slot: (item_type as u8, count). Empty slots use (`OBS_EMPTY_SLOT_ITEM`, 0).
    pub slots: Vec<(u8, u16)>,
}

/// Complete observation for a single agent.
///
/// `Default` produces an empty observation with zero-valued scalar fields and
/// empty `Vec`s. The default is used as a placeholder slot inside reusable
/// `StepResult` buffers (see [`crate::observation::StepResult`]) — agents are
/// observed by mutating the slot in place via `clear` + `extend` on the inner
/// `Vec`s, so the heap capacity allocated on the first `step` is reused across
/// every subsequent step on the hot path.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    ///
    /// On hex grids, 6-direction movement is projected to the nearest
    /// cardinal heading before observation export.
    #[serde(default)]
    pub heading: u8,
    /// Latest crop scan results (NDVI/thermal readings). Empty when agri disabled.
    #[serde(default)]
    pub crop_scan_results: Vec<CropScanResult>,
    /// Latest soil relay data. Empty when agri disabled.
    #[serde(default)]
    pub soil_readings: Vec<SoilReading>,
    /// Count of diseased tiles detected by this agent.
    #[serde(default)]
    pub disease_detections: u16,
    /// Whether a field report was generated this tick.
    #[serde(default)]
    pub report_ready: bool,
}

impl Observation {
    /// Copy `src` into `self`, reusing inner `Vec` allocations.
    ///
    /// `#[derive(Clone)]` implements `clone_from` as `*self = src.clone()`,
    /// which allocates a fresh `Vec` for every field on every call. This
    /// method uses [`Vec::clone_from`] so a warm [`StepResult`] /
    /// `Env::step_into` output buffer stays allocation-free after warmup.
    /// Destructuring `src` makes a new [`Observation`] field a compile error
    /// until it is copied here.
    pub fn copy_from(&mut self, src: &Self) {
        let Self {
            grid_view,
            view_width,
            view_height,
            inventory,
            health,
            stamina,
            position,
            messages,
            day_phase,
            task_progress,
            altitude,
            battery,
            morphology,
            heading,
            crop_scan_results,
            soil_readings,
            disease_detections,
            report_ready,
        } = src;
        let InventoryObservation { slots } = inventory;
        self.grid_view.clone_from(grid_view);
        self.view_width = *view_width;
        self.view_height = *view_height;
        self.inventory.slots.clone_from(slots);
        self.health = *health;
        self.stamina = *stamina;
        self.position = *position;
        self.messages.clone_from(messages);
        self.day_phase = *day_phase;
        self.task_progress.clone_from(task_progress);
        self.altitude = *altitude;
        self.battery = *battery;
        self.morphology = *morphology;
        self.heading = *heading;
        self.crop_scan_results.clone_from(crop_scan_results);
        self.soil_readings.clone_from(soil_readings);
        self.disease_detections = *disease_detections;
        self.report_ready = *report_ready;
    }

    /// Returns the total number of elements when flattened to a 1D array.
    /// Useful for defining observation space dimensions.
    pub fn flat_size(
        vision_radius: u8,
        carry_capacity: u8,
        comm_buffer_size: u8,
        num_predicates: u16,
        drone_enabled: bool,
    ) -> usize {
        Self::flat_size_full(
            vision_radius,
            carry_capacity,
            comm_buffer_size,
            num_predicates,
            drone_enabled,
            false,
        )
    }

    /// Returns the total number of elements when flattened to a 1D array,
    /// including agricultural observation fields when enabled.
    pub fn flat_size_full(
        vision_radius: u8,
        carry_capacity: u8,
        comm_buffer_size: u8,
        num_predicates: u16,
        drone_enabled: bool,
        agri_enabled: bool,
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
            + if agri_enabled {
                constants::OBS_AGRI_FIELDS_COUNT
            } else {
                0
            }
    }
}

/// Per-step results for *all* agents in a `WorldState` (one entry per
/// agent in `observations`, `rewards`, and `info.agents_alive`).
///
/// `Default` produces an empty result that is intended to be filled in place
/// by [`crate::WorldState::step_into`] (or the equivalent buffer-reusing API
/// in your binding). Reusing a `StepResult` across steps is what makes the
/// "zero allocation on hot path" contract hold — see
/// `crates/forge-bench/src/bin/allocation_audit.rs` for the audit that
/// gates the contract in CI.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

impl StepInfo {
    /// Copy `src` into `self`, reusing inner `Vec` allocations.
    ///
    /// See [`Observation::copy_from`]: derived `Clone::clone_from` reallocates
    /// every `Vec` on every call, which breaks the `Env::step_into` zero-alloc
    /// contract.
    pub fn copy_from(&mut self, src: &Self) {
        let Self {
            tick,
            agents_alive,
            tasks_completed,
            total_resources,
            day_phase,
        } = src;
        self.tick = *tick;
        self.agents_alive.clone_from(agents_alive);
        self.tasks_completed.clone_from(tasks_completed);
        self.total_resources = *total_resources;
        self.day_phase = *day_phase;
    }
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
        Self::new_full(comm_vocab_size, drone_actions_enabled, false, false)
    }

    /// Creates an action space with drone, agricultural, and hex action support.
    pub fn new_full(
        comm_vocab_size: u16,
        drone_actions_enabled: bool,
        agri_actions_enabled: bool,
        hex_actions_enabled: bool,
    ) -> Self {
        let n = crate::action::Action::space_size_full(
            comm_vocab_size,
            drone_actions_enabled,
            agri_actions_enabled,
            hex_actions_enabled,
        );
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
        if agri_actions_enabled && drone_actions_enabled {
            for i in 0..10 {
                names.push(format!("Spray Slot {}", i));
            }
            names.push("Scan Multispectral".to_string());
            names.push("Scan Thermal".to_string());
            names.push("Relay Soil Data".to_string());
            names.push("Generate Report".to_string());
        }
        if hex_actions_enabled {
            names.push("Move NE".to_string());
            names.push("Move E".to_string());
            names.push("Move SE".to_string());
            names.push("Move SW".to_string());
            names.push("Move W".to_string());
            names.push("Move NW".to_string());
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
    use crate::agriculture::{CropScanResult, SoilReading};

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
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
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
    fn test_observation_copy_from_reuses_vec_capacity() {
        let src = Observation {
            grid_view: vec![TileObservation::default(); 9],
            view_width: 3,
            view_height: 3,
            inventory: InventoryObservation {
                slots: vec![(1, 2), (3, 4)],
            },
            health: 0.5,
            stamina: 0.25,
            position: (1, 2),
            messages: vec![1],
            day_phase: 2,
            task_progress: vec![0.1, 0.2],
            altitude: 3,
            battery: 0.75,
            morphology: 1,
            heading: 2,
            crop_scan_results: vec![CropScanResult {
                position: (0, 1),
                ndvi: 0.4,
                thermal: 0.1,
                disease_flag: false,
            }],
            soil_readings: vec![SoilReading {
                node_id: 7,
                npk: [0.1, 0.2, 0.3],
                ph: 6.5,
                moisture: 0.4,
            }],
            disease_detections: 4,
            report_ready: true,
        };
        let mut dst = Observation::default();
        dst.copy_from(&src);
        let grid_cap = dst.grid_view.capacity();
        let slot_cap = dst.inventory.slots.capacity();
        let msg_cap = dst.messages.capacity();
        let task_cap = dst.task_progress.capacity();
        dst.copy_from(&src);
        assert_eq!(dst.grid_view.capacity(), grid_cap);
        assert_eq!(dst.inventory.slots.capacity(), slot_cap);
        assert_eq!(dst.messages.capacity(), msg_cap);
        assert_eq!(dst.task_progress.capacity(), task_cap);
        assert_eq!(dst.view_width, 3);
        assert_eq!(dst.health, 0.5);
        assert_eq!(dst.position, (1, 2));
        assert!(dst.report_ready);
        assert_eq!(dst.crop_scan_results.len(), 1);
        assert_eq!(dst.soil_readings[0].node_id, 7);
    }

    #[test]
    fn test_step_info_copy_from_reuses_vec_capacity() {
        let src = StepInfo {
            tick: 9,
            agents_alive: vec![true, false],
            tasks_completed: vec![vec![1, 2], vec![]],
            total_resources: 12,
            day_phase: 1,
        };
        let mut dst = StepInfo::default();
        dst.copy_from(&src);
        let alive_cap = dst.agents_alive.capacity();
        let tasks_cap = dst.tasks_completed.capacity();
        dst.copy_from(&src);
        assert_eq!(dst.agents_alive.capacity(), alive_cap);
        assert_eq!(dst.tasks_completed.capacity(), tasks_cap);
        assert_eq!(dst.tick, 9);
        assert_eq!(dst.total_resources, 12);
        assert_eq!(dst.tasks_completed[0], vec![1, 2]);
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
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
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
    fn test_observation_flat_size_with_agri() {
        let size_no_agri = Observation::flat_size_full(5, 10, 8, 4, true, false);
        let size_agri = Observation::flat_size_full(5, 10, 8, 4, true, true);
        assert_eq!(
            size_agri - size_no_agri,
            constants::OBS_AGRI_FIELDS_COUNT,
            "agri adds extra observation features"
        );
    }

    #[test]
    fn test_observation_agri_fields_default_via_serde() {
        // Deserialize without agri fields (backwards compat)
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
        assert!(
            obs.crop_scan_results.is_empty(),
            "crop_scan_results default should be empty"
        );
        assert!(
            obs.soil_readings.is_empty(),
            "soil_readings default should be empty"
        );
        assert_eq!(
            obs.disease_detections, 0,
            "disease_detections default should be 0"
        );
        assert!(!obs.report_ready, "report_ready default should be false");
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

    // ──────────────────────────────────────────────────────────────────
    // ActionSpace::new_full coverage — exercises the drone/agri/hex
    // branches that the default `new()` path skips.
    // ──────────────────────────────────────────────────────────────────

    /// Number of fixed drone action names appended in `new_full` (excluding
    /// the 10 DropPayload slots). Kept in sync with `observation.rs:285-293`.
    const DRONE_FIXED_ACTION_COUNT: usize = 9;
    /// Number of payload-slot drone actions.
    const DRONE_PAYLOAD_SLOTS: usize = 10;
    /// Total drone action surface added by `new_full`.
    const DRONE_TOTAL: usize = DRONE_FIXED_ACTION_COUNT + DRONE_PAYLOAD_SLOTS;
    /// Number of fixed agri action names (Multispectral/Thermal/Relay/Report).
    const AGRI_FIXED_ACTION_COUNT: usize = 4;
    /// Number of agri spray-slot actions.
    const AGRI_SPRAY_SLOTS: usize = 10;
    /// Total agri action surface.
    const AGRI_TOTAL: usize = AGRI_FIXED_ACTION_COUNT + AGRI_SPRAY_SLOTS;
    /// Number of hex movement action names appended.
    const HEX_ACTION_COUNT: usize = 6;

    #[test]
    fn test_action_space_new_full_drone_extends_names() {
        let comm = 0u16;
        let base = ActionSpace::new(comm, false);
        let with_drone = ActionSpace::new_full(comm, true, false, false);
        assert_eq!(
            with_drone.action_names.len(),
            base.action_names.len() + DRONE_TOTAL,
            "drone flag must append exactly {DRONE_TOTAL} names"
        );
        // Spot-check a few of the names introduced by the drone branch.
        for label in [
            "Ascend",
            "Descend",
            "Hover",
            "TakeOff",
            "Land",
            "Scan Up",
            "Scan Down",
            "Scan Left",
            "Scan Right",
        ] {
            assert!(
                with_drone.action_names.iter().any(|n| n == label),
                "missing drone label: {label}"
            );
        }
        for i in 0..DRONE_PAYLOAD_SLOTS {
            let expected = format!("DropPayload Slot {i}");
            assert!(
                with_drone.action_names.iter().any(|n| n == &expected),
                "missing {expected}"
            );
        }
    }

    #[test]
    fn test_action_space_new_full_agri_requires_drone() {
        let comm = 0u16;
        // Agri without drone: branch is gated and adds nothing.
        let only_agri = ActionSpace::new_full(comm, false, true, false);
        let base = ActionSpace::new(comm, false);
        assert_eq!(
            only_agri.action_names.len(),
            base.action_names.len(),
            "agri-only must not add names (drone is required)"
        );

        // Drone + agri: adds drone + agri name sets.
        let drone_agri = ActionSpace::new_full(comm, true, true, false);
        assert_eq!(
            drone_agri.action_names.len(),
            base.action_names.len() + DRONE_TOTAL + AGRI_TOTAL
        );
        for label in [
            "Scan Multispectral",
            "Scan Thermal",
            "Relay Soil Data",
            "Generate Report",
        ] {
            assert!(
                drone_agri.action_names.iter().any(|n| n == label),
                "missing agri label: {label}"
            );
        }
        for i in 0..AGRI_SPRAY_SLOTS {
            let expected = format!("Spray Slot {i}");
            assert!(
                drone_agri.action_names.iter().any(|n| n == &expected),
                "missing {expected}"
            );
        }
    }

    #[test]
    fn test_action_space_new_full_hex_appends_six_directions() {
        let comm = 0u16;
        let base = ActionSpace::new(comm, false);
        let hex = ActionSpace::new_full(comm, false, false, true);
        assert_eq!(
            hex.action_names.len(),
            base.action_names.len() + HEX_ACTION_COUNT,
            "hex flag must append exactly {HEX_ACTION_COUNT} direction names"
        );
        for label in [
            "Move NE", "Move E", "Move SE", "Move SW", "Move W", "Move NW",
        ] {
            assert!(
                hex.action_names.iter().any(|n| n == label),
                "missing hex label: {label}"
            );
        }
    }

    #[test]
    fn test_action_space_new_full_all_flags_compose() {
        let comm = 4u16;
        let full = ActionSpace::new_full(comm, true, true, true);
        // Drone + agri + hex labels should all be present.
        for label in [
            "Ascend",
            "DropPayload Slot 0",
            "Spray Slot 0",
            "Scan Multispectral",
            "Move NE",
            "Move NW",
        ] {
            assert!(
                full.action_names.iter().any(|n| n == label),
                "missing label under full flags: {label}"
            );
        }
        // n must match action_names length.
        assert_eq!(full.n as usize, full.action_names.len());
    }

    #[test]
    fn test_action_space_new_full_n_matches_space_size_full() {
        let comm = 3u16;
        for &drone in &[false, true] {
            for &agri in &[false, true] {
                for &hex in &[false, true] {
                    let space = ActionSpace::new_full(comm, drone, agri, hex);
                    let expected = crate::action::Action::space_size_full(comm, drone, agri, hex);
                    assert_eq!(
                        space.n, expected,
                        "n mismatch for drone={drone} agri={agri} hex={hex}"
                    );
                }
            }
        }
    }
}
