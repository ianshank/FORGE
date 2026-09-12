//! World state definition and initialization.

use std::collections::HashMap;
use std::sync::Arc;

use forge_civ::grid_topology::GridTopologyKind;
use forge_civ::{HexTopology, SquareTopology};
use forge_types::agriculture::{AgriScratch, CropState, SoilSensorNode};
use forge_types::config::{ForgeConfig, GridType};
use forge_types::entity::{Agent, Object};
use forge_types::grid::{Grid, Position};
use forge_types::resource::{RecipeBook, ResourceNode};
use forge_types::task::ActiveTask;
use forge_types::Action;
use forge_worldgen::WorldGenerator;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use tracing::{info, instrument};

use crate::physics::PhysicsScratch;
use crate::rng::ForgeRng;

/// The complete state of a FORGE simulation instance.
#[derive(Debug, Clone)]
pub struct WorldState {
    /// Current simulation tick.
    pub tick: u64,
    /// The 2D tile grid.
    pub grid: Grid,
    /// All agents in the simulation.
    pub agents: Vec<Agent>,
    /// All interactive objects.
    pub objects: Vec<Object>,
    /// All resource nodes.
    pub resources: Vec<ResourceNode>,
    /// Active tasks being tracked.
    pub tasks: Vec<ActiveTask>,
    /// The recipe book for crafting.
    pub recipe_book: RecipeBook,
    /// Day/night cycle phase (0-3: dawn, day, dusk, night).
    pub day_phase: u8,
    /// Deterministic RNG.
    pub rng: ForgeRng,
    /// Shared immutable configuration.
    pub config: Arc<ForgeConfig>,
    /// Whether the simulation has terminated.
    pub terminated: bool,
    /// Whether the simulation was truncated (max steps).
    pub truncated: bool,
    /// Rewards computed by the task evaluator for the most recent step.
    /// Consumed by `make_step_result()` and reset each tick.
    pub last_task_rewards: Option<Vec<f32>>,
    /// Pre-allocated scratch buffers for the physics system, avoiding
    /// per-tick heap allocations on the hot path.
    pub(crate) physics_scratch: PhysicsScratch,
    /// Grid topology dispatcher (square or hex). Used by all spatial systems.
    pub topology: GridTopologyKind,
    /// Per-tile crop state, parallel to `grid.tiles`. Only populated when `agri.enabled`.
    pub crop_states: Vec<CropState>,
    /// Ground-deployed IoT soil sensor nodes. Only populated when `agri.enabled`.
    pub soil_nodes: Vec<SoilSensorNode>,
    /// Pre-allocated scratch buffers for agricultural systems.
    pub(crate) agri_scratch: AgriScratch,
    /// Reusable padded actions buffer. Sized to `agents.len()` at the top of
    /// every step; replaces the per-step `actions.to_vec()` allocation.
    pub(crate) step_actions: Vec<Action>,
    /// Reusable validated actions buffer. Filled by `validate_actions` from
    /// the padded `step_actions` and consumed by every system below.
    pub(crate) validated_actions: Vec<Action>,
    /// Reusable per-agent flags from `compute_near_station`.
    pub(crate) near_station: Vec<bool>,
    /// Reusable object-id-to-type lookup for `compute_near_station`. Only
    /// holds entries for the duration of one tick; cleared at the top.
    pub(crate) crafting_object_map: HashMap<u32, forge_types::entity::ObjectType>,
    /// Reusable comm message queue for `process_communication`.
    pub(crate) comm_messages: Vec<(usize, u16)>,
    /// Reusable per-agent push snapshot for `physics::process_pushes`.
    /// Replaces the `SmallVec<[_; 8]>` that previously lived in
    /// `run_systems` and would heap-allocate when `agents.len() > 8`.
    pub(crate) push_scratch: Vec<crate::physics::AgentPushData>,
    /// Reusable discrete action ids for this tick's `validated_actions`.
    /// Filled only when `tasks` is non-empty (already an alloc-audit carve-out).
    pub(crate) task_action_ids: Vec<u32>,
}

impl WorldState {
    /// Creates a new world state from configuration.
    ///
    /// Validates the config and initializes the grid, spawns agents at
    /// default positions, and sets up the RNG.
    ///
    /// Returns [`ForgeError::Config`] if the configuration is invalid.
    ///
    /// # Breaking Change (v0.2.0)
    ///
    /// This method now returns `ForgeResult<WorldState>` instead of `WorldState`.
    /// Callers must handle the `Result` (e.g., `.unwrap()` or `?`).
    #[instrument(skip_all)]
    pub fn new(config: ForgeConfig) -> forge_types::ForgeResult<Self> {
        forge_types::validation::validate_config(&config)?;
        let config = Arc::new(config);
        let mut rng = ForgeRng::new(config.world.seed);

        // Use WorldGenerator for terrain, resources, and objects
        let generator = WorldGenerator::new(&config.world);
        let mut gen_rng = Pcg64Mcg::seed_from_u64(config.world.seed);
        let (grid, resources, objects, spawn_points) = generator.generate(&mut gen_rng);

        // Spawn agents at generated spawn points or deterministic fallback positions
        let mut agents = Vec::with_capacity(config.agents.num_agents as usize);
        for i in 0..config.agents.num_agents {
            let pos = if (i as usize) < spawn_points.len() {
                spawn_points[i as usize]
            } else {
                let x = rng.next_range(config.world.width as u32) as u16;
                let y = rng.next_range(config.world.height as u32) as u16;
                Position::new(x, y)
            };
            let mut agent = Agent::new(i, pos, &config.agents);

            // Assign morphology based on drone config
            if config.drone.enabled {
                let morphology = if i < config.drone.num_aerial {
                    forge_types::entity::AgentMorphology::Aerial
                } else if i < config.drone.num_aerial + config.drone.num_ground_vehicles {
                    forge_types::entity::AgentMorphology::GroundVehicle
                } else {
                    forge_types::entity::AgentMorphology::Ground
                };
                agent.morphology = morphology;
                match morphology {
                    forge_types::entity::AgentMorphology::Aerial => {
                        agent.capabilities.can_fly = true;
                        agent.capabilities.max_altitude = config.drone.max_altitude;
                        agent.battery = config.drone.starting_battery;
                        if let Some(home) = config.drone.spawn_home {
                            agent.position = home;
                        }
                    }
                    forge_types::entity::AgentMorphology::GroundVehicle => {
                        agent.capabilities.turn_radius = config.drone.vehicle_turn_radius;
                    }
                    forge_types::entity::AgentMorphology::Ground | _ => {}
                }
            }

            agents.push(agent);
        }

        let mut physics_scratch = PhysicsScratch::default();
        physics_scratch.ensure_capacity(agents.len());

        // Select grid topology based on configuration
        let topology = match config.world.grid_type {
            GridType::Square => GridTopologyKind::Square(SquareTopology),
            GridType::Hex => GridTopologyKind::Hex(HexTopology),
        };

        // Initialize agricultural state when enabled
        let (crop_states, soil_nodes, agri_scratch) = if config.agri.enabled {
            let crops = forge_worldgen::agriculture::generate_crop_states(&grid, &config.agri);
            let nodes = forge_worldgen::agriculture::spawn_soil_nodes(
                &grid,
                &config.agri,
                &mut Pcg64Mcg::seed_from_u64(config.world.seed.wrapping_add(0xA681)),
            );
            let scratch = AgriScratch::with_capacity(
                (config.agri.ndvi_scan_radius as usize * 2 + 1).pow(2),
                config.agri.num_soil_nodes as usize,
            );
            (crops, nodes, scratch)
        } else {
            (Vec::new(), Vec::new(), AgriScratch::default())
        };

        let agent_count = agents.len();
        let tasks = if config.task.enabled {
            config
                .task
                .scenario_tasks
                .iter()
                .cloned()
                .map(ActiveTask::from_definition)
                .collect()
        } else {
            Vec::new()
        };
        let mut state = Self {
            tick: 0,
            grid,
            agents,
            objects,
            resources,
            tasks,
            recipe_book: RecipeBook::default(),
            day_phase: 0,
            rng,
            config,
            terminated: false,
            truncated: false,
            last_task_rewards: None,
            physics_scratch,
            topology,
            crop_states,
            soil_nodes,
            agri_scratch,
            step_actions: Vec::with_capacity(agent_count),
            validated_actions: Vec::with_capacity(agent_count),
            near_station: Vec::with_capacity(agent_count),
            crafting_object_map: HashMap::new(),
            comm_messages: Vec::with_capacity(agent_count),
            push_scratch: Vec::with_capacity(agent_count),
            task_action_ids: Vec::with_capacity(agent_count),
        };

        // Place agents on the grid
        state.sync_agents_to_grid();

        info!(
            seed = state.config.world.seed,
            width = state.config.world.width,
            height = state.config.world.height,
            num_agents = state.config.agents.num_agents,
            resources = state.resources.len(),
            objects = state.objects.len(),
            "world created"
        );

        Ok(state)
    }

    /// Synchronizes agent positions to the grid tiles.
    pub(crate) fn sync_agents_to_grid(&mut self) {
        // Clear all agent references from grid
        for tile in self.grid.tiles.iter_mut() {
            tile.agent_id = None;
        }
        // Set agent positions
        for agent in &self.agents {
            if agent.alive {
                if let Some(tile) = self.grid.get_mut(agent.position.x, agent.position.y) {
                    tile.agent_id = Some(agent.id);
                }
            }
        }
    }
}
