//! World state and the main simulation entry point.
//!
//! `WorldState` holds all simulation data and provides the `step()` and
//! `reset()` methods that form the core API.

use std::sync::Arc;

use forge_types::agriculture::{AgriScratch, CropState, SoilSensorNode};
use forge_types::config::ForgeConfig;
use forge_types::constants::{OBS_EMPTY_SLOT_ITEM, OBS_NO_OBJECT, OBS_NO_RESOURCE};
use forge_types::entity::{Agent, Object};
use forge_types::grid::{Grid, Position};
use forge_types::observation::{
    InventoryObservation, Observation, StepInfo, StepResult, TileObservation,
};
use forge_types::resource::{RecipeBook, ResourceNode};
use forge_types::task::ActiveTask;
use forge_types::Action;
use forge_worldgen::WorldGenerator;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use serde::Serialize;
use tracing::{info, instrument, trace};

use crate::physics::PhysicsScratch;
use crate::rng::ForgeRng;
use crate::systems;

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
    /// Per-tile crop state, parallel to `grid.tiles`. Only populated when `agri.enabled`.
    pub crop_states: Vec<CropState>,
    /// Ground-deployed IoT soil sensor nodes. Only populated when `agri.enabled`.
    pub soil_nodes: Vec<SoilSensorNode>,
    /// Pre-allocated scratch buffers for agricultural systems.
    pub(crate) agri_scratch: AgriScratch,
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

        let mut state = Self {
            tick: 0,
            grid,
            agents,
            objects,
            resources,
            tasks: Vec::new(),
            recipe_book: RecipeBook::default(),
            day_phase: 0,
            rng,
            config,
            terminated: false,
            truncated: false,
            last_task_rewards: None,
            physics_scratch,
            crop_states,
            soil_nodes,
            agri_scratch,
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

    /// Advances the simulation by one tick with the given actions.
    ///
    /// Returns a StepResult containing observations, rewards, and termination info.
    #[instrument(skip_all)]
    pub fn step(&mut self, actions: &[Action]) -> StepResult {
        if self.terminated || self.truncated {
            return self.make_terminal_result();
        }

        trace!(tick = self.tick, num_actions = actions.len(), "step");

        // Pad or truncate actions to match agent count
        let mut padded_actions = actions.to_vec();
        padded_actions.resize(self.agents.len(), Action::Noop);

        // Run all systems
        systems::run_systems(self, &padded_actions);

        // Check truncation (max episode length)
        if self.config.task.max_episode_length > 0
            && self.tick >= self.config.task.max_episode_length
        {
            self.truncated = true;
        }

        // Check termination (all agents dead)
        if self.agents.iter().all(|a| !a.alive) {
            self.terminated = true;
        }

        // Generate observations and result
        self.make_step_result()
    }

    /// Resets the simulation to initial state with a new seed.
    #[instrument(skip_all)]
    pub fn reset(&mut self, seed: Option<u64>) -> StepResult {
        let new_seed = seed.unwrap_or_else(|| self.rng.next_u64());

        info!(seed = new_seed, "resetting world");

        let mut new_config = (*self.config).clone();
        new_config.world.seed = new_seed;

        let new_state = WorldState::new(new_config).expect("validated config must be valid");

        self.tick = new_state.tick;
        self.grid = new_state.grid;
        self.agents = new_state.agents;
        self.objects = new_state.objects;
        self.resources = new_state.resources;
        self.tasks = new_state.tasks;
        self.recipe_book = new_state.recipe_book;
        self.day_phase = new_state.day_phase;
        self.rng = new_state.rng;
        self.config = new_state.config;
        self.terminated = false;
        self.truncated = false;
        self.last_task_rewards = None;
        self.crop_states = new_state.crop_states;
        self.soil_nodes = new_state.soil_nodes;
        self.agri_scratch.clear();

        self.make_step_result()
    }

    /// Synchronizes agent positions to the grid tiles.
    fn sync_agents_to_grid(&mut self) {
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

    /// Generates an observation for a single agent.
    ///
    /// This constructs an ego-centric grid view and gathers inventory,
    /// health, stamina, messages, and task progress into an [`Observation`].
    /// Useful for adapters that need to bridge `WorldState`-based agents
    /// with the `AgentInterface` trait.
    pub fn generate_observation(&self, agent: &Agent) -> Observation {
        let vr = agent.vision_radius as i32;
        let view_side = (2 * vr + 1) as u16;
        let mut grid_view = Vec::with_capacity((view_side as usize) * (view_side as usize));

        for dy in -vr..=vr {
            for dx in -vr..=vr {
                let wx = agent.position.x as i32 + dx;
                let wy = agent.position.y as i32 + dy;

                if wx >= 0 && wx < self.grid.width as i32 && wy >= 0 && wy < self.grid.height as i32
                {
                    let tile = self.grid.get(wx as u16, wy as u16).unwrap();
                    grid_view.push(TileObservation {
                        terrain: tile.terrain as u8,
                        elevation: tile.elevation,
                        has_agent: tile.agent_id.is_some(),
                        has_object: tile.object_id.is_some(),
                        has_resource: tile.resource_id.is_some(),
                        object_type: tile.object_id.map_or(OBS_NO_OBJECT, |_| 0),
                        resource_type: tile.resource_id.map_or(OBS_NO_RESOURCE, |_| 0),
                    });
                } else {
                    // Out of bounds — show as wall
                    grid_view.push(TileObservation {
                        terrain: forge_types::TerrainType::Wall as u8,
                        elevation: 0,
                        has_agent: false,
                        has_object: false,
                        has_resource: false,
                        object_type: OBS_NO_OBJECT,
                        resource_type: OBS_NO_RESOURCE,
                    });
                }
            }
        }

        let inventory = InventoryObservation {
            slots: agent
                .inventory
                .slots
                .iter()
                .map(|slot| match slot {
                    Some(stack) => (stack.item_type as u8, stack.count),
                    None => (OBS_EMPTY_SLOT_ITEM, 0),
                })
                .collect(),
        };

        let max_health = self.config.agents.max_health as f32;
        let max_stamina = self.config.agents.max_stamina as f32;

        Observation {
            grid_view,
            view_width: view_side,
            view_height: view_side,
            inventory,
            health: if max_health > 0.0 {
                agent.health as f32 / max_health
            } else {
                0.0
            },
            stamina: if max_stamina > 0.0 {
                agent.stamina as f32 / max_stamina
            } else {
                0.0
            },
            position: (agent.position.x, agent.position.y),
            messages: agent.comm_buffer.to_vec(),
            day_phase: self.day_phase,
            task_progress: self
                .tasks
                .iter()
                .map(|t| {
                    if t.progress.is_empty() {
                        0.0
                    } else {
                        t.progress[0]
                    }
                })
                .collect(),
            altitude: agent.altitude,
            battery: if self.config.drone.enabled
                && agent.morphology == forge_types::entity::AgentMorphology::Aerial
            {
                let max = self.config.drone.max_battery as f32;
                if max > 0.0 {
                    (agent.battery as f32 / max).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            } else {
                1.0
            },
            morphology: agent.morphology as u8,
            heading: agent.heading as u8,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    /// Generates the StepResult for the current state.
    fn make_step_result(&mut self) -> StepResult {
        let observations: Vec<Observation> = self
            .agents
            .iter()
            .map(|agent| self.generate_observation(agent))
            .collect();

        let rewards = self
            .last_task_rewards
            .take()
            .unwrap_or_else(|| vec![0.0; self.agents.len()]);

        let info = StepInfo {
            tick: self.tick,
            agents_alive: self.agents.iter().map(|a| a.alive).collect(),
            tasks_completed: vec![Vec::new(); self.agents.len()],
            total_resources: self.resources.iter().map(|r| r.quantity as u32).sum(),
            day_phase: self.day_phase,
        };

        StepResult {
            observations,
            rewards,
            terminated: self.terminated,
            truncated: self.truncated,
            info,
        }
    }

    /// Generates a terminal StepResult.
    fn make_terminal_result(&mut self) -> StepResult {
        self.make_step_result()
    }

    /// Returns an ASCII debug representation of the world.
    #[instrument(skip_all)]
    pub fn to_debug_grid(&self) -> String {
        let mut result =
            String::with_capacity((self.grid.width as usize + 1) * self.grid.height as usize);

        for y in 0..self.grid.height {
            for x in 0..self.grid.width {
                let tile = self.grid.get(x, y).unwrap();
                let ch = if tile.agent_id.is_some() {
                    'A'
                } else if tile.object_id.is_some() {
                    'O'
                } else if tile.resource_id.is_some() {
                    'R'
                } else {
                    match tile.terrain {
                        forge_types::TerrainType::Ground => '.',
                        forge_types::TerrainType::Water => '~',
                        forge_types::TerrainType::Wall => '#',
                        forge_types::TerrainType::Lava => 'L',
                        forge_types::TerrainType::Ice => 'I',
                        forge_types::TerrainType::Sand => 'S',
                        forge_types::TerrainType::Forest => 'T',
                        forge_types::TerrainType::Mountain => 'M',
                        _ => '?',
                    }
                };
                result.push(ch);
            }
            result.push('\n');
        }

        result
    }

    /// Serializes the world state to bytes for MCTS state snapshots.
    #[instrument(skip_all)]
    pub fn to_bytes(&self) -> Vec<u8> {
        // Serialize the essential state (excluding config which is shared)
        let serializable = SerializableWorldState {
            tick: self.tick,
            grid: &self.grid,
            agents: &self.agents,
            objects: &self.objects,
            resources: &self.resources,
            day_phase: self.day_phase,
            rng_state: self.rng.save_state(),
            terminated: self.terminated,
            truncated: self.truncated,
        };
        bincode::serialize(&serializable).unwrap_or_default()
    }
}

/// Serializable subset of WorldState (excludes Arc<Config>).
#[derive(Serialize)]
struct SerializableWorldState<'a> {
    tick: u64,
    grid: &'a Grid,
    agents: &'a [Agent],
    objects: &'a [Object],
    resources: &'a [ResourceNode],
    day_phase: u8,
    rng_state: crate::rng::RngState,
    terminated: bool,
    truncated: bool,
}

/// Owned version for deserialization from bytes/JSON.
#[derive(serde::Deserialize)]
struct DeserializableWorldState {
    tick: u64,
    grid: Grid,
    agents: Vec<Agent>,
    objects: Vec<Object>,
    resources: Vec<ResourceNode>,
    day_phase: u8,
    rng_state: crate::rng::RngState,
    terminated: bool,
    truncated: bool,
}

impl WorldState {
    /// Deserializes a world state from bytes (bincode format).
    ///
    /// The config must be the same one used when the state was serialized.
    /// The RNG is reconstructed from the saved state to preserve determinism.
    #[instrument(skip_all)]
    pub fn from_bytes(bytes: &[u8], config: Arc<ForgeConfig>) -> forge_types::ForgeResult<Self> {
        let deserialized: DeserializableWorldState = bincode::deserialize(bytes)
            .map_err(|e| forge_types::ForgeError::Serialization(format!("bincode: {e}")))?;

        let mut physics_scratch = PhysicsScratch::default();
        physics_scratch.ensure_capacity(deserialized.agents.len());
        Ok(WorldState {
            tick: deserialized.tick,
            grid: deserialized.grid,
            agents: deserialized.agents,
            objects: deserialized.objects,
            resources: deserialized.resources,
            tasks: Vec::new(),
            recipe_book: RecipeBook::default(),
            day_phase: deserialized.day_phase,
            rng: ForgeRng::from_state(&deserialized.rng_state),
            config,
            terminated: deserialized.terminated,
            truncated: deserialized.truncated,
            last_task_rewards: None,
            physics_scratch,
            crop_states: Vec::new(),
            soil_nodes: Vec::new(),
            agri_scratch: AgriScratch::default(),
        })
    }

    /// Serializes the world state to JSON for human-readable saves.
    #[instrument(skip_all)]
    pub fn to_json(&self) -> forge_types::ForgeResult<String> {
        let serializable = SerializableWorldState {
            tick: self.tick,
            grid: &self.grid,
            agents: &self.agents,
            objects: &self.objects,
            resources: &self.resources,
            day_phase: self.day_phase,
            rng_state: self.rng.save_state(),
            terminated: self.terminated,
            truncated: self.truncated,
        };
        serde_json::to_string(&serializable)
            .map_err(|e| forge_types::ForgeError::Serialization(format!("json: {e}")))
    }

    /// Deserializes a world state from JSON.
    ///
    /// The config must match what was used when the state was serialized.
    #[instrument(skip_all)]
    pub fn from_json(json: &str, config: Arc<ForgeConfig>) -> forge_types::ForgeResult<Self> {
        let deserialized: DeserializableWorldState = serde_json::from_str(json)
            .map_err(|e| forge_types::ForgeError::Serialization(format!("json: {e}")))?;

        let mut physics_scratch = PhysicsScratch::default();
        physics_scratch.ensure_capacity(deserialized.agents.len());
        Ok(WorldState {
            tick: deserialized.tick,
            grid: deserialized.grid,
            agents: deserialized.agents,
            objects: deserialized.objects,
            resources: deserialized.resources,
            tasks: Vec::new(),
            recipe_book: RecipeBook::default(),
            day_phase: deserialized.day_phase,
            rng: ForgeRng::from_state(&deserialized.rng_state),
            config,
            terminated: deserialized.terminated,
            truncated: deserialized.truncated,
            last_task_rewards: None,
            physics_scratch,
            crop_states: Vec::new(),
            soil_nodes: Vec::new(),
            agri_scratch: AgriScratch::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;
    use forge_types::grid::Direction;

    fn make_test_world() -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.task.max_episode_length = 1000;
        WorldState::new(config).unwrap()
    }

    #[test]
    fn test_world_creation() {
        let world = make_test_world();
        assert_eq!(world.tick, 0);
        assert_eq!(world.agents.len(), 1);
        assert_eq!(world.grid.width, 16);
        assert_eq!(world.grid.height, 16);
        assert!(!world.terminated);
        assert!(!world.truncated);
    }

    #[test]
    fn test_world_step() {
        let mut world = make_test_world();
        let start_pos = world.agents[0].position;

        let result = world.step(&[Action::Move(Direction::Right)]);

        assert_eq!(world.tick, 1);
        assert_eq!(result.observations.len(), 1);
        assert_eq!(result.rewards.len(), 1);
        assert!(!result.terminated);
        assert!(!result.truncated);

        // Agent should have moved (unless at boundary)
        if start_pos.x + 1 < world.grid.width {
            assert_eq!(
                world.agents[0].position,
                Position::new(start_pos.x + 1, start_pos.y)
            );
        }
    }

    #[test]
    fn test_world_reset() {
        let mut world = make_test_world();
        world.step(&[Action::Move(Direction::Right)]);
        world.step(&[Action::Move(Direction::Right)]);
        assert_eq!(world.tick, 2);

        let result = world.reset(Some(99));
        assert_eq!(world.tick, 0);
        assert!(!world.terminated);
        assert!(!world.truncated);
        assert_eq!(result.observations.len(), 1);
    }

    #[test]
    fn test_world_truncation() {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.agents.default_vision_radius = 3;
        config.agents.num_agents = 1;
        config.task.max_episode_length = 5;
        let mut world = WorldState::new(config).unwrap();

        for _ in 0..5 {
            let result = world.step(&[Action::Noop]);
            if result.truncated {
                break;
            }
        }

        assert!(world.truncated);
    }

    #[test]
    fn test_determinism() {
        let config1 = {
            let mut c = ForgeConfig::default();
            c.world.width = 16;
            c.world.height = 16;
            c.world.seed = 42;
            c.agents.num_agents = 2;
            c
        };
        let config2 = config1.clone();

        let mut world1 = WorldState::new(config1).unwrap();
        let mut world2 = WorldState::new(config2).unwrap();

        // Same actions should produce identical states
        let action_sequence = vec![
            vec![
                Action::Move(Direction::Right),
                Action::Move(Direction::Down),
            ],
            vec![Action::Move(Direction::Up), Action::Move(Direction::Left)],
            vec![Action::Noop, Action::Move(Direction::Right)],
            vec![Action::Move(Direction::Down), Action::Move(Direction::Down)],
        ];

        for actions in &action_sequence {
            let r1 = world1.step(actions);
            let r2 = world2.step(actions);

            assert_eq!(world1.tick, world2.tick);
            for (a1, a2) in world1.agents.iter().zip(world2.agents.iter()) {
                assert_eq!(a1.position, a2.position);
                assert_eq!(a1.health, a2.health);
                assert_eq!(a1.stamina, a2.stamina);
            }
            assert_eq!(r1.terminated, r2.terminated);
            assert_eq!(r1.truncated, r2.truncated);
        }
    }

    #[test]
    fn test_observation_shape() {
        let world = make_test_world();
        let obs = world.generate_observation(&world.agents[0]);

        let vr = world.agents[0].vision_radius as u16;
        let expected_side = 2 * vr + 1;
        assert_eq!(obs.view_width, expected_side);
        assert_eq!(obs.view_height, expected_side);
        assert_eq!(
            obs.grid_view.len(),
            (expected_side as usize) * (expected_side as usize)
        );
    }

    #[test]
    fn test_debug_grid() {
        let world = make_test_world();
        let debug = world.to_debug_grid();
        assert!(!debug.is_empty());
        assert!(debug.contains('A')); // At least one agent
        assert!(debug.contains('.')); // At least some ground
    }

    #[test]
    fn test_serialization() {
        let world = make_test_world();
        let bytes = world.to_bytes();
        assert!(!bytes.is_empty());
        // Verify size is reasonable (should be well under 64KB for a 16x16 world)
        assert!(
            bytes.len() < 65536,
            "state too large: {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn test_step_with_wrong_action_count() {
        let mut world = make_test_world();
        // Too few actions — should be padded with Noop
        let result = world.step(&[]);
        assert!(!result.terminated);

        // Too many actions — extra should be ignored
        let result = world.step(&[Action::Noop, Action::Noop, Action::Noop]);
        assert!(!result.terminated);
    }

    #[test]
    fn test_multi_agent_world() {
        let mut config = ForgeConfig::default();
        config.world.width = 32;
        config.world.height = 32;
        config.world.seed = 42;
        config.agents.num_agents = 4;
        let world = WorldState::new(config).unwrap();

        assert_eq!(world.agents.len(), 4);
        // All agents should be alive
        assert!(world.agents.iter().all(|a| a.alive));
    }

    // ---- Edge case tests ----

    #[test]
    fn test_grid_with_min_dimension() {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.agents.default_vision_radius = 3;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        let world = WorldState::new(config).unwrap();

        assert_eq!(world.grid.width, 8);
        assert_eq!(world.grid.height, 8);
        assert_eq!(world.agents.len(), 1);
        assert!(world.agents[0].alive);
        // Agent should be within bounds
        assert!(world.agents[0].position.x < 8);
        assert!(world.agents[0].position.y < 8);
    }

    #[test]
    fn test_observation_agent_at_corner_origin() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        let mut world = WorldState::new(config).unwrap();

        // Move agent to (0,0)
        let old_pos = world.agents[0].position;
        if let Some(tile) = world.grid.get_mut(old_pos.x, old_pos.y) {
            tile.agent_id = None;
        }
        world.agents[0].position = Position::new(0, 0);
        world.grid.get_mut(0, 0).unwrap().agent_id = Some(0);

        let obs = world.generate_observation(&world.agents[0]);

        // Observation should still have the correct shape
        let vr = world.agents[0].vision_radius as u16;
        let expected_side = 2 * vr + 1;
        assert_eq!(obs.view_width, expected_side);
        assert_eq!(obs.view_height, expected_side);
        assert_eq!(
            obs.grid_view.len(),
            (expected_side as usize) * (expected_side as usize)
        );

        // Tiles beyond the boundary should show as walls
        // The top-left corner of the view (at offset -vr, -vr from agent at (0,0))
        // should be out of bounds (wall)
        let first_tile = &obs.grid_view[0];
        assert_eq!(first_tile.terrain, forge_types::TerrainType::Wall as u8);
    }

    #[test]
    fn test_reset_with_none_seed() {
        let mut world = make_test_world();
        let initial_seed = world.config.world.seed;

        // Step a few times to advance the RNG
        world.step(&[Action::Noop]);
        world.step(&[Action::Noop]);

        // Reset with None — should derive a new seed from the RNG
        let result = world.reset(None);
        assert_eq!(world.tick, 0);
        assert!(!world.terminated);
        assert!(!world.truncated);
        assert_eq!(result.observations.len(), 1);

        // The seed should have changed (overwhelmingly likely)
        assert_ne!(
            world.config.world.seed, initial_seed,
            "reset(None) should use a derived seed"
        );
    }

    #[test]
    fn test_multiple_resets_produce_different_states() {
        let mut world = make_test_world();

        // First reset with a specific seed
        world.reset(Some(100));
        let pos_after_first = world.agents[0].position;

        // Second reset with a different seed
        world.reset(Some(200));
        let pos_after_second = world.agents[0].position;

        // Third reset with another seed
        world.reset(Some(300));
        let pos_after_third = world.agents[0].position;

        // At least two of the three positions should differ
        // (technically all could coincide, but with a 16x16 grid that is very unlikely)
        let all_same = pos_after_first == pos_after_second && pos_after_second == pos_after_third;
        assert!(
            !all_same,
            "multiple resets with different seeds should produce different states"
        );
    }

    #[test]
    fn test_to_debug_grid_with_empty_world() {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.agents.default_vision_radius = 3;
        config.world.seed = 42;
        config.agents.num_agents = 0;
        config.task.enabled = false;
        let world = WorldState::new(config).unwrap();

        let debug = world.to_debug_grid();
        assert!(!debug.is_empty());
        // With no agents, should be all ground tiles
        assert!(!debug.contains('A'));
        // Should have 8 rows (each 8 chars + newline)
        let lines: Vec<&str> = debug.lines().collect();
        assert_eq!(lines.len(), 8);
        for line in &lines {
            assert_eq!(line.len(), 8);
        }
    }

    #[test]
    fn test_to_bytes_round_trip_consistency() {
        let world = make_test_world();
        let bytes1 = world.to_bytes();
        let bytes2 = world.to_bytes();

        assert!(!bytes1.is_empty());
        assert_eq!(bytes1, bytes2, "serialization should be deterministic");
    }

    #[test]
    fn test_to_bytes_changes_after_step() {
        let mut world = make_test_world();
        let bytes_before = world.to_bytes();

        world.step(&[Action::Move(Direction::Right)]);
        let bytes_after = world.to_bytes();

        assert_ne!(
            bytes_before, bytes_after,
            "state bytes should change after stepping"
        );
    }

    #[test]
    fn test_step_after_termination_returns_terminal_result() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        let mut world = WorldState::new(config).unwrap();

        // Kill the agent to trigger termination
        world.agents[0].alive = false;
        let result = world.step(&[Action::Noop]);
        assert!(result.terminated);

        // Subsequent step should still return terminal result
        let result2 = world.step(&[Action::Noop]);
        assert!(result2.terminated);
    }

    #[test]
    fn test_world_min_dimension_width() {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.default_vision_radius = 3;
        let mut world = WorldState::new(config).unwrap();

        // Should still work with min width
        let result = world.step(&[Action::Noop]);
        assert!(!result.terminated);
        assert_eq!(world.grid.width, 8);
    }

    #[test]
    fn test_world_min_dimension_height() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 8;
        config.agents.default_vision_radius = 3;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        let mut world = WorldState::new(config).unwrap();

        let result = world.step(&[Action::Noop]);
        assert!(!result.terminated);
        assert_eq!(world.grid.height, 8);
    }

    #[test]
    fn test_observation_position_field() {
        let world = make_test_world();
        let obs = world.generate_observation(&world.agents[0]);

        assert_eq!(obs.position.0, world.agents[0].position.x);
        assert_eq!(obs.position.1, world.agents[0].position.y);
    }

    #[test]
    fn test_observation_health_and_stamina_normalized() {
        let world = make_test_world();
        let obs = world.generate_observation(&world.agents[0]);

        // Health and stamina should be between 0.0 and 1.0
        assert!(obs.health >= 0.0 && obs.health <= 1.0);
        assert!(obs.stamina >= 0.0 && obs.stamina <= 1.0);
    }

    // ---- from_bytes / from_json roundtrip tests ----

    #[test]
    fn test_from_bytes_roundtrip() {
        let mut world = make_test_world();
        world.step(&[Action::Noop]);
        world.step(&[Action::Move(Direction::Right)]);

        let bytes = world.to_bytes();
        let restored = WorldState::from_bytes(&bytes, world.config.clone()).unwrap();

        assert_eq!(restored.tick, world.tick);
        assert_eq!(restored.day_phase, world.day_phase);
        assert_eq!(restored.agents.len(), world.agents.len());
        assert_eq!(restored.agents[0].position, world.agents[0].position);
        assert_eq!(restored.agents[0].health, world.agents[0].health);
        assert_eq!(restored.terminated, world.terminated);
        assert_eq!(restored.truncated, world.truncated);
    }

    #[test]
    fn test_from_bytes_rng_preserves_sequence() {
        let mut world = make_test_world();
        for _ in 0..5 {
            world.step(&[Action::Noop]);
        }

        let bytes = world.to_bytes();
        let mut restored = WorldState::from_bytes(&bytes, world.config.clone()).unwrap();

        // After restoration, stepping both should produce the same results
        let result_original = world.step(&[Action::Noop]);
        let result_restored = restored.step(&[Action::Noop]);
        assert_eq!(
            result_original.observations[0].position,
            result_restored.observations[0].position,
        );
    }

    #[test]
    fn test_from_bytes_invalid_data() {
        let config = Arc::new(ForgeConfig::default());
        let result = WorldState::from_bytes(&[0, 1, 2, 3], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_json_roundtrip() {
        let mut world = make_test_world();
        world.step(&[Action::Noop]);

        let json = world.to_json().unwrap();
        assert!(!json.is_empty());

        let restored = WorldState::from_json(&json, world.config.clone()).unwrap();
        assert_eq!(restored.tick, world.tick);
        assert_eq!(restored.agents[0].position, world.agents[0].position);
        assert_eq!(restored.day_phase, world.day_phase);
    }

    #[test]
    fn test_from_json_invalid_data() {
        let config = Arc::new(ForgeConfig::default());
        let result = WorldState::from_json("not valid json", config);
        assert!(result.is_err());
    }

    // ---- Proptest: world invariants ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn make_world_with_seed(seed: u64) -> WorldState {
            let mut config = ForgeConfig::default();
            config.world.width = 16;
            config.world.height = 16;
            config.world.seed = seed;
            config.agents.num_agents = 2;
            config.task.max_episode_length = 100;
            WorldState::new(config).unwrap()
        }

        proptest! {
            /// Same seed + same actions = identical world state.
            #[test]
            fn step_determinism(seed in 0u64..10_000) {
                let mut w1 = make_world_with_seed(seed);
                let mut w2 = make_world_with_seed(seed);

                let actions = vec![Action::Noop, Action::Move(Direction::Right)];

                for _ in 0..5 {
                    w1.step(&actions);
                    w2.step(&actions);
                }

                prop_assert_eq!(w1.tick, w2.tick);
                for (a, b) in w1.agents.iter().zip(w2.agents.iter()) {
                    prop_assert_eq!(a.position, b.position);
                    prop_assert_eq!(a.health, b.health);
                    prop_assert_eq!(a.stamina, b.stamina);
                    prop_assert_eq!(a.alive, b.alive);
                }
            }

            /// After any number of steps, all agent positions are in bounds.
            #[test]
            fn agents_in_bounds_after_steps(
                seed in 0u64..10_000,
                steps in 1u32..20,
            ) {
                let mut world = make_world_with_seed(seed);
                for _ in 0..steps {
                    world.step(&[Action::Move(Direction::Right), Action::Move(Direction::Down)]);
                }
                for agent in &world.agents {
                    prop_assert!(agent.position.x < world.grid.width);
                    prop_assert!(agent.position.y < world.grid.height);
                }
            }

            /// Serialization roundtrip preserves essential state.
            #[test]
            fn bytes_roundtrip(seed in 0u64..10_000) {
                let mut world = make_world_with_seed(seed);
                world.step(&[Action::Noop, Action::Noop]);

                let bytes = world.to_bytes();
                let restored = WorldState::from_bytes(&bytes, world.config.clone()).unwrap();

                prop_assert_eq!(restored.tick, world.tick);
                prop_assert_eq!(restored.agents.len(), world.agents.len());
                for (a, b) in restored.agents.iter().zip(world.agents.iter()) {
                    prop_assert_eq!(a.position, b.position);
                }
            }
        }
    }
}
