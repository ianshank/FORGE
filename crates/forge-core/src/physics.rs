//! Physics system for the FORGE simulation.
//!
//! Handles movement, collision detection, stamina costs, and projectile
//! trajectories. Uses the grid topology abstraction for neighbor lookups
//! so that both square and hex grids are supported.
//! All arithmetic is integer-based for determinism.

use forge_civ::grid_topology::{GridTopology, GridTopologyKind};
use forge_types::config::{DroneConfig, PhysicsConfig};
use forge_types::entity::{Agent, AgentMorphology};
use forge_types::grid::{Direction, Grid, HexDirection, Position, TerrainType};
use forge_types::Action;
use tracing::{instrument, trace, warn};

/// Converts a cardinal Direction to a target position using the topology.
///
/// On square grids, maps Direction to the corresponding topology direction index.
/// On hex grids, returns None — 4-direction Move is invalid; use MoveHex instead.
#[inline]
fn move_target(
    pos: Position,
    dir: Direction,
    topology: &GridTopologyKind,
    width: u16,
    height: u16,
) -> Option<Position> {
    match topology {
        GridTopologyKind::Square(_) => {
            let idx = match dir {
                Direction::Up => 0,
                Direction::Down => 1,
                Direction::Left => 2,
                Direction::Right => 3,
            };
            topology.neighbor(pos, idx, width, height)
        }
        GridTopologyKind::Hex(_) => None,
    }
}

/// Converts a hex direction to a target position using the topology.
///
/// On hex grids, maps HexDirection to the corresponding topology direction index.
/// On square grids, returns None — 6-direction MoveHex is invalid; use Move instead.
#[inline]
fn hex_move_target(
    pos: Position,
    dir: HexDirection,
    topology: &GridTopologyKind,
    width: u16,
    height: u16,
) -> Option<Position> {
    match topology {
        GridTopologyKind::Hex(_) => topology.neighbor(pos, dir as u8, width, height),
        GridTopologyKind::Square(_) => None,
    }
}

/// Looks up the vehicle-specific terrain cost for a given terrain type.
///
/// Returns the fixed-point movement cost multiplier from `DroneConfig::vehicle_terrain_costs`,
/// or falls back to the terrain's default movement cost if the drone config is absent or
/// the terrain index is out of range.
///
/// `i32::MAX` indicates impassable terrain.
#[inline]
fn vehicle_terrain_cost(terrain: TerrainType, drone_config: Option<&DroneConfig>) -> i32 {
    if let Some(dc) = drone_config {
        let idx = terrain as usize;
        if idx < dc.vehicle_terrain_costs.len() {
            dc.vehicle_terrain_costs[idx]
        } else {
            terrain.movement_cost()
        }
    } else {
        terrain.movement_cost()
    }
}

/// Result of processing a single agent's movement action.
#[derive(Debug, Clone, PartialEq)]
pub enum MoveResult {
    /// Agent successfully moved to the new position.
    Moved(Position),
    /// Movement was blocked (wall, occupied tile, boundary, etc.).
    Blocked,
    /// Agent doesn't have enough stamina.
    NoStamina,
    /// Terrain is not walkable.
    Impassable,
}

/// Pre-allocated scratch buffers for the physics system.
///
/// Stored in `WorldState` and reused across ticks to avoid per-tick heap
/// allocations on the hot path. Call [`PhysicsScratch::ensure_capacity`]
/// after changing the agent count.
#[derive(Debug, Clone, Default)]
pub(crate) struct PhysicsScratch {
    /// Scratch buffer for move results.
    pub(crate) results: Vec<MoveResult>,
    /// Scratch buffer for desired positions.
    pub(crate) desired_positions: Vec<Option<Position>>,
    /// Scratch buffer for occupied target tracking.
    pub(crate) occupied_targets: Vec<(usize, Position)>,
}

impl PhysicsScratch {
    /// Ensures all buffers have at least `agent_count` capacity.
    pub(crate) fn ensure_capacity(&mut self, agent_count: usize) {
        if self.results.capacity() < agent_count {
            self.results.reserve(agent_count - self.results.capacity());
        }
        if self.desired_positions.capacity() < agent_count {
            self.desired_positions
                .reserve(agent_count - self.desired_positions.capacity());
        }
        if self.occupied_targets.capacity() < agent_count {
            self.occupied_targets
                .reserve(agent_count - self.occupied_targets.capacity());
        }
    }
}

/// Processes movement actions for all agents.
///
/// This is the core physics step. It validates moves, checks collisions,
/// applies stamina costs, and updates agent positions on the grid.
///
/// Movement is processed in agent order (agent 0 first). Ties in movement
/// to the same tile are resolved by agent priority (lower ID wins).
///
/// When drone mechanics are enabled, morphology-aware rules apply:
/// - **Aerial (airborne)**: ignores ground terrain, drains battery instead of stamina,
///   only collides with agents at the same altitude.
/// - **GroundVehicle**: uses vehicle-specific terrain costs, blocked by Forest/Mountain/Water.
/// - **Ground**: unchanged default behavior.
#[instrument(skip_all)]
pub fn process_movements(
    agents: &mut [Agent],
    grid: &mut Grid,
    actions: &[Action],
    config: &PhysicsConfig,
    drone_config: Option<&DroneConfig>,
    topology: &GridTopologyKind,
) -> Vec<MoveResult> {
    let mut scratch = PhysicsScratch::default();
    process_movements_with_scratch(
        agents,
        grid,
        actions,
        config,
        drone_config,
        &mut scratch,
        topology,
    );
    std::mem::take(&mut scratch.results)
}

/// Processes movement actions using pre-allocated scratch buffers.
///
/// Like [`process_movements`], but avoids heap allocation by reusing
/// the provided [`PhysicsScratch`]. Prefer this on the hot path.
#[instrument(skip_all)]
pub(crate) fn process_movements_with_scratch(
    agents: &mut [Agent],
    grid: &mut Grid,
    actions: &[Action],
    config: &PhysicsConfig,
    drone_config: Option<&DroneConfig>,
    scratch: &mut PhysicsScratch,
    topology: &GridTopologyKind,
) {
    scratch.results.clear();
    scratch.desired_positions.clear();
    scratch.occupied_targets.clear();
    scratch.ensure_capacity(agents.len());

    // First pass: compute desired positions
    for (agent, action) in agents.iter().zip(actions.iter()) {
        let pos = if !agent.alive {
            None
        } else {
            match action {
                Action::Move(dir) => {
                    move_target(agent.position, *dir, topology, grid.width, grid.height)
                }
                Action::MoveHex(dir) => {
                    hex_move_target(agent.position, *dir, topology, grid.width, grid.height)
                }
                _ => None,
            }
        };
        scratch.desired_positions.push(pos);
    }

    // Snapshot agent positions and altitudes for aerial collision detection
    // Uses stack-allocated SmallVec to avoid heap allocation for small agent counts
    let agents_snapshot: smallvec::SmallVec<
        [(Position, u8, bool); forge_types::constants::PHYSICS_SMALLVEC_CAPACITY],
    > = agents
        .iter()
        .map(|a| (a.position, a.altitude, a.alive))
        .collect();

    // Second pass: detect conflicts (two agents wanting same tile)
    for (i, pos) in scratch.desired_positions.iter().enumerate() {
        if let Some(p) = pos {
            scratch.occupied_targets.push((i, *p));
        }
    }

    // Third pass: apply movements
    for (i, agent) in agents.iter_mut().enumerate() {
        if !agent.alive {
            scratch.results.push(MoveResult::Blocked);
            continue;
        }

        let action = &actions[i];
        let target_pos = match action {
            Action::Move(dir) => {
                move_target(agent.position, *dir, topology, grid.width, grid.height)
            }
            Action::MoveHex(dir) => {
                hex_move_target(agent.position, *dir, topology, grid.width, grid.height)
            }
            _ => {
                scratch.results.push(MoveResult::Blocked);
                continue;
            }
        };

        let is_airborne = agent.morphology == AgentMorphology::Aerial && agent.altitude > 0;

        // Energy check: airborne Aerial agents use battery, others use stamina
        let base_cost = config.stamina_cost_move;
        if is_airborne {
            if agent.battery < base_cost {
                trace!(agent_id = agent.id, "movement blocked: no battery");
                scratch.results.push(MoveResult::NoStamina);
                continue;
            }
        } else if agent.stamina < base_cost {
            trace!(agent_id = agent.id, "movement blocked: no stamina");
            scratch.results.push(MoveResult::NoStamina);
            continue;
        }

        // Get target position
        let target = match target_pos {
            Some(pos) => pos,
            None => {
                trace!(
                    agent_id = agent.id,
                    "movement blocked: boundary or invalid grid type"
                );
                scratch.results.push(MoveResult::Blocked);
                continue;
            }
        };

        // Terrain and collision checks depend on morphology and altitude
        let target_tile = grid.get(target.x, target.y).unwrap();

        if is_airborne {
            // Airborne Aerial: ignore ground terrain, only collide with agents at same altitude
            if config.collision_enabled {
                // Check for aerial collision at same altitude (scan agents, not grid)
                let aerial_conflict =
                    agents_snapshot
                        .iter()
                        .enumerate()
                        .any(|(j, (pos, alt, alive))| {
                            j != i && *alive && *pos == target && *alt == agent.altitude
                        });
                if aerial_conflict {
                    trace!(
                        agent_id = agent.id,
                        altitude = agent.altitude,
                        "movement blocked: aerial collision at same altitude"
                    );
                    scratch.results.push(MoveResult::Blocked);
                    continue;
                }
            }

            // Check if another airborne agent (with higher priority) is also moving here at same altitude
            let conflict = scratch
                .occupied_targets
                .iter()
                .any(|(other_idx, other_pos)| *other_idx < i && *other_pos == target);
            if conflict {
                trace!(
                    agent_id = agent.id,
                    altitude = agent.altitude,
                    "movement blocked: aerial conflict with higher priority agent"
                );
                scratch.results.push(MoveResult::Blocked);
                continue;
            }

            // Deduct battery for airborne movement
            agent.battery = (agent.battery - base_cost).max(0);
        } else {
            // Ground-level movement (Ground, GroundVehicle, or grounded Aerial)

            // Terrain walkability check (with vehicle-specific terrain costs)
            let terrain_passable = match agent.morphology {
                AgentMorphology::GroundVehicle => {
                    vehicle_terrain_cost(target_tile.terrain, drone_config) != i32::MAX
                }
                _ => target_tile.terrain.is_walkable(),
            };

            if !terrain_passable {
                trace!(
                    agent_id = agent.id,
                    terrain = ?target_tile.terrain,
                    morphology = ?agent.morphology,
                    "movement blocked: impassable terrain"
                );
                scratch.results.push(MoveResult::Impassable);
                continue;
            }

            // Ground-level collision check (existing logic)
            if config.collision_enabled {
                if let Some(occupant) = target_tile.agent_id {
                    if occupant != agent.id {
                        trace!(
                            agent_id = agent.id,
                            occupant_id = occupant,
                            "movement blocked: tile occupied"
                        );
                        scratch.results.push(MoveResult::Blocked);
                        continue;
                    }
                }

                // Check if another agent (with higher priority) is also moving here
                let conflict = scratch
                    .occupied_targets
                    .iter()
                    .any(|(other_idx, other_pos)| *other_idx < i && *other_pos == target);
                if conflict {
                    trace!(
                        agent_id = agent.id,
                        "movement blocked: conflict with higher priority agent"
                    );
                    scratch.results.push(MoveResult::Blocked);
                    continue;
                }
            }

            // Apply terrain movement cost
            let terrain_cost = match agent.morphology {
                AgentMorphology::GroundVehicle => {
                    vehicle_terrain_cost(target_tile.terrain, drone_config)
                }
                _ => target_tile.terrain.movement_cost(),
            };

            let total_cost = if terrain_cost == i32::MAX {
                base_cost
            } else {
                ((base_cost as i64 * terrain_cost as i64) >> 16) as i32
            };

            // Deduct stamina
            agent.stamina = (agent.stamina - total_cost).max(0);
        }

        // Update heading for Move actions (MoveHex does not update heading)
        if let Action::Move(dir) = action {
            agent.heading = *dir;
        }

        // Clear old position on grid (only for ground-level agents)
        if !is_airborne {
            if let Some(tile) = grid.get_mut(agent.position.x, agent.position.y) {
                if tile.agent_id == Some(agent.id) {
                    tile.agent_id = None;
                }
            }
        }

        // Update agent position
        let old_pos = agent.position;
        agent.position = target;

        // Set new position on grid (only for ground-level agents)
        if !is_airborne {
            if let Some(tile) = grid.get_mut(target.x, target.y) {
                tile.agent_id = Some(agent.id);
            }
        }

        trace!(
            agent_id = agent.id,
            old_x = old_pos.x,
            old_y = old_pos.y,
            new_x = target.x,
            new_y = target.y,
            morphology = ?agent.morphology,
            altitude = agent.altitude,
            "agent moved"
        );
        scratch.results.push(MoveResult::Moved(target));
    }
}

/// Regenerates stamina for all agents.
#[instrument(skip_all)]
pub fn regenerate_stamina(agents: &mut [Agent], config: &PhysicsConfig, max_stamina: i32) {
    for agent in agents.iter_mut() {
        if !agent.alive {
            continue;
        }
        agent.stamina = (agent.stamina + config.stamina_regen_rate).min(max_stamina);
    }
}

/// Minimal agent snapshot for push processing.
///
/// Contains only the fields needed to evaluate push actions, avoiding a full
/// `Vec<Agent>` clone. Extracted once per tick before mutable borrows begin.
#[derive(Debug, Clone, Copy)]
pub struct AgentPushData {
    /// Agent identifier.
    pub id: u32,
    /// Current position.
    pub position: Position,
    /// Whether the agent is alive.
    pub alive: bool,
}

impl AgentPushData {
    /// Creates push data from an agent reference.
    #[inline]
    #[instrument(skip_all)]
    pub fn from_agent(agent: &Agent) -> Self {
        Self {
            id: agent.id,
            position: agent.position,
            alive: agent.alive,
        }
    }
}

/// Processes push actions -- agents pushing objects.
///
/// Accepts [`AgentPushData`] slices instead of full `&[Agent]` references,
/// eliminating the need to clone the entire agent vec each tick.
#[instrument(skip_all)]
pub fn process_pushes(
    agent_data: &[AgentPushData],
    grid: &mut Grid,
    objects: &mut [forge_types::Object],
    actions: &[Action],
    config: &PhysicsConfig,
) {
    if !config.collision_enabled {
        return;
    }

    for (i, action) in actions.iter().enumerate() {
        if let Action::Push(direction) = action {
            if i >= agent_data.len() {
                continue;
            }
            let agent = &agent_data[i];
            if !agent.alive {
                continue;
            }

            // Find adjacent tile in push direction
            let push_from = match agent.position.offset(*direction, grid.width, grid.height) {
                Some(pos) => pos,
                None => continue,
            };

            // Check if there's an object to push
            let tile = match grid.get(push_from.x, push_from.y) {
                Some(t) => t,
                None => continue,
            };

            let obj_id = match tile.object_id {
                Some(id) => id,
                None => continue,
            };

            // Check if the destination tile is free
            let push_to = match push_from.offset(*direction, grid.width, grid.height) {
                Some(pos) => pos,
                None => {
                    warn!(agent_id = agent.id, "push blocked: boundary");
                    continue;
                }
            };

            let dest_tile = match grid.get(push_to.x, push_to.y) {
                Some(t) => t,
                None => continue,
            };

            if !dest_tile.terrain.is_walkable()
                || dest_tile.object_id.is_some()
                || dest_tile.agent_id.is_some()
            {
                continue;
            }

            // Move the object
            if let Some(obj) = objects.iter_mut().find(|o| o.id == obj_id) {
                // Clear old tile
                if let Some(t) = grid.get_mut(push_from.x, push_from.y) {
                    t.object_id = None;
                }
                // Set new tile
                if let Some(t) = grid.get_mut(push_to.x, push_to.y) {
                    t.object_id = Some(obj_id);
                }
                obj.position = push_to;
                trace!(
                    agent_id = agent.id,
                    object_id = obj_id,
                    "object pushed to ({}, {})",
                    push_to.x,
                    push_to.y
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_civ::grid_topology::GridTopologyKind;
    use forge_civ::SquareTopology;
    use forge_types::config::AgentConfig;
    use forge_types::grid::{Direction, TerrainType};

    fn topo() -> GridTopologyKind {
        GridTopologyKind::Square(SquareTopology)
    }

    fn make_test_grid(width: u16, height: u16) -> Grid {
        Grid::new(width, height)
    }

    fn make_test_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        let mut agent = Agent::new(id, Position::new(x, y), &config);
        agent.stamina = 655360; // 10.0 in fixed point
        agent
    }

    fn default_physics() -> PhysicsConfig {
        PhysicsConfig::default()
    }

    #[test]
    fn test_basic_movement() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Moved(Position::new(5, 4)));
        assert_eq!(agents[0].position, Position::new(5, 4));
        assert!(grid.get(5, 5).unwrap().agent_id.is_none());
        assert_eq!(grid.get(5, 4).unwrap().agent_id, Some(0));
    }

    #[test]
    fn test_movement_all_directions() {
        for (dir, expected) in [
            (Direction::Up, Position::new(5, 4)),
            (Direction::Down, Position::new(5, 6)),
            (Direction::Left, Position::new(4, 5)),
            (Direction::Right, Position::new(6, 5)),
        ] {
            let mut grid = make_test_grid(16, 16);
            let mut agents = [make_test_agent(0, 5, 5)];
            grid.get_mut(5, 5).unwrap().agent_id = Some(0);

            let actions = vec![Action::Move(dir)];
            let results = process_movements(
                &mut agents,
                &mut grid,
                &actions,
                &default_physics(),
                None,
                &topo(),
            );

            assert_eq!(
                results[0],
                MoveResult::Moved(expected),
                "direction: {:?}",
                dir
            );
        }
    }

    #[test]
    fn test_collision_with_wall() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 4).unwrap().terrain = TerrainType::Wall;
        let mut agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Impassable);
        assert_eq!(agents[0].position, Position::new(5, 5)); // didn't move
    }

    #[test]
    fn test_collision_with_boundary() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 0, 0)];
        grid.get_mut(0, 0).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Blocked);
        assert_eq!(agents[0].position, Position::new(0, 0));
    }

    #[test]
    fn test_collision_with_agent() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 5, 4)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);

        let actions = vec![Action::Move(Direction::Up), Action::Noop];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Blocked);
        assert_eq!(agents[0].position, Position::new(5, 5));
    }

    #[test]
    fn test_movement_drains_stamina() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let initial_stamina = agents[0].stamina;
        let actions = vec![Action::Move(Direction::Up)];
        process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert!(agents[0].stamina < initial_stamina);
    }

    #[test]
    fn test_no_stamina_blocks_movement() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_test_agent(0, 5, 5)];
        agents[0].stamina = 0;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::NoStamina);
    }

    #[test]
    fn test_stamina_regeneration() {
        let config = default_physics();
        let agent_config = AgentConfig::default();
        let mut agents = [make_test_agent(0, 5, 5)];
        agents[0].stamina = 100000;

        regenerate_stamina(&mut agents, &config, agent_config.max_stamina);
        assert!(agents[0].stamina > 100000);
    }

    #[test]
    fn test_stamina_regen_capped() {
        let config = default_physics();
        let max = 655360;
        let mut agents = [make_test_agent(0, 5, 5)];
        agents[0].stamina = max;

        regenerate_stamina(&mut agents, &config, max);
        assert_eq!(agents[0].stamina, max);
    }

    #[test]
    fn test_dead_agent_cannot_move() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_test_agent(0, 5, 5)];
        agents[0].alive = false;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Blocked);
        assert_eq!(agents[0].position, Position::new(5, 5));
    }

    #[test]
    fn test_noop_no_movement() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let initial_stamina = agents[0].stamina;
        let actions = vec![Action::Noop];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Blocked);
        assert_eq!(agents[0].position, Position::new(5, 5));
        assert_eq!(agents[0].stamina, initial_stamina); // no stamina cost for noop
    }

    #[test]
    fn test_terrain_movement_cost() {
        // Forest costs 2x stamina
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 4).unwrap().terrain = TerrainType::Forest;
        let mut agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let initial_stamina = agents[0].stamina;
        let actions = vec![Action::Move(Direction::Up)];
        process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        let stamina_used = initial_stamina - agents[0].stamina;

        // Now test normal ground
        let mut grid2 = make_test_grid(16, 16);
        let mut agents2 = vec![make_test_agent(0, 5, 5)];
        agents2[0].stamina = initial_stamina;
        grid2.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions2 = vec![Action::Move(Direction::Up)];
        process_movements(
            &mut agents2,
            &mut grid2,
            &actions2,
            &default_physics(),
            None,
            &topo(),
        );

        let normal_stamina_used = initial_stamina - agents2[0].stamina;

        // Forest should cost more
        assert!(stamina_used > normal_stamina_used);
    }

    #[test]
    fn test_water_impassable() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 4).unwrap().terrain = TerrainType::Water;
        let mut agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Impassable);
    }

    #[test]
    fn test_two_agents_independent_movement() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 2, 2), make_test_agent(1, 8, 8)];
        grid.get_mut(2, 2).unwrap().agent_id = Some(0);
        grid.get_mut(8, 8).unwrap().agent_id = Some(1);

        let actions = vec![
            Action::Move(Direction::Right),
            Action::Move(Direction::Left),
        ];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            None,
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Moved(Position::new(3, 2)));
        assert_eq!(results[1], MoveResult::Moved(Position::new(7, 8)));
    }

    // ---- process_pushes tests ----

    fn make_test_object(id: u32, x: u16, y: u16) -> forge_types::Object {
        forge_types::Object {
            id,
            position: Position::new(x, y),
            object_type: forge_types::entity::ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: forge_types::entity::ObjectState::Active,
        }
    }

    #[test]
    fn test_push_object_right() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 6, 5)];
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        let actions = vec![Action::Push(Direction::Right)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        assert_eq!(objects[0].position, Position::new(7, 5));
        assert!(grid.get(6, 5).unwrap().object_id.is_none());
        assert_eq!(grid.get(7, 5).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_object_up() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 5, 4)];
        grid.get_mut(5, 4).unwrap().object_id = Some(0);

        let actions = vec![Action::Push(Direction::Up)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        assert_eq!(objects[0].position, Position::new(5, 3));
        assert!(grid.get(5, 4).unwrap().object_id.is_none());
        assert_eq!(grid.get(5, 3).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_object_down() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 5, 6)];
        grid.get_mut(5, 6).unwrap().object_id = Some(0);

        let actions = vec![Action::Push(Direction::Down)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        assert_eq!(objects[0].position, Position::new(5, 7));
        assert!(grid.get(5, 6).unwrap().object_id.is_none());
        assert_eq!(grid.get(5, 7).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_object_left() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 4, 5)];
        grid.get_mut(4, 5).unwrap().object_id = Some(0);

        let actions = vec![Action::Push(Direction::Left)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        assert_eq!(objects[0].position, Position::new(3, 5));
        assert!(grid.get(4, 5).unwrap().object_id.is_none());
        assert_eq!(grid.get(3, 5).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_against_grid_boundary() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 0, 1)];
        grid.get_mut(0, 1).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 0, 0)];
        grid.get_mut(0, 0).unwrap().object_id = Some(0);

        let actions = vec![Action::Push(Direction::Up)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        // Object should NOT have moved — destination is out of bounds
        assert_eq!(objects[0].position, Position::new(0, 0));
        assert_eq!(grid.get(0, 0).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_against_wall() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 6, 5)];
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Wall;

        let actions = vec![Action::Push(Direction::Right)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        // Object should NOT have moved (wall blocks)
        assert_eq!(objects[0].position, Position::new(6, 5));
        assert_eq!(grid.get(6, 5).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_against_water() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 6, 5)];
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Water;

        let actions = vec![Action::Push(Direction::Right)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        // Object should NOT have moved (water is impassable)
        assert_eq!(objects[0].position, Position::new(6, 5));
        assert_eq!(grid.get(6, 5).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_with_collision_disabled() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 6, 5)];
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        let mut config = default_physics();
        config.collision_enabled = false;

        let actions = vec![Action::Push(Direction::Right)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(&push_data, &mut grid, &mut objects, &actions, &config);

        // Should skip entirely when collision is disabled
        assert_eq!(objects[0].position, Position::new(6, 5));
        assert_eq!(grid.get(6, 5).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_no_object_at_position() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects: Vec<forge_types::Object> = Vec::new();

        let actions = vec![Action::Push(Direction::Right)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        // Nothing to push, no crash
        assert!(objects.is_empty());
    }

    #[test]
    fn test_push_dead_agent_skipped() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_test_agent(0, 5, 5)];
        agents[0].alive = false;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 6, 5)];
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        let actions = vec![Action::Push(Direction::Right)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        // Dead agent's push should be skipped
        assert_eq!(objects[0].position, Position::new(6, 5));
        assert_eq!(grid.get(6, 5).unwrap().object_id, Some(0));
    }

    #[test]
    fn test_push_blocked_by_another_object() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut objects = vec![make_test_object(0, 6, 5), make_test_object(1, 7, 5)];
        grid.get_mut(6, 5).unwrap().object_id = Some(0);
        grid.get_mut(7, 5).unwrap().object_id = Some(1);

        let actions = vec![Action::Push(Direction::Right)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        // Push should fail — destination has another object
        assert_eq!(objects[0].position, Position::new(6, 5));
    }

    #[test]
    fn test_push_blocked_by_agent_on_destination() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 5, 5), make_test_agent(1, 7, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(7, 5).unwrap().agent_id = Some(1);

        let mut objects = vec![make_test_object(0, 6, 5)];
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        let actions = vec![Action::Push(Direction::Right), Action::Noop];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        // Push should fail — destination tile has an agent
        assert_eq!(objects[0].position, Position::new(6, 5));
        assert_eq!(grid.get(6, 5).unwrap().object_id, Some(0));
    }

    // ---- Aerial movement tests ----

    fn make_aerial_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        let mut agent = Agent::new(id, Position::new(x, y), &config);
        agent.morphology = AgentMorphology::Aerial;
        agent.altitude = 3;
        agent.battery = 655360;
        agent.stamina = 655360;
        agent
    }

    fn default_drone_config() -> forge_types::config::DroneConfig {
        forge_types::config::DroneConfig {
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_airborne_ignores_wall_terrain() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 4).unwrap().terrain = TerrainType::Wall;
        let mut agents = [make_aerial_agent(0, 5, 5)];

        let dc = default_drone_config();
        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            Some(&dc),
            &topo(),
        );

        // Airborne agents should ignore wall terrain
        assert_eq!(results[0], MoveResult::Moved(Position::new(5, 4)));
    }

    #[test]
    fn test_airborne_uses_battery_not_stamina() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_aerial_agent(0, 5, 5)];
        let initial_stamina = agents[0].stamina;
        let initial_battery = agents[0].battery;

        let dc = default_drone_config();
        let actions = vec![Action::Move(Direction::Right)];
        process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            Some(&dc),
            &topo(),
        );

        assert_eq!(
            agents[0].stamina, initial_stamina,
            "stamina should be unchanged"
        );
        assert!(
            agents[0].battery < initial_battery,
            "battery should decrease"
        );
    }

    #[test]
    fn test_airborne_no_battery_blocks_movement() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_aerial_agent(0, 5, 5)];
        agents[0].battery = 0;

        let dc = default_drone_config();
        let actions = vec![Action::Move(Direction::Right)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            Some(&dc),
            &topo(),
        );

        assert_eq!(results[0], MoveResult::NoStamina);
    }

    #[test]
    fn test_aerial_collision_same_altitude() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_aerial_agent(0, 5, 5), make_aerial_agent(1, 5, 4)];
        // Both at altitude 3

        let dc = default_drone_config();
        let actions = vec![Action::Move(Direction::Up), Action::Noop];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            Some(&dc),
            &topo(),
        );

        assert_eq!(
            results[0],
            MoveResult::Blocked,
            "should collide at same altitude"
        );
    }

    #[test]
    fn test_aerial_no_collision_different_altitude() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_aerial_agent(0, 5, 5), make_aerial_agent(1, 5, 4)];
        agents[0].altitude = 3;
        agents[1].altitude = 5; // different altitude

        let dc = default_drone_config();
        let actions = vec![Action::Move(Direction::Up), Action::Noop];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            Some(&dc),
            &topo(),
        );

        assert_eq!(
            results[0],
            MoveResult::Moved(Position::new(5, 4)),
            "should pass through agent at different altitude"
        );
    }

    #[test]
    fn test_ground_vehicle_blocked_by_forest() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 4).unwrap().terrain = TerrainType::Forest;
        let mut agents = [make_test_agent(0, 5, 5)];
        agents[0].morphology = AgentMorphology::GroundVehicle;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let dc = default_drone_config();
        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            Some(&dc),
            &topo(),
        );

        assert_eq!(results[0], MoveResult::Impassable);
    }

    #[test]
    fn test_ground_vehicle_faster_on_ground_terrain() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = [make_test_agent(0, 5, 5)];
        agents[0].morphology = AgentMorphology::GroundVehicle;
        agents[0].stamina = 655360;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let initial_stamina = agents[0].stamina;
        let dc = default_drone_config();
        let actions = vec![Action::Move(Direction::Right)];
        process_movements(
            &mut agents,
            &mut grid,
            &actions,
            &default_physics(),
            Some(&dc),
            &topo(),
        );

        let vehicle_cost = initial_stamina - agents[0].stamina;

        // Now test ground agent same movement
        let mut grid2 = make_test_grid(16, 16);
        let mut agents2 = [make_test_agent(0, 5, 5)];
        agents2[0].stamina = 655360;
        grid2.get_mut(5, 5).unwrap().agent_id = Some(0);
        let actions2 = vec![Action::Move(Direction::Right)];
        process_movements(
            &mut agents2,
            &mut grid2,
            &actions2,
            &default_physics(),
            None,
            &topo(),
        );
        let ground_cost = initial_stamina - agents2[0].stamina;

        // Vehicle should be faster (lower cost) on ground terrain
        assert!(
            vehicle_cost < ground_cost,
            "vehicle cost {} should be less than ground cost {}",
            vehicle_cost,
            ground_cost
        );
    }

    #[test]
    fn test_push_agent_at_boundary_no_adjacent_tile() {
        let mut grid = make_test_grid(16, 16);
        let agents = [make_test_agent(0, 15, 5)];
        grid.get_mut(15, 5).unwrap().agent_id = Some(0);

        let mut objects: Vec<forge_types::Object> = Vec::new();

        let actions = vec![Action::Push(Direction::Right)];
        let push_data: Vec<AgentPushData> = agents.iter().map(AgentPushData::from_agent).collect();
        process_pushes(
            &push_data,
            &mut grid,
            &mut objects,
            &actions,
            &default_physics(),
        );

        // No crash, push_from is out of bounds so nothing happens
        assert!(objects.is_empty());
    }

    // ---- Proptest: physics invariants ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_direction() -> impl Strategy<Value = Direction> {
            prop_oneof![
                Just(Direction::Up),
                Just(Direction::Down),
                Just(Direction::Left),
                Just(Direction::Right),
            ]
        }

        proptest! {
            /// Movement is deterministic: same setup produces same result.
            #[test]
            fn movement_determinism(
                x in 1u16..14,
                y in 1u16..14,
                dir in arb_direction(),
            ) {
                let run = || {
                    let mut grid = make_test_grid(16, 16);
                    let mut agents = [make_test_agent(0, x, y)];
                    grid.get_mut(x, y).unwrap().agent_id = Some(0);
                    let actions = vec![Action::Move(dir)];
                    let results = process_movements(
                        &mut agents, &mut grid, &actions, &default_physics(), None, &topo(),
                    );
                    (results, agents[0].position, agents[0].stamina)
                };
                let (r1, p1, s1) = run();
                let (r2, p2, s2) = run();
                prop_assert_eq!(r1, r2);
                prop_assert_eq!(p1, p2);
                prop_assert_eq!(s1, s2);
            }

            /// After movement, agent is always within grid bounds.
            #[test]
            fn agent_stays_in_bounds(
                x in 0u16..16,
                y in 0u16..16,
                dir in arb_direction(),
            ) {
                let mut grid = make_test_grid(16, 16);
                let mut agents = [make_test_agent(0, x, y)];
                grid.get_mut(x, y).unwrap().agent_id = Some(0);
                let actions = vec![Action::Move(dir)];
                process_movements(&mut agents, &mut grid, &actions, &default_physics(), None, &topo());
                prop_assert!(agents[0].position.x < 16);
                prop_assert!(agents[0].position.y < 16);
            }

            /// Stamina never goes negative after movement.
            #[test]
            fn stamina_non_negative(
                stamina in 0i32..1_000_000,
                dir in arb_direction(),
            ) {
                let mut grid = make_test_grid(16, 16);
                let mut agents = [make_test_agent(0, 8, 8)];
                agents[0].stamina = stamina;
                grid.get_mut(8, 8).unwrap().agent_id = Some(0);
                let actions = vec![Action::Move(dir)];
                process_movements(&mut agents, &mut grid, &actions, &default_physics(), None, &topo());
                prop_assert!(agents[0].stamina >= 0);
            }

            /// Stamina regen never exceeds max.
            #[test]
            fn stamina_regen_capped(
                stamina in 0i32..1_000_000,
                max in 100_000i32..2_000_000,
            ) {
                let config = default_physics();
                let mut agents = [make_test_agent(0, 5, 5)];
                agents[0].stamina = stamina;
                regenerate_stamina(&mut agents, &config, max);
                prop_assert!(agents[0].stamina <= max);
            }
        }
    }
}
