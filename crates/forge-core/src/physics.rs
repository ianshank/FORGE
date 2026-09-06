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

/// Projects a hex direction onto the nearest cardinal heading.
///
/// This preserves compatibility with the existing `Agent.heading: Direction`
/// field until heading becomes topology-aware.
#[inline]
fn heading_from_hex(dir: HexDirection) -> Direction {
    match dir {
        HexDirection::NE | HexDirection::NW => Direction::Up,
        HexDirection::E => Direction::Right,
        HexDirection::SE | HexDirection::SW => Direction::Down,
        HexDirection::W => Direction::Left,
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
    /// Snapshot of agent `(position, altitude, alive)` taken at the start
    /// of the movement pass. Aerial collision detection scans this slice
    /// while the third pass mutates `agents`, so a snapshot is required;
    /// reusing the buffer keeps the step zero-allocation at all agent
    /// counts (a per-step `SmallVec` here previously spilled to the heap
    /// once `num_agents` exceeded its inline capacity).
    pub(crate) agents_snapshot: Vec<(Position, u8, bool)>,
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
        if self.agents_snapshot.capacity() < agent_count {
            self.agents_snapshot
                .reserve(agent_count - self.agents_snapshot.capacity());
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
    scratch.agents_snapshot.clear();
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

    // Snapshot agent positions and altitudes for aerial collision detection.
    // The snapshot lives in `scratch.agents_snapshot` and is reused across
    // ticks via `ensure_capacity`, so this remains zero-allocation on the
    // hot path regardless of `num_agents` (the previous local SmallVec
    // spilled to heap once the count exceeded its inline capacity).
    scratch
        .agents_snapshot
        .extend(agents.iter().map(|a| (a.position, a.altitude, a.alive)));

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
        let target_tile = grid
            .get(target.x, target.y)
            .expect("invariant: target position is bounds-checked by neighbor()");

        if is_airborne {
            // Airborne Aerial: ignore ground terrain, only collide with agents at same altitude
            if config.collision_enabled {
                // Check for aerial collision at same altitude (scan agents, not grid).
                // Reads `scratch.agents_snapshot` directly so the borrow stays
                // disjoint from `scratch.results` writes below.
                let aerial_conflict =
                    scratch
                        .agents_snapshot
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

        // Update heading for successful movement actions.
        // Hex headings are projected to cardinal directions for compatibility
        // with the current `Direction`-typed heading field.
        match action {
            Action::Move(dir) => {
                agent.heading = *dir;
            }
            Action::MoveHex(dir) => {
                agent.heading = heading_from_hex(*dir);
            }
            _ => {}
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
#[path = "physics/tests.rs"]
mod tests;
