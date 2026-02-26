//! Physics system for the FORGE simulation.
//!
//! Handles movement, collision detection, stamina costs, and projectile
//! trajectories. Uses the grid directly for position tracking.
//! All arithmetic is integer-based for determinism.

use forge_types::config::PhysicsConfig;
use forge_types::entity::Agent;
use forge_types::grid::{Grid, Position};
use forge_types::Action;
use tracing::{trace, warn};

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

/// Processes movement actions for all agents.
///
/// This is the core physics step. It validates moves, checks collisions,
/// applies stamina costs, and updates agent positions on the grid.
///
/// Movement is processed in agent order (agent 0 first). Ties in movement
/// to the same tile are resolved by agent priority (lower ID wins).
pub fn process_movements(
    agents: &mut [Agent],
    grid: &mut Grid,
    actions: &[Action],
    config: &PhysicsConfig,
) -> Vec<MoveResult> {
    let mut results = Vec::with_capacity(agents.len());

    // First pass: compute desired positions
    let desired_positions: Vec<Option<Position>> = agents
        .iter()
        .zip(actions.iter())
        .map(|(agent, action)| {
            if !agent.alive {
                return None;
            }
            match action {
                Action::Move(dir) => agent.position.offset(*dir, grid.width, grid.height),
                _ => None,
            }
        })
        .collect();

    // Second pass: detect conflicts (two agents wanting same tile)
    let mut occupied_targets: Vec<(usize, Position)> = Vec::new();
    for (i, pos) in desired_positions.iter().enumerate() {
        if let Some(p) = pos {
            occupied_targets.push((i, *p));
        }
    }

    // Third pass: apply movements
    for (i, agent) in agents.iter_mut().enumerate() {
        if !agent.alive {
            results.push(MoveResult::Blocked);
            continue;
        }

        let action = &actions[i];
        let direction = match action {
            Action::Move(dir) => *dir,
            _ => {
                results.push(MoveResult::Blocked);
                continue;
            }
        };

        // Check stamina
        if !config.collision_enabled {
            // Simplified mode without collision
        }

        let stamina_cost = config.stamina_cost_move;
        if agent.stamina < stamina_cost {
            trace!(agent_id = agent.id, "movement blocked: no stamina");
            results.push(MoveResult::NoStamina);
            continue;
        }

        // Get target position
        let target = match agent.position.offset(direction, grid.width, grid.height) {
            Some(pos) => pos,
            None => {
                trace!(
                    agent_id = agent.id,
                    ?direction,
                    "movement blocked: boundary"
                );
                results.push(MoveResult::Blocked);
                continue;
            }
        };

        // Check terrain walkability
        let target_tile = grid.get(target.x, target.y).unwrap();
        if !target_tile.terrain.is_walkable() {
            trace!(
                agent_id = agent.id,
                terrain = ?target_tile.terrain,
                "movement blocked: impassable terrain"
            );
            results.push(MoveResult::Impassable);
            continue;
        }

        // Check collision with other agents
        if config.collision_enabled {
            if let Some(occupant) = target_tile.agent_id {
                if occupant != agent.id {
                    trace!(
                        agent_id = agent.id,
                        occupant_id = occupant,
                        "movement blocked: tile occupied"
                    );
                    results.push(MoveResult::Blocked);
                    continue;
                }
            }

            // Check if another agent (with higher priority) is also moving here
            let conflict = occupied_targets
                .iter()
                .any(|(other_idx, other_pos)| *other_idx < i && *other_pos == target);
            if conflict {
                trace!(
                    agent_id = agent.id,
                    "movement blocked: conflict with higher priority agent"
                );
                results.push(MoveResult::Blocked);
                continue;
            }
        }

        // Apply terrain movement cost
        let terrain_cost = target_tile.terrain.movement_cost();
        let total_cost = if terrain_cost == i32::MAX {
            stamina_cost
        } else {
            // Multiply base cost by terrain multiplier (both fixed-point)
            // Fixed-point multiply: (a * b) >> 16
            ((stamina_cost as i64 * terrain_cost as i64) >> 16) as i32
        };

        // Deduct stamina
        agent.stamina = (agent.stamina - total_cost).max(0);

        // Clear old position on grid
        if let Some(tile) = grid.get_mut(agent.position.x, agent.position.y) {
            if tile.agent_id == Some(agent.id) {
                tile.agent_id = None;
            }
        }

        // Update agent position
        let old_pos = agent.position;
        agent.position = target;

        // Set new position on grid
        if let Some(tile) = grid.get_mut(target.x, target.y) {
            tile.agent_id = Some(agent.id);
        }

        trace!(
            agent_id = agent.id,
            old_x = old_pos.x,
            old_y = old_pos.y,
            new_x = target.x,
            new_y = target.y,
            stamina_cost = total_cost,
            "agent moved"
        );
        results.push(MoveResult::Moved(target));
    }

    results
}

/// Regenerates stamina for all agents.
pub fn regenerate_stamina(agents: &mut [Agent], config: &PhysicsConfig, max_stamina: i32) {
    for agent in agents.iter_mut() {
        if !agent.alive {
            continue;
        }
        agent.stamina = (agent.stamina + config.stamina_regen_rate).min(max_stamina);
    }
}

/// Processes push actions — agents pushing objects.
pub fn process_pushes(
    agents: &[Agent],
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
            let agent = &agents[i];
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
    use forge_types::config::AgentConfig;
    use forge_types::grid::{Direction, TerrainType};

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
        let mut agents = vec![make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

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
            let mut agents = vec![make_test_agent(0, 5, 5)];
            grid.get_mut(5, 5).unwrap().agent_id = Some(0);

            let actions = vec![Action::Move(dir)];
            let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

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
        let mut agents = vec![make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

        assert_eq!(results[0], MoveResult::Impassable);
        assert_eq!(agents[0].position, Position::new(5, 5)); // didn't move
    }

    #[test]
    fn test_collision_with_boundary() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 0, 0)];
        grid.get_mut(0, 0).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

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
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

        assert_eq!(results[0], MoveResult::Blocked);
        assert_eq!(agents[0].position, Position::new(5, 5));
    }

    #[test]
    fn test_movement_drains_stamina() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let initial_stamina = agents[0].stamina;
        let actions = vec![Action::Move(Direction::Up)];
        process_movements(&mut agents, &mut grid, &actions, &default_physics());

        assert!(agents[0].stamina < initial_stamina);
    }

    #[test]
    fn test_no_stamina_blocks_movement() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].stamina = 0;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

        assert_eq!(results[0], MoveResult::NoStamina);
    }

    #[test]
    fn test_stamina_regeneration() {
        let config = default_physics();
        let agent_config = AgentConfig::default();
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].stamina = 100000;

        regenerate_stamina(&mut agents, &config, agent_config.max_stamina);
        assert!(agents[0].stamina > 100000);
    }

    #[test]
    fn test_stamina_regen_capped() {
        let config = default_physics();
        let max = 655360;
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].stamina = max;

        regenerate_stamina(&mut agents, &config, max);
        assert_eq!(agents[0].stamina, max);
    }

    #[test]
    fn test_dead_agent_cannot_move() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].alive = false;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

        assert_eq!(results[0], MoveResult::Blocked);
        assert_eq!(agents[0].position, Position::new(5, 5));
    }

    #[test]
    fn test_noop_no_movement() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let initial_stamina = agents[0].stamina;
        let actions = vec![Action::Noop];
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

        assert_eq!(results[0], MoveResult::Blocked);
        assert_eq!(agents[0].position, Position::new(5, 5));
        assert_eq!(agents[0].stamina, initial_stamina); // no stamina cost for noop
    }

    #[test]
    fn test_terrain_movement_cost() {
        // Forest costs 2x stamina
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 4).unwrap().terrain = TerrainType::Forest;
        let mut agents = vec![make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let initial_stamina = agents[0].stamina;
        let actions = vec![Action::Move(Direction::Up)];
        process_movements(&mut agents, &mut grid, &actions, &default_physics());

        let stamina_used = initial_stamina - agents[0].stamina;

        // Now test normal ground
        let mut grid2 = make_test_grid(16, 16);
        let mut agents2 = vec![make_test_agent(0, 5, 5)];
        agents2[0].stamina = initial_stamina;
        grid2.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions2 = vec![Action::Move(Direction::Up)];
        process_movements(&mut agents2, &mut grid2, &actions2, &default_physics());

        let normal_stamina_used = initial_stamina - agents2[0].stamina;

        // Forest should cost more
        assert!(stamina_used > normal_stamina_used);
    }

    #[test]
    fn test_water_impassable() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 4).unwrap().terrain = TerrainType::Water;
        let mut agents = vec![make_test_agent(0, 5, 5)];
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(Direction::Up)];
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

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
        let results = process_movements(&mut agents, &mut grid, &actions, &default_physics());

        assert_eq!(results[0], MoveResult::Moved(Position::new(3, 2)));
        assert_eq!(results[1], MoveResult::Moved(Position::new(7, 8)));
    }
}
