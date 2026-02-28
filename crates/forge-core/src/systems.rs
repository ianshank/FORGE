//! System runner for the FORGE simulation step pipeline.
//!
//! Systems are run in a fixed deterministic order each tick.
//! This module coordinates the execution of all systems.

use forge_types::entity::ObjectType;
use forge_types::grid::Direction;
use forge_types::Action;
use tracing::{debug, instrument, trace, warn};

use crate::combat;
use crate::communication;
use crate::crafting;
use crate::day_night;
use crate::physics;
use crate::resource;
use crate::visibility;
use crate::world::WorldState;

/// Runs all simulation systems for a single tick.
///
/// Systems execute in this fixed order:
/// 1. Validate actions
/// 2. Physics (movement, collision, push)
/// 3. Stamina regeneration
/// 4. Resource system (harvest, deplete, respawn) — Phase 1
/// 5. Crafting system — Phase 1
/// 6. Combat system — Phase 1
/// 7. Communication system — Phase 3
/// 8. Day/night system — Phase 3 (before visibility so phase affects vision)
/// 9. Visibility system — Phase 3 (applies day/night vision modifier)
/// 10. Task system — Phase 4
/// 11. Generate observations
#[instrument(skip_all)]
pub fn run_systems(state: &mut WorldState, actions: &[Action]) {
    trace!(tick = state.tick, "running systems");

    // 1. Validate actions (replace invalid actions with Noop)
    let validated_actions = validate_actions(actions, state);

    // 2. Physics: movement and collision
    let _move_results = physics::process_movements(
        &mut state.agents,
        &mut state.grid,
        &validated_actions,
        &state.config.physics,
    );

    // 2b. Physics: push processing — extract minimal data to avoid cloning
    let push_data: smallvec::SmallVec<[physics::AgentPushData; 8]> = state
        .agents
        .iter()
        .map(physics::AgentPushData::from_agent)
        .collect();
    physics::process_pushes(
        &push_data,
        &mut state.grid,
        &mut state.objects,
        &validated_actions,
        &state.config.physics,
    );

    // 3. Stamina regeneration
    physics::regenerate_stamina(
        &mut state.agents,
        &state.config.physics,
        state.config.agents.max_stamina,
    );

    // 4. Resource system: harvesting and respawn
    resource::process_harvesting(
        &mut state.agents,
        &state.grid,
        &mut state.resources,
        &validated_actions,
    );
    resource::tick_respawn(&mut state.resources);

    // 5. Crafting system
    let near_station = compute_near_station(&state.agents, &state.grid, &state.objects);
    crafting::process_crafting(
        &mut state.agents,
        &validated_actions,
        &state.recipe_book,
        &near_station,
    );

    // 6. Combat system
    combat::process_combat(&mut state.agents, &state.grid, &validated_actions);
    combat::apply_environmental_damage(&mut state.agents, &state.grid);

    // 7. Communication system
    communication::process_communication(
        &mut state.agents,
        &validated_actions,
        &state.config.agents,
    );

    // 8. Day/night system (compute before visibility so phase affects vision range)
    state.day_phase = day_night::compute_day_phase(state.tick, &state.config.world);

    // 9. Visibility system (applies day/night vision modifier)
    visibility::update_visibility(&state.agents, &mut state.grid, state.day_phase);

    // 10. Task evaluation and reward computation
    if !state.tasks.is_empty() {
        let task_result = forge_task::evaluator::evaluate_tasks(
            &mut state.tasks,
            &state.agents,
            state.tick,
            state.config.task.reward_scale,
            &[],
            Some(&state.grid),
            Some(&state.objects),
            state.config.agents.max_health,
        );
        state.last_task_rewards = Some(task_result.rewards);
        if task_result.should_terminate {
            debug!(tick = state.tick, "task system triggered termination");
            state.terminated = true;
        }
    }

    // Increment tick
    state.tick += 1;

    trace!(tick = state.tick, "systems complete");
}

/// Computes per-agent boolean indicating whether each agent is adjacent to
/// or standing on a tile containing a CraftingStation object.
#[instrument(skip_all)]
fn compute_near_station(
    agents: &[forge_types::entity::Agent],
    grid: &forge_types::grid::Grid,
    objects: &[forge_types::entity::Object],
) -> Vec<bool> {
    agents
        .iter()
        .map(|agent| {
            if !agent.alive {
                return false;
            }

            // Check current tile and all 4 adjacent tiles
            let positions_to_check = std::iter::once(agent.position).chain(
                Direction::all()
                    .into_iter()
                    .filter_map(|dir| agent.position.offset(dir, grid.width, grid.height)),
            );

            for pos in positions_to_check {
                if let Some(tile) = grid.get(pos.x, pos.y) {
                    if let Some(obj_id) = tile.object_id {
                        if let Some(obj) = objects.iter().find(|o| o.id == obj_id) {
                            if obj.object_type == ObjectType::CraftingStation {
                                return true;
                            }
                        }
                    }
                }
            }

            false
        })
        .collect()
}

/// Validates actions and replaces invalid ones with Noop.
#[instrument(skip_all)]
fn validate_actions(actions: &[Action], state: &WorldState) -> Vec<Action> {
    actions
        .iter()
        .enumerate()
        .map(|(i, action)| {
            if i >= state.agents.len() {
                return Action::Noop;
            }

            let agent = &state.agents[i];
            if !agent.alive {
                return Action::Noop;
            }

            match action {
                Action::Move(dir) => {
                    // Basic validation: direction is valid (always true for enum)
                    Action::Move(*dir)
                }
                Action::Communicate(token) => {
                    let vocab_size = state.config.agents.comm_vocab_size;
                    if *token < vocab_size {
                        Action::Communicate(*token)
                    } else {
                        warn!(
                            agent_id = agent.id,
                            token, vocab_size, "comm token out of range, falling back to Noop"
                        );
                        Action::Noop
                    }
                }
                Action::Drop(slot) => {
                    if (*slot as usize) < agent.inventory.capacity() {
                        Action::Drop(*slot)
                    } else {
                        warn!(
                            agent_id = agent.id,
                            slot,
                            capacity = agent.inventory.capacity(),
                            "drop slot out of range, falling back to Noop"
                        );
                        Action::Noop
                    }
                }
                Action::Use(slot) => {
                    if (*slot as usize) < agent.inventory.capacity() {
                        Action::Use(*slot)
                    } else {
                        warn!(
                            agent_id = agent.id,
                            slot,
                            capacity = agent.inventory.capacity(),
                            "use slot out of range, falling back to Noop"
                        );
                        Action::Noop
                    }
                }
                _ => action.clone(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;
    use forge_types::grid::Position;

    #[test]
    fn test_validate_actions_dead_agent() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let mut state = WorldState::new(config).unwrap();
        state.agents[0].alive = false;

        let actions = vec![Action::Move(forge_types::Direction::Up)];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Noop);
    }

    #[test]
    fn test_validate_actions_excess_actions() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        let actions = vec![Action::Noop, Action::Noop, Action::Noop]; // 3 actions for 1 agent
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated.len(), 3);
        // Extra actions are noop
        assert_eq!(validated[1], Action::Noop);
        assert_eq!(validated[2], Action::Noop);
    }

    #[test]
    fn test_systems_increment_tick() {
        let config = ForgeConfig::default();
        let mut state = WorldState::new(config).unwrap();
        let initial_tick = state.tick;

        let actions = vec![Action::Noop];
        run_systems(&mut state, &actions);

        assert_eq!(state.tick, initial_tick + 1);
    }

    #[test]
    fn test_systems_movement() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.agents.num_agents = 1;
        let mut state = WorldState::new(config).unwrap();

        // Place agent at (5, 5)
        let start_pos = Position::new(5, 5);
        state.agents[0].position = start_pos;
        state.grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(forge_types::Direction::Right)];
        run_systems(&mut state, &actions);

        assert_eq!(state.agents[0].position, Position::new(6, 5));
    }

    // ---- compute_near_station tests ----

    fn make_station_object(id: u32, x: u16, y: u16) -> forge_types::Object {
        forge_types::Object {
            id,
            position: Position::new(x, y),
            object_type: ObjectType::CraftingStation,
            mass: 65536,
            durability: 655360,
            state: forge_types::entity::ObjectState::Active,
        }
    }

    fn make_agent_at(id: u32, x: u16, y: u16) -> forge_types::entity::Agent {
        let config = forge_types::config::AgentConfig::default();
        forge_types::entity::Agent::new(id, Position::new(x, y), &config)
    }

    #[test]
    fn test_near_station_adjacent_right() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);
        let station = make_station_object(0, 6, 5);
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[station]);
        assert!(result[0], "agent should be near station to the right");
    }

    #[test]
    fn test_near_station_adjacent_left() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);
        let station = make_station_object(0, 4, 5);
        grid.get_mut(4, 5).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[station]);
        assert!(result[0], "agent should be near station to the left");
    }

    #[test]
    fn test_near_station_adjacent_up() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);
        let station = make_station_object(0, 5, 4);
        grid.get_mut(5, 4).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[station]);
        assert!(result[0], "agent should be near station above");
    }

    #[test]
    fn test_near_station_adjacent_down() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);
        let station = make_station_object(0, 5, 6);
        grid.get_mut(5, 6).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[station]);
        assert!(result[0], "agent should be near station below");
    }

    #[test]
    fn test_near_station_same_tile() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);
        let station = make_station_object(0, 5, 5);
        grid.get_mut(5, 5).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[station]);
        assert!(result[0], "agent should be near station on same tile");
    }

    #[test]
    fn test_near_station_no_objects() {
        let grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);

        let result = compute_near_station(&[agent], &grid, &[]);
        assert!(!result[0], "no objects means not near any station");
    }

    #[test]
    fn test_near_station_non_station_object() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);
        // Boulder is not a CraftingStation
        let boulder = forge_types::Object {
            id: 0,
            position: Position::new(6, 5),
            object_type: forge_types::entity::ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: forge_types::entity::ObjectState::Active,
        };
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[boulder]);
        assert!(!result[0], "boulder is not a crafting station");
    }

    #[test]
    fn test_near_station_at_grid_boundary() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        // Agent at (0,0) — Up and Left offsets go out of bounds
        let agent = make_agent_at(0, 0, 0);
        let station = make_station_object(0, 1, 0);
        grid.get_mut(1, 0).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[station]);
        assert!(result[0], "agent at corner should see adjacent station");
    }

    #[test]
    fn test_near_station_dead_agent() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let mut agent = make_agent_at(0, 5, 5);
        agent.alive = false;
        let station = make_station_object(0, 6, 5);
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[station]);
        assert!(!result[0], "dead agent should not be near station");
    }

    #[test]
    fn test_near_station_diagonal_not_adjacent() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);
        // Station at (6,6) — diagonal, not cardinal adjacent
        let station = make_station_object(0, 6, 6);
        grid.get_mut(6, 6).unwrap().object_id = Some(0);

        let result = compute_near_station(&[agent], &grid, &[station]);
        assert!(
            !result[0],
            "diagonal is not adjacent in cardinal directions"
        );
    }

    // ---- validate_actions additional tests ----

    #[test]
    fn test_validate_actions_empty() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        let actions: Vec<Action> = vec![];
        let validated = validate_actions(&actions, &state);
        assert!(validated.is_empty());
    }

    #[test]
    fn test_validate_actions_mixed_valid_invalid() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 2;
        config.agents.comm_vocab_size = 10;
        let mut state = WorldState::new(config).unwrap();
        state.agents[1].alive = false;

        let actions = vec![
            Action::Move(forge_types::Direction::Up),   // valid
            Action::Move(forge_types::Direction::Down), // invalid: dead agent
        ];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Move(forge_types::Direction::Up));
        assert_eq!(validated[1], Action::Noop);
    }

    #[test]
    fn test_validate_actions_invalid_drop_slot() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        // carry_capacity is 10 by default, so slot 10 is out of range
        let actions = vec![Action::Drop(10)];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Noop);
    }

    #[test]
    fn test_validate_actions_valid_drop_slot() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        // slot 0 should be valid
        let actions = vec![Action::Drop(0)];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Drop(0));
    }

    #[test]
    fn test_validate_actions_invalid_use_slot() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        // carry_capacity is 10, slot 255 is out of range
        let actions = vec![Action::Use(255)];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Noop);
    }

    #[test]
    fn test_validate_actions_valid_use_slot() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        let actions = vec![Action::Use(9)];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Use(9));
    }

    #[test]
    fn test_validate_actions_invalid_comm_token() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 10;
        let state = WorldState::new(config).unwrap();

        // Token 10 is beyond vocab_size of 10 (valid: 0..9)
        let actions = vec![Action::Communicate(10)];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Noop);
    }

    #[test]
    fn test_validate_actions_valid_comm_token() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 10;
        let state = WorldState::new(config).unwrap();

        let actions = vec![Action::Communicate(9)];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Communicate(9));
    }

    #[test]
    fn test_validate_actions_noop_passthrough() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        let actions = vec![Action::Noop];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Noop);
    }

    #[test]
    fn test_validate_actions_pickup_passthrough() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        let actions = vec![Action::PickUp];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::PickUp);
    }
}
