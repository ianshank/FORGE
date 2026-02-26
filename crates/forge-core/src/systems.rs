//! System runner for the FORGE simulation step pipeline.
//!
//! Systems are run in a fixed deterministic order each tick.
//! This module coordinates the execution of all systems.

use forge_types::entity::ObjectType;
use forge_types::grid::Direction;
use forge_types::Action;
use tracing::trace;

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
/// 8. Visibility system — Phase 3
/// 9. Day/night system — Phase 3
/// 10. Task system — Phase 4
/// 11. Generate observations
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

    // 2b. Physics: push processing
    physics::process_pushes(
        &state.agents.clone(), // TODO: avoid clone in Phase optimization
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

    // 8. Visibility system
    visibility::update_visibility(&state.agents, &mut state.grid);

    // 9. Day/night system
    state.day_phase = day_night::compute_day_phase(state.tick, &state.config.world);

    // Increment tick
    state.tick += 1;

    trace!(tick = state.tick, "systems complete");
}

/// Computes per-agent boolean indicating whether each agent is adjacent to
/// or standing on a tile containing a CraftingStation object.
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
                        Action::Noop
                    }
                }
                Action::Drop(slot) => {
                    if (*slot as usize) < agent.inventory.capacity() {
                        Action::Drop(*slot)
                    } else {
                        Action::Noop
                    }
                }
                Action::Use(slot) => {
                    if (*slot as usize) < agent.inventory.capacity() {
                        Action::Use(*slot)
                    } else {
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
        let mut state = WorldState::new(config);
        state.agents[0].alive = false;

        let actions = vec![Action::Move(forge_types::Direction::Up)];
        let validated = validate_actions(&actions, &state);
        assert_eq!(validated[0], Action::Noop);
    }

    #[test]
    fn test_validate_actions_excess_actions() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config);

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
        let mut state = WorldState::new(config);
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
        let mut state = WorldState::new(config);

        // Place agent at (5, 5)
        let start_pos = Position::new(5, 5);
        state.agents[0].position = start_pos;
        state.grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let actions = vec![Action::Move(forge_types::Direction::Right)];
        run_systems(&mut state, &actions);

        assert_eq!(state.agents[0].position, Position::new(6, 5));
    }
}
