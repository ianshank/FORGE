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
/// Reads padded actions from `state.step_actions` (populated by
/// [`WorldState::step_into`]) and writes validated actions into
/// `state.validated_actions`. Every per-tick scratch buffer required by
/// the pipeline lives on `WorldState`, so this function performs no heap
/// allocations on the hot path.
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
#[instrument(skip_all)]
pub fn run_systems(state: &mut WorldState) {
    trace!(tick = state.tick, "running systems");

    // 1. Validate actions (replace invalid actions with Noop). Reads
    //    from `state.step_actions`, writes into `state.validated_actions`.
    validate_actions_into(state);

    // 2. Physics: movement and collision (uses pre-allocated scratch buffers)
    let drone_config_ref = if state.config.drone.enabled {
        Some(&state.config.drone)
    } else {
        None
    };
    physics::process_movements_with_scratch(
        &mut state.agents,
        &mut state.grid,
        &state.validated_actions,
        &state.config.physics,
        drone_config_ref,
        &mut state.physics_scratch,
        &state.topology,
    );

    // 2b. Physics: push processing — extract minimal data to avoid cloning.
    //     SmallVec inline capacity 8 means up to 8 agents allocates nothing.
    let push_data: smallvec::SmallVec<[physics::AgentPushData; 8]> = state
        .agents
        .iter()
        .map(physics::AgentPushData::from_agent)
        .collect();
    physics::process_pushes(
        &push_data,
        &mut state.grid,
        &mut state.objects,
        &state.validated_actions,
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
        &state.validated_actions,
    );
    resource::tick_respawn(&mut state.resources);

    // 5. Crafting system. `compute_near_station_into` writes per-agent
    //    flags into `state.near_station` and uses `state.crafting_object_map`
    //    as the lookup scratch.
    compute_near_station_into(state);
    crafting::process_crafting(
        &mut state.agents,
        &state.validated_actions,
        &state.recipe_book,
        &state.near_station,
    );

    // 6. Combat system
    combat::process_combat(
        &mut state.agents,
        &state.grid,
        &state.validated_actions,
        &state.topology,
    );
    combat::apply_environmental_damage(&mut state.agents, &state.grid);

    // 6b. Drone systems (altitude, battery, payload) — only when enabled
    if state.config.drone.enabled {
        crate::drone::process_altitude_changes(
            &mut state.agents,
            &state.validated_actions,
            &state.config.drone,
        );
        crate::drone::process_battery_drain(&mut state.agents, &state.config.drone);
        crate::drone::process_battery_recharge(&mut state.agents, &state.config.drone);
        crate::drone::process_payload_drops(
            &mut state.agents,
            &mut state.grid,
            &state.validated_actions,
            &state.config.drone,
        );
    }

    // 6c. Agricultural systems — only when enabled
    if state.config.agri.enabled {
        crate::agriculture::process_crop_growth(
            &mut state.crop_states,
            &state.grid,
            &state.config.agri,
            state.tick,
            &mut state.agri_scratch.disease_spread_candidates,
            &state.topology,
        );
        crate::agriculture::process_spraying(
            &mut state.agents,
            &mut state.crop_states,
            &state.grid,
            &state.validated_actions,
            &state.config.agri,
        );
        crate::agriculture::process_multispectral_scan(
            &mut state.agents,
            &mut state.crop_states,
            &state.grid,
            &state.validated_actions,
            &state.config.agri,
            state.tick,
            &mut state.agri_scratch.scan_results,
        );
        crate::agriculture::process_thermal_scan(
            &mut state.agents,
            &state.crop_states,
            &state.grid,
            &state.validated_actions,
            &state.config.agri,
            &mut state.agri_scratch.scan_results,
        );
        crate::agriculture::process_soil_relay(
            &state.agents,
            &mut state.soil_nodes,
            &state.validated_actions,
            &state.config.agri,
            state.tick,
            &mut state.agri_scratch.soil_readings,
        );

        // Report generation: reuse the per-agent flag buffer carried on
        // `AgriScratch` so the agri pipeline doesn't allocate per tick.
        state.agri_scratch.report_flags.clear();
        state
            .agri_scratch
            .report_flags
            .resize(state.agents.len(), false);
        crate::agriculture::process_report_generation(
            &mut state.agents,
            &state.validated_actions,
            &state.config.agri,
            &mut state.agri_scratch.report_flags,
        );
    }

    // 7. Communication system. `comm_messages` is the per-tick scratch
    //    queue; `process_communication` clears it on entry and refills it.
    communication::process_communication(
        &mut state.agents,
        &state.validated_actions,
        &state.config.agents,
        &mut state.comm_messages,
    );

    // 8. Day/night system (compute before visibility so phase affects vision range)
    state.day_phase = day_night::compute_day_phase(state.tick, &state.config.world);

    // 9. Visibility system (applies day/night vision modifier)
    visibility::update_visibility(
        &state.agents,
        &mut state.grid,
        state.day_phase,
        &state.topology,
    );

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

/// Core logic for [`compute_near_station_into`], split out so tests can
/// drive it without constructing a full [`WorldState`].
///
/// Reuses the supplied `near` and `map` buffers — `near.clear()` first,
/// then pushes one bool per agent; `map.clear()` then refills with
/// `(object.id -> object.object_type)`.
fn compute_near_station_buf(
    agents: &[forge_types::entity::Agent],
    grid: &forge_types::grid::Grid,
    objects: &[forge_types::entity::Object],
    near: &mut Vec<bool>,
    map: &mut std::collections::HashMap<u32, ObjectType>,
) {
    near.clear();
    near.reserve(agents.len());
    map.clear();
    for o in objects {
        map.insert(o.id, o.object_type);
    }

    for agent in agents {
        if !agent.alive {
            near.push(false);
            continue;
        }

        // Check current tile and all 4 adjacent tiles
        let positions_to_check = std::iter::once(agent.position).chain(
            Direction::all()
                .into_iter()
                .filter_map(|dir| agent.position.offset(dir, grid.width, grid.height)),
        );

        let mut found_station = false;
        for pos in positions_to_check {
            if let Some(tile) = grid.get(pos.x, pos.y) {
                if let Some(obj_id) = tile.object_id {
                    if let Some(obj_type) = map.get(&obj_id) {
                        if *obj_type == ObjectType::CraftingStation {
                            found_station = true;
                            break;
                        }
                    }
                }
            }
        }

        near.push(found_station);
    }
}

/// Computes per-agent flags for "agent is adjacent to or on a CraftingStation".
///
/// Writes results into `state.near_station`, reusing the buffer allocated
/// on `WorldState`. Uses `state.crafting_object_map` as a transient
/// `ObjectId -> ObjectType` lookup so the per-tick `HashMap::new()` and
/// `Vec::with_capacity` allocations from earlier revisions are gone.
#[instrument(skip_all)]
fn compute_near_station_into(state: &mut WorldState) {
    compute_near_station_buf(
        &state.agents,
        &state.grid,
        &state.objects,
        &mut state.near_station,
        &mut state.crafting_object_map,
    );
}

/// Validates the padded actions in `state.step_actions` and writes the
/// per-agent validated form into `state.validated_actions`.
///
/// Always produces exactly one action per agent — out-of-range slot/token
/// values fall back to `Noop`. Reuses the `validated_actions` buffer on
/// `WorldState` so the previous per-tick `Vec::with_capacity` allocation
/// is gone.
#[instrument(skip_all)]
fn validate_actions_into(state: &mut WorldState) {
    state.validated_actions.clear();
    state.validated_actions.reserve(state.agents.len());

    for (i, agent) in state.agents.iter().enumerate() {
        // Get action for this agent, default to Noop if not provided
        let action = if let Some(a) = state.step_actions.get(i) {
            a
        } else {
            trace!(
                agent_id = agent.id,
                agent_idx = i,
                "no action provided, defaulting to Noop"
            );
            &Action::Noop
        };

        let validated = if !agent.alive {
            Action::Noop
        } else {
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
                // Drone altitude actions: only valid for Aerial morphology when drone enabled
                Action::Ascend
                | Action::Descend
                | Action::Hover
                | Action::TakeOff
                | Action::Land => {
                    if !state.config.drone.enabled
                        || agent.morphology != forge_types::entity::AgentMorphology::Aerial
                    {
                        trace!(
                            agent_id = agent.id,
                            ?action,
                            "drone action on non-aerial agent or drone disabled, falling back to Noop"
                        );
                        Action::Noop
                    } else {
                        action.clone()
                    }
                }
                // Scan: any morphology can scan when drone enabled
                Action::Scan(_) => {
                    if !state.config.drone.enabled {
                        Action::Noop
                    } else {
                        action.clone()
                    }
                }
                // Agricultural actions: require agri + drone enabled, aerial morphology, airborne
                Action::Spray(slot) => {
                    if !state.config.agri.enabled
                        || !state.config.drone.enabled
                        || agent.morphology != forge_types::entity::AgentMorphology::Aerial
                        || agent.altitude == 0
                        || (*slot as usize) >= agent.inventory.capacity()
                    {
                        Action::Noop
                    } else {
                        Action::Spray(*slot)
                    }
                }
                Action::ScanMultispectral | Action::ScanThermal => {
                    if !state.config.agri.enabled
                        || !state.config.drone.enabled
                        || agent.morphology != forge_types::entity::AgentMorphology::Aerial
                        || agent.altitude == 0
                    {
                        Action::Noop
                    } else {
                        action.clone()
                    }
                }
                Action::RelaySoilData => {
                    if !state.config.agri.enabled {
                        Action::Noop
                    } else {
                        action.clone()
                    }
                }
                Action::GenerateReport => {
                    if !state.config.agri.enabled {
                        Action::Noop
                    } else {
                        action.clone()
                    }
                }
                // DropPayload: only valid for airborne Aerial agents (altitude > 0)
                Action::DropPayload(slot) => {
                    if !state.config.drone.enabled
                        || agent.morphology != forge_types::entity::AgentMorphology::Aerial
                        || agent.altitude == 0
                    {
                        Action::Noop
                    } else if (*slot as usize) >= agent.inventory.capacity() {
                        warn!(
                            agent_id = agent.id,
                            slot,
                            capacity = agent.inventory.capacity(),
                            "drop payload slot out of range, falling back to Noop"
                        );
                        Action::Noop
                    } else {
                        Action::DropPayload(*slot)
                    }
                }
                _ => action.clone(),
            }
        };

        state.validated_actions.push(validated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;
    use forge_types::grid::Position;
    use std::collections::HashMap;

    /// Test wrapper around [`validate_actions_into`] that mirrors the
    /// pre-refactor `validate_actions(actions, state) -> Vec<Action>`
    /// signature. Stages `actions` into `state.step_actions` and returns
    /// a clone of the validated buffer.
    fn validate_actions(actions: &[Action], state: &WorldState) -> Vec<Action> {
        let mut state = state.clone();
        state.step_actions.clear();
        let n = state.agents.len();
        let take = actions.len().min(n);
        state.step_actions.extend_from_slice(&actions[..take]);
        state.step_actions.resize(n, Action::Noop);
        validate_actions_into(&mut state);
        state.validated_actions
    }

    /// Test wrapper around [`compute_near_station_buf`] that returns an
    /// owned `Vec<bool>` for the legacy assertion style.
    fn compute_near_station(
        agents: &[forge_types::entity::Agent],
        grid: &forge_types::grid::Grid,
        objects: &[forge_types::entity::Object],
    ) -> Vec<bool> {
        let mut near = Vec::new();
        let mut map: HashMap<u32, ObjectType> = HashMap::new();
        compute_near_station_buf(agents, grid, objects, &mut near, &mut map);
        near
    }

    /// Test wrapper around [`run_systems`] that mirrors the pre-refactor
    /// `run_systems(state, actions)` signature.
    fn run_systems_with_actions(state: &mut WorldState, actions: &[Action]) {
        state.step_actions.clear();
        let n = state.agents.len();
        let take = actions.len().min(n);
        state.step_actions.extend_from_slice(&actions[..take]);
        state.step_actions.resize(n, Action::Noop);
        run_systems(state);
    }

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
        // Should always return exactly one action per agent, ignoring excess actions
        assert_eq!(validated.len(), 1);
        assert_eq!(validated[0], Action::Noop);
    }

    #[test]
    fn test_validate_actions_insufficient_actions() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 3;
        let state = WorldState::new(config).unwrap();

        let actions = vec![Action::Noop]; // Only 1 action for 3 agents
        let validated = validate_actions(&actions, &state);
        // Should return one action per agent, missing ones default to Noop
        assert_eq!(validated.len(), 3);
        assert_eq!(validated[0], Action::Noop);
        assert_eq!(validated[1], Action::Noop);
        assert_eq!(validated[2], Action::Noop);
    }

    #[test]
    fn test_systems_increment_tick() {
        let config = ForgeConfig::default();
        let mut state = WorldState::new(config).unwrap();
        let initial_tick = state.tick;

        let actions = vec![Action::Noop];
        run_systems_with_actions(&mut state, &actions);

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
        run_systems_with_actions(&mut state, &actions);

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
        // Should return one action per agent (defaulting to Noop if no actions provided)
        assert_eq!(validated.len(), 1);
        assert_eq!(validated[0], Action::Noop);
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

    // ---- Drone action validation tests ----

    #[test]
    fn test_validate_drone_actions_disabled() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        // drone.enabled is false by default
        let state = WorldState::new(config).unwrap();

        // Drone actions should become Noop when drone is disabled
        for action in [
            Action::Ascend,
            Action::Descend,
            Action::Hover,
            Action::TakeOff,
            Action::Land,
        ] {
            let validated = validate_actions(std::slice::from_ref(&action), &state);
            assert_eq!(
                validated[0],
                Action::Noop,
                "{:?} should be Noop when drone disabled",
                action
            );
        }
    }

    #[test]
    fn test_validate_scan_disabled() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        let state = WorldState::new(config).unwrap();

        let validated = validate_actions(&[Action::Scan(forge_types::Direction::Up)], &state);
        assert_eq!(validated[0], Action::Noop);
    }

    #[test]
    fn test_validate_drone_actions_non_aerial() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        config.drone.enabled = true;
        // Agent is Ground morphology by default, drone actions should become Noop
        let state = WorldState::new(config).unwrap();

        let validated = validate_actions(&[Action::TakeOff], &state);
        assert_eq!(validated[0], Action::Noop, "ground agent can't take off");
    }

    #[test]
    fn test_validate_drone_actions_aerial_valid() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        config.drone.enabled = true;
        config.drone.num_aerial = 1;
        let state = WorldState::new(config).unwrap();

        // Aerial agent should keep drone actions
        assert_eq!(
            state.agents[0].morphology,
            forge_types::entity::AgentMorphology::Aerial
        );
        let validated = validate_actions(&[Action::TakeOff], &state);
        assert_eq!(validated[0], Action::TakeOff);
    }

    #[test]
    fn test_validate_drop_payload_out_of_range() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        config.drone.enabled = true;
        config.drone.num_aerial = 1;
        let state = WorldState::new(config).unwrap();

        let validated = validate_actions(&[Action::DropPayload(99)], &state);
        assert_eq!(validated[0], Action::Noop, "slot 99 out of range");
    }

    #[test]
    fn test_validate_drop_payload_valid() {
        let mut config = ForgeConfig::default();
        config.agents.num_agents = 1;
        config.drone.enabled = true;
        config.drone.num_aerial = 1;
        let mut state = WorldState::new(config).unwrap();
        // Agent must be airborne for DropPayload to be valid
        state.agents[0].altitude = 3;

        let validated = validate_actions(&[Action::DropPayload(0)], &state);
        assert_eq!(validated[0], Action::DropPayload(0));
    }

    #[test]
    fn test_compute_near_station_multiple_stations() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agent = make_agent_at(0, 5, 5);
        let station1 = make_station_object(0, 6, 5);
        let station2 = make_station_object(1, 5, 4);
        grid.get_mut(6, 5).unwrap().object_id = Some(0);
        grid.get_mut(5, 4).unwrap().object_id = Some(1);

        let result = compute_near_station(&[agent], &grid, &[station1, station2]);
        assert!(result[0], "agent should see at least one station");
    }

    #[test]
    fn test_compute_near_station_multiple_agents() {
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let agents = [make_agent_at(0, 5, 5), make_agent_at(1, 10, 10)];
        let station = make_station_object(0, 6, 5);
        grid.get_mut(6, 5).unwrap().object_id = Some(0);

        let result = compute_near_station(&agents, &grid, &[station]);
        assert!(result[0], "agent 0 near station");
        assert!(!result[1], "agent 1 not near station");
    }
}
