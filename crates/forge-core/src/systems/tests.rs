use super::*;
use forge_types::config::ForgeConfig;
use forge_types::grid::Position;
use std::collections::HashMap;

/// Test wrapper around [`validate_actions_into`] that mirrors the
/// pre-refactor `validate_actions(actions, state) -> Vec<Action>`
/// signature. Clones `state` locally, stages `actions` into
/// `state.step_actions`, and moves the validated buffer out of the
/// cloned local `state`. The caller's `WorldState` is untouched.
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

#[test]
fn test_geofence_noops_move_outside_margin() {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.geofence_enabled = true;
    config.world.geofence_margin = 1;
    config.agents.num_agents = 1;
    let mut state = WorldState::new(config).unwrap();
    state.agents[0].position = Position::new(1, 1);

    let validated = validate_actions(&[Action::Move(forge_types::Direction::Left)], &state);
    assert_eq!(validated[0], Action::Noop);
}

#[test]
fn test_geofence_allows_depot_tile() {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.geofence_enabled = true;
    config.world.geofence_margin = 1;
    config.agents.num_agents = 1;
    config.drone.enabled = true;
    config.drone.num_aerial = 1;
    config.drone.spawn_home = Some(Position::new(0, 1));
    let mut state = WorldState::new(config).unwrap();
    state.agents[0].position = Position::new(1, 1);

    let validated = validate_actions(&[Action::Move(forge_types::Direction::Left)], &state);
    assert_eq!(validated[0], Action::Move(forge_types::Direction::Left));
}

#[test]
fn test_battery_floor_noops_takeoff_but_allows_land() {
    let mut config = ForgeConfig::default();
    config.agents.num_agents = 1;
    config.drone.enabled = true;
    config.drone.num_aerial = 1;
    config.drone.battery_action_floor = 50_000;
    let mut state = WorldState::new(config).unwrap();
    state.agents[0].battery = 0;
    state.agents[0].altitude = 2;

    assert_eq!(
        validate_actions(&[Action::TakeOff], &state)[0],
        Action::Noop
    );
    assert_eq!(validate_actions(&[Action::Land], &state)[0], Action::Land);
}

#[test]
fn test_ascend_at_max_altitude_is_noop() {
    let mut config = ForgeConfig::default();
    config.agents.num_agents = 1;
    config.drone.enabled = true;
    config.drone.num_aerial = 1;
    let mut state = WorldState::new(config.clone()).unwrap();
    state.agents[0].altitude = config.drone.max_altitude;

    assert_eq!(validate_actions(&[Action::Ascend], &state)[0], Action::Noop);
}

#[test]
fn test_spawn_home_places_aerial_agent() {
    let mut config = ForgeConfig::default();
    config.agents.num_agents = 1;
    config.drone.enabled = true;
    config.drone.num_aerial = 1;
    config.drone.spawn_home = Some(Position::new(3, 4));
    let state = WorldState::new(config).unwrap();
    assert_eq!(state.agents[0].position, Position::new(3, 4));
}

#[test]
fn test_charger_constraint_same_seed_no_recharge_off_pad() {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.seed = 7;
    config.agents.num_agents = 1;
    config.drone.enabled = true;
    config.drone.num_aerial = 1;
    config.drone.restrict_recharge_to_chargers = true;
    config.drone.charger_tiles = vec![Position::new(0, 0)];
    config.drone.spawn_home = Some(Position::new(0, 0));

    let mut off_pad = WorldState::new(config.clone()).unwrap();
    off_pad.agents[0].altitude = 0;
    off_pad.agents[0].position = Position::new(4, 4);
    off_pad.agents[0].battery = 1_000;
    off_pad.step(&[Action::Noop]);
    assert_eq!(off_pad.agents[0].battery, 1_000);

    let mut on_pad = WorldState::new(config).unwrap();
    on_pad.agents[0].altitude = 0;
    on_pad.agents[0].battery = 1_000;
    on_pad.step(&[Action::Noop]);
    assert!(on_pad.agents[0].battery > 1_000);
}

#[test]
fn test_orchard_coverage_scenario_constructs_world() {
    let compiled = forge_types::scenario::compile_high_level_scenario(
        include_str!("../../../../configs/scenarios/orchard_coverage.toml"),
        "orchard_coverage",
    )
    .unwrap();
    let state = WorldState::new(compiled.forge_config).unwrap();
    assert_eq!(state.agents[0].position, Position::new(0, 0));
    assert!(state.config.drone.restrict_recharge_to_chargers);
    assert!(!state.tasks.is_empty());
}
