use super::*;
use forge_types::config::AgentConfig;

fn make_agent(id: u32, x: u16, y: u16) -> Agent {
    Agent::new(id, Position::new(x, y), &AgentConfig::default())
}

fn make_ctx(agents: &[Agent], tick: u64) -> EvalContext<'_> {
    EvalContext::new(agents, tick)
}

#[test]
fn test_agent_at_satisfied() {
    let agents = vec![make_agent(0, 5, 5)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentAt(0, Position::new(5, 5)), &ctx);
    assert!(result.satisfied);
    assert_eq!(result.progress, 1.0);
}

#[test]
fn test_agent_at_unsatisfied() {
    let agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentAt(0, Position::new(5, 5)), &ctx);
    assert!(!result.satisfied);
    assert!(result.progress > 0.0); // some progress toward target
}

#[test]
fn test_agent_has_satisfied() {
    let mut agents = vec![make_agent(0, 0, 0)];
    agents[0].inventory.add_item(ItemType::Wood, 5);
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentHas(0, ItemType::Wood, 3), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_agent_has_partial_progress() {
    let mut agents = vec![make_agent(0, 0, 0)];
    agents[0].inventory.add_item(ItemType::Wood, 2);
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentHas(0, ItemType::Wood, 4), &ctx);
    assert!(!result.satisfied);
    assert!((result.progress - 0.5).abs() < 0.01);
}

#[test]
fn test_agent_near_satisfied() {
    let agents = vec![make_agent(0, 5, 5), make_agent(1, 6, 5)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 3), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_agent_near_unsatisfied() {
    let agents = vec![make_agent(0, 0, 0), make_agent(1, 50, 50)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 3), &ctx);
    assert!(!result.satisfied);
}

#[test]
fn test_agent_near_exact_max_distance_satisfied() {
    let agents = vec![make_agent(0, 0, 0), make_agent(1, 2, 1)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 3), &ctx);
    assert!(result.satisfied);
    assert_eq!(result.progress, 1.0);
}

#[test]
fn test_agent_near_far_distance_progress_clamps_to_zero() {
    let agents = vec![make_agent(0, 0, 0), make_agent(1, 200, 200)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 1), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_time_elapsed() {
    let agents = vec![];
    let ctx = make_ctx(&agents, 100);
    let result = evaluate_predicate(&Predicate::TimeElapsed(50), &ctx);
    assert!(result.satisfied);

    let ctx2 = make_ctx(&agents, 25);
    let result2 = evaluate_predicate(&Predicate::TimeElapsed(50), &ctx2);
    assert!(!result2.satisfied);
    assert!((result2.progress - 0.5).abs() < 0.01);
}

#[test]
fn test_health_above() {
    let agents = vec![make_agent(0, 0, 0)]; // full health
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::HealthAbove(0, 0.5), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_team_alive() {
    let mut agents = vec![make_agent(0, 0, 0), make_agent(1, 5, 5)];
    agents[0].team = 1;
    agents[1].team = 1;
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::TeamAlive(1), &ctx);
    assert!(result.satisfied);

    // Kill one agent
    let mut agents2 = agents.clone();
    agents2[0].alive = false;
    let ctx2 = make_ctx(&agents2, 0);
    let result2 = evaluate_predicate(&Predicate::TeamAlive(1), &ctx2);
    assert!(!result2.satisfied);
    assert!((result2.progress - 0.5).abs() < 0.01);
}

#[test]
fn test_unknown_agent() {
    let agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentAt(99, Position::new(0, 0)), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_resource_count_predicate() {
    let mut agents = vec![make_agent(0, 0, 0)];
    agents[0].inventory.add_item(ItemType::Stone, 5);
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::ResourceCount(0, ItemType::Stone, 3), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_agent_has_zero_count() {
    let agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    // Requesting 0 items should always be satisfied
    let result = evaluate_predicate(&Predicate::AgentHas(0, ItemType::Wood, 0), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_agent_has_nonexistent_agent() {
    let agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentHas(99, ItemType::Wood, 1), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_agent_near_both_missing() {
    let agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    // Agent 5 and 6 don't exist
    let result = evaluate_predicate(&Predicate::AgentNear(5, 6, 3), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_time_elapsed_zero_deadline() {
    let agents = vec![];
    let ctx = make_ctx(&agents, 0);
    // target_tick == 0 should be satisfied immediately
    let result = evaluate_predicate(&Predicate::TimeElapsed(0), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_health_above_low_health() {
    let mut agents = vec![make_agent(0, 0, 0)];
    // Set health to very low value
    agents[0].health = 1;
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::HealthAbove(0, 0.5), &ctx);
    assert!(!result.satisfied);
    assert!(result.progress >= 0.0);
}

#[test]
fn test_health_above_zero_threshold() {
    let agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    // threshold == 0.0 should always be satisfied
    let result = evaluate_predicate(&Predicate::HealthAbove(0, 0.0), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_health_above_nonexistent_agent() {
    let agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::HealthAbove(99, 0.5), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_team_alive_empty_team() {
    let agents = vec![make_agent(0, 0, 0)];
    // Agent 0 defaults to team 0, so team 99 is empty
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::TeamAlive(99), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_agent_on_terrain_invalid_terrain_id() {
    let agents = vec![make_agent(0, 3, 3)];
    let grid = Grid::new(16, 16);
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: Some(&grid),
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    // terrain_id 255 is invalid
    let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 255), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_agent_on_terrain_next_after_last_valid_id_is_unsatisfied() {
    let agents = vec![make_agent(0, 3, 3)];
    let grid = Grid::new(16, 16);
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: Some(&grid),
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 8), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_agent_on_terrain_nonexistent_agent() {
    let agents = vec![make_agent(0, 3, 3)];
    let grid = Grid::new(16, 16);
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: Some(&grid),
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::AgentOnTerrain(99, 0), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_agent_on_terrain_all_terrain_ids() {
    let agents = vec![make_agent(0, 3, 3)];
    let grid = Grid::new(16, 16);
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: Some(&grid),
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    // Test all valid terrain IDs (0-7) don't panic
    for terrain_id in 0..=7 {
        let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, terrain_id), &ctx);
        // Only terrain_id 0 (Ground) should be satisfied on default grid
        if terrain_id == 0 {
            assert!(result.satisfied);
        } else {
            assert!(!result.satisfied);
        }
    }
}

#[test]
fn test_object_at_nonexistent_object() {
    use forge_types::entity::ObjectType;
    let objects = vec![Object {
        id: 0,
        position: Position::new(5, 5),
        object_type: ObjectType::Boulder,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Active,
    }];
    let ctx = EvalContext {
        agents: &[],
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    // Object 99 doesn't exist
    let result = evaluate_predicate(&Predicate::ObjectAt(99, Position::new(5, 5)), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_object_in_state_no_objects() {
    let ctx = EvalContext {
        agents: &[],
        tick: 0,
        grid: None,
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectInState(0, "Active".to_string()), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_object_in_state_invalid_state_name() {
    use forge_types::entity::ObjectType;
    let objects = vec![Object {
        id: 0,
        position: Position::new(0, 0),
        object_type: ObjectType::Boulder,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Active,
    }];
    let ctx = EvalContext {
        agents: &[],
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(
        &Predicate::ObjectInState(0, "InvalidState".to_string()),
        &ctx,
    );
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_object_in_state_nonexistent_object() {
    use forge_types::entity::ObjectType;
    let objects = vec![Object {
        id: 0,
        position: Position::new(0, 0),
        object_type: ObjectType::Boulder,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Active,
    }];
    let ctx = EvalContext {
        agents: &[],
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectInState(99, "Active".to_string()), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_object_in_state_all_valid_states() {
    use forge_types::entity::ObjectType;
    let objects = vec![Object {
        id: 0,
        position: Position::new(0, 0),
        object_type: ObjectType::Door,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Closed,
    }];
    let ctx = EvalContext {
        agents: &[],
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    // Test all valid state names
    for state_name in &[
        "Active",
        "active",
        "Inactive",
        "inactive",
        "Open",
        "open",
        "Closed",
        "closed",
        "Destroyed",
        "destroyed",
    ] {
        let result = evaluate_predicate(&Predicate::ObjectInState(0, state_name.to_string()), &ctx);
        // Only "Closed" and "closed" should match
        if *state_name == "Closed" || *state_name == "closed" {
            assert!(
                result.satisfied,
                "Expected satisfied for state '{}'",
                state_name
            );
        } else {
            assert!(
                !result.satisfied,
                "Expected unsatisfied for state '{}'",
                state_name
            );
        }
    }
}

// ---- AgentOnTerrain tests ----

#[test]
fn test_agent_on_terrain_satisfied() {
    let agents = vec![make_agent(0, 3, 3)];
    let mut grid = Grid::new(16, 16);
    grid.get_mut(3, 3).unwrap().terrain = TerrainType::Forest;
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: Some(&grid),
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    // Forest = terrain_id 6
    let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 6), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_agent_on_terrain_unsatisfied() {
    let agents = vec![make_agent(0, 3, 3)];
    let grid = Grid::new(16, 16); // all Ground = 0
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: Some(&grid),
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    // Forest = terrain_id 6, but agent is on Ground
    let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 6), &ctx);
    assert!(!result.satisfied);
}

#[test]
fn test_agent_on_terrain_no_grid() {
    let agents = vec![make_agent(0, 3, 3)];
    let ctx = make_ctx(&agents, 0); // grid is None
    let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 0), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

// ---- ObjectAt tests ----

#[test]
fn test_object_at_satisfied() {
    use forge_types::entity::ObjectType;
    let agents = vec![];
    let objects = vec![Object {
        id: 0,
        position: Position::new(5, 5),
        object_type: ObjectType::Boulder,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Active,
    }];
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectAt(0, Position::new(5, 5)), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_object_at_unsatisfied() {
    use forge_types::entity::ObjectType;
    let agents = vec![];
    let objects = vec![Object {
        id: 0,
        position: Position::new(1, 1),
        object_type: ObjectType::Boulder,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Active,
    }];
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectAt(0, Position::new(5, 5)), &ctx);
    assert!(!result.satisfied);
    assert!(result.progress > 0.0); // partial progress from proximity
}

#[test]
fn test_object_at_no_objects() {
    let agents = vec![];
    let ctx = make_ctx(&agents, 0); // objects is None
    let result = evaluate_predicate(&Predicate::ObjectAt(0, Position::new(5, 5)), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

// ---- ObjectInState tests ----

#[test]
fn test_object_in_state_satisfied() {
    use forge_types::entity::ObjectType;
    let agents = vec![];
    let objects = vec![Object {
        id: 0,
        position: Position::new(0, 0),
        object_type: ObjectType::Door,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Open,
    }];
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectInState(0, "Open".to_string()), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_object_in_state_unsatisfied() {
    use forge_types::entity::ObjectType;
    let agents = vec![];
    let objects = vec![Object {
        id: 0,
        position: Position::new(0, 0),
        object_type: ObjectType::Door,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Closed,
    }];
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectInState(0, "Open".to_string()), &ctx);
    assert!(!result.satisfied);
}

#[test]
fn test_object_in_state_case_insensitive() {
    use forge_types::entity::ObjectType;
    let agents = vec![];
    let objects = vec![Object {
        id: 0,
        position: Position::new(0, 0),
        object_type: ObjectType::Boulder,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Active,
    }];
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectInState(0, "active".to_string()), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_object_in_state_uppercase_name_is_rejected() {
    use forge_types::entity::ObjectType;
    let agents = vec![];
    let objects = vec![Object {
        id: 0,
        position: Position::new(0, 0),
        object_type: ObjectType::Boulder,
        mass: 65536,
        durability: 655360,
        state: ObjectState::Active,
    }];
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: None,
        objects: Some(&objects),
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectInState(0, "ACTIVE".to_string()), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

// ---- Drone / agri predicates ----

#[test]
fn test_agent_at_altitude_satisfied() {
    let mut agents = vec![make_agent(0, 3, 3)];
    agents[0].altitude = 5;
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentAtAltitude(0, 5), &ctx);
    assert!(result.satisfied);
    assert_eq!(result.progress, 1.0);
}

#[test]
fn test_agent_at_altitude_unsatisfied() {
    let agents = vec![make_agent(0, 3, 3)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentAtAltitude(0, 5), &ctx);
    assert!(!result.satisfied);
    assert!(result.progress < 1.0);
}

#[test]
fn test_battery_above_satisfied() {
    let mut agents = vec![make_agent(0, 3, 3)];
    agents[0].battery = forge_types::constants::DEFAULT_MAX_BATTERY;
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::BatteryAbove(0, 0.5), &ctx);
    assert!(result.satisfied);
}

#[test]
fn test_battery_above_unsatisfied() {
    let mut agents = vec![make_agent(0, 3, 3)];
    agents[0].battery = 0;
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::BatteryAbove(0, 0.5), &ctx);
    assert!(!result.satisfied);
}

#[test]
fn test_agent_airborne_and_landed() {
    let mut agents = vec![make_agent(0, 3, 3)];
    let ctx = make_ctx(&agents, 0);
    assert!(evaluate_predicate(&Predicate::AgentLanded(0), &ctx).satisfied);
    assert!(!evaluate_predicate(&Predicate::AgentAirborne(0), &ctx).satisfied);

    agents[0].altitude = 2;
    let ctx = make_ctx(&agents, 0);
    assert!(evaluate_predicate(&Predicate::AgentAirborne(0), &ctx).satisfied);
    assert!(!evaluate_predicate(&Predicate::AgentLanded(0), &ctx).satisfied);
}

#[test]
fn test_soil_data_collected() {
    use forge_types::agriculture::SoilSensorNode;
    let agents = vec![make_agent(0, 0, 0)];
    let mut nodes = vec![
        SoilSensorNode::new(0, Position::new(1, 1)),
        SoilSensorNode::new(1, Position::new(2, 2)),
    ];
    nodes[0].collected = true;
    nodes[1].collected = true;
    let ctx = EvalContext {
        soil_nodes: Some(&nodes),
        ..EvalContext::new(&agents, 0)
    };
    assert!(evaluate_predicate(&Predicate::SoilDataCollected(0, 2), &ctx).satisfied);
    assert!(!evaluate_predicate(&Predicate::SoilDataCollected(0, 3), &ctx).satisfied);
}

#[test]
fn test_field_report_generated() {
    let mut agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    assert!(!evaluate_predicate(&Predicate::FieldReportGenerated(0), &ctx).satisfied);
    agents[0].generated_field_report = true;
    let ctx = make_ctx(&agents, 0);
    assert!(evaluate_predicate(&Predicate::FieldReportGenerated(0), &ctx).satisfied);
}

#[test]
fn test_agent_on_terrain_position_outside_grid() {
    // Agent at position (200, 200) on a 16x16 grid — grid.get returns None
    let agents = vec![make_agent(0, 200, 200)];
    let grid = Grid::new(16, 16);
    let ctx = EvalContext {
        agents: &agents,
        tick: 0,
        grid: Some(&grid),
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 0), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_object_at_with_none_objects() {
    let ctx = EvalContext {
        agents: &[],
        tick: 0,
        grid: None,
        objects: None,
        crop_states: None,
        soil_nodes: None,
        max_battery: forge_types::constants::DEFAULT_MAX_BATTERY,
    };
    let result = evaluate_predicate(&Predicate::ObjectAt(0, Position::new(5, 5)), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_predicate_result_unsatisfied_clamps_progress() {
    let result = PredicateResult::unsatisfied(-0.5);
    assert_eq!(result.progress, 0.0);

    let result2 = PredicateResult::unsatisfied(1.5);
    assert_eq!(result2.progress, 1.0);
}

#[test]
fn test_agent_near_one_missing() {
    // Only agent 0 exists, agent 1 missing
    let agents = vec![make_agent(0, 5, 5)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 3), &ctx);
    assert!(!result.satisfied);
    assert_eq!(result.progress, 0.0);
}

#[test]
fn test_agent_at_far_away() {
    // Agent very far from target — progress should be near 0
    let agents = vec![make_agent(0, 0, 0)];
    let ctx = make_ctx(&agents, 0);
    let result = evaluate_predicate(&Predicate::AgentAt(0, Position::new(200, 200)), &ctx);
    assert!(!result.satisfied);
    // Manhattan distance 400 / max_dist 100 -> clamped to 1.0 -> progress 0.0
    assert_eq!(result.progress, 0.0);
}
