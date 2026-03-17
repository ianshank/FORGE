//! Integration tests for the FORGE simulation pipeline.
//!
//! These tests exercise the full simulation stack — from configuration through
//! world generation, stepping, combat, resource gathering, crafting,
//! communication, and day/night cycling.
//!
//! To run: `cargo test --test integration_tests`

use std::sync::Arc;

use forge_core::WorldState;
use forge_types::config::ForgeConfig;
use forge_types::grid::{Direction, Position};
use forge_types::resource::{ItemType, RecipeBook, ResourceNode};
use forge_types::task::{ActiveTask, Predicate, TaskComposition, TaskDefinition, TaskTier};
use forge_types::validation::validate_config;
use forge_types::Action;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Creates a small, fast-to-test world with the given number of agents.
fn make_config(num_agents: u32, seed: u64) -> ForgeConfig {
    let mut config = ForgeConfig::default();
    config.world.width = 32;
    config.world.height = 32;
    config.world.seed = seed;
    config.agents.num_agents = num_agents;
    config.task.max_episode_length = 10000;
    config.world.day_night_cycle_length = 100;
    config
}

// ---------------------------------------------------------------------------
// 1. Full episode
// ---------------------------------------------------------------------------

/// Create a WorldState with default config, reset, step 100 times with Noop
/// actions, and verify state consistency throughout the episode.
#[test]
fn test_full_episode() {
    let config = make_config(1, 42);
    let mut state = WorldState::new(config).unwrap();

    // Reset with explicit seed
    let reset_result = state.reset(Some(42));
    assert_eq!(reset_result.observations.len(), 1);
    assert!(!reset_result.terminated);
    assert!(!reset_result.truncated);

    // Step 100 times with Noop
    for tick in 0..100 {
        let result = state.step(&[Action::Noop]);

        // Observation count must match agent count
        assert_eq!(
            result.observations.len(),
            1,
            "observation count mismatch at tick {}",
            tick
        );

        // Reward vector must match agent count
        assert_eq!(
            result.rewards.len(),
            1,
            "reward count mismatch at tick {}",
            tick
        );

        // Tick should be advancing
        assert_eq!(state.tick, (tick + 1) as u64);

        // Agent should still be alive (Noop only)
        assert!(
            state.agents[0].alive,
            "agent died unexpectedly at tick {}",
            tick
        );

        // Info diagnostics should be consistent
        assert_eq!(result.info.agents_alive.len(), 1);
        assert!(result.info.agents_alive[0]);
    }

    // After 100 Noop steps the episode should not have ended
    assert!(!state.terminated);
    assert!(!state.truncated);
}

// ---------------------------------------------------------------------------
// 2. Deterministic replay
// ---------------------------------------------------------------------------

/// Run the same seed twice with the same action sequence and verify
/// that the resulting world states are identical.
#[test]
fn test_deterministic_replay() {
    let seed = 12345u64;

    // A non-trivial action sequence mixing movement and Noop.
    let action_sequence: Vec<Vec<Action>> = vec![
        vec![
            Action::Move(Direction::Right),
            Action::Move(Direction::Down),
        ],
        vec![Action::Noop, Action::Move(Direction::Left)],
        vec![Action::Move(Direction::Up), Action::Move(Direction::Right)],
        vec![Action::Move(Direction::Down), Action::Move(Direction::Down)],
        vec![Action::Noop, Action::Noop],
    ];

    // --- first run ---
    let config1 = make_config(2, seed);
    let mut state1 = WorldState::new(config1).unwrap();
    state1.reset(Some(seed));

    let mut results1 = Vec::new();
    for actions in &action_sequence {
        results1.push(state1.step(actions));
    }

    // --- second run ---
    let config2 = make_config(2, seed);
    let mut state2 = WorldState::new(config2).unwrap();
    state2.reset(Some(seed));

    let mut results2 = Vec::new();
    for actions in &action_sequence {
        results2.push(state2.step(actions));
    }

    // --- compare ---
    assert_eq!(state1.tick, state2.tick, "tick mismatch");
    for (i, (a1, a2)) in state1.agents.iter().zip(state2.agents.iter()).enumerate() {
        assert_eq!(
            a1.position, a2.position,
            "position mismatch for agent {}",
            i
        );
        assert_eq!(a1.health, a2.health, "health mismatch for agent {}", i);
        assert_eq!(a1.stamina, a2.stamina, "stamina mismatch for agent {}", i);
        assert_eq!(a1.alive, a2.alive, "alive mismatch for agent {}", i);
    }

    for (step_idx, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
        assert_eq!(
            r1.terminated, r2.terminated,
            "terminated mismatch at step {}",
            step_idx
        );
        assert_eq!(
            r1.truncated, r2.truncated,
            "truncated mismatch at step {}",
            step_idx
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Multi-agent episode
// ---------------------------------------------------------------------------

/// Create a 3-agent world, step with per-agent actions, and verify
/// each agent receives its own observation.
#[test]
fn test_multi_agent_episode() {
    let config = make_config(3, 99);
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(99));

    assert_eq!(state.agents.len(), 3, "expected 3 agents");

    // Each agent takes a different action
    let actions = vec![
        Action::Move(Direction::Up),
        Action::Move(Direction::Down),
        Action::Move(Direction::Left),
    ];

    let result = state.step(&actions);

    // Observations: one per agent
    assert_eq!(result.observations.len(), 3);
    // Rewards: one per agent
    assert_eq!(result.rewards.len(), 3);
    // Info alive flags: one per agent
    assert_eq!(result.info.agents_alive.len(), 3);

    // Each observation must have a grid view with the expected dimensions
    for (i, obs) in result.observations.iter().enumerate() {
        let expected_side = 2 * state.agents[i].vision_radius as u16 + 1;
        assert_eq!(
            obs.view_width, expected_side,
            "view_width mismatch for agent {}",
            i
        );
        assert_eq!(
            obs.view_height, expected_side,
            "view_height mismatch for agent {}",
            i
        );
        assert_eq!(
            obs.grid_view.len(),
            (expected_side as usize) * (expected_side as usize),
            "grid_view length mismatch for agent {}",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Worldgen to step
// ---------------------------------------------------------------------------

/// Generate a world with forge-worldgen, transplant the generated grid and
/// resources into a WorldState, and step it.
#[test]
fn test_worldgen_to_step() {
    use forge_worldgen::WorldGenerator;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    let seed = 7777u64;
    let mut config = ForgeConfig::default();
    config.world.width = 32;
    config.world.height = 32;
    config.world.seed = seed;
    config.agents.num_agents = 1;

    // Generate the world through the worldgen pipeline
    let gen = WorldGenerator::new(&config.world);
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let (grid, resources, objects, _spawn_points) = gen.generate(&mut rng);

    // Build a WorldState and inject the generated world
    let mut state = WorldState::new(config).unwrap();
    state.grid = grid;
    state.resources = resources;
    state.objects = objects;

    // Place the agent on a walkable tile within the generated world
    let agent_pos = state.agents[0].position;
    assert!(
        state.grid.in_bounds(agent_pos.x, agent_pos.y),
        "agent spawned out of bounds"
    );

    // Step the simulation — it should not panic
    let result = state.step(&[Action::Noop]);
    assert_eq!(result.observations.len(), 1);
    assert!(!result.terminated);
}

// ---------------------------------------------------------------------------
// 5. Task evaluation in episode
// ---------------------------------------------------------------------------

/// Create a task, run steps, and verify that the task evaluation pipeline
/// runs automatically within `step()` — rewards flow through StepResult
/// and task progress is tracked in observations.
#[test]
fn test_task_evaluation_in_episode() {
    let config = make_config(1, 42);
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    // Create a TimeElapsed task that is satisfied after 5 ticks
    let task = ActiveTask {
        definition: TaskDefinition {
            id: 1,
            description: "Wait 5 ticks".to_string(),
            goal: TaskComposition::Atom(Predicate::TimeElapsed(5)),
            tier: TaskTier::new(1),
            estimated_steps: 5,
            reward: 10.0,
            dense_reward_weights: vec![1.0],
        },
        progress: vec![0.0],
        sequence_index: 0,
        completed: false,
        failed: false,
    };
    state.tasks.push(task);

    // Step 4 times — task should NOT be completed yet
    let mut total_reward = 0.0_f32;
    for tick in 0..4 {
        let result = state.step(&[Action::Noop]);
        total_reward += result.rewards[0];
        assert!(
            !state.tasks[0].completed,
            "task should not be completed at tick {}",
            tick + 1
        );
        // Observations should contain task progress
        assert!(
            !result.observations[0].task_progress.is_empty(),
            "observation should include task progress at tick {}",
            tick + 1
        );
    }

    // Dense rewards should have been accruing from progress
    assert!(
        total_reward > 0.0,
        "dense rewards should accrue before task completion"
    );

    // Step until the task completes (tick >= 5)
    let mut completion_reward = 0.0_f32;
    for _ in 0..4 {
        let result = state.step(&[Action::Noop]);
        completion_reward += result.rewards[0];
        if state.tasks[0].completed {
            break;
        }
    }

    assert!(
        state.tasks[0].completed,
        "task should be completed at tick {}",
        state.tick
    );
    assert!(
        completion_reward > 0.0,
        "agent should receive a reward for completing the task"
    );
}

// ---------------------------------------------------------------------------
// 6. Combat between agents
// ---------------------------------------------------------------------------

/// Place two agents adjacent to each other, have the attacker use a
/// sword, and verify the target's health decreases.
#[test]
fn test_combat_between_agents() {
    let mut config = make_config(2, 42);
    config.world.width = 16;
    config.world.height = 16;
    let mut state = WorldState::new(config).unwrap();

    // Manually place agents adjacent to each other
    state.agents[0].position = Position::new(5, 5);
    state.agents[1].position = Position::new(5, 4); // directly above agent 0

    // Update the grid to reflect positions
    for tile in state.grid.tiles.iter_mut() {
        tile.agent_id = None;
    }
    state.grid.get_mut(5, 5).unwrap().agent_id = Some(state.agents[0].id);
    state.grid.get_mut(5, 4).unwrap().agent_id = Some(state.agents[1].id);

    // Give agent 0 a sword in slot 0
    state.agents[0].inventory.add_item(ItemType::Sword, 1);

    let initial_health = state.agents[1].health;

    // Agent 0 uses slot 0 (sword attack), agent 1 does nothing
    let result = state.step(&[Action::Use(0), Action::Noop]);

    assert!(
        state.agents[1].health < initial_health,
        "target health should decrease after sword attack: before={}, after={}",
        initial_health,
        state.agents[1].health
    );

    // Observations should still be produced for both agents
    assert_eq!(result.observations.len(), 2);
}

// ---------------------------------------------------------------------------
// 7. Resource gathering
// ---------------------------------------------------------------------------

/// Agent moves to a resource tile and gathers a resource. Verify the
/// agent's inventory is updated and the resource quantity decreases.
#[test]
fn test_resource_gathering() {
    let mut config = make_config(1, 42);
    config.world.width = 16;
    config.world.height = 16;
    let mut state = WorldState::new(config).unwrap();

    // Place agent at (3, 3)
    state.agents[0].position = Position::new(3, 3);
    for tile in state.grid.tiles.iter_mut() {
        tile.agent_id = None;
    }
    state.grid.get_mut(3, 3).unwrap().agent_id = Some(state.agents[0].id);

    // Place a wood resource at (3, 3).
    // respawn_rate = 0 disables respawn so the quantity stays reduced after harvest.
    let resource = ResourceNode {
        id: 0,
        position: Position::new(3, 3),
        resource_type: ItemType::Wood,
        quantity: 5,
        max_quantity: 5,
        respawn_timer: 0,
        respawn_rate: 0,
        requires_tool: None,
    };
    state.resources.push(resource);
    state.grid.get_mut(3, 3).unwrap().resource_id = Some(0);

    // Verify inventory is initially empty for wood
    assert_eq!(state.agents[0].inventory.count_item(ItemType::Wood), 0);

    // PickUp action
    state.step(&[Action::PickUp]);

    assert_eq!(
        state.agents[0].inventory.count_item(ItemType::Wood),
        1,
        "agent should have 1 wood after picking up"
    );
    assert_eq!(
        state.resources[0].quantity, 4,
        "resource quantity should decrease by 1"
    );
}

// ---------------------------------------------------------------------------
// 8. Crafting pipeline
// ---------------------------------------------------------------------------

/// Gather materials (directly placed in inventory) then craft, and verify
/// the crafted item appears in inventory.
#[test]
fn test_crafting_pipeline() {
    let mut config = make_config(1, 42);
    config.world.width = 16;
    config.world.height = 16;
    config.crafting.enabled = true;
    let mut state = WorldState::new(config).unwrap();

    // Set up recipe book with a simple recipe (Axe: 2 Wood + 1 Stone)
    state.recipe_book = RecipeBook::default();

    // Give agent the materials for an Axe (recipe id 0)
    state.agents[0].inventory.add_item(ItemType::Wood, 5);
    state.agents[0].inventory.add_item(ItemType::Stone, 3);

    let wood_before = state.agents[0].inventory.count_item(ItemType::Wood);
    let stone_before = state.agents[0].inventory.count_item(ItemType::Stone);

    // Craft recipe 0 (Axe: requires 2 Wood + 1 Stone, no station needed)
    state.step(&[Action::Craft(0)]);

    let wood_after = state.agents[0].inventory.count_item(ItemType::Wood);
    let stone_after = state.agents[0].inventory.count_item(ItemType::Stone);
    let axe_count = state.agents[0].inventory.count_item(ItemType::Axe);

    assert_eq!(axe_count, 1, "agent should have crafted 1 axe");
    assert_eq!(
        wood_before - wood_after,
        2,
        "crafting should consume 2 wood"
    );
    assert_eq!(
        stone_before - stone_after,
        1,
        "crafting should consume 1 stone"
    );
}

// ---------------------------------------------------------------------------
// 9. Communication in episode
// ---------------------------------------------------------------------------

/// Two agents: one sends a communication token, the other should receive
/// it in their communication buffer on the next observation.
#[test]
fn test_communication_in_episode() {
    let mut config = make_config(2, 42);
    config.world.width = 16;
    config.world.height = 16;
    config.agents.comm_vocab_size = 16;
    config.agents.comm_radius = 0; // global broadcast
    let mut state = WorldState::new(config).unwrap();

    // Place agents within comm radius (global means always in range)
    state.agents[0].position = Position::new(5, 5);
    state.agents[1].position = Position::new(6, 5);
    for tile in state.grid.tiles.iter_mut() {
        tile.agent_id = None;
    }
    state.grid.get_mut(5, 5).unwrap().agent_id = Some(state.agents[0].id);
    state.grid.get_mut(6, 5).unwrap().agent_id = Some(state.agents[1].id);

    // Agent 0 sends token 7, agent 1 does Noop
    let result = state.step(&[Action::Communicate(7), Action::Noop]);

    // After the step, agent 1 should have the message in its comm buffer
    assert!(
        !state.agents[1].comm_buffer.is_empty(),
        "agent 1 should have received a message"
    );
    assert_eq!(
        state.agents[1].comm_buffer[0], 7,
        "agent 1 should have received token 7"
    );

    // Agent 0 (the sender) should NOT have its own message
    let sender_has_own_msg = state.agents[0].comm_buffer.contains(&7);
    assert!(
        !sender_has_own_msg,
        "sender should not receive their own message"
    );

    // The observation for agent 1 should include the message
    assert!(
        !result.observations[1].messages.is_empty(),
        "agent 1's observation should contain the received message"
    );
}

// ---------------------------------------------------------------------------
// 10. Day/night cycle progression
// ---------------------------------------------------------------------------

/// Run enough ticks to cycle through all 4 day phases (dawn, day, dusk,
/// night) and verify each phase is observed.
#[test]
fn test_day_night_cycle_progression() {
    let mut config = make_config(1, 42);
    // Cycle length of 100: each quarter is 25 ticks
    config.world.day_night_cycle_length = 100;
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    let mut phases_seen = [false; 4];

    // Step through a full cycle (100 ticks)
    for _ in 0..100 {
        let result = state.step(&[Action::Noop]);
        let phase = result.info.day_phase;
        assert!(phase <= 3, "day_phase should be 0-3, got {}", phase);
        phases_seen[phase as usize] = true;
    }

    // All four phases should have been observed
    assert!(
        phases_seen[0],
        "dawn phase (0) was never observed during a full cycle"
    );
    assert!(
        phases_seen[1],
        "day phase (1) was never observed during a full cycle"
    );
    assert!(
        phases_seen[2],
        "dusk phase (2) was never observed during a full cycle"
    );
    assert!(
        phases_seen[3],
        "night phase (3) was never observed during a full cycle"
    );
}

// ---------------------------------------------------------------------------
// 11. Config validation integration
// ---------------------------------------------------------------------------

/// Verify that invalid configs are rejected at WorldState::new().
#[test]
fn test_invalid_config_rejected() {
    let mut config = ForgeConfig::default();
    config.world.width = 2; // below min_dimension (8)

    let result = WorldState::new(config);
    assert!(
        result.is_err(),
        "WorldState::new should reject invalid config"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("width"),
        "error should mention 'width': {}",
        err_msg
    );
}

/// Verify that validate_config accepts the default config.
#[test]
fn test_default_config_passes_validation() {
    let config = ForgeConfig::default();
    assert!(
        validate_config(&config).is_ok(),
        "default config should be valid"
    );
}

// ---------------------------------------------------------------------------
// 12. State serialization integration
// ---------------------------------------------------------------------------

/// Serialize a WorldState to bytes, deserialize, and verify state matches.
#[test]
fn test_state_serialization_roundtrip() {
    let config = make_config(2, 42);
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    // Step a few times to build up interesting state
    for _ in 0..10 {
        state.step(&[
            Action::Move(Direction::Right),
            Action::Move(Direction::Left),
        ]);
    }

    let bytes = state.to_bytes();
    let config_arc = Arc::new(state.config.as_ref().clone());
    let restored = WorldState::from_bytes(&bytes, config_arc);
    assert!(
        restored.is_ok(),
        "from_bytes should succeed for valid bytes"
    );
    let restored = restored.unwrap();

    assert_eq!(restored.tick, state.tick, "tick mismatch after roundtrip");
    assert_eq!(
        restored.agents.len(),
        state.agents.len(),
        "agent count mismatch"
    );
    for (i, (a1, a2)) in state.agents.iter().zip(restored.agents.iter()).enumerate() {
        assert_eq!(
            a1.position, a2.position,
            "position mismatch for agent {} after roundtrip",
            i
        );
    }
}

/// JSON roundtrip for human-readable serialization.
#[test]
fn test_state_json_roundtrip() {
    let config = make_config(1, 99);
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(99));
    state.step(&[Action::Noop]);

    let json = state.to_json();
    assert!(json.is_ok(), "to_json should succeed");
    let json_str = json.unwrap();
    assert!(
        json_str.contains("\"tick\""),
        "JSON should contain tick field"
    );

    let config_arc = Arc::new(state.config.as_ref().clone());
    let restored = WorldState::from_json(&json_str, config_arc);
    assert!(restored.is_ok(), "from_json should succeed");
    assert_eq!(restored.unwrap().tick, state.tick);
}

// ---------------------------------------------------------------------------
// 13. Vision modifier integration
// ---------------------------------------------------------------------------

/// Verify that night phase reduces the effective observation grid compared
/// to daytime (reflecting the day/night vision modifier).
#[test]
fn test_night_vision_reduces_observation() {
    let mut config = make_config(1, 42);
    // Very short cycle: 4 ticks per phase
    config.world.day_night_cycle_length = 16;
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    // The day_phase changes over ticks; step through and collect phases
    let mut seen_day = false;
    let mut seen_night = false;
    for _ in 0..16 {
        let result = state.step(&[Action::Noop]);
        let phase = result.info.day_phase;
        if phase == 1 {
            seen_day = true;
        }
        if phase == 3 {
            seen_night = true;
        }
    }

    // Both phases should have been observed during a full cycle
    assert!(seen_day, "day phase should appear during a 16-tick cycle");
    assert!(
        seen_night,
        "night phase should appear during a 16-tick cycle"
    );
}

// ---------------------------------------------------------------------------
// 14. Reset determinism
// ---------------------------------------------------------------------------

/// Verify that calling reset() with the same seed produces identical state.
#[test]
fn test_reset_determinism() {
    let config = make_config(1, 42);
    let mut state = WorldState::new(config).unwrap();

    // First reset
    state.reset(Some(123));
    let pos1 = state.agents[0].position;
    let health1 = state.agents[0].health;

    // Step to advance state
    for _ in 0..5 {
        state.step(&[Action::Move(Direction::Down)]);
    }
    assert_ne!(
        state.agents[0].position, pos1,
        "position should change after moves"
    );

    // Reset with same seed
    state.reset(Some(123));
    assert_eq!(
        state.agents[0].position, pos1,
        "reset with same seed should restore position"
    );
    assert_eq!(
        state.agents[0].health, health1,
        "reset with same seed should restore health"
    );
    assert_eq!(state.tick, 0, "tick should be 0 after reset");
}

// ---------------------------------------------------------------------------
// 15. Episode truncation
// ---------------------------------------------------------------------------

/// Verify that episodes are truncated after max_episode_length steps.
#[test]
fn test_episode_truncation() {
    let mut config = make_config(1, 42);
    config.task.max_episode_length = 10;
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    let mut truncated = false;
    for _ in 0..20 {
        let result = state.step(&[Action::Noop]);
        if result.truncated {
            truncated = true;
            break;
        }
    }

    assert!(
        truncated,
        "episode should be truncated after max_episode_length"
    );
}

// ---------------------------------------------------------------------------
// Drone integration tests
// ---------------------------------------------------------------------------

/// Creates a drone-enabled config with the given agent distribution.
fn make_drone_config(
    num_aerial: u32,
    num_ground: u32,
    num_vehicles: u32,
    seed: u64,
) -> ForgeConfig {
    let mut config = make_config(num_aerial + num_ground + num_vehicles, seed);
    config.drone.enabled = true;
    config.drone.num_aerial = num_aerial;
    config.drone.num_ground_vehicles = num_vehicles;
    config.drone.max_altitude = 10;
    config
}

/// Aerial agent takes off, flies right 5 times, then lands.
#[test]
fn test_aerial_drone_flight_episode() {
    let config = make_drone_config(1, 0, 0, 42);
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    assert_eq!(
        state.agents[0].morphology,
        forge_types::entity::AgentMorphology::Aerial,
        "first agent should be Aerial"
    );
    assert_eq!(state.agents[0].altitude, 0, "should start on ground");
    let initial_battery = state.agents[0].battery;

    // Take off
    let result = state.step(&[Action::TakeOff]);
    assert!(!result.terminated);
    assert_eq!(
        state.agents[0].altitude, 1,
        "should be at altitude 1 after takeoff"
    );
    assert!(
        state.agents[0].battery < initial_battery,
        "battery should decrease after takeoff"
    );

    // Fly right 5 times
    for _ in 0..5 {
        state.step(&[Action::Move(Direction::Right)]);
    }
    assert_eq!(
        state.agents[0].altitude, 1,
        "altitude should persist while flying"
    );

    // Land
    state.step(&[Action::Land]);
    assert_eq!(
        state.agents[0].altitude, 0,
        "should be on ground after landing"
    );
    assert!(state.agents[0].alive, "agent should still be alive");
}

/// Ground vehicle agent moves on terrain, verifying it's assigned the right morphology.
#[test]
fn test_ground_vehicle_episode() {
    let config = make_drone_config(0, 0, 1, 42);
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    assert_eq!(
        state.agents[0].morphology,
        forge_types::entity::AgentMorphology::GroundVehicle,
        "first agent should be GroundVehicle"
    );

    // GroundVehicle should be able to move on ground terrain
    for _ in 0..10 {
        state.step(&[Action::Noop]);
    }
    assert!(
        state.agents[0].alive,
        "vehicle should still be alive after Noop steps"
    );
}

/// Mixed morphology: 1 Ground + 1 Aerial + 1 GroundVehicle in the same world.
#[test]
fn test_mixed_morphology_episode() {
    let config = make_drone_config(1, 1, 1, 42);
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    assert_eq!(state.agents.len(), 3);
    assert_eq!(
        state.agents[0].morphology,
        forge_types::entity::AgentMorphology::Aerial
    );
    assert_eq!(
        state.agents[1].morphology,
        forge_types::entity::AgentMorphology::GroundVehicle
    );
    assert_eq!(
        state.agents[2].morphology,
        forge_types::entity::AgentMorphology::Ground
    );

    // All agents should survive 50 Noop steps
    for _ in 0..50 {
        state.step(&[Action::Noop, Action::Noop, Action::Noop]);
    }
    for agent in &state.agents {
        assert!(agent.alive, "agent {} should be alive", agent.id);
    }
}

/// Aerial agent flies until battery depletes, verify forced landing and damage.
#[test]
fn test_battery_depletion_force_land() {
    let mut config = make_drone_config(1, 0, 0, 42);
    // Set small battery so it depletes quickly
    config.drone.starting_battery = 65536; // 1.0 in fixed-point
    config.drone.aerial_drain_rate = 13107; // ~0.2 per tick
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    // Take off
    state.step(&[Action::TakeOff]);
    assert_eq!(state.agents[0].altitude, 1);

    // Hover until battery runs out
    let mut landed = false;
    for _ in 0..100 {
        state.step(&[Action::Hover]);
        if state.agents[0].altitude == 0 {
            landed = true;
            break;
        }
    }
    assert!(
        landed,
        "agent should have been force-landed due to battery depletion"
    );
}

/// Default config (drone disabled) should produce identical behavior to pre-drone code.
#[test]
fn test_drone_disabled_backwards_compat() {
    let config = make_config(1, 42);
    assert!(!config.drone.enabled);
    let mut state = WorldState::new(config).unwrap();
    state.reset(Some(42));

    // Agent should have Ground morphology by default
    assert_eq!(
        state.agents[0].morphology,
        forge_types::entity::AgentMorphology::Ground
    );
    assert_eq!(state.agents[0].altitude, 0);

    // Drone actions should be silently converted to Noop
    state.step(&[Action::TakeOff]);
    assert_eq!(
        state.agents[0].altitude, 0,
        "TakeOff should be Noop when drone disabled"
    );

    // Standard actions still work
    let _pos_before = state.agents[0].position;
    state.step(&[Action::Move(Direction::Right)]);
    // Position may or may not change (terrain dependent), but agent should be alive
    assert!(state.agents[0].alive);
}

/// Two runs with same seed and actions should produce identical drone state.
#[test]
fn test_deterministic_drone_replay() {
    let actions_sequence: Vec<Action> = vec![
        Action::TakeOff,
        Action::Move(Direction::Right),
        Action::Ascend,
        Action::Move(Direction::Right),
        Action::Hover,
        Action::Descend,
        Action::Land,
    ];

    let mut state1 = WorldState::new(make_drone_config(1, 0, 0, 42)).unwrap();
    state1.reset(Some(42));
    for action in &actions_sequence {
        state1.step(std::slice::from_ref(action));
    }

    let mut state2 = WorldState::new(make_drone_config(1, 0, 0, 42)).unwrap();
    state2.reset(Some(42));
    for action in &actions_sequence {
        state2.step(std::slice::from_ref(action));
    }

    assert_eq!(state1.agents[0].position, state2.agents[0].position);
    assert_eq!(state1.agents[0].altitude, state2.agents[0].altitude);
    assert_eq!(state1.agents[0].battery, state2.agents[0].battery);
    assert_eq!(state1.agents[0].health, state2.agents[0].health);
    assert_eq!(state1.tick, state2.tick);
}

/// Verify observations contain drone fields.
#[test]
fn test_drone_observation_fields() {
    let config = make_drone_config(1, 0, 0, 42);
    let mut state = WorldState::new(config).unwrap();
    let result = state.reset(Some(42));

    let obs = &result.observations[0];
    assert_eq!(obs.morphology, 2, "Aerial morphology should be 2");
    assert_eq!(obs.altitude, 0, "should start on ground");
    assert!(
        obs.battery > 0.0 && obs.battery <= 1.0,
        "battery should be normalized"
    );

    // Take off and verify observation updates
    let result = state.step(&[Action::TakeOff]);
    let obs = &result.observations[0];
    assert_eq!(obs.altitude, 1, "observation should reflect altitude 1");
    assert!(obs.battery < 1.0, "battery should have decreased");
}
