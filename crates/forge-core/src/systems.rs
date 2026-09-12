//! System runner for the FORGE simulation step pipeline.
//!
//! Systems are run in a fixed deterministic order each tick.
//! This module coordinates the execution of all systems.

use forge_civ::grid_topology::GridTopology;
use forge_types::entity::{Agent, AgentMorphology, ObjectType};
use forge_types::grid::{Direction, Position};
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

    // 2b. Physics: push processing — extract minimal data to avoid
    //     cloning the full `Agent` vec. Uses the `push_scratch` buffer
    //     on `WorldState` so the snapshot is allocation-free for any
    //     agent count after warmup.
    state.push_scratch.clear();
    state
        .push_scratch
        .extend(state.agents.iter().map(physics::AgentPushData::from_agent));
    physics::process_pushes(
        &state.push_scratch,
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
        state.task_action_ids.clear();
        let comm_vocab = state.config.agents.comm_vocab_size;
        let drone_enabled = state.config.drone.enabled;
        let agri_enabled = state.config.agri.enabled && drone_enabled;
        let hex_enabled = matches!(
            state.config.world.grid_type,
            forge_types::config::GridType::Hex
        );
        for action in &state.validated_actions {
            match action.try_to_discrete_configured(
                comm_vocab,
                drone_enabled,
                agri_enabled,
                hex_enabled,
            ) {
                Ok(id) => state.task_action_ids.push(id),
                Err(e) => {
                    warn!(error = %e, "skipping unencodable validated action for Without");
                }
            }
        }
        let crop_states = if state.crop_states.is_empty() {
            None
        } else {
            Some(state.crop_states.as_slice())
        };
        let soil_nodes = if state.soil_nodes.is_empty() {
            None
        } else {
            Some(state.soil_nodes.as_slice())
        };
        let ctx = forge_task::predicate::EvalContext {
            agents: &state.agents,
            tick: state.tick,
            grid: Some(&state.grid),
            objects: Some(&state.objects),
            crop_states,
            soil_nodes,
            max_battery: state.config.drone.max_battery,
        };
        let task_result = forge_task::evaluator::evaluate_tasks(
            &mut state.tasks,
            &ctx,
            state.config.task.reward_scale,
            &state.task_action_ids,
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
            apply_process_constraints(agent, validate_action_shape(agent, action, state), state)
        };

        state.validated_actions.push(validated);
    }
}

fn validate_action_shape(agent: &Agent, action: &Action, state: &WorldState) -> Action {
    match action {
        Action::Move(dir) => Action::Move(*dir),
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
        Action::Ascend | Action::Descend | Action::Hover | Action::TakeOff | Action::Land => {
            if !state.config.drone.enabled || agent.morphology != AgentMorphology::Aerial {
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
        Action::Scan(_) => {
            if !state.config.drone.enabled {
                Action::Noop
            } else {
                action.clone()
            }
        }
        Action::Spray(slot) => {
            if !state.config.agri.enabled
                || !state.config.drone.enabled
                || agent.morphology != AgentMorphology::Aerial
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
                || agent.morphology != AgentMorphology::Aerial
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
        Action::DropPayload(slot) => {
            if !state.config.drone.enabled
                || agent.morphology != AgentMorphology::Aerial
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
}

fn apply_process_constraints(agent: &Agent, action: Action, state: &WorldState) -> Action {
    if matches!(action, Action::Noop) {
        return action;
    }
    let drone = &state.config.drone;
    if drone.enabled && agent.morphology == AgentMorphology::Aerial {
        if matches!(action, Action::Ascend) && agent.altitude >= drone.max_altitude {
            trace!(
                agent_id = agent.id,
                altitude = agent.altitude,
                max_altitude = drone.max_altitude,
                "ascend at max altitude, falling back to Noop"
            );
            return Action::Noop;
        }
        if agent.battery < drone.battery_action_floor && energy_constrained_action(&action) {
            debug!(
                agent_id = agent.id,
                battery = agent.battery,
                floor = drone.battery_action_floor,
                ?action,
                "battery below action floor, falling back to Noop"
            );
            return Action::Noop;
        }
    }
    if state.config.world.geofence_enabled && is_locomotion(&action) {
        match movement_target(state, agent.position, &action) {
            Some(next) if geofence_allows(state, next) => {}
            other => {
                debug!(
                    agent_id = agent.id,
                    ?action,
                    x = agent.position.x,
                    y = agent.position.y,
                    target = ?other,
                    "geofence rejected locomotion, falling back to Noop"
                );
                return Action::Noop;
            }
        }
    }
    action
}

fn energy_constrained_action(action: &Action) -> bool {
    matches!(
        action,
        Action::TakeOff
            | Action::Ascend
            | Action::Hover
            | Action::Scan(_)
            | Action::ScanMultispectral
            | Action::ScanThermal
            | Action::Spray(_)
            | Action::GenerateReport
            | Action::DropPayload(_)
    )
}

fn is_locomotion(action: &Action) -> bool {
    matches!(action, Action::Move(_) | Action::MoveHex(_))
}

fn movement_target(state: &WorldState, pos: Position, action: &Action) -> Option<Position> {
    let width = state.config.world.width;
    let height = state.config.world.height;
    match action {
        Action::Move(dir) => pos.offset(*dir, width, height),
        Action::MoveHex(dir) => state.topology.neighbor(pos, *dir as u8, width, height),
        _ => None,
    }
}

fn geofence_allows(state: &WorldState, pos: Position) -> bool {
    if state.config.world.allows_position(pos) {
        return true;
    }
    if state.config.drone.spawn_home == Some(pos) {
        return true;
    }
    state.config.drone.restrict_recharge_to_chargers && state.config.drone.allows_recharge_at(pos)
}

#[cfg(test)]
#[path = "systems/tests.rs"]
mod tests;
