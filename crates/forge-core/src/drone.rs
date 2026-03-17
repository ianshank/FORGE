//! Drone physics: altitude changes, battery drain, battery recharge, and payload drops.
//!
//! All functions in this module are no-ops when called on non-aerial agents.
//! The caller (systems.rs) gates these calls behind `config.drone.enabled`.

use forge_types::config::DroneConfig;
use forge_types::entity::{Agent, AgentMorphology};
use forge_types::grid::Grid;
use forge_types::Action;
use tracing::{debug, instrument, trace, warn};

/// Processes altitude-changing actions (TakeOff, Land, Ascend, Descend, Hover) for all agents.
///
/// Only affects agents with `morphology == Aerial`. Invalid actions are silently ignored
/// (they should already be filtered to Noop by validate_actions, but defense-in-depth).
#[instrument(skip_all)]
pub fn process_altitude_changes(agents: &mut [Agent], actions: &[Action], config: &DroneConfig) {
    for (i, agent) in agents.iter_mut().enumerate() {
        if !agent.alive || agent.morphology != AgentMorphology::Aerial {
            continue;
        }

        let action = actions.get(i).unwrap_or(&Action::Noop);

        match action {
            Action::TakeOff => {
                if agent.altitude == 0 && agent.battery >= config.ascend_cost {
                    let old = agent.altitude;
                    agent.altitude = 1;
                    agent.battery -= config.ascend_cost;
                    trace!(
                        agent_id = agent.id,
                        old_altitude = old,
                        new_altitude = 1,
                        "takeoff"
                    );
                }
            }
            Action::Land => {
                if agent.altitude > 0 {
                    let old = agent.altitude;
                    agent.altitude = 0;
                    trace!(agent_id = agent.id, old_altitude = old, "landed");
                }
            }
            Action::Ascend => {
                if agent.altitude < config.max_altitude && agent.battery >= config.ascend_cost {
                    agent.altitude += 1;
                    agent.battery -= config.ascend_cost;
                    trace!(agent_id = agent.id, altitude = agent.altitude, "ascended");
                }
            }
            Action::Descend => {
                if agent.altitude > 0 {
                    agent.altitude -= 1;
                    agent.battery -= config.descend_cost.min(agent.battery);
                    trace!(agent_id = agent.id, altitude = agent.altitude, "descended");
                }
            }
            Action::Hover => {
                if agent.altitude > 0 {
                    agent.battery -= config.hover_cost.min(agent.battery);
                    trace!(agent_id = agent.id, altitude = agent.altitude, "hovering");
                }
            }
            _ => {}
        }
    }
}

/// Drains battery for all airborne aerial agents each tick.
///
/// If an agent's battery reaches zero while airborne, forces an emergency landing
/// and applies fall damage proportional to altitude (1.0 damage per altitude level).
#[instrument(skip_all)]
pub fn process_battery_drain(agents: &mut [Agent], config: &DroneConfig) {
    let fall_damage_per_level = forge_types::constants::FIXED_POINT_ONE;

    for agent in agents.iter_mut() {
        if !agent.alive || agent.morphology != AgentMorphology::Aerial || agent.altitude == 0 {
            continue;
        }

        // Drain battery
        agent.battery = (agent.battery - config.aerial_drain_rate).max(0);

        // Force landing if battery depleted
        if agent.battery == 0 {
            let fall_altitude = agent.altitude;
            let damage = fall_altitude as i32 * fall_damage_per_level;
            agent.altitude = 0;
            agent.health -= damage;
            if agent.health <= 0 {
                agent.alive = false;
            }
            warn!(
                agent_id = agent.id,
                fall_altitude,
                damage,
                health = agent.health,
                "forced landing due to battery depletion"
            );
        }
    }
}

/// Recharges battery for landed aerial agents.
///
/// Only recharges agents with `morphology == Aerial` and `altitude == 0`.
/// Battery is clamped to `max_battery`.
#[instrument(skip_all)]
pub fn process_battery_recharge(agents: &mut [Agent], config: &DroneConfig) {
    for agent in agents.iter_mut() {
        if !agent.alive || agent.morphology != AgentMorphology::Aerial || agent.altitude != 0 {
            continue;
        }

        agent.battery = (agent.battery + config.recharge_rate).min(config.max_battery);
        trace!(
            agent_id = agent.id,
            battery = agent.battery,
            "battery recharging"
        );
    }
}

/// Processes DropPayload actions for aerial agents.
///
/// Removes the item from the agent's inventory slot and creates a resource
/// node on the ground tile below. Only works when the agent is airborne.
#[instrument(skip_all)]
pub fn process_payload_drops(
    agents: &mut [Agent],
    grid: &mut Grid,
    actions: &[Action],
    config: &DroneConfig,
) {
    let _ = config; // reserved for future payload-specific config
    for (i, agent) in agents.iter_mut().enumerate() {
        if !agent.alive || agent.morphology != AgentMorphology::Aerial || agent.altitude == 0 {
            continue;
        }

        let action = actions.get(i).unwrap_or(&Action::Noop);
        if let Action::DropPayload(slot) = action {
            let slot_idx = *slot as usize;
            if slot_idx < agent.inventory.capacity() {
                if let Some(_stack) = agent.inventory.get_slot(slot_idx) {
                    debug!(
                        agent_id = agent.id,
                        slot = slot_idx,
                        position = ?(agent.position.x, agent.position.y),
                        altitude = agent.altitude,
                        "payload dropped"
                    );
                    // Remove item from inventory — it falls to the ground tile
                    // The item becomes available for pickup by ground agents
                    agent.inventory.slots[slot_idx] = None;
                    // Note: full resource node creation deferred to worldgen integration (Phase 3)
                    // For now, the item is simply removed from inventory
                    let _ = grid; // will be used when ground item placement is implemented
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::{AgentConfig, DroneConfig};
    use forge_types::entity::{Agent, AgentMorphology};
    use forge_types::grid::Position;
    use forge_types::resource::ItemType;

    fn make_drone_config() -> DroneConfig {
        DroneConfig {
            enabled: true,
            ..Default::default()
        }
    }

    fn make_aerial_agent(id: u32, x: u16, y: u16) -> Agent {
        let agent_config = AgentConfig::default();
        let mut agent = Agent::new(id, Position::new(x, y), &agent_config);
        agent.morphology = AgentMorphology::Aerial;
        agent.capabilities.can_fly = true;
        agent.capabilities.max_altitude = 10;
        agent.battery = forge_types::constants::DEFAULT_STARTING_BATTERY;
        agent
    }

    fn make_ground_agent(id: u32, x: u16, y: u16) -> Agent {
        let agent_config = AgentConfig::default();
        Agent::new(id, Position::new(x, y), &agent_config)
    }

    // ---- TakeOff tests ----

    #[test]
    fn test_takeoff_sets_altitude_1() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        let actions = vec![Action::TakeOff];
        let initial_battery = agents[0].battery;

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 1);
        assert_eq!(agents[0].battery, initial_battery - config.ascend_cost);
    }

    #[test]
    fn test_takeoff_already_airborne_noop() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        let actions = vec![Action::TakeOff];
        let prev_alt = agents[0].altitude;
        let prev_bat = agents[0].battery;

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, prev_alt);
        assert_eq!(agents[0].battery, prev_bat);
    }

    #[test]
    fn test_takeoff_non_aerial_noop() {
        let config = make_drone_config();
        let mut agents = vec![make_ground_agent(0, 5, 5)];
        let actions = vec![Action::TakeOff];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 0);
    }

    #[test]
    fn test_takeoff_insufficient_battery() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].battery = config.ascend_cost - 1;
        let actions = vec![Action::TakeOff];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 0);
    }

    // ---- Land tests ----

    #[test]
    fn test_land_sets_altitude_0() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 5;
        let actions = vec![Action::Land];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 0);
    }

    #[test]
    fn test_land_already_ground_noop() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        let actions = vec![Action::Land];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 0);
    }

    // ---- Ascend tests ----

    #[test]
    fn test_ascend_increments_altitude() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        let actions = vec![Action::Ascend];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 4);
    }

    #[test]
    fn test_ascend_at_max_blocked() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = config.max_altitude;
        let actions = vec![Action::Ascend];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, config.max_altitude);
    }

    // ---- Descend tests ----

    #[test]
    fn test_descend_decrements_altitude() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        let actions = vec![Action::Descend];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 2);
    }

    #[test]
    fn test_descend_at_ground_noop() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 0;
        let actions = vec![Action::Descend];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 0);
    }

    // ---- Battery drain tests ----

    #[test]
    fn test_battery_drain_airborne() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 2;
        let initial_battery = agents[0].battery;

        process_battery_drain(&mut agents, &config);

        assert_eq!(
            agents[0].battery,
            initial_battery - config.aerial_drain_rate
        );
    }

    #[test]
    fn test_battery_drain_ground_zero() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 0;
        let initial_battery = agents[0].battery;

        process_battery_drain(&mut agents, &config);

        assert_eq!(agents[0].battery, initial_battery);
    }

    #[test]
    fn test_battery_depletion_force_land() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        agents[0].battery = config.aerial_drain_rate; // exactly enough for 1 tick

        process_battery_drain(&mut agents, &config);

        // Battery should be 0, agent force-landed
        assert_eq!(agents[0].battery, 0);
        assert_eq!(agents[0].altitude, 0);
        // Fall damage: 3 * FIXED_POINT_ONE = 3 * 65536 = 196608
        let expected_damage = 3 * forge_types::constants::FIXED_POINT_ONE;
        let expected_health = forge_types::constants::DEFAULT_STARTING_HEALTH - expected_damage;
        assert_eq!(agents[0].health, expected_health);
    }

    #[test]
    fn test_battery_depletion_kills_at_high_altitude() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 10; // high altitude
        agents[0].battery = 1; // nearly depleted

        process_battery_drain(&mut agents, &config);

        // 10 * 65536 = 655360 damage = exactly starting health
        assert_eq!(agents[0].altitude, 0);
        assert!(!agents[0].alive);
    }

    // ---- Battery recharge tests ----

    #[test]
    fn test_battery_recharge_when_landed() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 0;
        agents[0].battery = 100000;

        process_battery_recharge(&mut agents, &config);

        assert_eq!(agents[0].battery, 100000 + config.recharge_rate);
    }

    #[test]
    fn test_battery_recharge_capped_at_max() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 0;
        agents[0].battery = config.max_battery - 1;

        process_battery_recharge(&mut agents, &config);

        assert_eq!(agents[0].battery, config.max_battery);
    }

    #[test]
    fn test_battery_no_recharge_when_airborne() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        agents[0].battery = 100000;

        process_battery_recharge(&mut agents, &config);

        assert_eq!(agents[0].battery, 100000);
    }

    // ---- Hover test ----

    #[test]
    fn test_hover_drains_extra_battery() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        let initial_battery = agents[0].battery;
        let actions = vec![Action::Hover];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].battery, initial_battery - config.hover_cost);
        assert_eq!(agents[0].altitude, 3); // stays at same altitude
    }

    // ---- Payload drop tests ----

    #[test]
    fn test_payload_drop_removes_item() {
        let config = make_drone_config();
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        agents[0].inventory.add_item(ItemType::Wood, 5);
        let actions = vec![Action::DropPayload(0)];

        process_payload_drops(&mut agents, &mut grid, &actions, &config);

        assert!(agents[0].inventory.get_slot(0).is_none());
    }

    #[test]
    fn test_payload_drop_ground_noop() {
        let config = make_drone_config();
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 0; // on ground
        agents[0].inventory.add_item(ItemType::Wood, 5);
        let actions = vec![Action::DropPayload(0)];

        process_payload_drops(&mut agents, &mut grid, &actions, &config);

        // Item should still be in inventory since agent is on ground
        assert!(agents[0].inventory.get_slot(0).is_some());
    }

    #[test]
    fn test_payload_drop_non_aerial_noop() {
        let config = make_drone_config();
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let mut agents = vec![make_ground_agent(0, 5, 5)];
        agents[0].inventory.add_item(ItemType::Wood, 5);
        let actions = vec![Action::DropPayload(0)];

        process_payload_drops(&mut agents, &mut grid, &actions, &config);

        assert!(agents[0].inventory.get_slot(0).is_some());
    }

    // ---- Dead agent tests ----

    #[test]
    fn test_dead_agent_ignored() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].alive = false;
        agents[0].altitude = 3;
        let actions = vec![Action::Ascend];

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 3); // unchanged
    }
}
