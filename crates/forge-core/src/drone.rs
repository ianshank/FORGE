//! Drone physics: altitude changes, battery drain, battery recharge, and payload drops.
//!
//! All functions in this module are no-ops when called on non-aerial agents.
//! The caller (systems.rs) gates these calls behind `config.drone.enabled`.

use forge_types::config::DroneConfig;
use forge_types::entity::{Agent, AgentMorphology};
use forge_types::grid::Grid;
use forge_types::Action;
use tracing::{debug, instrument, trace, warn};

/// Deducts battery from an agent, clamping to zero.
///
/// Returns the actual amount deducted (may be less than `cost` if battery was low).
#[inline]
fn deduct_battery(agent: &mut Agent, cost: i32) -> i32 {
    let actual = cost.min(agent.battery);
    agent.battery -= actual;
    actual
}

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
                    deduct_battery(agent, config.ascend_cost);
                    trace!(
                        agent_id = agent.id,
                        old_altitude = old,
                        new_altitude = 1,
                        "takeoff"
                    );
                } else if agent.altitude == 0 {
                    debug!(
                        agent_id = agent.id,
                        battery = agent.battery,
                        required = config.ascend_cost,
                        "takeoff blocked: insufficient battery"
                    );
                }
            }
            Action::Land if agent.altitude > 0 => {
                let old = agent.altitude;
                agent.altitude = 0;
                trace!(agent_id = agent.id, old_altitude = old, "landed");
            }
            Action::Ascend => {
                if agent.altitude < config.max_altitude && agent.battery >= config.ascend_cost {
                    agent.altitude += 1;
                    deduct_battery(agent, config.ascend_cost);
                    trace!(agent_id = agent.id, altitude = agent.altitude, "ascended");
                } else if agent.battery < config.ascend_cost {
                    debug!(
                        agent_id = agent.id,
                        battery = agent.battery,
                        required = config.ascend_cost,
                        "ascend blocked: insufficient battery"
                    );
                }
            }
            Action::Descend if agent.altitude > 0 => {
                agent.altitude -= 1;
                deduct_battery(agent, config.descend_cost);
                trace!(agent_id = agent.id, altitude = agent.altitude, "descended");
            }
            Action::Hover if agent.altitude > 0 => {
                deduct_battery(agent, config.hover_cost);
                trace!(agent_id = agent.id, altitude = agent.altitude, "hovering");
            }
            _ => {}
        }
    }
}

/// Drains battery for all airborne aerial agents each tick.
///
/// If an agent's battery reaches zero while airborne, forces an emergency landing
/// and applies fall damage proportional to altitude using `config.fall_damage_per_level`.
#[instrument(skip_all)]
pub fn process_battery_drain(agents: &mut [Agent], config: &DroneConfig) {
    for agent in agents.iter_mut() {
        if !agent.alive || agent.morphology != AgentMorphology::Aerial || agent.altitude == 0 {
            continue;
        }

        // Drain battery
        agent.battery = (agent.battery - config.aerial_drain_rate).max(0);

        // Force landing if battery depleted
        if agent.battery == 0 {
            let fall_altitude = agent.altitude;
            let damage = fall_altitude as i32 * config.fall_damage_per_level;
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

        let old_battery = agent.battery;
        if !config.allows_recharge_at(agent.position) {
            trace!(
                agent_id = agent.id,
                x = agent.position.x,
                y = agent.position.y,
                "landed off charger; skip recharge"
            );
            continue;
        }
        agent.battery = (agent.battery + config.recharge_rate).min(config.max_battery);
        if agent.battery == config.max_battery && old_battery < config.max_battery {
            debug!(
                agent_id = agent.id,
                battery = agent.battery,
                "battery fully recharged"
            );
        } else {
            trace!(
                agent_id = agent.id,
                battery = agent.battery,
                "battery recharging"
            );
        }
    }
}

/// Processes DropPayload actions for aerial agents.
///
/// Removes the item from the agent's inventory slot. Only works when the agent
/// is airborne. Ground item placement is deferred to worldgen integration (Phase 3).
#[instrument(skip_all)]
pub fn process_payload_drops(
    agents: &mut [Agent],
    grid: &mut Grid,
    actions: &[Action],
    _config: &DroneConfig,
) {
    for (i, agent) in agents.iter_mut().enumerate() {
        if !agent.alive || agent.morphology != AgentMorphology::Aerial || agent.altitude == 0 {
            continue;
        }

        let action = actions.get(i).unwrap_or(&Action::Noop);
        if let Action::DropPayload(slot) = action {
            let slot_idx = *slot as usize;
            if slot_idx >= agent.inventory.capacity() {
                warn!(
                    agent_id = agent.id,
                    slot = slot_idx,
                    capacity = agent.inventory.capacity(),
                    "payload drop: slot out of range"
                );
                continue;
            }
            if agent.inventory.get_slot(slot_idx).is_some() {
                debug!(
                    agent_id = agent.id,
                    slot = slot_idx,
                    position = ?(agent.position.x, agent.position.y),
                    altitude = agent.altitude,
                    "payload dropped"
                );
                // Remove item from inventory — it falls to the ground tile
                agent.inventory.slots[slot_idx] = None;
                // Note: full resource node creation deferred to worldgen integration (Phase 3)
                let _ = grid; // will be used when ground item placement is implemented
            } else {
                trace!(
                    agent_id = agent.id,
                    slot = slot_idx,
                    "payload drop: slot empty, no-op"
                );
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

    #[test]
    fn test_battery_no_recharge_off_charger_when_restricted() {
        let mut config = make_drone_config();
        config.restrict_recharge_to_chargers = true;
        config.charger_tiles = vec![Position::new(0, 0)];
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 0;
        agents[0].battery = 100000;

        process_battery_recharge(&mut agents, &config);

        assert_eq!(agents[0].battery, 100000);
    }

    #[test]
    fn test_battery_recharge_on_charger_when_restricted() {
        let mut config = make_drone_config();
        config.restrict_recharge_to_chargers = true;
        config.charger_tiles = vec![Position::new(5, 5)];
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 0;
        agents[0].battery = 100000;

        process_battery_recharge(&mut agents, &config);

        assert_eq!(agents[0].battery, 100000 + config.recharge_rate);
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

    // ---- Multiple agent tests ----

    #[test]
    fn test_battery_drain_multiple_agents() {
        let config = make_drone_config();
        let mut agents = vec![
            make_aerial_agent(0, 5, 5),
            make_aerial_agent(1, 8, 8),
            make_ground_agent(2, 3, 3),
        ];
        agents[0].altitude = 2;
        agents[1].altitude = 5;
        let bat0 = agents[0].battery;
        let bat1 = agents[1].battery;
        let bat2 = agents[2].battery;

        process_battery_drain(&mut agents, &config);

        assert_eq!(agents[0].battery, bat0 - config.aerial_drain_rate);
        assert_eq!(agents[1].battery, bat1 - config.aerial_drain_rate);
        assert_eq!(agents[2].battery, bat2, "ground agent battery unchanged");
    }

    #[test]
    fn test_altitude_changes_multiple_agents() {
        let config = make_drone_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5), make_aerial_agent(1, 8, 8)];
        let actions = vec![Action::TakeOff, Action::Ascend];
        agents[1].altitude = 2;

        process_altitude_changes(&mut agents, &actions, &config);

        assert_eq!(agents[0].altitude, 1, "agent 0 should take off");
        assert_eq!(agents[1].altitude, 3, "agent 1 should ascend from 2 to 3");
    }

    // ---- Payload drop edge cases ----

    #[test]
    fn test_payload_drop_empty_slot_noop() {
        let config = make_drone_config();
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        // Slot 0 is empty
        let actions = vec![Action::DropPayload(0)];

        process_payload_drops(&mut agents, &mut grid, &actions, &config);

        assert!(agents[0].inventory.get_slot(0).is_none());
    }

    #[test]
    fn test_payload_drop_out_of_range_slot() {
        let config = make_drone_config();
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        agents[0].inventory.add_item(ItemType::Wood, 5);
        let actions = vec![Action::DropPayload(99)]; // out of range

        process_payload_drops(&mut agents, &mut grid, &actions, &config);

        // Item in slot 0 should be unchanged
        assert!(agents[0].inventory.get_slot(0).is_some());
    }

    #[test]
    fn test_payload_drop_different_slots() {
        let config = make_drone_config();
        let mut grid = forge_types::grid::Grid::new(16, 16);
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        agents[0].inventory.add_item(ItemType::Wood, 5);
        agents[0].inventory.add_item(ItemType::Stone, 3);

        // Drop from slot 1
        let actions = vec![Action::DropPayload(1)];
        process_payload_drops(&mut agents, &mut grid, &actions, &config);

        assert!(
            agents[0].inventory.get_slot(0).is_some(),
            "slot 0 untouched"
        );
        assert!(agents[0].inventory.get_slot(1).is_none(), "slot 1 dropped");
    }

    // ---- Configurable fall damage test ----

    #[test]
    fn test_custom_fall_damage_per_level() {
        let mut config = make_drone_config();
        // Double the fall damage
        config.fall_damage_per_level = forge_types::constants::FIXED_POINT_ONE * 2;
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        agents[0].altitude = 3;
        agents[0].battery = 1;

        process_battery_drain(&mut agents, &config);

        // 3 * 2*FIXED_POINT_ONE = 6 * 65536 = 393216 damage
        let expected_damage = 3 * config.fall_damage_per_level;
        let expected_health = forge_types::constants::DEFAULT_STARTING_HEALTH - expected_damage;
        assert_eq!(agents[0].health, expected_health);
    }

    // ---- deduct_battery helper test ----

    #[test]
    fn test_deduct_battery_clamps_to_zero() {
        let agent_config = AgentConfig::default();
        let mut agent = Agent::new(0, Position::new(5, 5), &agent_config);
        agent.battery = 100;

        let actual = deduct_battery(&mut agent, 200);
        assert_eq!(actual, 100, "should only deduct what's available");
        assert_eq!(agent.battery, 0, "battery should be 0");
    }

    #[test]
    fn test_deduct_battery_exact() {
        let agent_config = AgentConfig::default();
        let mut agent = Agent::new(0, Position::new(5, 5), &agent_config);
        agent.battery = 500;

        let actual = deduct_battery(&mut agent, 300);
        assert_eq!(actual, 300);
        assert_eq!(agent.battery, 200);
    }

    // ---- Property-based tests ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn battery_stays_non_negative_after_drain(
                initial_battery in 0..1_000_000i32,
                drain_rate in 0..200_000i32,
            ) {
                let mut config = make_drone_config();
                config.aerial_drain_rate = drain_rate;
                let mut agents = vec![make_aerial_agent(0, 5, 5)];
                agents[0].battery = initial_battery;
                agents[0].altitude = 1;

                process_battery_drain(&mut agents, &config);

                prop_assert!(agents[0].battery >= 0, "battery must never go negative");
            }

            #[test]
            fn altitude_stays_in_range_after_changes(
                initial_alt in 0u8..10,
                action_idx in 0usize..5,
            ) {
                let config = make_drone_config();
                let mut agents = vec![make_aerial_agent(0, 5, 5)];
                agents[0].altitude = initial_alt;

                let action = match action_idx {
                    0 => Action::TakeOff,
                    1 => Action::Land,
                    2 => Action::Ascend,
                    3 => Action::Descend,
                    _ => Action::Hover,
                };
                let actions = vec![action];

                process_altitude_changes(&mut agents, &actions, &config);

                prop_assert!(
                    agents[0].altitude <= config.max_altitude,
                    "altitude {} exceeds max {}",
                    agents[0].altitude,
                    config.max_altitude
                );
            }

            #[test]
            fn recharge_never_exceeds_max(
                initial_battery in 0..700_000i32,
                recharge_rate in 0..100_000i32,
            ) {
                let mut config = make_drone_config();
                config.recharge_rate = recharge_rate;
                let mut agents = vec![make_aerial_agent(0, 5, 5)];
                agents[0].altitude = 0; // must be landed to recharge
                agents[0].battery = initial_battery;

                process_battery_recharge(&mut agents, &config);

                prop_assert!(
                    agents[0].battery <= config.max_battery,
                    "battery {} exceeds max {}",
                    agents[0].battery,
                    config.max_battery
                );
            }

            #[test]
            fn fall_damage_scales_with_altitude(alt in 1u8..10) {
                let config = make_drone_config();
                let mut agents = vec![make_aerial_agent(0, 5, 5)];
                agents[0].altitude = alt;
                agents[0].battery = 1; // will deplete

                let initial_health = agents[0].health;
                process_battery_drain(&mut agents, &config);

                let expected_damage = alt as i32 * config.fall_damage_per_level;
                prop_assert_eq!(
                    agents[0].health,
                    initial_health - expected_damage,
                    "fall damage should scale linearly"
                );
            }
        }
    }
}
