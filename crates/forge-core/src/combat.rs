//! Combat system for the FORGE simulation.
//!
//! Handles melee combat (sword attacks), health tracking, agent death,
//! and environmental damage from hazardous terrain such as lava.

use forge_civ::grid_topology::{GridTopology, GridTopologyKind};
use forge_types::constants::{DEFAULT_LAVA_DAMAGE, DEFAULT_SWORD_DAMAGE};
use forge_types::entity::Agent;
use forge_types::grid::{Grid, TerrainType};
use forge_types::resource::ItemType;
use forge_types::Action;
use tracing::{instrument, trace};

/// Processes combat damage from Use actions when using a weapon.
///
/// When an agent performs `Use(slot)` and the item in that slot is a Sword:
/// - Checks all 4 adjacent tiles in canonical order (Up, Down, Left, Right)
///   for a living agent
/// - Deals `DEFAULT_SWORD_DAMAGE` to the first adjacent agent found
/// - If the target's health drops to 0 or below, marks them as dead
///
/// Adjacent tiles are checked in a fixed deterministic order to ensure
/// reproducible behavior.
#[instrument(skip_all)]
pub fn process_combat(
    agents: &mut [Agent],
    grid: &Grid,
    actions: &[Action],
    topology: &GridTopologyKind,
) {
    // Collect attack intents first to avoid borrow issues
    let mut attacks: Vec<(usize, usize)> = Vec::new(); // (attacker_idx, target_idx)

    for (i, action) in actions.iter().enumerate() {
        let slot = match action {
            Action::Use(s) => *s as usize,
            _ => continue,
        };

        if i >= agents.len() {
            continue;
        }

        let agent = &agents[i];
        if !agent.alive {
            continue;
        }

        // Check if the item in the slot is a sword
        let has_sword = match agent.inventory.get_slot(slot) {
            Some(stack) => stack.item_type == ItemType::Sword,
            None => false,
        };

        if !has_sword {
            continue;
        }

        // Check all adjacent tiles for a target agent (4 for square, 6 for hex)
        let attacker_pos = agent.position;
        let mut target_idx = None;

        for adj_pos in topology.neighbors(attacker_pos, grid.width, grid.height) {
            if let Some(tile) = grid.get(adj_pos.x, adj_pos.y) {
                if let Some(target_id) = tile.agent_id {
                    // Find the agent index for this target
                    if let Some(idx) = agents.iter().position(|a| a.id == target_id && a.alive) {
                        // Don't attack yourself
                        if idx != i {
                            target_idx = Some(idx);
                            break;
                        }
                    }
                }
            }
        }

        if let Some(target) = target_idx {
            attacks.push((i, target));
            trace!(
                attacker_id = agents[i].id,
                target_id = agents[target].id,
                damage = DEFAULT_SWORD_DAMAGE,
                "sword attack"
            );
        } else {
            trace!(agent_id = agents[i].id, "sword attack: no adjacent target");
        }
    }

    // Apply damage
    for (attacker_idx, target_idx) in attacks {
        let damage = DEFAULT_SWORD_DAMAGE;
        agents[target_idx].health -= damage;

        trace!(
            target_id = agents[target_idx].id,
            remaining_health = agents[target_idx].health,
            "damage applied"
        );

        if agents[target_idx].health <= 0 {
            agents[target_idx].health = 0;
            agents[target_idx].alive = false;
            trace!(
                agent_id = agents[target_idx].id,
                killed_by = agents[attacker_idx].id,
                "agent killed"
            );
        }
    }
}

/// Applies environmental damage to agents standing on hazardous terrain.
///
/// Currently handles:
/// - **Lava**: Agents on Lava terrain take `DEFAULT_LAVA_DAMAGE` per tick.
///   If their health drops to 0, they are marked as dead.
///
/// Note: Lava is normally not walkable (is_walkable() returns false),
/// but agents may end up on lava tiles through external placement,
/// terrain changes, or being pushed. This system handles those cases.
#[instrument(skip_all)]
pub fn apply_environmental_damage(agents: &mut [Agent], grid: &Grid) {
    for agent in agents.iter_mut() {
        if !agent.alive {
            continue;
        }

        let tile = match grid.get(agent.position.x, agent.position.y) {
            Some(t) => t,
            None => continue,
        };

        if tile.terrain == TerrainType::Lava {
            agent.health -= DEFAULT_LAVA_DAMAGE;

            trace!(
                agent_id = agent.id,
                damage = DEFAULT_LAVA_DAMAGE,
                remaining_health = agent.health,
                "lava damage"
            );

            if agent.health <= 0 {
                agent.health = 0;
                agent.alive = false;
                trace!(agent_id = agent.id, "agent killed by lava");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_civ::grid_topology::GridTopologyKind;
    use forge_civ::SquareTopology;
    use forge_types::config::AgentConfig;
    use forge_types::grid::Position;

    fn topo() -> GridTopologyKind {
        GridTopologyKind::Square(SquareTopology)
    }

    fn make_test_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        Agent::new(id, Position::new(x, y), &config)
    }

    fn make_test_grid(width: u16, height: u16) -> Grid {
        Grid::new(width, height)
    }

    // ---- Combat tests ----

    #[test]
    fn test_combat_sword_attack() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![
            make_test_agent(0, 5, 5), // attacker
            make_test_agent(1, 5, 4), // target (above attacker)
        ];
        // Give attacker a sword in slot 0
        agents[0].inventory.add_item(ItemType::Sword, 1);

        // Place agents on grid
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);

        let initial_health = agents[1].health;
        let actions = vec![Action::Use(0), Action::Noop];

        process_combat(&mut agents, &grid, &actions, &topo());

        assert!(agents[1].health < initial_health);
        assert_eq!(initial_health - agents[1].health, DEFAULT_SWORD_DAMAGE);
    }

    #[test]
    fn test_combat_agent_death() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 5, 4)];
        agents[0].inventory.add_item(ItemType::Sword, 1);
        // Set target health low enough to die from one hit
        agents[1].health = DEFAULT_SWORD_DAMAGE;

        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);

        let actions = vec![Action::Use(0), Action::Noop];

        process_combat(&mut agents, &grid, &actions, &topo());

        assert_eq!(agents[1].health, 0);
        assert!(!agents[1].alive);
    }

    #[test]
    fn test_combat_no_sword_no_damage() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 5, 4)];
        // Agent has wood in slot 0, not a sword
        agents[0].inventory.add_item(ItemType::Wood, 1);

        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);

        let initial_health = agents[1].health;
        let actions = vec![Action::Use(0), Action::Noop];

        process_combat(&mut agents, &grid, &actions, &topo());

        assert_eq!(agents[1].health, initial_health);
    }

    #[test]
    fn test_combat_no_adjacent_target() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![
            make_test_agent(0, 5, 5),
            make_test_agent(1, 10, 10), // far away
        ];
        agents[0].inventory.add_item(ItemType::Sword, 1);

        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(10, 10).unwrap().agent_id = Some(1);

        let initial_health = agents[1].health;
        let actions = vec![Action::Use(0), Action::Noop];

        process_combat(&mut agents, &grid, &actions, &topo());

        assert_eq!(agents[1].health, initial_health);
    }

    #[test]
    fn test_combat_dead_attacker_ignored() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 5, 4)];
        agents[0].inventory.add_item(ItemType::Sword, 1);
        agents[0].alive = false;

        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);

        let initial_health = agents[1].health;
        let actions = vec![Action::Use(0), Action::Noop];

        process_combat(&mut agents, &grid, &actions, &topo());

        assert_eq!(agents[1].health, initial_health);
    }

    #[test]
    fn test_combat_dead_target_not_attacked() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 5, 4)];
        agents[0].inventory.add_item(ItemType::Sword, 1);
        agents[1].alive = false;

        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);

        let initial_health = agents[1].health;
        let actions = vec![Action::Use(0), Action::Noop];

        process_combat(&mut agents, &grid, &actions, &topo());

        assert_eq!(agents[1].health, initial_health);
    }

    #[test]
    fn test_combat_empty_slot() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 5, 4)];
        // Slot 0 is empty

        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);

        let initial_health = agents[1].health;
        let actions = vec![Action::Use(0), Action::Noop];

        process_combat(&mut agents, &grid, &actions, &topo());

        assert_eq!(agents[1].health, initial_health);
    }

    #[test]
    fn test_combat_direction_priority() {
        // When multiple agents are adjacent, the one in the first checked direction
        // (Up) should be attacked
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![
            make_test_agent(0, 5, 5), // attacker
            make_test_agent(1, 5, 4), // Up
            make_test_agent(2, 5, 6), // Down
        ];
        agents[0].inventory.add_item(ItemType::Sword, 1);

        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);
        grid.get_mut(5, 6).unwrap().agent_id = Some(2);

        let initial_health_1 = agents[1].health;
        let initial_health_2 = agents[2].health;
        let actions = vec![Action::Use(0), Action::Noop, Action::Noop];

        process_combat(&mut agents, &grid, &actions, &topo());

        // Agent 1 (Up) should be attacked, agent 2 (Down) should be unharmed
        assert!(agents[1].health < initial_health_1);
        assert_eq!(agents[2].health, initial_health_2);
    }

    #[test]
    fn test_combat_mutual_attack() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 5, 4)];
        agents[0].inventory.add_item(ItemType::Sword, 1);
        agents[1].inventory.add_item(ItemType::Sword, 1);

        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(5, 4).unwrap().agent_id = Some(1);

        let initial_health_0 = agents[0].health;
        let initial_health_1 = agents[1].health;
        let actions = vec![Action::Use(0), Action::Use(0)]; // both attack

        process_combat(&mut agents, &grid, &actions, &topo());

        // Both should take damage
        assert!(agents[0].health < initial_health_0);
        assert!(agents[1].health < initial_health_1);
    }

    // ---- Environmental damage tests ----

    #[test]
    fn test_lava_damage() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 5).unwrap().terrain = TerrainType::Lava;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut agents = vec![make_test_agent(0, 5, 5)];
        let initial_health = agents[0].health;

        apply_environmental_damage(&mut agents, &grid);

        assert!(agents[0].health < initial_health);
        assert_eq!(initial_health - agents[0].health, DEFAULT_LAVA_DAMAGE);
    }

    #[test]
    fn test_lava_kills_agent() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 5).unwrap().terrain = TerrainType::Lava;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].health = DEFAULT_LAVA_DAMAGE; // just enough to die

        apply_environmental_damage(&mut agents, &grid);

        assert_eq!(agents[0].health, 0);
        assert!(!agents[0].alive);
    }

    #[test]
    fn test_no_damage_on_ground() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        // Ground is default terrain

        let mut agents = vec![make_test_agent(0, 5, 5)];
        let initial_health = agents[0].health;

        apply_environmental_damage(&mut agents, &grid);

        assert_eq!(agents[0].health, initial_health);
    }

    #[test]
    fn test_dead_agent_no_environmental_damage() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 5).unwrap().terrain = TerrainType::Lava;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].alive = false;
        let initial_health = agents[0].health;

        apply_environmental_damage(&mut agents, &grid);

        assert_eq!(agents[0].health, initial_health);
    }

    #[test]
    fn test_lava_cumulative_damage() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 5).unwrap().terrain = TerrainType::Lava;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);

        let mut agents = vec![make_test_agent(0, 5, 5)];
        let initial_health = agents[0].health;

        // Apply damage 3 times
        apply_environmental_damage(&mut agents, &grid);
        apply_environmental_damage(&mut agents, &grid);
        apply_environmental_damage(&mut agents, &grid);

        assert_eq!(agents[0].health, initial_health - 3 * DEFAULT_LAVA_DAMAGE);
        assert!(agents[0].alive); // still alive with default health (10.0)
    }

    #[test]
    fn test_multiple_agents_environmental_damage() {
        let mut grid = make_test_grid(16, 16);
        grid.get_mut(5, 5).unwrap().terrain = TerrainType::Lava;
        grid.get_mut(5, 5).unwrap().agent_id = Some(0);
        grid.get_mut(7, 7).unwrap().agent_id = Some(1);
        // Agent 1 is on ground

        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 7, 7)];
        let initial_health_0 = agents[0].health;
        let initial_health_1 = agents[1].health;

        apply_environmental_damage(&mut agents, &grid);

        // Agent 0 on lava should take damage
        assert!(agents[0].health < initial_health_0);
        // Agent 1 on ground should be fine
        assert_eq!(agents[1].health, initial_health_1);
    }

    // ---- Proptest: combat invariants ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Health never goes below zero (death check triggers at 0).
            #[test]
            fn health_non_negative_after_lava(
                initial_health in 1i32..1_000_000,
                ticks in 1u32..100,
            ) {
                let mut grid = make_test_grid(16, 16);
                grid.get_mut(5, 5).unwrap().terrain = TerrainType::Lava;
                grid.get_mut(5, 5).unwrap().agent_id = Some(0);

                let mut agents = vec![make_test_agent(0, 5, 5)];
                agents[0].health = initial_health;

                for _ in 0..ticks {
                    if !agents[0].alive {
                        break;
                    }
                    apply_environmental_damage(&mut agents, &grid);
                }

                prop_assert!(agents[0].health >= 0);
            }

            /// Combat is deterministic: same setup produces same result.
            #[test]
            fn combat_determinism(
                attacker_x in 2u16..14,
                attacker_y in 2u16..14,
            ) {
                let run = || {
                    let mut grid = make_test_grid(16, 16);
                    let target_y = attacker_y - 1; // Up direction
                    let mut agents = vec![
                        make_test_agent(0, attacker_x, attacker_y),
                        make_test_agent(1, attacker_x, target_y),
                    ];
                    agents[0].inventory.add_item(ItemType::Sword, 1);
                    grid.get_mut(attacker_x, attacker_y).unwrap().agent_id = Some(0);
                    grid.get_mut(attacker_x, target_y).unwrap().agent_id = Some(1);

                    let actions = vec![Action::Use(0), Action::Noop];
                    process_combat(&mut agents, &grid, &actions, &topo());
                    (agents[0].health, agents[1].health, agents[1].alive)
                };
                let r1 = run();
                let r2 = run();
                prop_assert_eq!(r1, r2);
            }
        }
    }
}
