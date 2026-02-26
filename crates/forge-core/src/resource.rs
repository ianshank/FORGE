//! Resource harvesting, depletion, and respawn system.
//!
//! Handles agents picking up resources from resource nodes in the world,
//! depletion tracking, and automatic respawn of depleted resource nodes.

use forge_types::entity::Agent;
use forge_types::grid::Grid;
use forge_types::resource::ResourceNode;
use forge_types::Action;
use tracing::{instrument, trace};

/// Processes resource harvesting for PickUp actions at resource nodes.
///
/// When an agent performs a `PickUp` action on a tile that contains a resource node:
/// - Checks if the resource node has remaining quantity
/// - Checks if the resource requires a tool and verifies the agent has it
/// - Checks if the agent's inventory has space
/// - Decrements the resource quantity by 1
/// - Adds the harvested item to the agent's inventory
///
/// All processing is deterministic and uses agent-order priority.
#[instrument(skip_all)]
pub fn process_harvesting(
    agents: &mut [Agent],
    grid: &Grid,
    resources: &mut [ResourceNode],
    actions: &[Action],
) {
    for (i, action) in actions.iter().enumerate() {
        if *action != Action::PickUp {
            continue;
        }

        if i >= agents.len() {
            continue;
        }

        let agent = &agents[i];
        if !agent.alive {
            continue;
        }

        // Look up the tile at the agent's position
        let tile = match grid.get(agent.position.x, agent.position.y) {
            Some(t) => t,
            None => continue,
        };

        // Check if there's a resource node on this tile
        let resource_id = match tile.resource_id {
            Some(id) => id,
            None => {
                trace!(
                    agent_id = agent.id,
                    "harvest failed: no resource at position"
                );
                continue;
            }
        };

        // Find the resource node
        let resource = match resources.iter_mut().find(|r| r.id == resource_id) {
            Some(r) => r,
            None => continue,
        };

        // Check if the resource is depleted
        if !resource.can_harvest() {
            trace!(
                agent_id = agent.id,
                resource_id = resource.id,
                "harvest failed: resource depleted"
            );
            continue;
        }

        // Check tool requirement
        if let Some(required_tool) = resource.requires_tool {
            if !agents[i].inventory.has_item(required_tool, 1) {
                trace!(
                    agent_id = agents[i].id,
                    resource_id = resource.id,
                    required_tool = ?required_tool,
                    "harvest failed: missing required tool"
                );
                continue;
            }
        }

        // Check inventory space
        let agent = &agents[i];
        if agent.inventory.is_full() && agent.inventory.count_item(resource.resource_type) == 0 {
            // Inventory is full and we can't stack with existing items
            trace!(
                agent_id = agent.id,
                resource_id = resource.id,
                "harvest failed: inventory full"
            );
            continue;
        }

        // Attempt to add item to inventory
        let item_type = resource.resource_type;
        let agent = &mut agents[i];
        if !agent.inventory.add_item(item_type, 1) {
            trace!(
                agent_id = agent.id,
                resource_id = resource.id,
                "harvest failed: could not add item to inventory"
            );
            continue;
        }

        // Decrement resource quantity
        resource.quantity -= 1;

        trace!(
            agent_id = agent.id,
            resource_id = resource.id,
            item_type = ?item_type,
            remaining = resource.quantity,
            "resource harvested"
        );
    }
}

/// Ticks resource respawn timers and replenishes depleted resources.
///
/// For each resource node:
/// - If the resource is below max quantity and the respawn timer is active,
///   decrement the timer
/// - When the timer reaches 0, add 1 unit of quantity (up to max_quantity)
///   and reset the timer to the respawn_rate
#[instrument(skip_all)]
pub fn tick_respawn(resources: &mut [ResourceNode]) {
    for resource in resources.iter_mut() {
        // Only process resources that are below max capacity
        if resource.quantity >= resource.max_quantity {
            continue;
        }

        // Skip resources with no respawn (rate of 0)
        if resource.respawn_rate == 0 {
            continue;
        }

        if resource.respawn_timer > 0 {
            resource.respawn_timer -= 1;
        }

        if resource.respawn_timer == 0 {
            // Replenish 1 unit
            resource.quantity = (resource.quantity + 1).min(resource.max_quantity);

            // Reset timer if still below max
            if resource.quantity < resource.max_quantity {
                resource.respawn_timer = resource.respawn_rate;
            }

            trace!(
                resource_id = resource.id,
                quantity = resource.quantity,
                max_quantity = resource.max_quantity,
                "resource respawned"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::AgentConfig;
    use forge_types::grid::Position;
    use forge_types::resource::ItemType;

    fn make_test_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        Agent::new(id, Position::new(x, y), &config)
    }

    fn make_test_grid(width: u16, height: u16) -> Grid {
        Grid::new(width, height)
    }

    fn make_test_resource(
        id: u32,
        x: u16,
        y: u16,
        item_type: ItemType,
        quantity: u16,
    ) -> ResourceNode {
        ResourceNode {
            id,
            position: Position::new(x, y),
            resource_type: item_type,
            quantity,
            max_quantity: 5,
            respawn_timer: 0,
            respawn_rate: 10,
            requires_tool: None,
        }
    }

    // ---- Harvesting tests ----

    #[test]
    fn test_harvest_success() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 3, 3)];
        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        grid.get_mut(3, 3).unwrap().resource_id = Some(0);

        let mut resources = vec![make_test_resource(0, 3, 3, ItemType::Wood, 3)];
        let actions = vec![Action::PickUp];

        process_harvesting(&mut agents, &grid, &mut resources, &actions);

        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 1);
        assert_eq!(resources[0].quantity, 2);
    }

    #[test]
    fn test_harvest_multiple_times() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 3, 3)];
        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        grid.get_mut(3, 3).unwrap().resource_id = Some(0);

        let mut resources = vec![make_test_resource(0, 3, 3, ItemType::Wood, 3)];
        let actions = vec![Action::PickUp];

        // Harvest 3 times
        for _ in 0..3 {
            process_harvesting(&mut agents, &grid, &mut resources, &actions);
        }

        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 3);
        assert_eq!(resources[0].quantity, 0);
    }

    #[test]
    fn test_harvest_depleted_resource() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 3, 3)];
        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        grid.get_mut(3, 3).unwrap().resource_id = Some(0);

        let mut resources = vec![make_test_resource(0, 3, 3, ItemType::Wood, 0)];
        let actions = vec![Action::PickUp];

        process_harvesting(&mut agents, &grid, &mut resources, &actions);

        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 0);
        assert_eq!(resources[0].quantity, 0);
    }

    #[test]
    fn test_harvest_requires_tool_success() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 3, 3)];
        // Give agent a pickaxe
        agents[0].inventory.add_item(ItemType::Pickaxe, 1);
        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        grid.get_mut(3, 3).unwrap().resource_id = Some(0);

        let mut resources = vec![ResourceNode {
            id: 0,
            position: Position::new(3, 3),
            resource_type: ItemType::Stone,
            quantity: 5,
            max_quantity: 5,
            respawn_timer: 0,
            respawn_rate: 10,
            requires_tool: Some(ItemType::Pickaxe),
        }];
        let actions = vec![Action::PickUp];

        process_harvesting(&mut agents, &grid, &mut resources, &actions);

        assert_eq!(agents[0].inventory.count_item(ItemType::Stone), 1);
        assert_eq!(resources[0].quantity, 4);
    }

    #[test]
    fn test_harvest_requires_tool_missing() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 3, 3)];
        // Agent has no pickaxe
        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        grid.get_mut(3, 3).unwrap().resource_id = Some(0);

        let mut resources = vec![ResourceNode {
            id: 0,
            position: Position::new(3, 3),
            resource_type: ItemType::Stone,
            quantity: 5,
            max_quantity: 5,
            respawn_timer: 0,
            respawn_rate: 10,
            requires_tool: Some(ItemType::Pickaxe),
        }];
        let actions = vec![Action::PickUp];

        process_harvesting(&mut agents, &grid, &mut resources, &actions);

        assert_eq!(agents[0].inventory.count_item(ItemType::Stone), 0);
        assert_eq!(resources[0].quantity, 5); // unchanged
    }

    #[test]
    fn test_harvest_inventory_full() {
        let mut grid = make_test_grid(16, 16);
        let config = AgentConfig {
            default_carry_capacity: 2,
            ..AgentConfig::default()
        };
        let mut agents = vec![Agent::new(0, Position::new(3, 3), &config)];
        // Fill inventory
        agents[0].inventory.add_item(ItemType::Ore, 1);
        agents[0].inventory.add_item(ItemType::Fiber, 1);
        assert!(agents[0].inventory.is_full());

        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        grid.get_mut(3, 3).unwrap().resource_id = Some(0);

        let mut resources = vec![make_test_resource(0, 3, 3, ItemType::Wood, 3)];
        let actions = vec![Action::PickUp];

        process_harvesting(&mut agents, &grid, &mut resources, &actions);

        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 0);
        assert_eq!(resources[0].quantity, 3); // unchanged
    }

    #[test]
    fn test_harvest_inventory_full_but_can_stack() {
        let mut grid = make_test_grid(16, 16);
        let config = AgentConfig {
            default_carry_capacity: 2,
            ..AgentConfig::default()
        };
        let mut agents = vec![Agent::new(0, Position::new(3, 3), &config)];
        // Fill inventory with wood and stone
        agents[0].inventory.add_item(ItemType::Wood, 1);
        agents[0].inventory.add_item(ItemType::Stone, 1);
        assert!(agents[0].inventory.is_full());

        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        grid.get_mut(3, 3).unwrap().resource_id = Some(0);

        // Resource yields wood -- can stack with existing wood
        let mut resources = vec![make_test_resource(0, 3, 3, ItemType::Wood, 3)];
        let actions = vec![Action::PickUp];

        process_harvesting(&mut agents, &grid, &mut resources, &actions);

        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 2);
        assert_eq!(resources[0].quantity, 2);
    }

    #[test]
    fn test_harvest_no_resource_at_position() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 3, 3)];
        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        // No resource_id set on this tile

        let mut resources: Vec<ResourceNode> = vec![];
        let actions = vec![Action::PickUp];

        process_harvesting(&mut agents, &grid, &mut resources, &actions);

        assert_eq!(agents[0].inventory.occupied_slots(), 0);
    }

    #[test]
    fn test_harvest_dead_agent_ignored() {
        let mut grid = make_test_grid(16, 16);
        let mut agents = vec![make_test_agent(0, 3, 3)];
        agents[0].alive = false;
        grid.get_mut(3, 3).unwrap().agent_id = Some(0);
        grid.get_mut(3, 3).unwrap().resource_id = Some(0);

        let mut resources = vec![make_test_resource(0, 3, 3, ItemType::Wood, 3)];
        let actions = vec![Action::PickUp];

        process_harvesting(&mut agents, &grid, &mut resources, &actions);

        assert_eq!(resources[0].quantity, 3); // unchanged
    }

    // ---- Respawn tests ----

    #[test]
    fn test_respawn_timer_decrement() {
        let mut resources = vec![ResourceNode {
            id: 0,
            position: Position::new(0, 0),
            resource_type: ItemType::Wood,
            quantity: 0,
            max_quantity: 5,
            respawn_timer: 5,
            respawn_rate: 10,
            requires_tool: None,
        }];

        tick_respawn(&mut resources);

        assert_eq!(resources[0].respawn_timer, 4);
        assert_eq!(resources[0].quantity, 0); // not yet
    }

    #[test]
    fn test_respawn_replenish_on_zero_timer() {
        let mut resources = vec![ResourceNode {
            id: 0,
            position: Position::new(0, 0),
            resource_type: ItemType::Wood,
            quantity: 0,
            max_quantity: 5,
            respawn_timer: 1,
            respawn_rate: 10,
            requires_tool: None,
        }];

        tick_respawn(&mut resources);

        // Timer was decremented from 1 to 0, then respawn triggered
        assert_eq!(resources[0].quantity, 1);
        // Timer reset because quantity (1) < max_quantity (5)
        assert_eq!(resources[0].respawn_timer, 10);
    }

    #[test]
    fn test_respawn_to_max() {
        let mut resources = vec![ResourceNode {
            id: 0,
            position: Position::new(0, 0),
            resource_type: ItemType::Wood,
            quantity: 4,
            max_quantity: 5,
            respawn_timer: 1,
            respawn_rate: 10,
            requires_tool: None,
        }];

        tick_respawn(&mut resources);

        assert_eq!(resources[0].quantity, 5);
        // Timer should NOT be reset since we're now at max
        // (the if condition quantity < max_quantity is false)
    }

    #[test]
    fn test_respawn_full_resource_no_change() {
        let mut resources = vec![ResourceNode {
            id: 0,
            position: Position::new(0, 0),
            resource_type: ItemType::Wood,
            quantity: 5,
            max_quantity: 5,
            respawn_timer: 0,
            respawn_rate: 10,
            requires_tool: None,
        }];

        tick_respawn(&mut resources);

        assert_eq!(resources[0].quantity, 5);
    }

    #[test]
    fn test_respawn_zero_rate_no_respawn() {
        let mut resources = vec![ResourceNode {
            id: 0,
            position: Position::new(0, 0),
            resource_type: ItemType::Wood,
            quantity: 0,
            max_quantity: 5,
            respawn_timer: 0,
            respawn_rate: 0, // no respawn
            requires_tool: None,
        }];

        tick_respawn(&mut resources);

        assert_eq!(resources[0].quantity, 0); // still depleted
    }

    #[test]
    fn test_respawn_complete_cycle() {
        let mut resources = vec![ResourceNode {
            id: 0,
            position: Position::new(0, 0),
            resource_type: ItemType::Stone,
            quantity: 0,
            max_quantity: 3,
            respawn_timer: 0,
            respawn_rate: 2,
            requires_tool: None,
        }];

        // Tick 1: timer is 0, quantity below max -> respawn 1, reset timer to 2
        tick_respawn(&mut resources);
        assert_eq!(resources[0].quantity, 1);
        assert_eq!(resources[0].respawn_timer, 2);

        // Tick 2: decrement timer to 1
        tick_respawn(&mut resources);
        assert_eq!(resources[0].quantity, 1);
        assert_eq!(resources[0].respawn_timer, 1);

        // Tick 3: decrement timer to 0, respawn another
        tick_respawn(&mut resources);
        assert_eq!(resources[0].quantity, 2);
        assert_eq!(resources[0].respawn_timer, 2);

        // Tick 4: decrement timer to 1
        tick_respawn(&mut resources);
        assert_eq!(resources[0].quantity, 2);
        assert_eq!(resources[0].respawn_timer, 1);

        // Tick 5: decrement timer to 0, respawn to max (3)
        tick_respawn(&mut resources);
        assert_eq!(resources[0].quantity, 3);
    }
}
