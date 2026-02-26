//! Crafting system for the FORGE simulation.
//!
//! Processes craft actions by looking up recipes in a recipe book,
//! verifying ingredient availability, station requirements, and
//! crafting level prerequisites. Consumes inputs and produces outputs.

use forge_types::entity::Agent;
use forge_types::resource::RecipeBook;
use forge_types::Action;
use tracing::trace;

/// Processes Craft actions using the recipe book.
///
/// When an agent performs a `Craft(recipe_id)` action:
/// - Looks up the recipe in the recipe book
/// - Checks that the agent has all required input items
/// - Checks that the agent meets the minimum crafting level
/// - Checks that the agent is near a crafting station (if required)
/// - Removes input items from the agent's inventory
/// - Adds the output item to the agent's inventory
///
/// The `near_station` slice provides a per-agent boolean indicating whether
/// the agent is adjacent to or on a crafting station tile.
pub fn process_crafting(
    agents: &mut [Agent],
    actions: &[Action],
    recipes: &RecipeBook,
    near_station: &[bool],
) {
    for (i, action) in actions.iter().enumerate() {
        let recipe_id = match action {
            Action::Craft(id) => *id,
            _ => continue,
        };

        if i >= agents.len() {
            continue;
        }

        let agent = &agents[i];
        if !agent.alive {
            continue;
        }

        // Look up the recipe
        let recipe = match recipes.get(recipe_id) {
            Some(r) => r,
            None => {
                trace!(
                    agent_id = agent.id,
                    recipe_id = recipe_id,
                    "craft failed: recipe not found"
                );
                continue;
            }
        };

        // Check crafting level
        if agent.capabilities.crafting_level < recipe.min_crafting_level {
            trace!(
                agent_id = agent.id,
                recipe_id = recipe_id,
                agent_level = agent.capabilities.crafting_level,
                required_level = recipe.min_crafting_level,
                "craft failed: insufficient crafting level"
            );
            continue;
        }

        // Check station requirement
        if recipe.requires_station {
            let is_near = near_station.get(i).copied().unwrap_or(false);
            if !is_near {
                trace!(
                    agent_id = agent.id,
                    recipe_id = recipe_id,
                    "craft failed: not near crafting station"
                );
                continue;
            }
        }

        // Check all required inputs are available
        let has_all_inputs = recipe
            .inputs
            .iter()
            .all(|(item_type, count)| agents[i].inventory.has_item(*item_type, *count));

        if !has_all_inputs {
            trace!(
                agent_id = agent.id,
                recipe_id = recipe_id,
                "craft failed: missing ingredients"
            );
            continue;
        }

        // Remove inputs from inventory
        let mut all_removed = true;
        for (item_type, count) in &recipe.inputs {
            if !agents[i].inventory.remove_item(*item_type, *count) {
                all_removed = false;
                break;
            }
        }

        if !all_removed {
            // This shouldn't happen since we checked above, but be safe
            trace!(
                agent_id = agents[i].id,
                recipe_id = recipe_id,
                "craft failed: could not remove inputs"
            );
            continue;
        }

        // Add output to inventory
        let (output_type, output_count) = recipe.output;
        if !agents[i].inventory.add_item(output_type, output_count) {
            // Output doesn't fit -- we already consumed inputs, this is a problem.
            // In a real game we might want to roll back. For determinism, we log and continue.
            trace!(
                agent_id = agents[i].id,
                recipe_id = recipe_id,
                "craft partially failed: inputs consumed but output does not fit in inventory"
            );
            continue;
        }

        trace!(
            agent_id = agents[i].id,
            recipe_id = recipe_id,
            output = ?output_type,
            output_count = output_count,
            "crafting successful"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::AgentConfig;
    use forge_types::grid::Position;
    use forge_types::resource::{CraftingRecipe, ItemType};

    fn make_test_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        Agent::new(id, Position::new(x, y), &config)
    }

    fn make_test_recipe_book() -> RecipeBook {
        RecipeBook {
            recipes: vec![
                CraftingRecipe {
                    id: 0,
                    name: "Axe".to_string(),
                    inputs: vec![(ItemType::Wood, 2), (ItemType::Stone, 1)],
                    output: (ItemType::Axe, 1),
                    requires_station: false,
                    min_crafting_level: 1,
                },
                CraftingRecipe {
                    id: 1,
                    name: "Sword".to_string(),
                    inputs: vec![(ItemType::Wood, 1), (ItemType::Ore, 2)],
                    output: (ItemType::Sword, 1),
                    requires_station: true,
                    min_crafting_level: 2,
                },
            ],
        }
    }

    #[test]
    fn test_craft_success() {
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].inventory.add_item(ItemType::Wood, 5);
        agents[0].inventory.add_item(ItemType::Stone, 3);

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Craft(0)]; // craft Axe
        let near_station = vec![false];

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        // Should have consumed 2 wood + 1 stone, produced 1 axe
        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 3);
        assert_eq!(agents[0].inventory.count_item(ItemType::Stone), 2);
        assert_eq!(agents[0].inventory.count_item(ItemType::Axe), 1);
    }

    #[test]
    fn test_craft_missing_ingredients() {
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].inventory.add_item(ItemType::Wood, 1); // need 2

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Craft(0)]; // craft Axe
        let near_station = vec![false];

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        // Nothing should have changed
        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 1);
        assert_eq!(agents[0].inventory.count_item(ItemType::Axe), 0);
    }

    #[test]
    fn test_craft_missing_station() {
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].capabilities.crafting_level = 2;
        agents[0].inventory.add_item(ItemType::Wood, 5);
        agents[0].inventory.add_item(ItemType::Ore, 5);

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Craft(1)]; // craft Sword (requires station)
        let near_station = vec![false]; // NOT near station

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        // Nothing consumed
        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 5);
        assert_eq!(agents[0].inventory.count_item(ItemType::Ore), 5);
        assert_eq!(agents[0].inventory.count_item(ItemType::Sword), 0);
    }

    #[test]
    fn test_craft_with_station() {
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].capabilities.crafting_level = 2;
        agents[0].inventory.add_item(ItemType::Wood, 5);
        agents[0].inventory.add_item(ItemType::Ore, 5);

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Craft(1)]; // craft Sword (requires station)
        let near_station = vec![true]; // IS near station

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 4);
        assert_eq!(agents[0].inventory.count_item(ItemType::Ore), 3);
        assert_eq!(agents[0].inventory.count_item(ItemType::Sword), 1);
    }

    #[test]
    fn test_craft_insufficient_crafting_level() {
        let mut agents = vec![make_test_agent(0, 5, 5)];
        // Default crafting_level is 1, sword requires level 2
        agents[0].inventory.add_item(ItemType::Wood, 5);
        agents[0].inventory.add_item(ItemType::Ore, 5);

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Craft(1)]; // craft Sword
        let near_station = vec![true];

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        // Nothing consumed -- level too low
        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 5);
        assert_eq!(agents[0].inventory.count_item(ItemType::Ore), 5);
        assert_eq!(agents[0].inventory.count_item(ItemType::Sword), 0);
    }

    #[test]
    fn test_craft_invalid_recipe() {
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].inventory.add_item(ItemType::Wood, 5);

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Craft(999)]; // nonexistent recipe
        let near_station = vec![false];

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        // Nothing consumed
        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 5);
    }

    #[test]
    fn test_craft_dead_agent_ignored() {
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].alive = false;
        agents[0].inventory.add_item(ItemType::Wood, 5);
        agents[0].inventory.add_item(ItemType::Stone, 3);

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Craft(0)];
        let near_station = vec![false];

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        // Nothing consumed
        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 5);
        assert_eq!(agents[0].inventory.count_item(ItemType::Axe), 0);
    }

    #[test]
    fn test_craft_noop_action_ignored() {
        let mut agents = vec![make_test_agent(0, 5, 5)];
        agents[0].inventory.add_item(ItemType::Wood, 5);
        agents[0].inventory.add_item(ItemType::Stone, 3);

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Noop];
        let near_station = vec![false];

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        // Nothing consumed
        assert_eq!(agents[0].inventory.count_item(ItemType::Wood), 5);
        assert_eq!(agents[0].inventory.count_item(ItemType::Axe), 0);
    }

    #[test]
    fn test_craft_multiple_agents() {
        let mut agents = vec![make_test_agent(0, 5, 5), make_test_agent(1, 7, 7)];
        agents[0].inventory.add_item(ItemType::Wood, 5);
        agents[0].inventory.add_item(ItemType::Stone, 3);
        agents[1].inventory.add_item(ItemType::Wood, 5);
        agents[1].inventory.add_item(ItemType::Stone, 3);

        let recipes = make_test_recipe_book();
        let actions = vec![Action::Craft(0), Action::Craft(0)]; // both craft Axe
        let near_station = vec![false, false];

        process_crafting(&mut agents, &actions, &recipes, &near_station);

        // Both should have crafted
        assert_eq!(agents[0].inventory.count_item(ItemType::Axe), 1);
        assert_eq!(agents[1].inventory.count_item(ItemType::Axe), 1);
    }
}
