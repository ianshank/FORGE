//! Resource and crafting types for the FORGE simulation.
//!
//! Resources are finite, harvestable world elements. The crafting system
//! combines resources into tools and structures via configurable recipes.

use serde::{Deserialize, Serialize};

use crate::grid::{Position, TerrainType};

/// Types of items that can be in inventory or used in crafting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
#[repr(u8)]
pub enum ItemType {
    /// Raw wood resource.
    Wood = 0,
    /// Raw stone resource.
    Stone = 1,
    /// Raw ore resource.
    Ore = 2,
    /// Raw fish resource.
    Fish = 3,
    /// Raw fiber resource.
    Fiber = 4,
    /// Raw clay resource.
    Clay = 5,

    /// Crafted axe tool (wood harvesting).
    Axe = 10,
    /// Crafted pickaxe tool (stone/ore mining).
    Pickaxe = 11,
    /// Crafted sword weapon (combat).
    Sword = 12,
    /// Crafted shield (defense).
    Shield = 13,
    /// Crafted plank (intermediate material).
    Plank = 14,
    /// Crafted bridge (terrain crossing).
    Bridge = 15,
    /// Crafted rope (utility).
    Rope = 16,
    /// Crafted brick (construction material).
    Brick = 17,
    /// Crafted key (unlocking).
    Key = 18,
    /// Crafted torch (illumination).
    Torch = 19,

    /// Cooked fish food item.
    CookedFish = 30,
    /// Bread food item.
    Bread = 31,
}

impl ItemType {
    /// Whether this is a raw (harvestable) resource.
    pub fn is_raw_resource(&self) -> bool {
        (*self as u8) < 10
    }

    /// Whether this is a crafted item.
    pub fn is_crafted(&self) -> bool {
        let val = *self as u8;
        (10..30).contains(&val)
    }

    /// Whether this item is a tool (provides capabilities).
    pub fn is_tool(&self) -> bool {
        matches!(
            self,
            ItemType::Axe | ItemType::Pickaxe | ItemType::Sword | ItemType::Shield
        )
    }

    /// Converts from u8.
    pub fn from_u8(val: u8) -> Option<ItemType> {
        match val {
            0 => Some(ItemType::Wood),
            1 => Some(ItemType::Stone),
            2 => Some(ItemType::Ore),
            3 => Some(ItemType::Fish),
            4 => Some(ItemType::Fiber),
            5 => Some(ItemType::Clay),
            10 => Some(ItemType::Axe),
            11 => Some(ItemType::Pickaxe),
            12 => Some(ItemType::Sword),
            13 => Some(ItemType::Shield),
            14 => Some(ItemType::Plank),
            15 => Some(ItemType::Bridge),
            16 => Some(ItemType::Rope),
            17 => Some(ItemType::Brick),
            18 => Some(ItemType::Key),
            19 => Some(ItemType::Torch),
            30 => Some(ItemType::CookedFish),
            31 => Some(ItemType::Bread),
            _ => None,
        }
    }
}

/// Resource type mapped to terrain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceType {
    /// The item that is harvested from this resource.
    pub yields: ItemType,
    /// The terrain this resource naturally appears on.
    pub terrain_affinity: TerrainType,
}

/// A resource node in the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceNode {
    /// Unique resource node identifier.
    pub id: u32,
    /// Grid position.
    pub position: Position,
    /// What this resource yields when harvested.
    pub resource_type: ItemType,
    /// Remaining quantity available.
    pub quantity: u16,
    /// Maximum quantity (for respawn calculation).
    pub max_quantity: u16,
    /// Ticks remaining until next respawn increment. 0 = can respawn.
    pub respawn_timer: u32,
    /// Ticks between respawn increments.
    pub respawn_rate: u32,
    /// Whether this resource requires a tool to harvest.
    pub requires_tool: Option<ItemType>,
}

impl ResourceNode {
    /// Whether this resource can currently be harvested.
    pub fn can_harvest(&self) -> bool {
        self.quantity > 0
    }

    /// Whether this resource is depleted.
    pub fn is_depleted(&self) -> bool {
        self.quantity == 0
    }
}

/// A crafting recipe defining how to combine items.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CraftingRecipe {
    /// Unique recipe identifier.
    pub id: u16,
    /// Required input items and their quantities.
    pub inputs: Vec<(ItemType, u16)>,
    /// Output item and quantity produced.
    pub output: (ItemType, u16),
    /// Whether this recipe requires a crafting station.
    pub requires_station: bool,
    /// Minimum crafting level required.
    pub min_crafting_level: u8,
    /// Display name for the recipe.
    pub name: String,
}

/// The set of all available crafting recipes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeBook {
    /// All available recipes.
    pub recipes: Vec<CraftingRecipe>,
}

impl RecipeBook {
    /// Creates an empty recipe book.
    pub fn new() -> Self {
        Self {
            recipes: Vec::new(),
        }
    }

    /// Creates the default recipe book with standard recipes.
    pub fn default_recipes() -> Self {
        let recipes = vec![
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
                name: "Pickaxe".to_string(),
                inputs: vec![(ItemType::Wood, 2), (ItemType::Stone, 2)],
                output: (ItemType::Pickaxe, 1),
                requires_station: false,
                min_crafting_level: 1,
            },
            CraftingRecipe {
                id: 2,
                name: "Plank".to_string(),
                inputs: vec![(ItemType::Wood, 2)],
                output: (ItemType::Plank, 2),
                requires_station: false,
                min_crafting_level: 1,
            },
            CraftingRecipe {
                id: 3,
                name: "Bridge".to_string(),
                inputs: vec![(ItemType::Plank, 4)],
                output: (ItemType::Bridge, 1),
                requires_station: false,
                min_crafting_level: 2,
            },
            CraftingRecipe {
                id: 4,
                name: "Sword".to_string(),
                inputs: vec![(ItemType::Wood, 1), (ItemType::Ore, 2)],
                output: (ItemType::Sword, 1),
                requires_station: true,
                min_crafting_level: 2,
            },
            CraftingRecipe {
                id: 5,
                name: "Rope".to_string(),
                inputs: vec![(ItemType::Fiber, 3)],
                output: (ItemType::Rope, 1),
                requires_station: false,
                min_crafting_level: 1,
            },
            CraftingRecipe {
                id: 6,
                name: "Brick".to_string(),
                inputs: vec![(ItemType::Clay, 2)],
                output: (ItemType::Brick, 1),
                requires_station: true,
                min_crafting_level: 1,
            },
            CraftingRecipe {
                id: 7,
                name: "Torch".to_string(),
                inputs: vec![(ItemType::Wood, 1), (ItemType::Fiber, 1)],
                output: (ItemType::Torch, 1),
                requires_station: false,
                min_crafting_level: 1,
            },
            CraftingRecipe {
                id: 8,
                name: "Shield".to_string(),
                inputs: vec![(ItemType::Plank, 2), (ItemType::Ore, 1)],
                output: (ItemType::Shield, 1),
                requires_station: true,
                min_crafting_level: 2,
            },
        ];
        Self { recipes }
    }

    /// Looks up a recipe by ID.
    pub fn get(&self, id: u16) -> Option<&CraftingRecipe> {
        self.recipes.iter().find(|r| r.id == id)
    }

    /// Returns the number of recipes.
    pub fn len(&self) -> usize {
        self.recipes.len()
    }

    /// Whether the recipe book is empty.
    pub fn is_empty(&self) -> bool {
        self.recipes.is_empty()
    }
}

impl Default for RecipeBook {
    fn default() -> Self {
        Self::default_recipes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_type_classification() {
        assert!(ItemType::Wood.is_raw_resource());
        assert!(ItemType::Stone.is_raw_resource());
        assert!(!ItemType::Axe.is_raw_resource());

        assert!(ItemType::Axe.is_crafted());
        assert!(ItemType::Bridge.is_crafted());
        assert!(!ItemType::Wood.is_crafted());

        assert!(ItemType::Axe.is_tool());
        assert!(ItemType::Sword.is_tool());
        assert!(!ItemType::Wood.is_tool());
    }

    #[test]
    fn test_resource_node_harvest() {
        let node = ResourceNode {
            id: 0,
            position: Position::new(5, 5),
            resource_type: ItemType::Wood,
            quantity: 3,
            max_quantity: 5,
            respawn_timer: 0,
            respawn_rate: 100,
            requires_tool: None,
        };
        assert!(node.can_harvest());
        assert!(!node.is_depleted());
    }

    #[test]
    fn test_resource_node_depleted() {
        let node = ResourceNode {
            id: 0,
            position: Position::new(0, 0),
            resource_type: ItemType::Stone,
            quantity: 0,
            max_quantity: 5,
            respawn_timer: 50,
            respawn_rate: 100,
            requires_tool: Some(ItemType::Pickaxe),
        };
        assert!(!node.can_harvest());
        assert!(node.is_depleted());
    }

    #[test]
    fn test_default_recipe_book() {
        let book = RecipeBook::default_recipes();
        assert!(!book.is_empty());
        assert!(book.get(0).is_some()); // Axe recipe exists

        // Verify axe recipe
        let axe = book.get(0).unwrap();
        assert_eq!(axe.output.0, ItemType::Axe);
        assert_eq!(axe.inputs.len(), 2);
    }

    #[test]
    fn test_recipe_book_lookup() {
        let book = RecipeBook::default_recipes();
        assert!(book.get(0).is_some());
        assert!(book.get(999).is_none());
    }

    #[test]
    fn test_item_type_from_u8() {
        assert_eq!(ItemType::from_u8(0), Some(ItemType::Wood));
        assert_eq!(ItemType::from_u8(10), Some(ItemType::Axe));
        assert_eq!(ItemType::from_u8(255), None);
    }

    #[test]
    fn test_recipe_book_empty() {
        let book = RecipeBook::new();
        assert!(book.is_empty());
        assert_eq!(book.len(), 0);
        assert!(book.get(0).is_none());
    }

    #[test]
    fn test_item_type_from_u8_all_valid() {
        // Raw resources: 0-5
        assert_eq!(ItemType::from_u8(0), Some(ItemType::Wood));
        assert_eq!(ItemType::from_u8(1), Some(ItemType::Stone));
        assert_eq!(ItemType::from_u8(2), Some(ItemType::Ore));
        assert_eq!(ItemType::from_u8(3), Some(ItemType::Fish));
        assert_eq!(ItemType::from_u8(4), Some(ItemType::Fiber));
        assert_eq!(ItemType::from_u8(5), Some(ItemType::Clay));

        // Crafted items: 10-19
        assert_eq!(ItemType::from_u8(10), Some(ItemType::Axe));
        assert_eq!(ItemType::from_u8(11), Some(ItemType::Pickaxe));
        assert_eq!(ItemType::from_u8(12), Some(ItemType::Sword));
        assert_eq!(ItemType::from_u8(13), Some(ItemType::Shield));
        assert_eq!(ItemType::from_u8(14), Some(ItemType::Plank));
        assert_eq!(ItemType::from_u8(15), Some(ItemType::Bridge));
        assert_eq!(ItemType::from_u8(16), Some(ItemType::Rope));
        assert_eq!(ItemType::from_u8(17), Some(ItemType::Brick));
        assert_eq!(ItemType::from_u8(18), Some(ItemType::Key));
        assert_eq!(ItemType::from_u8(19), Some(ItemType::Torch));

        // Food: 30-31
        assert_eq!(ItemType::from_u8(30), Some(ItemType::CookedFish));
        assert_eq!(ItemType::from_u8(31), Some(ItemType::Bread));

        // Invalid values in the gaps should return None.
        assert_eq!(ItemType::from_u8(6), None);
        assert_eq!(ItemType::from_u8(9), None);
        assert_eq!(ItemType::from_u8(20), None);
        assert_eq!(ItemType::from_u8(29), None);
        assert_eq!(ItemType::from_u8(32), None);
    }

    #[test]
    fn test_item_type_food_classification() {
        // CookedFish and Bread are food items (value >= 30).
        assert!(!ItemType::CookedFish.is_raw_resource());
        assert!(!ItemType::CookedFish.is_crafted()); // value 30 is not in 10..30
        assert!(!ItemType::CookedFish.is_tool());

        assert!(!ItemType::Bread.is_raw_resource());
        assert!(!ItemType::Bread.is_crafted()); // value 31 is not in 10..30
        assert!(!ItemType::Bread.is_tool());

        // Verify that actual crafted items are classified correctly.
        assert!(ItemType::Axe.is_crafted());
        assert!(ItemType::Torch.is_crafted());
    }
}
