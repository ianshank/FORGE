//! Entity types: agents, objects, and their properties.
//!
//! Entities are the dynamic actors in the FORGE world. Agents are player-controlled
//! or AI-controlled entities. Objects are interactive world elements.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::grid::Position;
use crate::resource::ItemType;

/// Unique identifier for an agent.
pub type AgentId = u32;

/// Unique identifier for an object.
pub type ObjectId = u32;

/// Unique identifier for a team.
pub type TeamId = u8;

/// Communication token — a discrete symbol from the vocabulary.
pub type CommToken = u16;

/// An agent in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Unique agent identifier.
    pub id: AgentId,
    /// Current grid position.
    pub position: Position,
    /// Current health (fixed-point i32, 16 fractional bits).
    pub health: i32,
    /// Current stamina (fixed-point i32, 16 fractional bits).
    pub stamina: i32,
    /// Agent's inventory.
    pub inventory: Inventory,
    /// Team assignment.
    pub team: TeamId,
    /// Vision radius in tiles.
    pub vision_radius: u8,
    /// Maximum inventory slots.
    pub carry_capacity: u8,
    /// Recent communication messages received.
    pub comm_buffer: SmallVec<[CommToken; 8]>,
    /// Agent capabilities (may differ in heterogeneous mode).
    pub capabilities: AgentCapabilities,
    /// Whether this agent is alive.
    pub alive: bool,
}

impl Agent {
    /// Creates a new agent with default capabilities at the given position.
    pub fn new(id: AgentId, position: Position, config: &crate::config::AgentConfig) -> Self {
        Self {
            id,
            position,
            health: config.starting_health,
            stamina: config.starting_stamina,
            inventory: Inventory::new(config.default_carry_capacity as usize),
            team: 0,
            vision_radius: config.default_vision_radius,
            carry_capacity: config.default_carry_capacity,
            comm_buffer: SmallVec::new(),
            capabilities: AgentCapabilities::default(),
            alive: true,
        }
    }

    /// Whether the agent can perform actions (alive and has stamina).
    pub fn can_act(&self) -> bool {
        self.alive && self.stamina > 0
    }
}

/// Capabilities that may vary between heterogeneous agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    /// Movement speed multiplier (fixed-point). 65536 = 1.0x.
    pub speed_multiplier: i32,
    /// Crafting skill level (affects which recipes are available).
    pub crafting_level: u8,
    /// Whether this agent can communicate.
    pub can_communicate: bool,
    /// Whether this agent can trade.
    pub can_trade: bool,
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self {
            speed_multiplier: 65536, // 1.0x
            crafting_level: 1,
            can_communicate: true,
            can_trade: true,
        }
    }
}

/// An agent's inventory, storing item stacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inventory {
    /// Item slots. Each slot can hold a stack of one item type.
    pub slots: Vec<Option<ItemStack>>,
}

impl Inventory {
    /// Creates a new empty inventory with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![None; capacity],
        }
    }

    /// Returns the number of slots.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Returns the number of occupied slots.
    pub fn occupied_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Whether the inventory is full.
    pub fn is_full(&self) -> bool {
        self.occupied_slots() >= self.capacity()
    }

    /// Attempts to add an item. Returns true if successful.
    pub fn add_item(&mut self, item_type: ItemType, count: u16) -> bool {
        // First, try to stack with existing items of the same type
        for stack in self.slots.iter_mut().flatten() {
            if stack.item_type == item_type {
                let space = crate::constants::MAX_STACK_SIZE - stack.count;
                if space >= count {
                    stack.count += count;
                    return true;
                }
            }
        }
        // Otherwise, find an empty slot
        for slot in self.slots.iter_mut() {
            if slot.is_none() {
                *slot = Some(ItemStack { item_type, count });
                return true;
            }
        }
        false
    }

    /// Removes items from inventory. Returns true if successful.
    pub fn remove_item(&mut self, item_type: ItemType, count: u16) -> bool {
        if self.count_item(item_type) < count {
            return false;
        }
        let mut remaining = count;
        for slot in self.slots.iter_mut() {
            if remaining == 0 {
                break;
            }
            if let Some(stack) = slot {
                if stack.item_type == item_type {
                    let take = remaining.min(stack.count);
                    stack.count -= take;
                    remaining -= take;
                    if stack.count == 0 {
                        *slot = None;
                    }
                }
            }
        }
        remaining == 0
    }

    /// Counts total items of a given type across all slots.
    pub fn count_item(&self, item_type: ItemType) -> u16 {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.item_type == item_type)
            .map(|s| s.count)
            .sum()
    }

    /// Returns the item stack at a given slot index.
    pub fn get_slot(&self, index: usize) -> Option<&ItemStack> {
        self.slots.get(index).and_then(|s| s.as_ref())
    }

    /// Whether the inventory contains at least `count` of the given item.
    pub fn has_item(&self, item_type: ItemType, count: u16) -> bool {
        self.count_item(item_type) >= count
    }
}

/// A stack of items in an inventory slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItemStack {
    /// The type of item.
    pub item_type: ItemType,
    /// Number of items in this stack.
    pub count: u16,
}

/// An interactive object in the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    /// Unique object identifier.
    pub id: ObjectId,
    /// Grid position.
    pub position: Position,
    /// The type of object.
    pub object_type: ObjectType,
    /// Mass (fixed-point). Affects pushability.
    pub mass: i32,
    /// Durability (fixed-point). 0 = destroyed.
    pub durability: i32,
    /// Current object state.
    pub state: ObjectState,
}

/// Types of interactive objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
#[repr(u8)]
pub enum ObjectType {
    /// Heavy pushable boulder.
    Boulder = 0,
    /// Openable/closeable door.
    Door = 1,
    /// Toggleable switch or lever.
    Switch = 2,
    /// Storage container for items.
    Container = 3,
    /// Station required for advanced crafting recipes.
    CraftingStation = 4,
    /// Placeable bridge for crossing water.
    Bridge = 5,
    /// Floor-activated pressure plate.
    PressurePlate = 6,
    /// Light-emitting torch object.
    Torch = 7,
}

/// State of an object.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ObjectState {
    /// Default active state.
    #[default]
    Active,
    /// Inactive / off.
    Inactive,
    /// Open (for doors, containers).
    Open,
    /// Closed (for doors, containers).
    Closed,
    /// Destroyed / broken.
    Destroyed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;

    #[test]
    fn test_agent_creation() {
        let config = AgentConfig::default();
        let agent = Agent::new(0, Position::new(5, 5), &config);
        assert_eq!(agent.id, 0);
        assert_eq!(agent.position, Position::new(5, 5));
        assert!(agent.alive);
        assert!(agent.can_act());
    }

    #[test]
    fn test_inventory_add_remove() {
        let mut inv = Inventory::new(3);
        assert!(inv.add_item(ItemType::Wood, 5));
        assert_eq!(inv.count_item(ItemType::Wood), 5);

        assert!(inv.remove_item(ItemType::Wood, 3));
        assert_eq!(inv.count_item(ItemType::Wood), 2);

        assert!(!inv.remove_item(ItemType::Wood, 10)); // not enough
        assert_eq!(inv.count_item(ItemType::Wood), 2); // unchanged
    }

    #[test]
    fn test_inventory_full() {
        let mut inv = Inventory::new(2);
        assert!(inv.add_item(ItemType::Wood, 1));
        assert!(inv.add_item(ItemType::Stone, 1));
        assert!(inv.is_full());
        assert!(!inv.add_item(ItemType::Ore, 1)); // full
    }

    #[test]
    fn test_inventory_stacking() {
        let mut inv = Inventory::new(2);
        assert!(inv.add_item(ItemType::Wood, 3));
        assert!(inv.add_item(ItemType::Wood, 5));
        assert_eq!(inv.count_item(ItemType::Wood), 8);
        assert_eq!(inv.occupied_slots(), 1); // stacked into one slot
    }

    #[test]
    fn test_inventory_has_item() {
        let mut inv = Inventory::new(5);
        inv.add_item(ItemType::Stone, 10);
        assert!(inv.has_item(ItemType::Stone, 5));
        assert!(inv.has_item(ItemType::Stone, 10));
        assert!(!inv.has_item(ItemType::Stone, 11));
        assert!(!inv.has_item(ItemType::Wood, 1));
    }

    #[test]
    fn test_object_state() {
        let obj = Object {
            id: 0,
            position: Position::new(1, 1),
            object_type: ObjectType::Door,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Closed,
        };
        assert_eq!(obj.state, ObjectState::Closed);
    }

    #[test]
    fn test_agent_capabilities_default() {
        let caps = AgentCapabilities::default();
        assert_eq!(caps.speed_multiplier, 65536);
        assert!(caps.can_communicate);
        assert!(caps.can_trade);
    }

    #[test]
    fn test_inventory_capacity() {
        let inv = Inventory::new(5);
        assert_eq!(inv.capacity(), 5);

        let inv_zero = Inventory::new(0);
        assert_eq!(inv_zero.capacity(), 0);

        let inv_large = Inventory::new(100);
        assert_eq!(inv_large.capacity(), 100);
    }

    #[test]
    fn test_inventory_occupied_slots() {
        let mut inv = Inventory::new(5);
        assert_eq!(inv.occupied_slots(), 0);

        inv.add_item(ItemType::Wood, 1);
        assert_eq!(inv.occupied_slots(), 1);

        inv.add_item(ItemType::Stone, 2);
        assert_eq!(inv.occupied_slots(), 2);

        inv.add_item(ItemType::Ore, 3);
        assert_eq!(inv.occupied_slots(), 3);
    }

    #[test]
    fn test_inventory_get_slot() {
        let mut inv = Inventory::new(3);
        inv.add_item(ItemType::Wood, 5);

        // Valid index with an item returns Some.
        let slot0 = inv.get_slot(0);
        assert!(slot0.is_some());
        let stack = slot0.unwrap();
        assert_eq!(stack.item_type, ItemType::Wood);
        assert_eq!(stack.count, 5);

        // Valid index but empty slot returns None.
        assert!(inv.get_slot(1).is_none());

        // Invalid index (out of bounds) returns None.
        assert!(inv.get_slot(10).is_none());
    }

    #[test]
    fn test_inventory_empty() {
        let mut inv = Inventory::new(0);
        assert_eq!(inv.capacity(), 0);
        assert!(inv.is_full());
        // Adding an item to a zero-capacity inventory should fail gracefully.
        assert!(!inv.add_item(ItemType::Wood, 1));
        assert_eq!(inv.occupied_slots(), 0);
        assert_eq!(inv.count_item(ItemType::Wood), 0);
    }
}
