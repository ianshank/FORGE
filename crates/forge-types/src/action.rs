//! Action types and action space definitions for the FORGE simulation.
//!
//! Actions are the interface through which agents interact with the world.
//! The discrete action space is designed for efficient GPU batching.

use serde::{Deserialize, Serialize};

use crate::entity::{AgentId, CommToken};
use crate::grid::Direction;
use crate::resource::ItemType;

/// An action that an agent can take in a single tick.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Do nothing this tick.
    Noop,
    /// Move one tile in the given direction.
    Move(Direction),
    /// Pick up an item or object at the current position.
    PickUp,
    /// Drop an item from the given inventory slot.
    Drop(u8),
    /// Use an item from the given inventory slot.
    Use(u8),
    /// Craft an item using the given recipe index.
    Craft(u16),
    /// Push an object in the given direction.
    Push(Direction),
    /// Throw an item from inventory slot in a direction.
    Throw(Direction, u8),
    /// Emit a communication token.
    Communicate(CommToken),
    /// Offer a trade to another agent.
    Trade(AgentId, TradeOffer),
    /// Context-sensitive interaction with adjacent tile.
    Interact,
}

impl Action {
    /// Converts a flat integer action index to an Action.
    ///
    /// The encoding is:
    ///   0: Noop
    ///   1-4: Move (Up, Down, Left, Right)
    ///   5: PickUp
    ///   6-15: Drop(slot 0-9)
    ///   16-25: Use(slot 0-9)
    ///   26: Craft(0) — recipe index encoded separately or via extended action
    ///   27-30: Push (Up, Down, Left, Right)
    ///   31: Interact
    ///   32+: Communication tokens
    pub fn from_discrete(action_id: u32, comm_vocab_size: u16) -> Option<Action> {
        match action_id {
            0 => Some(Action::Noop),
            1 => Some(Action::Move(Direction::Up)),
            2 => Some(Action::Move(Direction::Down)),
            3 => Some(Action::Move(Direction::Left)),
            4 => Some(Action::Move(Direction::Right)),
            5 => Some(Action::PickUp),
            6..=15 => Some(Action::Drop((action_id - 6) as u8)),
            16..=25 => Some(Action::Use((action_id - 16) as u8)),
            26 => Some(Action::Craft(0)),
            27 => Some(Action::Push(Direction::Up)),
            28 => Some(Action::Push(Direction::Down)),
            29 => Some(Action::Push(Direction::Left)),
            30 => Some(Action::Push(Direction::Right)),
            31 => Some(Action::Interact),
            n if n >= 32 && (n - 32) < comm_vocab_size as u32 => {
                Some(Action::Communicate((n - 32) as CommToken))
            }
            _ => None,
        }
    }

    /// Converts an Action to its discrete integer representation.
    pub fn to_discrete(&self) -> u32 {
        match self {
            Action::Noop => 0,
            Action::Move(Direction::Up) => 1,
            Action::Move(Direction::Down) => 2,
            Action::Move(Direction::Left) => 3,
            Action::Move(Direction::Right) => 4,
            Action::PickUp => 5,
            Action::Drop(slot) => 6 + *slot as u32,
            Action::Use(slot) => 16 + *slot as u32,
            Action::Craft(recipe) => 26 + *recipe as u32,
            Action::Push(Direction::Up) => 27,
            Action::Push(Direction::Down) => 28,
            Action::Push(Direction::Left) => 29,
            Action::Push(Direction::Right) => 30,
            Action::Interact => 31,
            Action::Communicate(token) => 32 + *token as u32,
            Action::Trade(_, _) => 0, // Trade uses extended action space
            Action::Throw(_, _) => 0, // Throw uses extended action space
        }
    }

    /// Returns the total size of the discrete action space.
    pub fn space_size(comm_vocab_size: u16) -> u32 {
        32 + comm_vocab_size as u32
    }
}

/// A trade offer between agents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TradeOffer {
    /// Items offered by the proposer.
    pub offer: Vec<(ItemType, u16)>,
    /// Items requested from the other agent.
    pub request: Vec<(ItemType, u16)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_discrete_roundtrip() {
        let vocab_size = 16;
        let actions = vec![
            Action::Noop,
            Action::Move(Direction::Up),
            Action::Move(Direction::Down),
            Action::Move(Direction::Left),
            Action::Move(Direction::Right),
            Action::PickUp,
            Action::Drop(0),
            Action::Drop(5),
            Action::Use(0),
            Action::Use(9),
            Action::Push(Direction::Up),
            Action::Push(Direction::Right),
            Action::Interact,
            Action::Communicate(0),
            Action::Communicate(15),
        ];

        for action in actions {
            let discrete = action.to_discrete();
            let recovered = Action::from_discrete(discrete, vocab_size).unwrap();
            assert_eq!(action, recovered, "roundtrip failed for {:?}", action);
        }
    }

    #[test]
    fn test_action_space_size() {
        assert_eq!(Action::space_size(0), 32);
        assert_eq!(Action::space_size(16), 48);
        assert_eq!(Action::space_size(256), 288);
    }

    #[test]
    fn test_invalid_discrete_action() {
        assert!(Action::from_discrete(1000, 16).is_none());
    }

    #[test]
    fn test_noop_is_zero() {
        assert_eq!(Action::Noop.to_discrete(), 0);
    }
}
