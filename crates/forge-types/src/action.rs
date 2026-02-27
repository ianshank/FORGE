//! Action types and action space definitions for the FORGE simulation.
//!
//! Actions are the interface through which agents interact with the world.
//! The discrete action space is designed for efficient GPU batching.

use serde::{Deserialize, Serialize};

use crate::entity::CommToken;
use crate::grid::Direction;

/// An action that an agent can take in a single tick.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// Emit a communication token.
    Communicate(CommToken),
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
    ///   26-34: Craft(recipe 0-8)
    ///   35-38: Push (Up, Down, Left, Right)
    ///   39: Interact
    ///   40+: Communication tokens
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
            26..=34 => Some(Action::Craft((action_id - 26) as u16)),
            35 => Some(Action::Push(Direction::Up)),
            36 => Some(Action::Push(Direction::Down)),
            37 => Some(Action::Push(Direction::Left)),
            38 => Some(Action::Push(Direction::Right)),
            39 => Some(Action::Interact),
            n if n >= 40 && (n - 40) < comm_vocab_size as u32 => {
                Some(Action::Communicate((n - 40) as CommToken))
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
            Action::Push(Direction::Up) => 35,
            Action::Push(Direction::Down) => 36,
            Action::Push(Direction::Left) => 37,
            Action::Push(Direction::Right) => 38,
            Action::Interact => 39,
            Action::Communicate(token) => 40 + *token as u32,
        }
    }

    /// Returns the total size of the discrete action space.
    pub fn space_size(comm_vocab_size: u16) -> u32 {
        40 + comm_vocab_size as u32
    }
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
            Action::Craft(0),
            Action::Craft(4),
            Action::Craft(8),
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
        assert_eq!(Action::space_size(0), 40);
        assert_eq!(Action::space_size(16), 56);
        assert_eq!(Action::space_size(256), 296);
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
