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
    /// Increase altitude by 1 (Aerial only).
    Ascend,
    /// Decrease altitude by 1 (Aerial only).
    Descend,
    /// Hover in place, maintaining altitude (Aerial only, costs battery).
    Hover,
    /// Transition from ground to airborne at altitude 1 (Aerial only).
    TakeOff,
    /// Land: set altitude to 0 (Aerial only).
    Land,
    /// Directional sensor sweep with extended range.
    Scan(Direction),
    /// Drop payload from inventory slot to ground tile below (Aerial only).
    DropPayload(u8),
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
    pub fn from_discrete(
        action_id: u32,
        comm_vocab_size: u16,
        drone_actions_enabled: bool,
    ) -> Option<Action> {
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
            n if drone_actions_enabled && n >= 40 + comm_vocab_size as u32 => {
                let drone_offset = n - 40 - comm_vocab_size as u32;
                match drone_offset {
                    0 => Some(Action::Ascend),
                    1 => Some(Action::Descend),
                    2 => Some(Action::Hover),
                    3 => Some(Action::TakeOff),
                    4 => Some(Action::Land),
                    5 => Some(Action::Scan(Direction::Up)),
                    6 => Some(Action::Scan(Direction::Down)),
                    7 => Some(Action::Scan(Direction::Left)),
                    8 => Some(Action::Scan(Direction::Right)),
                    d @ 9..=18 => Some(Action::DropPayload((d - 9) as u8)),
                    _ => None,
                }
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
            Action::Ascend => 40,
            Action::Descend => 41,
            Action::Hover => 42,
            Action::TakeOff => 43,
            Action::Land => 44,
            Action::Scan(Direction::Up) => 45,
            Action::Scan(Direction::Down) => 46,
            Action::Scan(Direction::Left) => 47,
            Action::Scan(Direction::Right) => 48,
            Action::DropPayload(slot) => 49 + *slot as u32,
        }
    }

    /// Converts an Action to its discrete integer representation, accounting for drone actions.
    ///
    /// Drone actions are encoded after communication tokens at offset `40 + comm_vocab_size`.
    pub fn to_discrete_full(&self, comm_vocab_size: u16) -> u32 {
        let base = self.to_discrete();
        match self {
            Action::Ascend => 40 + comm_vocab_size as u32,
            Action::Descend => 40 + comm_vocab_size as u32 + 1,
            Action::Hover => 40 + comm_vocab_size as u32 + 2,
            Action::TakeOff => 40 + comm_vocab_size as u32 + 3,
            Action::Land => 40 + comm_vocab_size as u32 + 4,
            Action::Scan(Direction::Up) => 40 + comm_vocab_size as u32 + 5,
            Action::Scan(Direction::Down) => 40 + comm_vocab_size as u32 + 6,
            Action::Scan(Direction::Left) => 40 + comm_vocab_size as u32 + 7,
            Action::Scan(Direction::Right) => 40 + comm_vocab_size as u32 + 8,
            Action::DropPayload(slot) => 40 + comm_vocab_size as u32 + 9 + *slot as u32,
            _ => base,
        }
    }

    /// Returns the total size of the discrete action space.
    ///
    /// When `drone_actions_enabled` is true, includes 19 additional actions
    /// for drone control (Ascend, Descend, Hover, TakeOff, Land, 4 Scan, 10 DropPayload).
    pub fn space_size(comm_vocab_size: u16, drone_actions_enabled: bool) -> u32 {
        let base = 40 + comm_vocab_size as u32;
        if drone_actions_enabled {
            base + crate::constants::DRONE_ACTION_COUNT
        } else {
            base
        }
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
            let recovered = Action::from_discrete(discrete, vocab_size, false).unwrap();
            assert_eq!(action, recovered, "roundtrip failed for {:?}", action);
        }
    }

    #[test]
    fn test_action_space_size() {
        assert_eq!(Action::space_size(0, false), 40);
        assert_eq!(Action::space_size(16, false), 56);
        assert_eq!(Action::space_size(256, false), 296);
    }

    #[test]
    fn test_invalid_discrete_action() {
        assert!(Action::from_discrete(1000, 16, false).is_none());
    }

    #[test]
    fn test_noop_is_zero() {
        assert_eq!(Action::Noop.to_discrete(), 0);
    }

    #[test]
    fn test_drone_action_discrete_roundtrip() {
        let vocab_size = 16;
        let drone_base = 40 + vocab_size as u32;
        let drone_actions: Vec<(u32, Action)> = vec![
            (drone_base, Action::Ascend),
            (drone_base + 1, Action::Descend),
            (drone_base + 2, Action::Hover),
            (drone_base + 3, Action::TakeOff),
            (drone_base + 4, Action::Land),
            (drone_base + 5, Action::Scan(Direction::Up)),
            (drone_base + 6, Action::Scan(Direction::Down)),
            (drone_base + 7, Action::Scan(Direction::Left)),
            (drone_base + 8, Action::Scan(Direction::Right)),
            (drone_base + 9, Action::DropPayload(0)),
            (drone_base + 18, Action::DropPayload(9)),
        ];
        for (id, expected) in &drone_actions {
            let decoded = Action::from_discrete(*id, vocab_size, true);
            assert_eq!(decoded.as_ref(), Some(expected), "failed for id {}", id);
        }
    }

    #[test]
    fn test_drone_to_discrete_full_roundtrip() {
        let vocab_size = 16;
        let actions = vec![
            Action::Ascend,
            Action::Descend,
            Action::Hover,
            Action::TakeOff,
            Action::Land,
            Action::Scan(Direction::Up),
            Action::Scan(Direction::Down),
            Action::Scan(Direction::Left),
            Action::Scan(Direction::Right),
            Action::DropPayload(0),
            Action::DropPayload(9),
        ];
        for action in &actions {
            let id = action.to_discrete_full(vocab_size);
            let decoded = Action::from_discrete(id, vocab_size, true).unwrap();
            assert_eq!(&decoded, action);
        }
    }

    #[test]
    fn test_space_size_without_drones() {
        assert_eq!(Action::space_size(16, false), 56);
    }

    #[test]
    fn test_space_size_with_drones() {
        assert_eq!(Action::space_size(16, true), 56 + 19);
    }

    #[test]
    fn test_drone_actions_not_decoded_when_disabled() {
        let vocab_size = 16;
        let drone_base = 40 + vocab_size as u32;
        assert!(Action::from_discrete(drone_base, vocab_size, false).is_none());
    }
}
