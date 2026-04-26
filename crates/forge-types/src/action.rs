//! Action types and action space definitions for the FORGE simulation.
//!
//! Actions are the interface through which agents interact with the world.
//! The discrete action space is designed for efficient GPU batching.

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::entity::CommToken;
use crate::error::ActionEncodingError;
use crate::grid::{Direction, HexDirection};

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
    /// Spray pesticide from inventory slot over target area (Aerial only, airborne).
    Spray(u8),
    /// Perform multispectral NDVI scan of nearby crop tiles (Aerial only, airborne).
    ScanMultispectral,
    /// Perform thermal scan for irrigation stress mapping (Aerial only, airborne).
    ScanThermal,
    /// Relay data from nearby ground-deployed soil sensor nodes.
    RelaySoilData,
    /// Generate an agronomic field report from collected scan data.
    GenerateReport,
    /// Move one tile in a hex direction (hex grid only; Noop on square grid).
    MoveHex(HexDirection),
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
    ///   40..40+comm_vocab_size: Communication tokens
    ///   40+comm_vocab_size..: Drone actions (when `drone_actions_enabled`)
    ///   40+comm_vocab_size+19..: Agricultural actions (when `agri_actions_enabled`)
    ///
    /// Returns `None` if `action_id` is out of range for the given configuration.
    pub fn from_discrete(
        action_id: u32,
        comm_vocab_size: u16,
        drone_actions_enabled: bool,
    ) -> Option<Action> {
        Self::from_discrete_full(
            action_id,
            comm_vocab_size,
            drone_actions_enabled,
            false,
            false,
        )
    }

    /// Converts a flat integer action index to an Action, with agricultural action support.
    ///
    /// Agricultural actions are encoded after drone actions when `agri_actions_enabled` is true.
    /// Hex movement actions are encoded after agricultural actions when `hex_actions_enabled` is true.
    pub fn from_discrete_full(
        action_id: u32,
        comm_vocab_size: u16,
        drone_actions_enabled: bool,
        agri_actions_enabled: bool,
        hex_actions_enabled: bool,
    ) -> Option<Action> {
        // Early bounds check
        if action_id
            >= Self::space_size_full(
                comm_vocab_size,
                drone_actions_enabled,
                agri_actions_enabled,
                hex_actions_enabled,
            )
        {
            return None;
        }
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
            n if n >= 40 + comm_vocab_size as u32 => {
                let drone_base = 40 + comm_vocab_size as u32;
                let offset = n - drone_base;

                // Drone actions block
                if drone_actions_enabled && offset < crate::constants::DRONE_ACTION_COUNT {
                    return match offset {
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
                    };
                }

                // Agri actions block
                let agri_base_offset = if drone_actions_enabled {
                    crate::constants::DRONE_ACTION_COUNT
                } else {
                    0
                };
                let agri_offset = offset.checked_sub(agri_base_offset)?;
                if agri_actions_enabled
                    && drone_actions_enabled
                    && agri_offset < crate::constants::AGRI_ACTION_COUNT
                {
                    return match agri_offset {
                        d @ 0..=9 => Some(Action::Spray(d as u8)),
                        10 => Some(Action::ScanMultispectral),
                        11 => Some(Action::ScanThermal),
                        12 => Some(Action::RelaySoilData),
                        13 => Some(Action::GenerateReport),
                        _ => None,
                    };
                }

                // Hex movement actions block
                let hex_base_offset = agri_base_offset
                    + if agri_actions_enabled && drone_actions_enabled {
                        crate::constants::AGRI_ACTION_COUNT
                    } else {
                        0
                    };
                let hex_offset = offset.checked_sub(hex_base_offset)?;
                if hex_actions_enabled && hex_offset < crate::constants::HEX_ACTION_COUNT {
                    return HexDirection::from_index(hex_offset as u8).map(Action::MoveHex);
                }

                None
            }
            _ => None,
        }
    }

    /// Converts an Action to its discrete integer representation (base actions only).
    ///
    /// **Important**: This method only produces correct, non-colliding IDs for base
    /// actions (Noop, Move, PickUp, Drop, Use, Craft, Push, Interact, Communicate).
    /// For drone actions (Ascend, Descend, Hover, TakeOff, Land, Scan, DropPayload),
    /// use [`Action::to_discrete_full`] which accounts for the communication
    /// vocabulary offset.
    ///
    /// # Panics
    ///
    /// Panics if called on a drone, agricultural, or hex-move action. Prefer
    /// [`Action::try_to_discrete`] for fallible callers.
    pub fn to_discrete(&self) -> u32 {
        // Delegate to the fallible variant so the encoding rules live in exactly
        // one place. The panic path is preserved for back-compat with callers
        // that have always been documented to pass base actions only.
        self.try_to_discrete()
            .unwrap_or_else(|e| panic!("Action::to_discrete: {e}"))
    }

    /// Fallible counterpart of [`Action::to_discrete`].
    ///
    /// Returns [`ActionEncodingError`] instead of panicking when called on a
    /// drone, agricultural, or hex-move action. This is the preferred entry
    /// point for new code that may receive actions from untrusted sources
    /// (e.g. external policies, replay loaders, RPC handlers).
    pub fn try_to_discrete(&self) -> Result<u32, ActionEncodingError> {
        match self {
            Action::Noop => Ok(0),
            Action::Move(Direction::Up) => Ok(1),
            Action::Move(Direction::Down) => Ok(2),
            Action::Move(Direction::Left) => Ok(3),
            Action::Move(Direction::Right) => Ok(4),
            Action::PickUp => Ok(5),
            Action::Drop(slot) => Ok(6 + *slot as u32),
            Action::Use(slot) => Ok(16 + *slot as u32),
            Action::Craft(recipe) => Ok(26 + *recipe as u32),
            Action::Push(Direction::Up) => Ok(35),
            Action::Push(Direction::Down) => Ok(36),
            Action::Push(Direction::Left) => Ok(37),
            Action::Push(Direction::Right) => Ok(38),
            Action::Interact => Ok(39),
            Action::Communicate(token) => Ok(40 + *token as u32),
            Action::Ascend
            | Action::Descend
            | Action::Hover
            | Action::TakeOff
            | Action::Land
            | Action::Scan(_)
            | Action::DropPayload(_) => {
                let err = ActionEncodingError::DroneActionRequiresFullEncoder {
                    action_name: drone_action_name(self),
                };
                warn!(target: "forge_types::action", action = ?self, "{err}");
                Err(err)
            }
            Action::Spray(_)
            | Action::ScanMultispectral
            | Action::ScanThermal
            | Action::RelaySoilData
            | Action::GenerateReport => {
                let err = ActionEncodingError::AgriActionUnsupported {
                    action_name: agri_action_name(self),
                    drone_actions_enabled: false,
                    agri_actions_enabled: false,
                };
                warn!(target: "forge_types::action", action = ?self, "{err}");
                Err(err)
            }
            Action::MoveHex(_) => {
                let err = ActionEncodingError::HexActionUnsupported {
                    hex_actions_enabled: false,
                };
                warn!(target: "forge_types::action", action = ?self, "{err}");
                Err(err)
            }
        }
    }

    /// Converts an Action to its discrete integer representation, accounting for drone actions.
    ///
    /// Drone actions are encoded after communication tokens at offset `40 + comm_vocab_size`.
    /// This is the canonical encoding method that produces non-colliding IDs for all actions
    /// including drone actions.
    ///
    /// For configuration-dependent spaces that enable agricultural or hex actions conditionally,
    /// prefer [`to_discrete_configured`] so offsets match the active action space layout.
    pub fn to_discrete_full(&self, comm_vocab_size: u16) -> u32 {
        self.to_discrete_configured(comm_vocab_size, true, true, true)
    }

    /// Fallible counterpart of [`Action::to_discrete_full`].
    ///
    /// All variants are enabled in the layout selected by this method, so the
    /// only failure modes are unreachable in practice. The `Result`-typed
    /// signature is provided so callers building generic encoder pipelines do
    /// not need to special-case the canonical path.
    pub fn try_to_discrete_full(&self, comm_vocab_size: u16) -> Result<u32, ActionEncodingError> {
        self.try_to_discrete_configured(comm_vocab_size, true, true, true)
    }

    /// Converts an Action to its discrete integer representation for a specific action-space layout.
    ///
    /// This is the correct encoder when the active action space is controlled by configuration,
    /// because agricultural actions depend on drone support and hex actions are only appended when
    /// hex movement is enabled.
    ///
    /// # Panics
    ///
    /// Panics if the action variant is not enabled in the requested layout (for
    /// example, [`Action::Spray`] with `agri_actions_enabled=false`). Prefer
    /// [`Action::try_to_discrete_configured`] for fallible callers.
    pub fn to_discrete_configured(
        &self,
        comm_vocab_size: u16,
        drone_actions_enabled: bool,
        agri_actions_enabled: bool,
        hex_actions_enabled: bool,
    ) -> u32 {
        // Delegate to the fallible variant so the encoding rules (and the
        // checked offset arithmetic) live in exactly one place.
        self.try_to_discrete_configured(
            comm_vocab_size,
            drone_actions_enabled,
            agri_actions_enabled,
            hex_actions_enabled,
        )
        .unwrap_or_else(|e| panic!("Action::to_discrete_configured: {e}"))
    }

    /// Fallible counterpart of [`Action::to_discrete_configured`].
    ///
    /// Returns an [`ActionEncodingError`] instead of panicking when the action
    /// variant is not enabled in the requested layout. This is the preferred
    /// entry point for new code that drives encoding from runtime
    /// configuration (e.g. RPC handlers, replay loaders, multi-config
    /// curricula).
    pub fn try_to_discrete_configured(
        &self,
        comm_vocab_size: u16,
        drone_actions_enabled: bool,
        agri_actions_enabled: bool,
        hex_actions_enabled: bool,
    ) -> Result<u32, ActionEncodingError> {
        let drone_base = 40u32 + comm_vocab_size as u32;
        let agri_base = drone_base
            + if drone_actions_enabled {
                crate::constants::DRONE_ACTION_COUNT
            } else {
                0
            };
        let hex_base = agri_base
            + if agri_actions_enabled && drone_actions_enabled {
                crate::constants::AGRI_ACTION_COUNT
            } else {
                0
            };
        match self {
            Action::Noop => Ok(0),
            Action::Move(Direction::Up) => Ok(1),
            Action::Move(Direction::Down) => Ok(2),
            Action::Move(Direction::Left) => Ok(3),
            Action::Move(Direction::Right) => Ok(4),
            Action::PickUp => Ok(5),
            Action::Drop(slot) => Ok(6 + *slot as u32),
            Action::Use(slot) => Ok(16 + *slot as u32),
            Action::Craft(recipe) => Ok(26 + *recipe as u32),
            Action::Push(Direction::Up) => Ok(35),
            Action::Push(Direction::Down) => Ok(36),
            Action::Push(Direction::Left) => Ok(37),
            Action::Push(Direction::Right) => Ok(38),
            Action::Interact => Ok(39),
            Action::Communicate(token) => Ok(40 + *token as u32),
            Action::Ascend => Ok(drone_base),
            Action::Descend => Ok(drone_base + 1),
            Action::Hover => Ok(drone_base + 2),
            Action::TakeOff => Ok(drone_base + 3),
            Action::Land => Ok(drone_base + 4),
            Action::Scan(Direction::Up) => Ok(drone_base + 5),
            Action::Scan(Direction::Down) => Ok(drone_base + 6),
            Action::Scan(Direction::Left) => Ok(drone_base + 7),
            Action::Scan(Direction::Right) => Ok(drone_base + 8),
            Action::DropPayload(slot) => Ok(drone_base + 9 + *slot as u32),
            // Agricultural actions: after drone actions
            Action::Spray(slot) => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    agri_base + *slot as u32
                })
            }
            Action::ScanMultispectral => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    agri_base + 10
                })
            }
            Action::ScanThermal => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    agri_base + 11
                })
            }
            Action::RelaySoilData => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    agri_base + 12
                })
            }
            Action::GenerateReport => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    agri_base + 13
                })
            }
            // Hex movement actions: after agricultural actions
            Action::MoveHex(dir) => {
                if !hex_actions_enabled {
                    let err = ActionEncodingError::HexActionUnsupported {
                        hex_actions_enabled,
                    };
                    warn!(target: "forge_types::action", action = ?self, "{err}");
                    return Err(err);
                }
                Ok(hex_base + *dir as u32)
            }
        }
    }

    /// Returns the total size of the discrete action space.
    ///
    /// When `drone_actions_enabled` is true, includes 19 additional actions
    /// for drone control (Ascend, Descend, Hover, TakeOff, Land, 4 Scan, 10 DropPayload).
    pub fn space_size(comm_vocab_size: u16, drone_actions_enabled: bool) -> u32 {
        Self::space_size_full(comm_vocab_size, drone_actions_enabled, false, false)
    }

    /// Returns the total size of the discrete action space with agricultural and hex actions.
    ///
    /// Agricultural actions are appended after drone actions and require
    /// `drone_actions_enabled` to be true (they depend on drone infrastructure).
    /// Hex movement actions are appended after agricultural actions when enabled.
    pub fn space_size_full(
        comm_vocab_size: u16,
        drone_actions_enabled: bool,
        agri_actions_enabled: bool,
        hex_actions_enabled: bool,
    ) -> u32 {
        let base = 40 + comm_vocab_size as u32;
        let drone = if drone_actions_enabled {
            crate::constants::DRONE_ACTION_COUNT
        } else {
            0
        };
        let agri = if agri_actions_enabled && drone_actions_enabled {
            crate::constants::AGRI_ACTION_COUNT
        } else {
            0
        };
        let hex = if hex_actions_enabled {
            crate::constants::HEX_ACTION_COUNT
        } else {
            0
        };
        base + drone + agri + hex
    }
}

/// Static, debug-style name for a drone-class [`Action`] variant.
///
/// Used to enrich [`ActionEncodingError`] without forcing a heap allocation.
/// Returns `"<non-drone>"` for non-drone actions; the encoder never calls this
/// helper on a non-drone variant, so that branch is unreachable in production.
#[inline]
fn drone_action_name(action: &Action) -> &'static str {
    match action {
        Action::Ascend => "Ascend",
        Action::Descend => "Descend",
        Action::Hover => "Hover",
        Action::TakeOff => "TakeOff",
        Action::Land => "Land",
        Action::Scan(_) => "Scan",
        Action::DropPayload(_) => "DropPayload",
        _ => "<non-drone>",
    }
}

/// Static, debug-style name for an agricultural [`Action`] variant.
///
/// See [`drone_action_name`] for the same allocation-free contract.
#[inline]
fn agri_action_name(action: &Action) -> &'static str {
    match action {
        Action::Spray(_) => "Spray",
        Action::ScanMultispectral => "ScanMultispectral",
        Action::ScanThermal => "ScanThermal",
        Action::RelaySoilData => "RelaySoilData",
        Action::GenerateReport => "GenerateReport",
        _ => "<non-agri>",
    }
}

/// Helper used by [`Action::try_to_discrete_configured`] to gate every
/// agricultural variant on `drone_actions_enabled && agri_actions_enabled`
/// without duplicating the error construction at every match arm.
///
/// `compute_id` is only invoked when both flags are enabled, so it can do the
/// final offset arithmetic unconditionally.
#[inline]
fn agri_check<F>(
    action: &Action,
    drone_actions_enabled: bool,
    agri_actions_enabled: bool,
    compute_id: F,
) -> Result<u32, ActionEncodingError>
where
    F: FnOnce() -> u32,
{
    if drone_actions_enabled && agri_actions_enabled {
        Ok(compute_id())
    } else {
        let err = ActionEncodingError::AgriActionUnsupported {
            action_name: agri_action_name(action),
            drone_actions_enabled,
            agri_actions_enabled,
        };
        warn!(target: "forge_types::action", action = ?action, "{err}");
        Err(err)
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

    #[test]
    fn test_from_discrete_bounds_check() {
        // Beyond space_size should return None
        let max_id = Action::space_size(16, true);
        assert!(Action::from_discrete(max_id, 16, true).is_none());
        assert!(Action::from_discrete(max_id + 1, 16, true).is_none());

        // Last valid action should succeed
        assert!(Action::from_discrete(max_id - 1, 16, true).is_some());
    }

    #[test]
    fn test_to_discrete_full_no_collision_with_comm_tokens() {
        let vocab_size = 16u16;
        // Verify drone actions don't collide with communication tokens
        let comm_ids: Vec<u32> = (0..vocab_size)
            .map(|t| Action::Communicate(t).to_discrete_full(vocab_size))
            .collect();
        let drone_actions = [
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
        ];
        for da in &drone_actions {
            let id = da.to_discrete_full(vocab_size);
            assert!(
                !comm_ids.contains(&id),
                "drone action {:?} (id={}) collides with communication token",
                da,
                id
            );
        }
    }

    #[test]
    fn test_space_size_zero_vocab() {
        assert_eq!(Action::space_size(0, false), 40);
        assert_eq!(Action::space_size(0, true), 40 + 19);
    }

    #[test]
    fn test_full_roundtrip_all_actions_with_drones() {
        let vocab_size = 8u16;
        let total = Action::space_size(vocab_size, true);
        for id in 0..total {
            let action = Action::from_discrete(id, vocab_size, true);
            assert!(action.is_some(), "id {} should decode to an action", id);
            let action = action.unwrap();
            let roundtrip_id = action.to_discrete_full(vocab_size);
            assert_eq!(
                roundtrip_id, id,
                "roundtrip failed for action {:?} (expected id={}, got id={})",
                action, id, roundtrip_id
            );
        }
    }

    #[test]
    fn test_hex_roundtrip_without_optional_drone_blocks() {
        let vocab_size = 8u16;
        let base = 40 + vocab_size as u32;
        for (index, dir) in HexDirection::ALL.iter().copied().enumerate() {
            let action = Action::MoveHex(dir);
            let id = action.to_discrete_configured(vocab_size, false, false, true);
            assert_eq!(id, base + index as u32);
            let decoded = Action::from_discrete_full(id, vocab_size, false, false, true).unwrap();
            assert_eq!(decoded, action);
        }
    }

    #[test]
    fn test_hex_roundtrip_after_drone_and_agri_blocks() {
        let vocab_size = 8u16;
        let base = 40
            + vocab_size as u32
            + crate::constants::DRONE_ACTION_COUNT
            + crate::constants::AGRI_ACTION_COUNT;
        for (index, dir) in HexDirection::ALL.iter().copied().enumerate() {
            let action = Action::MoveHex(dir);
            let id = action.to_discrete_configured(vocab_size, true, true, true);
            assert_eq!(id, base + index as u32);
            let decoded = Action::from_discrete_full(id, vocab_size, true, true, true).unwrap();
            assert_eq!(decoded, action);
        }
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn from_discrete_roundtrip(
                vocab_size in 0u16..64,
                action_id in 0u32..200,
            ) {
                let space = Action::space_size(vocab_size, true);
                if action_id < space {
                    let action = Action::from_discrete(action_id, vocab_size, true).unwrap();
                    let recovered_id = action.to_discrete_full(vocab_size);
                    prop_assert_eq!(recovered_id, action_id, "roundtrip failed");
                } else {
                    prop_assert!(Action::from_discrete(action_id, vocab_size, true).is_none());
                }
            }

            #[test]
            fn space_size_monotonic_in_vocab(
                vocab_size in 0u16..1024,
            ) {
                let without = Action::space_size(vocab_size, false);
                let with = Action::space_size(vocab_size, true);
                prop_assert!(with > without, "drone actions should increase space size");
                prop_assert_eq!(with - without, crate::constants::DRONE_ACTION_COUNT);
            }

            /// `try_to_discrete_configured` and `to_discrete_configured` agree
            /// on every input that doesn't fail (i.e. their happy paths are
            /// byte-identical). This guards against the panicking path silently
            /// drifting from the fallible path.
            #[test]
            fn try_and_panic_agree_on_happy_path(
                vocab_size in 0u16..32,
                drone in any::<bool>(),
                agri in any::<bool>(),
                hex in any::<bool>(),
                action_id in 0u32..256,
            ) {
                let space = Action::space_size_full(vocab_size, drone, agri, hex);
                if action_id >= space {
                    return Ok(());
                }
                let Some(action) = Action::from_discrete_full(action_id, vocab_size, drone, agri, hex) else {
                    return Ok(());
                };
                // try_* must succeed for any action that round-trips through the
                // configured layout — those are by construction enabled.
                let try_id = action
                    .try_to_discrete_configured(vocab_size, drone, agri, hex)
                    .expect("decoded action must re-encode under the same layout");
                let panic_id = action.to_discrete_configured(vocab_size, drone, agri, hex);
                prop_assert_eq!(try_id, panic_id);
                prop_assert_eq!(try_id, action_id);
            }
        }
    }

    /// Tests for the fallible action encoders ([`Action::try_to_discrete`],
    /// [`Action::try_to_discrete_configured`], [`Action::try_to_discrete_full`])
    /// and their interplay with [`crate::error::ActionEncodingError`].
    mod try_encoder_tests {
        use super::*;
        use crate::error::{ActionEncodingError, ForgeError};

        // ---- try_to_discrete: happy paths ----

        #[test]
        fn try_to_discrete_succeeds_for_base_actions() {
            let cases: &[(Action, u32)] = &[
                (Action::Noop, 0),
                (Action::Move(Direction::Up), 1),
                (Action::Move(Direction::Down), 2),
                (Action::PickUp, 5),
                (Action::Drop(0), 6),
                (Action::Drop(9), 15),
                (Action::Use(0), 16),
                (Action::Craft(0), 26),
                (Action::Push(Direction::Up), 35),
                (Action::Interact, 39),
                (Action::Communicate(0), 40),
                (Action::Communicate(15), 55),
            ];
            for (action, expected) in cases {
                let got = action
                    .try_to_discrete()
                    .expect("base action must encode without error");
                assert_eq!(got, *expected, "wrong id for {action:?}");
            }
        }

        // ---- try_to_discrete: error paths ----

        #[test]
        fn try_to_discrete_rejects_drone_actions() {
            let drone_actions = [
                Action::Ascend,
                Action::Descend,
                Action::Hover,
                Action::TakeOff,
                Action::Land,
                Action::Scan(Direction::Up),
                Action::DropPayload(0),
            ];
            for action in drone_actions {
                let err = action
                    .try_to_discrete()
                    .expect_err("drone action must not encode via try_to_discrete");
                assert!(
                    matches!(
                        err,
                        ActionEncodingError::DroneActionRequiresFullEncoder { .. }
                    ),
                    "wrong error variant for {action:?}: {err:?}",
                );
            }
        }

        #[test]
        fn try_to_discrete_rejects_agri_actions() {
            let agri_actions = [
                Action::Spray(0),
                Action::ScanMultispectral,
                Action::ScanThermal,
                Action::RelaySoilData,
                Action::GenerateReport,
            ];
            for action in agri_actions {
                let err = action
                    .try_to_discrete()
                    .expect_err("agri action must not encode via try_to_discrete");
                assert!(
                    matches!(err, ActionEncodingError::AgriActionUnsupported { .. }),
                    "wrong error variant for {action:?}: {err:?}",
                );
            }
        }

        #[test]
        fn try_to_discrete_rejects_hex_actions() {
            let action = Action::MoveHex(HexDirection::E);
            let err = action
                .try_to_discrete()
                .expect_err("hex action must not encode via try_to_discrete");
            assert!(
                matches!(err, ActionEncodingError::HexActionUnsupported { .. }),
                "wrong error variant: {err:?}",
            );
        }

        // ---- try_to_discrete_configured: error paths ----

        #[test]
        fn try_configured_rejects_agri_when_agri_disabled() {
            let action = Action::Spray(2);
            let err = action
                .try_to_discrete_configured(8, true, false, false)
                .expect_err("agri must require both drone and agri flags");
            match err {
                ActionEncodingError::AgriActionUnsupported {
                    action_name,
                    drone_actions_enabled,
                    agri_actions_enabled,
                } => {
                    assert_eq!(action_name, "Spray");
                    assert!(drone_actions_enabled);
                    assert!(!agri_actions_enabled);
                }
                other => panic!("wrong error variant: {other:?}"),
            }
        }

        #[test]
        fn try_configured_rejects_agri_when_drone_disabled() {
            let action = Action::ScanMultispectral;
            let err = action
                .try_to_discrete_configured(8, false, true, false)
                .expect_err("agri requires drone infrastructure");
            assert!(matches!(
                err,
                ActionEncodingError::AgriActionUnsupported {
                    drone_actions_enabled: false,
                    agri_actions_enabled: true,
                    ..
                }
            ));
        }

        #[test]
        fn try_configured_rejects_hex_when_hex_disabled() {
            let action = Action::MoveHex(HexDirection::NW);
            let err = action
                .try_to_discrete_configured(8, true, true, false)
                .expect_err("hex must require hex_actions_enabled");
            assert!(matches!(
                err,
                ActionEncodingError::HexActionUnsupported {
                    hex_actions_enabled: false,
                }
            ));
        }

        // ---- panicking variants delegate cleanly ----

        #[test]
        #[should_panic(expected = "Action::to_discrete:")]
        fn legacy_to_discrete_still_panics_on_drone() {
            let _ = Action::Ascend.to_discrete();
        }

        #[test]
        #[should_panic(expected = "Action::to_discrete_configured:")]
        fn legacy_to_discrete_configured_still_panics_when_disabled() {
            let _ = Action::Spray(0).to_discrete_configured(8, true, false, false);
        }

        // ---- error type plumbing ----

        #[test]
        fn action_encoding_error_converts_into_forge_error() {
            let err = Action::MoveHex(HexDirection::E)
                .try_to_discrete()
                .unwrap_err();
            let forge_err: ForgeError = err.into();
            assert!(matches!(forge_err, ForgeError::ActionEncoding(_)));
            let msg = forge_err.to_string();
            assert!(
                msg.contains("action encoding error"),
                "missing wrapper prefix in: {msg}",
            );
        }

        #[test]
        fn action_encoding_error_display_includes_action_name() {
            let err = Action::Hover.try_to_discrete().unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("Hover"), "expected action name in: {msg}");
        }

        #[test]
        fn try_to_discrete_full_succeeds_for_every_variant() {
            // Every variant is enabled in the canonical full layout, so there
            // are no failure modes; this guards against regressions if a new
            // variant is added without wiring it through.
            let vocab_size = 16u16;
            let space = Action::space_size_full(vocab_size, true, true, true);
            for id in 0..space {
                let action = Action::from_discrete_full(id, vocab_size, true, true, true).unwrap();
                let roundtrip = action
                    .try_to_discrete_full(vocab_size)
                    .expect("full layout must accept every decoded action");
                assert_eq!(roundtrip, id, "roundtrip failed for {action:?}");
            }
        }
    }
}
