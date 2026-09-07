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
            n if n >= crate::constants::ACTION_BASE_COUNT
                && (n - crate::constants::ACTION_BASE_COUNT) < comm_vocab_size as u32 =>
            {
                Some(Action::Communicate(
                    (n - crate::constants::ACTION_BASE_COUNT) as CommToken,
                ))
            }
            n if n >= crate::constants::ACTION_BASE_COUNT + comm_vocab_size as u32 => {
                let drone_base = crate::constants::ACTION_BASE_COUNT + comm_vocab_size as u32;
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
        // The base action layout has no `comm_vocab_size` parameter, so we
        // route through the configured encoder with vocab_size=0 to keep
        // bounds-checking semantics identical between the two entrypoints.
        // `try_to_discrete` is documented as base-action-only, so drone /
        // agri / hex variants will fall through to their typed errors below.
        match self {
            Action::Noop => Ok(0),
            Action::Move(Direction::Up) => Ok(1),
            Action::Move(Direction::Down) => Ok(2),
            Action::Move(Direction::Left) => Ok(3),
            Action::Move(Direction::Right) => Ok(4),
            Action::PickUp => Ok(5),
            Action::Drop(slot) => param_check(
                self,
                "Drop",
                *slot as u32,
                crate::constants::ACTION_DROP_SLOTS as u32,
                6 + *slot as u32,
            ),
            Action::Use(slot) => param_check(
                self,
                "Use",
                *slot as u32,
                crate::constants::ACTION_USE_SLOTS as u32,
                16 + *slot as u32,
            ),
            Action::Craft(recipe) => param_check(
                self,
                "Craft",
                *recipe as u32,
                crate::constants::ACTION_CRAFT_SLOTS as u32,
                26 + *recipe as u32,
            ),
            Action::Push(Direction::Up) => Ok(35),
            Action::Push(Direction::Down) => Ok(36),
            Action::Push(Direction::Left) => Ok(37),
            Action::Push(Direction::Right) => Ok(38),
            Action::Interact => Ok(39),
            // `try_to_discrete` cannot validate Communicate against
            // `comm_vocab_size` (no vocab parameter). Out-of-vocab tokens
            // collide with the drone block; callers that need a strict bound
            // check should use `try_to_discrete_configured` with the active
            // vocab size.
            Action::Communicate(token) => Ok(crate::constants::ACTION_BASE_COUNT + *token as u32),
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
    /// All layout flags are enabled, so layout-gating errors
    /// ([`ActionEncodingError::DroneActionRequiresFullEncoder`],
    /// [`ActionEncodingError::AgriActionUnsupported`],
    /// [`ActionEncodingError::HexActionUnsupported`]) are unreachable. The
    /// remaining failure mode is parameter bounds:
    /// [`ActionEncodingError::ParameterOutOfRange`] for `Drop`/`Use`/`Craft`/
    /// `Communicate`/`DropPayload`/`Spray` whose argument exceeds its allocated
    /// slot/vocab range. Callers that already validate parameters at their own
    /// boundary can safely `.expect()` the result; callers ingesting actions
    /// from untrusted sources should propagate the `Result`.
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
        // Delegate to the fallible variant so the encoding rules and
        // parameter bounds checks live in exactly one place.
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
        let drone_base = crate::constants::ACTION_BASE_COUNT + comm_vocab_size as u32;
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
            Action::Drop(slot) => param_check(
                self,
                "Drop",
                *slot as u32,
                crate::constants::ACTION_DROP_SLOTS as u32,
                6 + *slot as u32,
            ),
            Action::Use(slot) => param_check(
                self,
                "Use",
                *slot as u32,
                crate::constants::ACTION_USE_SLOTS as u32,
                16 + *slot as u32,
            ),
            Action::Craft(recipe) => param_check(
                self,
                "Craft",
                *recipe as u32,
                crate::constants::ACTION_CRAFT_SLOTS as u32,
                26 + *recipe as u32,
            ),
            Action::Push(Direction::Up) => Ok(35),
            Action::Push(Direction::Down) => Ok(36),
            Action::Push(Direction::Left) => Ok(37),
            Action::Push(Direction::Right) => Ok(38),
            Action::Interact => Ok(39),
            // Out-of-vocab tokens would otherwise collide with the drone
            // block (offset `40 + comm_vocab_size`), so we explicitly bound
            // the token here.
            Action::Communicate(token) => param_check(
                self,
                "Communicate",
                *token as u32,
                comm_vocab_size as u32,
                crate::constants::ACTION_BASE_COUNT + *token as u32,
            ),
            Action::Ascend => drone_check(self, drone_actions_enabled, || Ok(drone_base)),
            Action::Descend => drone_check(self, drone_actions_enabled, || Ok(drone_base + 1)),
            Action::Hover => drone_check(self, drone_actions_enabled, || Ok(drone_base + 2)),
            Action::TakeOff => drone_check(self, drone_actions_enabled, || Ok(drone_base + 3)),
            Action::Land => drone_check(self, drone_actions_enabled, || Ok(drone_base + 4)),
            Action::Scan(Direction::Up) => {
                drone_check(self, drone_actions_enabled, || Ok(drone_base + 5))
            }
            Action::Scan(Direction::Down) => {
                drone_check(self, drone_actions_enabled, || Ok(drone_base + 6))
            }
            Action::Scan(Direction::Left) => {
                drone_check(self, drone_actions_enabled, || Ok(drone_base + 7))
            }
            Action::Scan(Direction::Right) => {
                drone_check(self, drone_actions_enabled, || Ok(drone_base + 8))
            }
            Action::DropPayload(slot) => drone_check(self, drone_actions_enabled, || {
                param_check(
                    self,
                    "DropPayload",
                    *slot as u32,
                    crate::constants::ACTION_DROP_PAYLOAD_SLOTS as u32,
                    drone_base + 9 + *slot as u32,
                )
            }),
            // Agricultural actions: after drone actions
            Action::Spray(slot) => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    param_check(
                        self,
                        "Spray",
                        *slot as u32,
                        crate::constants::ACTION_SPRAY_SLOTS as u32,
                        agri_base + *slot as u32,
                    )
                })
            }
            Action::ScanMultispectral => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    Ok(agri_base + 10)
                })
            }
            Action::ScanThermal => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    Ok(agri_base + 11)
                })
            }
            Action::RelaySoilData => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    Ok(agri_base + 12)
                })
            }
            Action::GenerateReport => {
                agri_check(self, drone_actions_enabled, agri_actions_enabled, || {
                    Ok(agri_base + 13)
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
        let base = crate::constants::ACTION_BASE_COUNT + comm_vocab_size as u32;
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

/// Helper used by [`Action::try_to_discrete_configured`] to gate every drone
/// variant on `drone_actions_enabled`, symmetric with [`agri_check`].
///
/// Without this guard the configured encoder would happily encode
/// `Action::Ascend` to `40 + comm_vocab_size` even when drone actions are
/// disabled — but that ID equals `space_size_full(_, drone=false, _, _)`, so
/// it is out-of-bounds for the active action space and disagrees with
/// [`Action::from_discrete_full`] which returns `None` for the same input.
///
/// `compute_id` is only invoked when drone actions are enabled. The closure
/// returns its own `Result` so it can perform a downstream parameter-bounds
/// check (e.g. for `DropPayload`'s slot) without fighting the layout-gating
/// logic.
#[inline]
fn drone_check<F>(
    action: &Action,
    drone_actions_enabled: bool,
    compute_id: F,
) -> Result<u32, ActionEncodingError>
where
    F: FnOnce() -> Result<u32, ActionEncodingError>,
{
    if drone_actions_enabled {
        compute_id()
    } else {
        let err = ActionEncodingError::DroneActionRequiresFullEncoder {
            action_name: drone_action_name(action),
        };
        warn!(target: "forge_types::action", action = ?action, "{err}");
        Err(err)
    }
}

/// Helper used by [`Action::try_to_discrete_configured`] to gate every
/// agricultural variant on `drone_actions_enabled && agri_actions_enabled`
/// without duplicating the error construction at every match arm.
///
/// `compute_id` is only invoked when both flags are enabled. The closure
/// returns its own `Result` so it can perform a downstream parameter-bounds
/// check (see [`param_check`]) without fighting the layout-gating logic.
#[inline]
fn agri_check<F>(
    action: &Action,
    drone_actions_enabled: bool,
    agri_actions_enabled: bool,
    compute_id: F,
) -> Result<u32, ActionEncodingError>
where
    F: FnOnce() -> Result<u32, ActionEncodingError>,
{
    if drone_actions_enabled && agri_actions_enabled {
        compute_id()
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

/// Helper that validates an action parameter (inventory slot, recipe index,
/// communication token, drone payload slot, etc.) against its allocated range
/// in the discrete action space. Returns the supplied `id` on success, or a
/// typed [`ActionEncodingError::ParameterOutOfRange`] on failure.
///
/// Without this check, an out-of-range parameter silently produces a
/// valid-looking discrete ID that collides with the next action block — for
/// example `Action::Drop(slot=10)` would otherwise encode to ID 16, which is
/// the slot for `Action::Use(slot=0)`.
#[inline]
fn param_check(
    action: &Action,
    name: &'static str,
    value: u32,
    limit: u32,
    id: u32,
) -> Result<u32, ActionEncodingError> {
    if value >= limit {
        let err = ActionEncodingError::ParameterOutOfRange {
            action_name: name,
            value,
            max: limit.saturating_sub(1),
        };
        warn!(target: "forge_types::action", action = ?action, "{err}");
        Err(err)
    } else {
        Ok(id)
    }
}

#[cfg(test)]
#[path = "action/tests.rs"]
mod tests;
