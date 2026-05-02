//! High-level actuator commands the bridge can emit.
//!
//! Commands are deliberately abstract — they describe *what* to do
//! (`drive_direction`, `engage_sweeper`) rather than *how* (PWM duty cycles,
//! servo PWM widths). The concrete encoding is the [`crate::ActuatorDriver`]
//! implementation's responsibility, which keeps this crate hardware-agnostic
//! and reusable across drive trains, gripper geometries, and bus protocols.

use serde::{Deserialize, Serialize};

/// Cardinal direction for drive primitives.
///
/// Matches the FORGE world's grid-aligned movement model. Hex domains can
/// reuse [`ActuatorCommand::Custom`] with a hex-direction payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CardinalDirection {
    /// Move toward decreasing y / north.
    Up,
    /// Move toward increasing y / south.
    Down,
    /// Move toward decreasing x / west.
    Left,
    /// Move toward increasing x / east.
    Right,
}

/// A single high-level actuator command.
///
/// The `kind` discriminator means TOML mappings stay readable
/// (`{ kind = "drive_direction", direction = "up", distance_mm = 30 }`) and
/// new variants can be added in a backwards-compatible way: existing TOML
/// files that don't reference a new variant continue to deserialize.
///
/// Variants intentionally cover the union of common manipulator + mobile-base
/// primitives. Domain-specific routines (e.g. "sink_edge_sweep") use
/// [`ActuatorCommand::Custom`] so the core enum doesn't grow per-deployment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActuatorCommand {
    /// Drive the chassis a fixed distance in a cardinal direction.
    DriveDirection {
        /// Direction of travel.
        direction: CardinalDirection,
        /// Distance in millimetres. Drivers are free to clamp to their
        /// minimum step, but should never exceed this distance.
        distance_mm: u32,
    },
    /// Open the gripper (release inventory / let go).
    OpenGripper,
    /// Close the gripper (pick up).
    CloseGripper,
    /// Lower the sweeper bar / squeegee onto the work surface.
    EngageSweeper,
    /// Raise the sweeper bar / squeegee off the work surface.
    DisengageSweeper,
    /// Run a brief vibration burst (e.g. to dislodge stuck debris).
    Vibrate {
        /// Duration in milliseconds.
        duration_ms: u32,
    },
    /// Stop all motion immediately and hold position.
    Halt,
    /// Trigger a domain-specific subroutine identified by `name`. Drivers
    /// that don't recognise the name should return
    /// [`crate::ActuatorError::DriverFailure`] so the caller can route to a
    /// safe-pose fallback.
    Custom {
        /// Subroutine name (e.g. `"sink_edge_sweep"`, `"dock_align"`).
        name: String,
        /// Optional opaque payload (driver-defined encoding).
        #[serde(default)]
        payload: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drive_direction_roundtrips_through_toml() {
        let cmd = ActuatorCommand::DriveDirection {
            direction: CardinalDirection::Up,
            distance_mm: 30,
        };
        let toml_str = toml::to_string(&cmd).unwrap();
        let parsed: ActuatorCommand = toml::from_str(&toml_str).unwrap();
        assert_eq!(cmd, parsed);
    }

    #[test]
    fn unit_variant_roundtrips_through_toml() {
        for cmd in [
            ActuatorCommand::OpenGripper,
            ActuatorCommand::CloseGripper,
            ActuatorCommand::EngageSweeper,
            ActuatorCommand::DisengageSweeper,
            ActuatorCommand::Halt,
        ] {
            let s = toml::to_string(&cmd).unwrap();
            let parsed: ActuatorCommand = toml::from_str(&s).unwrap();
            assert_eq!(cmd, parsed);
        }
    }

    #[test]
    fn vibrate_roundtrips_through_toml() {
        let cmd = ActuatorCommand::Vibrate { duration_ms: 250 };
        let s = toml::to_string(&cmd).unwrap();
        let parsed: ActuatorCommand = toml::from_str(&s).unwrap();
        assert_eq!(cmd, parsed);
    }

    #[test]
    fn custom_with_payload_roundtrips_through_toml() {
        let cmd = ActuatorCommand::Custom {
            name: "sink_edge_sweep".to_string(),
            payload: Some("angle=15".to_string()),
        };
        let s = toml::to_string(&cmd).unwrap();
        let parsed: ActuatorCommand = toml::from_str(&s).unwrap();
        assert_eq!(cmd, parsed);
    }

    #[test]
    fn custom_without_payload_defaults_to_none() {
        // Backwards-compat: an older TOML that omits `payload` must still parse.
        let cmd: ActuatorCommand = toml::from_str(
            r#"
kind = "custom"
name = "dock_align"
"#,
        )
        .unwrap();
        assert_eq!(
            cmd,
            ActuatorCommand::Custom {
                name: "dock_align".to_string(),
                payload: None,
            }
        );
    }

    #[test]
    fn cardinal_direction_serde_uses_snake_case() {
        // Bare enums can't sit at TOML's top level, so we verify the rename
        // by serialising a containing struct and checking the rendered field.
        #[derive(Serialize)]
        struct Wrap {
            direction: CardinalDirection,
        }
        let s = toml::to_string(&Wrap {
            direction: CardinalDirection::Right,
        })
        .unwrap();
        assert!(
            s.contains("\"right\""),
            "expected snake_case serialization, got {s:?}"
        );
    }

    #[test]
    fn unknown_command_kind_is_rejected() {
        let err = toml::from_str::<ActuatorCommand>(
            r#"
kind = "teleport"
"#,
        )
        .unwrap_err();
        // Unknown variant must produce a deserialization error rather than
        // silently being treated as Halt or Custom — otherwise typos in TOML
        // mappings would map silently to no-ops on hardware.
        assert!(err.to_string().to_lowercase().contains("unknown"));
    }
}
