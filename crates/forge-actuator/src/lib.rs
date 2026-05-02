#![deny(missing_docs)]
#![deny(clippy::all)]
//! Action-id → actuator-command bridge for FORGE edge deployments.
//!
//! FORGE policies emit a single discrete `action_id: u32` per decision via
//! [`forge_types::agent_interface::AgentResponse`]. Physical robots need that
//! id translated into a sequence of hardware commands — drive a wheel, close
//! a gripper, lower a sweeper bar, etc. This crate provides that translation
//! layer in a way that is:
//!
//! * **Config-driven.** No action-id-to-command mapping is hard-coded in
//!   Rust. Mappings are loaded from TOML files (see
//!   [`mapping::ActionMapping::from_toml_str`]). Swapping the mapping file
//!   reconfigures the robot for a different domain (kitchen counter, patrol
//!   pickup, agri sampling) without recompiling.
//! * **Hardware-agnostic.** [`bridge::ActuatorDriver`] is the trait every
//!   physical or simulated end-effector implements. The bundled
//!   [`bridge::MockDriver`] captures every command in memory for tests; a
//!   production driver would marshal commands onto a serial bus, ROS topic,
//!   or GPIO pin — outside the scope of this crate.
//! * **Reusable across domains.** [`bridge::MappedActuator`] is the only
//!   bridge implementation; "kitchen-counter robot" is the
//!   `configs/actuator/openclaw_kitchen.toml` mapping plus a real driver,
//!   not a separate code path.
//!
//! See `docs/kitchen_counter_robot.md` for a worked example wiring this
//! crate to a Raspberry Pi 5 + Hailo AI HAT 2 + OpenClaw stack.

pub mod bridge;
pub mod command;
pub mod error;
pub mod mapping;

pub use bridge::{
    ActuatorBridge, ActuatorDriver, DispatchResult, DispatchSource, MappedActuator, MockDriver,
};
pub use command::{ActuatorCommand, CardinalDirection};
pub use error::ActuatorError;
pub use mapping::{ActionMapping, LookupSource};
