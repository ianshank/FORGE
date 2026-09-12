#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-core
//!
//! Core simulation engine for the FORGE platform.
//!
//! This crate implements the deterministic simulation step function,
//! physics, and all game systems. The step function is designed for
//! <1µs per environment step on a single CPU core.

pub mod agriculture;
pub mod baselines;
pub mod combat;
pub mod communication;
pub mod crafting;
pub mod day_night;
pub mod drone;
pub mod events;
pub mod physics;
pub mod prelude;
pub mod replay;
pub mod resource;
pub mod rng;
pub mod sensor;
pub mod systems;
pub mod visibility;
pub mod world;

// Re-export the main entry points
pub use world::WorldState;
