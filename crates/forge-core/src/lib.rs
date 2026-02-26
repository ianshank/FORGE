//! # forge-core
//!
//! Core simulation engine for the FORGE platform.
//!
//! This crate implements the deterministic simulation step function,
//! physics, and all game systems. The step function is designed for
//! <1µs per environment step on a single CPU core.

pub mod combat;
pub mod crafting;
pub mod physics;
pub mod resource;
pub mod rng;
pub mod systems;
pub mod world;

// Re-export the main entry points
pub use world::WorldState;
