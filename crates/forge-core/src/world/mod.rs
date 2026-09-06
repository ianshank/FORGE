//! World state and the main simulation entry point.
//!
//! `WorldState` holds all simulation data and provides the `step()` and
//! `reset()` methods that form the core API.

mod debug;
mod observation;
mod reset;
mod serialize;
mod state;
mod step;

#[cfg(test)]
mod tests;

pub use state::WorldState;
