//! Adapters for bridging FORGE and MangoMAS action/observation spaces.
//!
//! FORGE uses discrete actions on a 2D grid; MangoMAS uses continuous
//! action spaces (2D for car, 4D for drone). These adapters provide
//! bidirectional mapping.

pub mod action_adapter;
pub mod config_adapter;
pub mod observation_adapter;

pub use action_adapter::{ActionAdapter, DiscreteGridAdapter};
pub use config_adapter::ConfigAdapter;
pub use observation_adapter::{FlatStateAdapter, ObservationAdapter};
