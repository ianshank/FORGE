//! Grid topology abstraction and pathfinding for the FORGE simulation platform.
//!
//! This crate provides a `GridTopology` trait that abstracts grid-shape-specific
//! operations (neighbor iteration, distance, line-of-sight). Both square and hex
//! grids use the same `Grid` struct (flat `Vec<Tile>` storage) but different
//! topology implementations.
//!
//! # Grid Types
//!
//! - **Square**: 4-neighbor cardinal directions (Up/Down/Left/Right), Chebyshev distance
//! - **Hex**: 6-neighbor hex directions (odd-r offset), cube-coordinate distance and LOS
//!
//! The active topology is selected via [`GridType`](forge_types::config::GridType)
//! in configuration and dispatched at zero cost through [`GridTopologyKind`].

// Enforce the workspace-wide documentation convention (every other crate uses
// `#![deny(missing_docs)]`; forge-civ was the lone exception). All public items
// must carry doc comments.
#![deny(missing_docs)]

pub mod grid_topology;
pub mod hex;
pub mod hex_direction;
pub mod pathfinding;
pub mod square;

pub use grid_topology::{GridTopology, GridTopologyKind};
pub use hex::HexTopology;
pub use hex_direction::HexDirection;
pub use square::SquareTopology;
