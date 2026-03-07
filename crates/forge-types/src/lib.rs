#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-types
//!
//! Shared types, traits, and configuration for the FORGE simulation platform.
//!
//! This crate defines all the core data structures used across FORGE crates:
//! - Configuration structs (all parameters configurable, no hard-coded values)
//! - Grid and tile types for world representation
//! - Entity types (agents, objects) and their properties
//! - Action and observation space definitions
//! - Resource and crafting types
//! - Task predicate DSL
//! - Error types
//! - Default constants

pub mod action;
pub mod config;
pub mod constants;
pub mod entity;
pub mod error;
pub mod grid;
pub mod intent;
pub mod observation;
pub mod prelude;
pub mod resource;
pub mod task;
pub mod validation;

// Re-export commonly used types at crate root
pub use action::Action;
pub use config::ForgeConfig;
pub use entity::{Agent, AgentId, Object, ObjectId};
pub use error::{ForgeError, ForgeResult};
pub use grid::{Direction, Grid, Position, TerrainType, Tile};
pub use observation::{Observation, StepResult};
pub use resource::{CraftingRecipe, ItemType, RecipeBook, ResourceNode};
pub use intent::{AgentIntent, IntentDeclaration, IntentLabel};
pub use task::{ActiveTask, Predicate, TaskComposition, TaskDefinition, TaskTier};
