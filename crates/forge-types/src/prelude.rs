//! Convenience re-exports for common forge-types items.
//!
//! ```rust,no_run
//! use forge_types::prelude::*;
//! ```

pub use crate::action::Action;
pub use crate::config::ForgeConfig;
pub use crate::entity::{Agent, AgentId, Object, ObjectId};
pub use crate::error::{ForgeError, ForgeResult};
pub use crate::grid::{Direction, Grid, Position, TerrainType, Tile};
pub use crate::observation::{Observation, StepResult};
pub use crate::resource::{CraftingRecipe, ItemType, RecipeBook, ResourceNode};
pub use crate::task::{ActiveTask, Predicate, TaskComposition, TaskDefinition, TaskTier};
