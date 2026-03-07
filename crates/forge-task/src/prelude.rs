//! Convenience re-exports for common forge-task items.
//!
//! ```rust,no_run
//! use forge_task::prelude::*;
//! ```

pub use crate::composer::evaluate_composition;
pub use crate::curriculum::CurriculumController;
pub use crate::evaluator::evaluate_tasks;
pub use crate::generator::{generate_task, TaskGenConfig};
pub use crate::predicate::evaluate_predicate;
