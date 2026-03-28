//! Platform-specific curriculum definitions with adaptive difficulty.
//!
//! Provides car (5-tier) and drone (5-tier) curriculum progressions
//! built on FORGE's task DSL and adaptive curriculum controller.

pub mod platform_curriculum;
pub mod task_mapping;

pub use platform_curriculum::{PlatformCurriculum, TierDefinition};
pub use task_mapping::TaskDslMapper;
