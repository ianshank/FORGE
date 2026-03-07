#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-task
//!
//! Task DSL, procedural task generation, and curriculum controller
//! for the FORGE platform.
//!
//! This crate provides:
//! - **Predicate evaluation** (`predicate`): Atomic predicates checked against world state
//! - **Composition** (`composer`): Composite task evaluation (AND, OR, SEQUENCE, etc.)
//! - **Evaluation** (`evaluator`): Full task evaluation with reward computation
//! - **Difficulty** (`difficulty`): Difficulty estimation and tier assignment

pub mod composer;
pub mod curriculum;
pub mod difficulty;
pub mod evaluator;
pub mod generator;
pub mod predicate;
pub mod prelude;
