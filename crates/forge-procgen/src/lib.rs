#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-procgen
//!
//! Procedural content generation for FORGE scenarios.
//!
//! This crate provides deterministic procedural generation of maps, objectives,
//! team compositions, and adaptive curriculum control for training environments.

pub mod curriculum;
pub mod grammar;
pub mod map_generator;
pub mod objective_generator;
pub mod seed;
pub mod team_composer;
