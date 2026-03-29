#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-scenario
//!
//! Scenario registry, composition, and marketplace for the FORGE platform.
//!
//! This crate provides:
//! - **Config** ([`config`]): Scenario configuration with metadata (tags, tier, author)
//! - **Registry** ([`registry`]): Index, search, and retrieve scenarios
//! - **Compose** ([`compose`]): Merge multiple TOML configs for scenario layering

pub mod compose;
pub mod config;
pub mod registry;
