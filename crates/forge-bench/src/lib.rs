#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-bench
//!
//! Performance benchmarks for the FORGE simulation platform.
//!
//! The library surface exposes helpers shared by bench targets and the
//! `allocation_audit` binary so that new bench files (procgen,
//! memory_social, etc.) inherit the same environment-variable contract
//! and defaults without duplicating parsing logic.

pub mod env;
