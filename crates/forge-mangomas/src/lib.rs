#![deny(missing_docs)]
#![deny(clippy::all)]

//! MangoMAS agent training integration layer for FORGE.
//!
//! This crate bridges the FORGE simulation platform with MangoMAS autonomous
//! agent systems. It provides:
//!
//! - **Adapters**: Bidirectional action/observation space mapping between
//!   FORGE's discrete grid world and MangoMAS's continuous action spaces.
//! - **Sweep**: MCTS hyperparameter sweep infrastructure for finding optimal
//!   planning parameters at high throughput.
//! - **Transfer**: BDI intention mapping, constitutional constraint mapping,
//!   and RSSM sequence extraction for pre-training MangoMAS cognitive layers.
//! - **Curriculum**: Platform-specific (car/drone) curriculum definitions
//!   with adaptive difficulty, built on FORGE's task DSL.
//! - **Batch Runner**: Headless high-throughput episode collection using
//!   parallel environments via rayon.
//!
//! # Architecture
//!
//! > **Maturity**: `[Research]` — Research Stack Component — multi-agent scenario collection and benchmarks.
//!
//! All constants flow through config structs with `Default` implementations.
//! No hard-coded values. The integration layer does not depend on MangoMAS
//! directly — it defines adapter interfaces that MangoMAS consumers implement.

pub mod adapters;
pub mod batch_runner;
pub mod config;
pub mod curriculum;
pub mod error;
pub mod swarm;
pub mod sweep;
pub mod transfer;
