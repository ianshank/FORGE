#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-replay
//!
//! Compact deterministic replay and trajectory storage for the FORGE platform.
//!
//! This crate provides two complementary storage formats:
//!
//! - **Compact replay** ([`compact`]): Stores only seed + config + action sequence.
//!   Given FORGE's byte-identical determinism, this is sufficient to reconstruct
//!   the full simulation state. Extremely space-efficient.
//!
//! - **Trajectory** ([`trajectory`]): Full observation-action-reward records per step,
//!   suitable for ML training pipelines and HuggingFace Datasets export.
//!
//! - **Export** ([`export`]): Conversion to external formats (CSV, JSON Lines).
//!
//! # Architecture
//!
//! All types are serializable via serde. Compact replays use bincode for binary
//! storage. Config structs implement `Default` for programmatic construction.
//! No hard-coded values.

pub mod compact;
pub mod config;
pub mod coverage;
pub mod export;
/// Authoritative append-only journal for capturing events prior to export.
pub mod journal;
pub mod trajectory;
pub mod v2;

/// HuggingFace Datasets (Parquet) export. Available only with the `hf`
/// cargo feature enabled; pulls in `arrow` + `parquet`.
#[cfg(feature = "hf")]
pub mod hf;
