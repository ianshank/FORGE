//! Episode runner foundation for FORGE Minecraft RL integration (Phase 4).
//!
//! This crate currently ships the **foundation** for `forge-mc-runner`:
//! configuration, the `model_manifest.json` schema + atomic loader, the
//! hot-reload watcher that observes manifest version bumps, and the
//! `TrajectoryWriter` that buffers `TrajectoryV2` steps and saves them
//! atomically at episode boundaries.
//!
//! The full episode-driving `Runner<E: FlatObsEnv, M: LatentForwardModel>`
//! and the `LatentPlanner` are deferred to a follow-up PR; the building
//! blocks here are independently testable and let the runner be wired up
//! without further plumbing changes.
//!
//! ## Crate layout
//!
//! - [`config`] — [`RunnerConfig`] (episode loop knobs, paths, ports)
//! - [`manifest`] — [`ModelManifest`] (the swap signal: version + sha256s)
//! - [`hot_reload`] — [`HotReloadWatcher`] (between-episode polling)
//! - [`trajectory`] — [`TrajectoryWriter`] (TrajectoryV2 file I/O)
//! - [`error`] — [`RunnerError`] (single thiserror enum)
//!
//! ## Hot-reload discipline (per plan §3.4)
//!
//! The watcher only emits a reload event when the manifest's monotonic
//! `version` strictly increases. The contract — *poll only between
//! episodes* — is enforced by the caller, not the watcher. A doc
//! comment on `HotReloadWatcher::poll` calls this out.

#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod hot_reload;
pub mod manifest;
#[cfg(feature = "onnx-reload")]
pub mod onnx_reload;
pub mod runner;
pub mod trajectory;

pub use config::RunnerConfig;
pub use error::RunnerError;
pub use hot_reload::{HotReloadWatcher, ReloadEvent};
pub use manifest::{ModelFileEntry, ModelManifest, ModelManifestFiles, MANIFEST_SCHEMA_VERSION};
#[cfg(feature = "onnx-reload")]
pub use onnx_reload::{config_from_manifest, into_reload_fn};
pub use runner::{EpisodeOutcome, ReloadFn, Runner, RunnerOutcome};
pub use trajectory::TrajectoryWriter;
