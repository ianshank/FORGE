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
//! - [`integrity`] — bundle path containment + sha256 verification
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

#![deny(missing_docs)]

pub mod config;
pub mod error;
pub mod hot_reload;
pub mod integrity;
#[cfg(feature = "mc-live")]
pub mod live;
pub mod manifest;
pub mod metrics;
#[cfg(feature = "onnx-reload")]
pub mod onnx_reload;
pub mod random_baseline;
pub mod runner;
pub mod trajectory;

pub use config::{RunnerConfig, EPISODES_ENV_VAR, RANDOM_ACTIONS_ENV_VAR, SCHEMA_ID_ENV_VAR};
pub use error::{
    parse_transient_env_error, RunnerError, TRANSIENT_ENV_CODE_MESSAGE_SEP,
    TRANSIENT_ENV_DISPLAY_PREFIX,
};
pub use hot_reload::{HotReloadWatcher, ReloadEvent};
pub use integrity::{
    file_sha256_hex, resolve_bundle_path, verify_bundle, VerifiedBundle, DIGEST_CHUNK_BYTES,
};
#[cfg(feature = "mc-live")]
pub use live::run_live;
pub use manifest::{ModelFileEntry, ModelManifest, ModelManifestFiles, MANIFEST_SCHEMA_VERSION};
pub use metrics::{
    serve_metrics, MetricsError, MetricsRecorder, METRIC_REASON_ENV_STEP, METRIC_REASON_PLANNER,
};
#[cfg(feature = "onnx-reload")]
pub use onnx_reload::{config_from_manifest, into_reload_fn};
pub use random_baseline::{sample_random_action, RandomLatentModel};
pub use runner::{
    format_episode_id, EpisodeOutcome, ReloadFn, Runner, RunnerOutcome, EPISODE_ID_PAD_WIDTH,
    EPISODE_ID_PREFIX,
};
pub use trajectory::TrajectoryWriter;
