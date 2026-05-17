#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-env
//!
//! Generic environment abstraction for FORGE agents.
//!
//! This crate defines the [`Env`] trait — a minimal reset/step interface —
//! plus space-description types ([`ObsSpec`], [`ActionSpec`]) and an
//! optional [`StepInto`] trait for zero-allocation step variants.
//!
//! `forge-env` is intentionally free of FORGE-specific types so it can be
//! implemented by any environment (FORGE's own `WorldState`, Minecraft via
//! mineflayer, or any future backend). The companion crates
//! `forge-env-forge` and `forge-env-mc` provide concrete impls.
//!
//! ## Zero-allocation contract
//!
//! Environments that can fill outputs into a caller-owned buffer should
//! implement [`StepInto`] in addition to [`Env`]. Wire-bound envs (e.g.
//! `MinecraftEnv`) intentionally do **not** implement `StepInto`: their
//! step path performs unavoidable network I/O, so the zero-alloc CI gate
//! in `crates/forge-bench/src/bin/allocation_audit.rs` excludes them by
//! module path.
//!
//! ## Logging
//!
//! Implementors are expected to annotate `reset` and `step` with
//! `#[tracing::instrument(skip_all, fields(env = %self.name()))]` (or
//! equivalent) to participate in FORGE's structured-logging conventions.

pub mod env;
pub mod error;
pub mod spec;

pub use env::{Env, FlatObsEnv, StepInto, StepOutput};
pub use error::EnvError;
pub use spec::{ActionSpec, DType, ObsSpec};
