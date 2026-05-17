#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-env
//!
//! Generic environment abstraction for FORGE agents.
//!
//! This crate defines the [`Env`] trait — a buffer-filling reset/step
//! interface — plus space-description types ([`ObsSpec`], [`ActionSpec`]).
//!
//! `forge-env` is intentionally free of FORGE-specific types so it can be
//! implemented by any environment (FORGE's own `WorldState`, Minecraft via
//! mineflayer, or any future backend). The companion crates
//! `forge-env-forge` and `forge-env-mc` provide concrete impls.
//!
//! ## Zero-allocation contract
//!
//! Both required methods — [`Env::reset_into`] and [`Env::step_into`] — are
//! buffer-filling: the caller owns the output and passes a mutable
//! reference. Reusing the same buffer across calls makes the hot path
//! allocation-free.
//!
//! The CI gate in `crates/forge-bench/src/bin/allocation_audit.rs`
//! verifies zero heap-allocation on the [`Env::step_into`] hot path.
//!
//! ## Logging
//!
//! Implementors are expected to annotate `reset_into` and `step_into` with
//! `#[tracing::instrument(skip_all, fields(env = %self.name()))]` (or
//! equivalent) to participate in FORGE's structured-logging conventions.

pub mod env;
pub mod error;
pub mod spec;

pub use env::{Env, FlatObsEnv, StepOutput};
pub use error::EnvError;
pub use spec::{ActionSpec, DType, ObsSpec};
