#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-social
//!
//! > **Maturity**: `[Research]` — Research Stack Component — multi-agent social and trust dynamics.
//!
//! Social interaction primitives for the FORGE platform.
//!
//! This crate provides the social layer that enables agents to develop
//! relationships, form alliances, and receive social reward signals:
//!
//! - **Trust** (`trust`): Pairwise trust matrix updated by observed behavior
//! - **Reputation** (`reputation`): Public reputation scores derived from action history
//! - **Alliance** (`alliance`): Dynamic team formation based on trust thresholds
//! - **Social rewards** (`social_reward`): Cooperation and reputation-based reward signals
//!
//! # Architecture
//!
//! Following Botvinick's framework, intelligence is shaped by social environment.
//! This crate provides the social substrate that makes Data-like social reasoning
//! possible — agents learn not just task performance but relational behavior.

pub mod alliance;
pub mod config;
pub mod prelude;
pub mod reputation;
pub mod social_reward;
pub mod trust;

/// Test utilities shared across proptest modules.
#[cfg(test)]
pub(crate) mod test_util {
    /// PCG-style LCG step for deterministic pseudo-random values in proptests.
    ///
    /// Using a simple LCG avoids pulling in a full RNG crate in test code while
    /// giving proptest control over the seed via its built-in `any::<u64>()`.
    pub(crate) const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

    /// Advance a PCG-style LCG state by one step.
    #[inline]
    pub(crate) fn lcg_next(state: u64) -> u64 {
        state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1)
    }
}
