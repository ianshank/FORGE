#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-social
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
