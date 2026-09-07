#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-memory
//!
//! > **Maturity**: `[Research]` — Research Stack Component — persistent cognitive memory architecture.
//!
//! Persistent agent memory system for the FORGE platform.
//!
//! This crate provides three types of memory that enable agents to maintain
//! identity and continuity across episodes:
//!
//! - **Semantic memory** (`semantic`): Facts and concepts learned by the agent
//! - **Episodic memory** (`episodic`): Records of past events and experiences
//! - **Preference memory** (`preference`): Learned action tendencies and values
//!
//! All memory types support decay (forgetting), reinforcement, and bounded storage
//! with eviction of the weakest entries. The [`InMemoryStore`](store::InMemoryStore)
//! combines all three into a single per-agent memory system with persistence support.
//!
//! # Architecture
//!
//! This crate follows the Botvinick framework for multi-level cognitive modeling:
//! memory is not a separate module bolted on after training, but a constitutive
//! part of the agent's reasoning — analogous to Data's positronic memory that
//! shapes his identity across episodes.

pub mod config;
pub mod episodic;
pub mod error;
pub mod preference;
pub mod prelude;
pub mod semantic;
pub mod store;
