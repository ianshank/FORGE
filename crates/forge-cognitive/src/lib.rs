#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-cognitive
//!
//! > **Maturity**: `[Research]` — Research Stack Component — LLM-driven deliberation and reasoning.
//!
//! LLM-backed cognitive agent for the FORGE platform.
//!
//! This crate provides:
//! - **Provider trait** (`provider`): Provider-agnostic interface for LLM backends
//! - **Cognitive agent** (`agent`): Agent that reasons through structured prompts
//! - **Prompt construction** (`prompt`): Builder for observation + memory + social prompts
//! - **Reasoning traces** (`reasoning`): Structured chain-of-thought records
//!
//! # Architecture
//!
//! The cognitive core follows Botvinick's "machine-emulator" concept: a general
//! substrate that can rapidly adapt to new tasks. The provider-agnostic design
//! allows swapping between mock (for testing), local (for speed), and frontier
//! models (for capability) without changing agent code.

pub mod agent;
pub mod config;
pub mod prelude;
pub mod prompt;
pub mod provider;
pub mod reasoning;
