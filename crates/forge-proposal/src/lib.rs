#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-proposal
//!
//! > **Maturity**: `[Experimental]` — Experimental Component — SBIR research proposal generation.
//!
//! SBIR proposal template system for the FORGE simulation platform.
//!
//! This crate provides a configurable, validated proposal generation system
//! supporting multiple federal agencies (DoD, NSF, AFWERX, DARPA) with:
//!
//! - **Agency profiles** ([`agency`]): Configurable constraints per agency
//! - **Cover page** ([`cover_page`]): Title, PI, company, cost, duration
//! - **Technical volume** ([`technical`]): 6 composable sections
//! - **Cost volume** ([`cost`]): Labor, materials, travel, indirect costs
//! - **Supporting docs** ([`supporting`]): Registration, PI commitment, DMP
//! - **Section composition** ([`section`]): Recursive tree structure
//! - **Validation** ([`validation`]): Agency-specific constraint checking
//! - **Rendering** ([`render`]): Structured Markdown output
//! - **Builder** ([`proposal`]): Fluent API for proposal construction
//!
//! # Configurable Defaults
//!
//! Core defaults and reusable configuration values are defined as named constants
//! in [`constants`]. Agency profiles, page limits, cost ranges, and rates are
//! configurable via the corresponding config structs.
//!
//! # Example
//!
//! ```
//! use forge_proposal::prelude::*;
//!
//! let proposal = ProposalBuilder::new()
//!     .agency(AgencyProfile::dod_phase_i())
//!     .cover_page(CoverPage {
//!         title: "MCTS-Guided AMR".to_string(),
//!         topic_number: "N252-088".to_string(),
//!         company_name: "AlphaGalerkin Inc.".to_string(),
//!         pi_name: "Dr. Smith".to_string(),
//!         uei: "ABC123".to_string(),
//!         ..Default::default()
//!     })
//!     .build()
//!     .unwrap();
//!
//! let md = proposal.render_markdown();
//! assert!(md.contains("Cover Page"));
//! ```

pub mod agency;
pub mod config;
pub mod constants;
pub mod cost;
pub mod cover_page;
pub mod error;
pub mod prelude;
pub mod proposal;
pub mod render;
pub mod section;
pub mod supporting;
pub mod technical;
pub mod validation;

// Re-export key types at crate root
pub use agency::{AgencyId, AgencyProfile};
pub use config::ProposalConfig;
pub use error::{ProposalError, ProposalResult};
pub use proposal::{Proposal, ProposalBuilder};
