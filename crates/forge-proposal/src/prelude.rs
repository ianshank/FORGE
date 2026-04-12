//! Convenience re-exports for common proposal types.
//!
//! ```ignore
//! use forge_proposal::prelude::*;
//! ```

pub use crate::agency::{AgencyId, AgencyProfile};
pub use crate::config::ProposalConfig;
pub use crate::cost::{
    CostVolume, IndirectCosts, LaborItem, MaterialItem, SubcontractItem, TravelItem,
};
pub use crate::cover_page::CoverPage;
pub use crate::error::{ProposalError, ProposalResult, ValidationError};
pub use crate::proposal::{Proposal, ProposalBuilder};
pub use crate::render::{render_summary, render_to_markdown};
pub use crate::section::{SectionComposition, SectionContent};
pub use crate::supporting::{
    DataManagementPlan, PICommitment, Registration, SubcontractPlan, SupportingDocumentation,
};
pub use crate::technical::{
    Deliverable, InnovationSection, Milestone, MonthBlock, NoveltyClaim, Objective,
    PIQualification, PriorResult, ProblemSection, RelatedWorkSection, TechnicalApproachSection,
    TechnicalMeritSection, TechnicalVolume, WorkPlanSection,
};
pub use crate::validation::{
    cost_in_range, pages_within_limit, validate_proposal, validate_proposal_config,
    validate_proposal_strict, ValidationIssue, ValidationReport,
};
