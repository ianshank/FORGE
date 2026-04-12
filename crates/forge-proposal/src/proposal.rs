//! Top-level proposal aggregate and builder.
//!
//! [`Proposal`] is the root structure containing all proposal sections.
//! [`ProposalBuilder`] provides a fluent API for constructing proposals
//! with validation on build.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::agency::AgencyProfile;
use crate::config::ProposalConfig;
use crate::cost::CostVolume;
use crate::cover_page::CoverPage;
use crate::error::{ProposalError, ProposalResult};
use crate::supporting::SupportingDocumentation;
use crate::technical::TechnicalVolume;
use crate::validation;

/// A complete SBIR proposal.
///
/// Contains all four major sections: cover page, technical volume,
/// cost volume, and supporting documentation, along with the
/// agency profile and rendering configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Agency-specific constraints.
    pub agency: AgencyProfile,
    /// Cover page information.
    pub cover_page: CoverPage,
    /// Technical volume (6 sections).
    pub technical: TechnicalVolume,
    /// Cost breakdown.
    pub cost: CostVolume,
    /// Supporting documentation.
    pub supporting: SupportingDocumentation,
    /// Proposal configuration (rendering, validation, limits).
    #[serde(default)]
    pub config: ProposalConfig,
}

impl Proposal {
    /// Validates this proposal against its agency profile.
    ///
    /// Returns a validation report with all issues found.
    #[instrument(skip(self))]
    pub fn validate(&self) -> validation::ValidationReport {
        validation::validate_proposal(self, &self.config)
    }

    /// Validates this proposal strictly, returning the first blocking error.
    #[instrument(skip(self))]
    pub fn validate_strict(&self) -> ProposalResult<()> {
        validation::validate_proposal_strict(self, &self.config)
    }

    /// Renders this proposal to Markdown.
    #[instrument(skip(self))]
    pub fn render_markdown(&self) -> String {
        crate::render::render_to_markdown(self, &self.config.render)
    }

    /// Serializes this proposal to a TOML string.
    #[instrument(skip(self))]
    pub fn to_toml(&self) -> ProposalResult<String> {
        toml::to_string_pretty(self)
            .map_err(|e| ProposalError::Serialization(format!("TOML serialization error: {e}")))
    }

    /// Deserializes a proposal from a TOML string.
    #[instrument(skip_all)]
    pub fn from_toml_str(toml_str: &str) -> ProposalResult<Self> {
        toml::from_str(toml_str)
            .map_err(|e| ProposalError::Serialization(format!("TOML parse error: {e}")))
    }

    /// Loads a proposal from a TOML file.
    #[instrument(skip_all)]
    pub fn from_toml(path: impl AsRef<Path>) -> ProposalResult<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ProposalError::Serialization(
                format!("failed to read {}: {e}", path.as_ref().display(),),
            )
        })?;
        Self::from_toml_str(&content)
    }

    /// Returns the total estimated page count for the entire proposal.
    pub fn total_estimated_pages(&self) -> f32 {
        self.cover_page.estimated_pages()
            + self
                .technical
                .estimated_pages(self.config.validation.words_per_page)
            + self.cost.estimated_pages()
            + self.supporting.estimated_pages()
    }

    /// Returns the total proposed cost.
    pub fn total_cost(&self) -> u64 {
        self.cost.total_cost()
    }
}

impl std::fmt::Display for Proposal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [{}] — ${}, {} months",
            self.cover_page.title,
            self.agency.name,
            self.cover_page.proposed_cost,
            self.cover_page.duration_months,
        )
    }
}

/// Builder for constructing proposals with validation.
///
/// # Example
///
/// ```ignore
/// let proposal = ProposalBuilder::new()
///     .agency(AgencyProfile::dod_phase_i())
///     .cover_page(cover)
///     .technical(technical_volume)
///     .cost(cost_volume)
///     .supporting(docs)
///     .build()?;
/// ```
pub struct ProposalBuilder {
    agency: Option<AgencyProfile>,
    cover_page: Option<CoverPage>,
    technical: Option<TechnicalVolume>,
    cost: Option<CostVolume>,
    supporting: Option<SupportingDocumentation>,
    config: ProposalConfig,
}

impl ProposalBuilder {
    /// Creates a new proposal builder with default configuration.
    pub fn new() -> Self {
        Self {
            agency: None,
            cover_page: None,
            technical: None,
            cost: None,
            supporting: None,
            config: ProposalConfig::default(),
        }
    }

    /// Sets the agency profile.
    pub fn agency(mut self, profile: AgencyProfile) -> Self {
        self.agency = Some(profile);
        self
    }

    /// Sets the cover page.
    pub fn cover_page(mut self, cover: CoverPage) -> Self {
        self.cover_page = Some(cover);
        self
    }

    /// Sets the technical volume.
    pub fn technical(mut self, tech: TechnicalVolume) -> Self {
        self.technical = Some(tech);
        self
    }

    /// Sets the cost volume.
    pub fn cost(mut self, cost: CostVolume) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Sets the supporting documentation.
    pub fn supporting(mut self, docs: SupportingDocumentation) -> Self {
        self.supporting = Some(docs);
        self
    }

    /// Sets the proposal configuration.
    pub fn config(mut self, config: ProposalConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds the proposal, returning an error if required fields are missing.
    ///
    /// Does NOT run content validation — call [`Proposal::validate()`] after build
    /// to check agency constraints.
    #[instrument(skip(self))]
    pub fn build(self) -> ProposalResult<Proposal> {
        let agency = self
            .agency
            .ok_or_else(|| ProposalError::BuilderMissing("agency".to_string()))?;
        let cover_page = self
            .cover_page
            .ok_or_else(|| ProposalError::BuilderMissing("cover_page".to_string()))?;
        let technical = self.technical.unwrap_or_default();
        let cost = self.cost.unwrap_or_default();
        let supporting = self.supporting.unwrap_or_default();

        validation::validate_proposal_config(&self.config)?;

        Ok(Proposal {
            agency,
            cover_page,
            technical,
            cost,
            supporting,
            config: self.config,
        })
    }
}

impl Default for ProposalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agency::AgencyId;
    use crate::cost::LaborItem;

    fn make_cover() -> CoverPage {
        CoverPage {
            title: "MCTS-Guided AMR".to_string(),
            topic_number: "N252-088".to_string(),
            company_name: "AlphaGalerkin Inc.".to_string(),
            pi_name: "Dr. Smith".to_string(),
            uei: "ABC123".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_builder_minimal() {
        let proposal = ProposalBuilder::new()
            .agency(AgencyProfile::dod_phase_i())
            .cover_page(make_cover())
            .build()
            .unwrap();
        assert_eq!(proposal.agency.id, AgencyId::DodPhaseI);
        assert_eq!(proposal.cover_page.title, "MCTS-Guided AMR");
    }

    #[test]
    fn test_builder_missing_agency() {
        let result = ProposalBuilder::new().cover_page(make_cover()).build();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ProposalError::BuilderMissing(f) if f == "agency"));
    }

    #[test]
    fn test_builder_missing_cover_page() {
        let result = ProposalBuilder::new()
            .agency(AgencyProfile::dod_phase_i())
            .build();
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProposalError::BuilderMissing(f) if f == "cover_page")
        );
    }

    #[test]
    fn test_builder_full() {
        let proposal = ProposalBuilder::new()
            .agency(AgencyProfile::nsf_phase_i())
            .cover_page(make_cover())
            .technical(TechnicalVolume::default())
            .cost(CostVolume {
                labor: vec![LaborItem {
                    category: "PI".to_string(),
                    hours: 1000,
                    hourly_rate: 100,
                }],
                ..Default::default()
            })
            .supporting(SupportingDocumentation::default())
            .config(ProposalConfig::default())
            .build()
            .unwrap();
        assert_eq!(proposal.agency.id, AgencyId::NsfPhaseI);
        assert!(proposal.total_cost() > 0);
    }

    #[test]
    fn test_proposal_display() {
        let proposal = ProposalBuilder::new()
            .agency(AgencyProfile::dod_phase_i())
            .cover_page(CoverPage {
                title: "Test Title".to_string(),
                proposed_cost: 200_000,
                duration_months: 6,
                ..make_cover()
            })
            .build()
            .unwrap();
        let display = format!("{proposal}");
        assert!(display.contains("Test Title"));
        assert!(display.contains("DoD"));
        assert!(display.contains("200000"));
        assert!(display.contains("6 months"));
    }

    #[test]
    fn test_proposal_toml_roundtrip() {
        let proposal = ProposalBuilder::new()
            .agency(AgencyProfile::dod_phase_i())
            .cover_page(make_cover())
            .build()
            .unwrap();

        let toml_str = proposal.to_toml().unwrap();
        let deser = Proposal::from_toml_str(&toml_str).unwrap();
        assert_eq!(deser.cover_page.title, proposal.cover_page.title);
        assert_eq!(deser.agency.id, proposal.agency.id);
    }

    #[test]
    fn test_proposal_from_toml_file_missing() {
        let result = Proposal::from_toml("/nonexistent/path.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_proposal_total_estimated_pages() {
        let proposal = ProposalBuilder::new()
            .agency(AgencyProfile::dod_phase_i())
            .cover_page(make_cover())
            .build()
            .unwrap();
        // Default proposal has at least cover page (1 page)
        assert!(proposal.total_estimated_pages() >= 1.0);
    }

    #[test]
    fn test_proposal_render_markdown() {
        let proposal = ProposalBuilder::new()
            .agency(AgencyProfile::dod_phase_i())
            .cover_page(make_cover())
            .build()
            .unwrap();
        let md = proposal.render_markdown();
        assert!(!md.is_empty());
        assert!(md.contains("Cover Page"));
        assert!(md.contains("Technical Volume"));
        assert!(md.contains("Cost Volume"));
    }

    #[test]
    fn test_proposal_validate() {
        let proposal = ProposalBuilder::new()
            .agency(AgencyProfile::dod_phase_i())
            .cover_page(make_cover())
            .build()
            .unwrap();
        let _report = proposal.validate();
        // Default proposal may have some issues (e.g., missing sections)
        // Just verify validate() doesn't panic
    }

    #[test]
    fn test_builder_default() {
        let builder = ProposalBuilder::default();
        // Should be same as new()
        let result = builder
            .agency(AgencyProfile::dod_phase_i())
            .cover_page(make_cover())
            .build();
        assert!(result.is_ok());
    }
}
