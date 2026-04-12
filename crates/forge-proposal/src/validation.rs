//! Proposal validation.
//!
//! Validates proposal content against agency-specific constraints.
//! Follows the centralized validation pattern from `forge-types/src/validation.rs`.

use tracing::instrument;

use crate::agency::AgencyProfile;
use crate::config::ProposalConfig;
use crate::error::{ProposalError, ProposalResult, ValidationError};
use crate::proposal::Proposal;

/// A single validation issue with severity.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// The validation error.
    pub error: ValidationError,
    /// Whether this is a blocking issue.
    pub is_blocking: bool,
}

/// Result of validating a proposal: accumulated issues.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// All issues found during validation.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Returns true if there are no blocking issues.
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.is_blocking)
    }

    /// Returns the number of blocking issues.
    pub fn blocking_count(&self) -> usize {
        self.issues.iter().filter(|i| i.is_blocking).count()
    }

    /// Returns the number of warning (non-blocking) issues.
    pub fn warning_count(&self) -> usize {
        self.issues.iter().filter(|i| !i.is_blocking).count()
    }

    /// Returns the total number of issues.
    pub fn total_count(&self) -> usize {
        self.issues.len()
    }
}

/// Validates the proposal configuration itself.
///
/// Checks that all config values are within valid ranges.
/// Uses `ConfigError::OutOfRange` for range violations to stay consistent
/// with `forge-types` error handling.
#[instrument(skip_all)]
pub fn validate_proposal_config(config: &ProposalConfig) -> ProposalResult<()> {
    if config.validation.words_per_page == 0 {
        return Err(ProposalError::Config(
            forge_types::error::ConfigError::OutOfRange {
                field: "words_per_page".to_string(),
                value: config.validation.words_per_page.to_string(),
                min: "1".to_string(),
                max: u32::MAX.to_string(),
            },
        ));
    }
    if config.limits.max_sections == 0 {
        return Err(ProposalError::Config(
            forge_types::error::ConfigError::OutOfRange {
                field: "max_sections".to_string(),
                value: config.limits.max_sections.to_string(),
                min: "1".to_string(),
                max: usize::MAX.to_string(),
            },
        ));
    }
    if config.limits.profit_rate_max < 0.0 || config.limits.profit_rate_max > 1.0 {
        return Err(ProposalError::Config(
            forge_types::error::ConfigError::OutOfRange {
                field: "profit_rate_max".to_string(),
                value: config.limits.profit_rate_max.to_string(),
                min: "0.0".to_string(),
                max: "1.0".to_string(),
            },
        ));
    }
    if config.render.heading_level_offset < 1 || config.render.heading_level_offset > 5 {
        return Err(ProposalError::Config(
            forge_types::error::ConfigError::OutOfRange {
                field: "heading_level_offset".to_string(),
                value: config.render.heading_level_offset.to_string(),
                min: "1".to_string(),
                max: "5".to_string(),
            },
        ));
    }
    Ok(())
}

/// Validates a complete proposal against its agency profile.
///
/// Returns a [`ValidationReport`] with all issues found.
/// In strict mode (per config), returns on the first blocking error.
#[instrument(skip_all)]
pub fn validate_proposal(proposal: &Proposal, config: &ProposalConfig) -> ValidationReport {
    let mut report = ValidationReport::default();
    let profile = &proposal.agency;

    // Check cover page required fields
    for field in proposal.cover_page.missing_fields() {
        report.issues.push(ValidationIssue {
            error: ValidationError::MissingField {
                field: format!("cover_page.{field}"),
            },
            is_blocking: true,
        });
        if config.validation.strict {
            return report;
        }
    }

    // Check technical volume page limit (guard against zero words_per_page)
    if config.validation.words_per_page > 0 {
        let tech_pages = proposal
            .technical
            .estimated_pages(config.validation.words_per_page);
        if tech_pages > profile.technical_page_limit as f32 {
            report.issues.push(ValidationIssue {
                error: ValidationError::PageLimitExceeded {
                    section: "technical_volume".to_string(),
                    actual: tech_pages,
                    limit: profile.technical_page_limit,
                },
                is_blocking: true,
            });
            if config.validation.strict {
                return report;
            }
        }
    }

    // Check cost range
    let total_cost = proposal.cost.total_cost();
    if total_cost < profile.cost_range_min || total_cost > profile.cost_range_max {
        report.issues.push(ValidationIssue {
            error: ValidationError::CostOutOfRange {
                total: total_cost,
                min: profile.cost_range_min,
                max: profile.cost_range_max,
            },
            is_blocking: total_cost > 0, // zero cost is a warning for draft proposals
        });
        if config.validation.strict && total_cost > 0 {
            return report;
        }
    }

    // Check duration consistency — only block if work plan exceeds declared duration
    let work_plan_months = proposal.technical.work_plan.total_months();
    if work_plan_months > 0 && work_plan_months > proposal.cover_page.duration_months {
        report.issues.push(ValidationIssue {
            error: ValidationError::DurationMismatch {
                work_plan_months,
                declared_months: proposal.cover_page.duration_months,
            },
            is_blocking: true,
        });
        if config.validation.strict {
            return report;
        }
    }

    // Check PI effort
    if proposal.cover_page.pi_effort_percent < profile.pi_min_effort_percent {
        report.issues.push(ValidationIssue {
            error: ValidationError::InsufficientPIEffort {
                actual: proposal.cover_page.pi_effort_percent,
                required: profile.pi_min_effort_percent,
            },
            is_blocking: true,
        });
        if config.validation.strict {
            return report;
        }
    }

    // Check subcontracts allowed
    if !profile.allows_subcontracts && proposal.cost.total_subcontracts() > 0 {
        report.issues.push(ValidationIssue {
            error: ValidationError::SubcontractLimitExceeded {
                actual: proposal.cost.subcontract_percentage(),
                limit: 0,
            },
            is_blocking: true,
        });
        if config.validation.strict {
            return report;
        }
    }

    // Check subcontract limits
    let sub_pct = proposal.cost.subcontract_percentage();
    if sub_pct > profile.subcontract_limit_percent as f32 {
        report.issues.push(ValidationIssue {
            error: ValidationError::SubcontractLimitExceeded {
                actual: sub_pct,
                limit: profile.subcontract_limit_percent,
            },
            is_blocking: true,
        });
        if config.validation.strict {
            return report;
        }
    }

    // Check profit rate limit
    if proposal.cost.indirect.profit_rate > config.limits.profit_rate_max {
        report.issues.push(ValidationIssue {
            error: ValidationError::ProfitRateExceeded {
                actual: proposal.cost.indirect.profit_rate,
                max: config.limits.profit_rate_max,
            },
            is_blocking: true,
        });
        if config.validation.strict {
            return report;
        }
    }

    // Check required sections have content
    for section_name in &profile.required_sections {
        let is_empty = match section_name.as_str() {
            "problem" => proposal.technical.problem.description.is_empty(),
            "approach" => {
                proposal.technical.approach.overview.is_empty()
                    && proposal.technical.approach.objectives.is_empty()
            }
            "innovation" => {
                proposal.technical.innovation.summary.is_empty()
                    && proposal.technical.innovation.claims.is_empty()
            }
            "merit" => {
                proposal.technical.merit.overview.is_empty()
                    && proposal.technical.merit.prior_results.is_empty()
            }
            "work_plan" => proposal.technical.work_plan.months.is_empty(),
            "related_work" => {
                proposal.technical.related_work.overview.is_empty()
                    && proposal.technical.related_work.pi.name.is_empty()
            }
            "cost_volume" => {
                proposal.cost.labor.is_empty()
                    && proposal.cost.materials.is_empty()
                    && proposal.cost.travel.is_empty()
                    && proposal.cost.subcontracts.is_empty()
            }
            "cover_page" => proposal.cover_page.title.is_empty(),
            // Unknown required sections fail closed — report as missing
            // since we cannot verify their content
            _ => true,
        };
        if is_empty {
            report.issues.push(ValidationIssue {
                error: ValidationError::MissingSectionContent {
                    section: section_name.clone(),
                },
                is_blocking: true,
            });
            if config.validation.strict {
                return report;
            }
        }
    }

    // Check deliverable count
    let total_deliverables = proposal.technical.work_plan.total_deliverables();
    if total_deliverables > config.limits.max_deliverables as usize {
        report.issues.push(ValidationIssue {
            error: ValidationError::TooManyDeliverables {
                actual: total_deliverables,
                max: config.limits.max_deliverables,
            },
            is_blocking: false,
        });
    }

    report
}

/// Validates a proposal strictly, returning the first blocking error.
///
/// Convenience wrapper that enables strict mode and converts to a `ProposalResult`.
#[instrument(skip_all)]
pub fn validate_proposal_strict(
    proposal: &Proposal,
    config: &ProposalConfig,
) -> ProposalResult<()> {
    let mut strict_config = config.clone();
    strict_config.validation.strict = true;
    let report = validate_proposal(proposal, &strict_config);
    if let Some(issue) = report.issues.into_iter().find(|i| i.is_blocking) {
        Err(ProposalError::Validation(issue.error))
    } else {
        Ok(())
    }
}

/// Checks if a cost is within an agency's allowed range.
#[inline]
pub fn cost_in_range(cost: u64, profile: &AgencyProfile) -> bool {
    cost >= profile.cost_range_min && cost <= profile.cost_range_max
}

/// Checks if a page count is within an agency's limit.
#[inline]
pub fn pages_within_limit(pages: f32, profile: &AgencyProfile) -> bool {
    pages <= profile.technical_page_limit as f32
}

/// Estimates word count from text content.
pub fn estimate_word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Converts word count to estimated page count.
pub fn words_to_pages(words: usize, words_per_page: u32) -> f32 {
    if words_per_page == 0 {
        return 0.0;
    }
    words as f32 / words_per_page as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agency::AgencyProfile;
    use crate::constants;
    use crate::cost::{CostVolume, LaborItem, SubcontractItem};
    use crate::cover_page::CoverPage;
    use crate::supporting::SupportingDocumentation;
    use crate::technical::TechnicalVolume;

    fn make_valid_proposal() -> Proposal {
        Proposal {
            agency: AgencyProfile::dod_phase_i(),
            cover_page: CoverPage {
                title: "MCTS-Guided AMR".to_string(),
                topic_number: "N252-088".to_string(),
                company_name: "AlphaGalerkin Inc.".to_string(),
                pi_name: "Dr. Smith".to_string(),
                pi_effort_percent: 55,
                duration_months: constants::DEFAULT_DOD_PHASE_I_DURATION_MONTHS,
                proposed_cost: 200_000,
                uei: "ABC123".to_string(),
                ..Default::default()
            },
            technical: TechnicalVolume::default(),
            cost: CostVolume {
                labor: vec![LaborItem {
                    category: "PI".to_string(),
                    hours: 1000,
                    hourly_rate: 100,
                }],
                ..Default::default()
            },
            supporting: SupportingDocumentation::default(),
            config: ProposalConfig::default(),
        }
    }

    #[test]
    fn test_validate_config_defaults_pass() {
        let config = ProposalConfig::default();
        assert!(validate_proposal_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_zero_words_per_page() {
        let mut config = ProposalConfig::default();
        config.validation.words_per_page = 0;
        let err = validate_proposal_config(&config).unwrap_err();
        assert!(matches!(err, ProposalError::Config(_)));
    }

    #[test]
    fn test_validate_config_zero_max_sections() {
        let mut config = ProposalConfig::default();
        config.limits.max_sections = 0;
        let err = validate_proposal_config(&config).unwrap_err();
        assert!(matches!(err, ProposalError::Config(_)));
    }

    #[test]
    fn test_validate_config_invalid_profit_rate() {
        let mut config = ProposalConfig::default();
        config.limits.profit_rate_max = 1.5;
        let err = validate_proposal_config(&config).unwrap_err();
        assert!(matches!(err, ProposalError::Config(_)));
    }

    #[test]
    fn test_validate_proposal_reports_missing_cover_fields() {
        let proposal = Proposal {
            agency: AgencyProfile::dod_phase_i(),
            cover_page: CoverPage::default(),
            technical: TechnicalVolume::default(),
            cost: CostVolume::default(),
            supporting: SupportingDocumentation::default(),
            config: ProposalConfig::default(),
        };
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        assert!(!report.is_valid());
        assert!(report.blocking_count() > 0);
    }

    #[test]
    fn test_validate_cost_out_of_range() {
        let mut proposal = make_valid_proposal();
        // Make cost exceed DoD max ($250K)
        proposal.cost = CostVolume {
            labor: vec![LaborItem {
                category: "PI".to_string(),
                hours: 5000,
                hourly_rate: 200,
            }],
            ..Default::default()
        };
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_cost_error = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::CostOutOfRange { .. }));
        assert!(has_cost_error, "expected cost out of range error");
    }

    #[test]
    fn test_validate_pi_effort_insufficient() {
        let mut proposal = make_valid_proposal();
        proposal.cover_page.pi_effort_percent = 30; // below 51% minimum
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_pi_error = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::InsufficientPIEffort { .. }));
        assert!(has_pi_error, "expected PI effort error");
    }

    #[test]
    fn test_validate_subcontract_limit() {
        let mut proposal = make_valid_proposal();
        proposal.cost.subcontracts = vec![SubcontractItem {
            organization: "BigSub".to_string(),
            description: "Most of the work".to_string(),
            cost: 200_000,
        }];
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_sub_error = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::SubcontractLimitExceeded { .. }));
        assert!(has_sub_error, "expected subcontract limit error");
    }

    #[test]
    fn test_validate_missing_required_sections() {
        let proposal = Proposal {
            agency: AgencyProfile::dod_phase_i(),
            cover_page: CoverPage {
                title: "Test".to_string(),
                topic_number: "T001".to_string(),
                company_name: "Co".to_string(),
                pi_name: "PI".to_string(),
                uei: "UEI".to_string(),
                ..Default::default()
            },
            technical: TechnicalVolume::default(),
            cost: CostVolume::default(),
            supporting: SupportingDocumentation::default(),
            config: ProposalConfig::default(),
        };
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_missing = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::MissingSectionContent { .. }));
        assert!(has_missing, "expected missing section content errors");
    }

    #[test]
    fn test_validate_strict_returns_first_error() {
        let proposal = Proposal {
            agency: AgencyProfile::dod_phase_i(),
            cover_page: CoverPage::default(), // missing many fields
            technical: TechnicalVolume::default(),
            cost: CostVolume::default(),
            supporting: SupportingDocumentation::default(),
            config: ProposalConfig::default(),
        };
        let mut config = ProposalConfig::default();
        config.validation.strict = true;
        let report = validate_proposal(&proposal, &config);
        // In strict mode, should stop after first blocking error
        assert_eq!(report.blocking_count(), 1);
    }

    #[test]
    fn test_validate_proposal_strict_fn() {
        let proposal = Proposal {
            agency: AgencyProfile::dod_phase_i(),
            cover_page: CoverPage::default(),
            technical: TechnicalVolume::default(),
            cost: CostVolume::default(),
            supporting: SupportingDocumentation::default(),
            config: ProposalConfig::default(),
        };
        let result = validate_proposal_strict(&proposal, &ProposalConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_report_empty() {
        let report = ValidationReport::default();
        assert!(report.is_valid());
        assert_eq!(report.blocking_count(), 0);
        assert_eq!(report.warning_count(), 0);
        assert_eq!(report.total_count(), 0);
    }

    #[test]
    fn test_cost_in_range() {
        let profile = AgencyProfile::dod_phase_i();
        assert!(cost_in_range(200_000, &profile));
        assert!(!cost_in_range(50_000, &profile));
        assert!(!cost_in_range(500_000, &profile));
    }

    #[test]
    fn test_pages_within_limit() {
        let profile = AgencyProfile::dod_phase_i();
        assert!(pages_within_limit(5.0, &profile));
        assert!(pages_within_limit(10.0, &profile));
        assert!(!pages_within_limit(15.0, &profile));
    }

    #[test]
    fn test_estimate_word_count() {
        assert_eq!(estimate_word_count("hello world"), 2);
        assert_eq!(estimate_word_count(""), 0);
        assert_eq!(estimate_word_count("one two three four five"), 5);
    }

    #[test]
    fn test_words_to_pages() {
        assert!((words_to_pages(300, 300) - 1.0).abs() < f32::EPSILON);
        assert!((words_to_pages(600, 300) - 2.0).abs() < f32::EPSILON);
        assert!((words_to_pages(0, 300)).abs() < f32::EPSILON);
        assert!((words_to_pages(100, 0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_duration_mismatch() {
        use crate::technical::{MonthBlock, WorkPlanSection};
        let mut proposal = make_valid_proposal();
        proposal.cover_page.duration_months = 6;
        proposal.technical.work_plan = WorkPlanSection {
            months: vec![MonthBlock {
                start_month: 1,
                end_month: 8, // 8 months but declared 6
                milestone: "M1".to_string(),
                deliverables: vec![],
            }],
        };
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_duration = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::DurationMismatch { .. }));
        assert!(has_duration, "expected duration mismatch error");
    }

    #[test]
    fn test_validate_page_limit_exceeded() {
        use crate::technical::ProblemSection;
        let mut proposal = make_valid_proposal();
        // Fill problem section with enough words to exceed the 10-page limit
        proposal.technical.problem = ProblemSection {
            description: "word ".repeat(5000),
            ..Default::default()
        };
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_page_error = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::PageLimitExceeded { .. }));
        assert!(has_page_error, "expected page limit exceeded error");
    }

    #[test]
    fn test_validate_page_limit_strict_returns_early() {
        use crate::technical::ProblemSection;
        let mut proposal = make_valid_proposal();
        proposal.technical.problem = ProblemSection {
            description: "word ".repeat(5000),
            ..Default::default()
        };
        let mut config = ProposalConfig::default();
        config.validation.strict = true;
        let report = validate_proposal(&proposal, &config);
        assert_eq!(
            report.blocking_count(),
            1,
            "strict mode should stop after first error"
        );
    }

    #[test]
    fn test_validate_cost_strict_returns_early() {
        let mut proposal = make_valid_proposal();
        proposal.cost = CostVolume {
            labor: vec![LaborItem {
                category: "PI".to_string(),
                hours: 5000,
                hourly_rate: 200,
            }],
            ..Default::default()
        };
        let mut config = ProposalConfig::default();
        config.validation.strict = true;
        let report = validate_proposal(&proposal, &config);
        assert!(report.blocking_count() >= 1);
    }

    #[test]
    fn test_validate_duration_strict_returns_early() {
        use crate::technical::{MonthBlock, WorkPlanSection};
        let mut proposal = make_valid_proposal();
        proposal.technical.work_plan = WorkPlanSection {
            months: vec![MonthBlock {
                start_month: 1,
                end_month: 12,
                milestone: "M1".to_string(),
                deliverables: vec![],
            }],
        };
        let mut config = ProposalConfig::default();
        config.validation.strict = true;
        let report = validate_proposal(&proposal, &config);
        assert!(report.blocking_count() >= 1);
    }

    #[test]
    fn test_validate_pi_effort_strict() {
        let mut proposal = make_valid_proposal();
        proposal.cover_page.pi_effort_percent = 10;
        let mut config = ProposalConfig::default();
        config.validation.strict = true;
        let report = validate_proposal(&proposal, &config);
        assert!(report.blocking_count() >= 1);
    }

    #[test]
    fn test_validate_subcontract_strict() {
        let mut proposal = make_valid_proposal();
        proposal.cost.subcontracts = vec![SubcontractItem {
            organization: "Big".to_string(),
            description: "Work".to_string(),
            cost: 500_000,
        }];
        let mut config = ProposalConfig::default();
        config.validation.strict = true;
        let report = validate_proposal(&proposal, &config);
        assert!(report.blocking_count() >= 1);
    }

    #[test]
    fn test_validate_subcontracts_not_allowed() {
        let mut proposal = make_valid_proposal();
        proposal.agency.allows_subcontracts = false;
        proposal.cost.subcontracts = vec![SubcontractItem {
            organization: "Sub".to_string(),
            description: "Work".to_string(),
            cost: 1_000,
        }];
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_sub_error = report.issues.iter().any(|i| {
            matches!(
                &i.error,
                ValidationError::SubcontractLimitExceeded { limit: 0, .. }
            )
        });
        assert!(
            has_sub_error,
            "should block subcontracts when allows_subcontracts is false"
        );
    }

    #[test]
    fn test_validate_config_invalid_heading_level() {
        let mut config = ProposalConfig::default();
        config.render.heading_level_offset = 0;
        let err = validate_proposal_config(&config).unwrap_err();
        assert!(matches!(err, ProposalError::Config(_)));

        let mut config2 = ProposalConfig::default();
        config2.render.heading_level_offset = 6;
        let err2 = validate_proposal_config(&config2).unwrap_err();
        assert!(matches!(err2, ProposalError::Config(_)));
    }

    #[test]
    fn test_validate_profit_rate_exceeded() {
        let mut proposal = make_valid_proposal();
        proposal.cost.indirect.profit_rate = 0.25; // exceeds default 0.10 max
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_profit_error = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::ProfitRateExceeded { .. }));
        assert!(
            has_profit_error,
            "should catch profit rate exceeding configured maximum"
        );
    }

    #[test]
    fn test_validate_unknown_required_section_fails_closed() {
        let mut proposal = make_valid_proposal();
        proposal.agency.required_sections = vec!["unknown_section".to_string()];
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_missing = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::MissingSectionContent { .. }));
        assert!(
            has_missing,
            "unknown required sections should fail closed and report as missing"
        );
    }

    #[test]
    fn test_validate_deliverable_count_warning() {
        use crate::technical::{Deliverable, MonthBlock, WorkPlanSection};
        let mut proposal = make_valid_proposal();
        proposal.technical.work_plan = WorkPlanSection {
            months: vec![MonthBlock {
                start_month: 1,
                end_month: 6,
                milestone: "M1".to_string(),
                deliverables: (0..60)
                    .map(|i| Deliverable {
                        name: format!("D{i}"),
                        description: "desc".to_string(),
                    })
                    .collect(),
            }],
        };
        let report = validate_proposal(&proposal, &ProposalConfig::default());
        let has_deliverable_warning = report
            .issues
            .iter()
            .any(|i| matches!(&i.error, ValidationError::TooManyDeliverables { .. }));
        assert!(
            has_deliverable_warning,
            "should have a warning for too many deliverables"
        );
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_cost_in_range_boundaries(
                cost in 0u64..2_000_000,
            ) {
                let profile = AgencyProfile::dod_phase_i();
                let in_range = cost_in_range(cost, &profile);
                if cost >= profile.cost_range_min && cost <= profile.cost_range_max {
                    prop_assert!(in_range);
                } else {
                    prop_assert!(!in_range);
                }
            }

            #[test]
            fn prop_pages_within_limit_boundaries(
                pages in 0.0f32..50.0,
            ) {
                let profile = AgencyProfile::dod_phase_i();
                let within = pages_within_limit(pages, &profile);
                if pages <= profile.technical_page_limit as f32 {
                    prop_assert!(within);
                } else {
                    prop_assert!(!within);
                }
            }

            #[test]
            fn prop_words_to_pages_proportional(
                words in 0usize..10_000,
                wpp in 1u32..1_000,
            ) {
                let pages = words_to_pages(words, wpp);
                prop_assert!(pages >= 0.0);
                let expected = words as f32 / wpp as f32;
                prop_assert!((pages - expected).abs() < 0.01);
            }

            #[test]
            fn prop_words_to_pages_zero_wpp(words in 0usize..10_000) {
                let pages = words_to_pages(words, 0);
                prop_assert!((pages).abs() < f32::EPSILON);
            }

            #[test]
            fn prop_estimate_word_count_leq_len(
                text in "[a-z ]{0,200}",
            ) {
                let count = estimate_word_count(&text);
                prop_assert!(count <= text.len());
            }

            #[test]
            fn prop_empty_report_is_valid(_ in 0u8..1) {
                let report = ValidationReport::default();
                prop_assert!(report.is_valid());
                prop_assert_eq!(report.blocking_count(), 0);
                prop_assert_eq!(report.warning_count(), 0);
            }
        }
    }
}
