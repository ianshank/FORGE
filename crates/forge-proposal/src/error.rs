//! Error types for the SBIR proposal template system.
//!
//! All errors use `thiserror` derive macros for clean error propagation.
//! [`ProposalError`] is the top-level error type; [`ValidationError`] provides
//! detailed validation issues that can be accumulated for batch reporting.

use forge_types::error::ConfigError;
use thiserror::Error;

/// Top-level error type for proposal operations.
#[derive(Debug, Error)]
pub enum ProposalError {
    /// Configuration validation failed.
    #[error("proposal config error: {0}")]
    Config(#[from] ConfigError),

    /// Proposal content validation failed.
    #[error("proposal validation error: {0}")]
    Validation(#[from] ValidationError),

    /// Serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Rendering failed.
    #[error("render error: {0}")]
    Render(String),

    /// Builder was missing a required field.
    #[error("builder error: missing required field '{0}'")]
    BuilderMissing(String),
}

/// Detailed validation issues for proposal content.
#[derive(Debug, Error, Clone)]
pub enum ValidationError {
    /// A section exceeds its page limit.
    #[error(
        "page limit exceeded: section '{section}' has {actual} estimated pages, limit is {limit}"
    )]
    PageLimitExceeded {
        /// Name of the section that exceeded the limit.
        section: String,
        /// Actual estimated page count.
        actual: f32,
        /// Maximum allowed pages.
        limit: u32,
    },

    /// Total proposal cost is outside the agency's allowed range.
    #[error("cost out of range: total ${total} not in [${min}, ${max}]")]
    CostOutOfRange {
        /// Actual total cost.
        total: u64,
        /// Minimum allowed cost.
        min: u64,
        /// Maximum allowed cost.
        max: u64,
    },

    /// Work plan duration does not match the declared proposal duration.
    #[error("duration mismatch: work plan covers {work_plan_months} months, declared duration is {declared_months}")]
    DurationMismatch {
        /// Number of months covered by the work plan.
        work_plan_months: u32,
        /// Declared proposal duration in months.
        declared_months: u32,
    },

    /// A required section has no content.
    #[error("missing required section content: '{section}'")]
    MissingSectionContent {
        /// Name of the section missing content.
        section: String,
    },

    /// A required field is empty or missing.
    #[error("missing required field: '{field}'")]
    MissingField {
        /// Name of the missing field.
        field: String,
    },

    /// Subcontract costs exceed the agency-imposed limit.
    #[error("subcontract percentage {actual:.1}% exceeds limit {limit}%")]
    SubcontractLimitExceeded {
        /// Actual subcontract percentage.
        actual: f32,
        /// Maximum allowed subcontract percentage.
        limit: u8,
    },

    /// PI effort commitment is below the agency minimum.
    #[error("PI effort {actual}% is below minimum {required}%")]
    InsufficientPIEffort {
        /// Actual PI effort percentage.
        actual: u8,
        /// Minimum required PI effort percentage.
        required: u8,
    },

    /// Work plan has too many deliverables.
    #[error("too many deliverables: {actual} exceeds maximum {max}")]
    TooManyDeliverables {
        /// Actual deliverable count.
        actual: usize,
        /// Maximum allowed deliverables.
        max: u16,
    },
}

/// Result type alias for proposal operations.
pub type ProposalResult<T> = Result<T, ProposalError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposal_error_display_config() {
        let err = ProposalError::Config(ConfigError::OutOfRange {
            field: "page_limit".to_string(),
            value: "999".to_string(),
            min: "1".to_string(),
            max: "100".to_string(),
        });
        let msg = err.to_string();
        assert!(msg.contains("proposal config error"));
        assert!(msg.contains("page_limit"));
    }

    #[test]
    fn test_proposal_error_display_validation() {
        let err = ProposalError::Validation(ValidationError::PageLimitExceeded {
            section: "technical".to_string(),
            actual: 15.0,
            limit: 10,
        });
        let msg = err.to_string();
        assert!(msg.contains("proposal validation error"));
        assert!(msg.contains("technical"));
    }

    #[test]
    fn test_proposal_error_display_serialization() {
        let err = ProposalError::Serialization("invalid TOML at line 5".to_string());
        let msg = err.to_string();
        assert!(msg.contains("serialization error"));
        assert!(msg.contains("invalid TOML at line 5"));
    }

    #[test]
    fn test_proposal_error_display_render() {
        let err = ProposalError::Render("template not found".to_string());
        let msg = err.to_string();
        assert!(msg.contains("render error"));
        assert!(msg.contains("template not found"));
    }

    #[test]
    fn test_proposal_error_display_builder_missing() {
        let err = ProposalError::BuilderMissing("cover_page".to_string());
        let msg = err.to_string();
        assert!(msg.contains("builder error"));
        assert!(msg.contains("cover_page"));
    }

    #[test]
    fn test_validation_error_page_limit() {
        let err = ValidationError::PageLimitExceeded {
            section: "approach".to_string(),
            actual: 12.5,
            limit: 10,
        };
        let msg = err.to_string();
        assert!(msg.contains("page limit exceeded"));
        assert!(msg.contains("approach"));
        assert!(msg.contains("12.5"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn test_validation_error_cost_out_of_range() {
        let err = ValidationError::CostOutOfRange {
            total: 500_000,
            min: 150_000,
            max: 250_000,
        };
        let msg = err.to_string();
        assert!(msg.contains("cost out of range"));
        assert!(msg.contains("500000"));
    }

    #[test]
    fn test_validation_error_duration_mismatch() {
        let err = ValidationError::DurationMismatch {
            work_plan_months: 8,
            declared_months: 6,
        };
        let msg = err.to_string();
        assert!(msg.contains("duration mismatch"));
        assert!(msg.contains("8"));
        assert!(msg.contains("6"));
    }

    #[test]
    fn test_validation_error_missing_section() {
        let err = ValidationError::MissingSectionContent {
            section: "innovation".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("missing required section"));
        assert!(msg.contains("innovation"));
    }

    #[test]
    fn test_validation_error_missing_field() {
        let err = ValidationError::MissingField {
            field: "pi_name".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("missing required field"));
        assert!(msg.contains("pi_name"));
    }

    #[test]
    fn test_validation_error_subcontract_limit() {
        let err = ValidationError::SubcontractLimitExceeded {
            actual: 45.0,
            limit: 33,
        };
        let msg = err.to_string();
        assert!(msg.contains("subcontract percentage"));
        assert!(msg.contains("45.0"));
        assert!(msg.contains("33"));
    }

    #[test]
    fn test_validation_error_pi_effort() {
        let err = ValidationError::InsufficientPIEffort {
            actual: 30,
            required: 51,
        };
        let msg = err.to_string();
        assert!(msg.contains("PI effort"));
        assert!(msg.contains("30"));
        assert!(msg.contains("51"));
    }

    #[test]
    fn test_validation_error_too_many_deliverables() {
        let err = ValidationError::TooManyDeliverables {
            actual: 75,
            max: 50,
        };
        let msg = err.to_string();
        assert!(msg.contains("too many deliverables"));
        assert!(msg.contains("75"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_config_error_conversion() {
        let config_err = ConfigError::ParseError("bad input".to_string());
        let proposal_err: ProposalError = config_err.into();
        assert!(matches!(proposal_err, ProposalError::Config(_)));
    }

    #[test]
    fn test_validation_error_conversion() {
        let val_err = ValidationError::MissingField {
            field: "title".to_string(),
        };
        let proposal_err: ProposalError = val_err.into();
        assert!(matches!(proposal_err, ProposalError::Validation(_)));
    }

    #[test]
    fn test_proposal_result_ok() {
        let result: ProposalResult<u32> = Ok(42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_proposal_result_err() {
        let result: ProposalResult<u32> = Err(ProposalError::Serialization("fail".to_string()));
        assert!(result.is_err());
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_validation_error_display_non_empty(
                actual in 0.0f32..100.0,
                limit in 1u32..100,
            ) {
                let err = ValidationError::PageLimitExceeded {
                    section: "test".to_string(),
                    actual,
                    limit,
                };
                let msg = err.to_string();
                prop_assert!(!msg.is_empty());
                prop_assert!(msg.contains("page limit exceeded"));
            }

            #[test]
            fn prop_cost_error_display_contains_values(
                total in 0u64..2_000_000,
                min in 0u64..500_000,
                extra in 0u64..500_000,
            ) {
                let max = min + extra;
                let err = ValidationError::CostOutOfRange { total, min, max };
                let msg = err.to_string();
                prop_assert!(msg.contains("cost out of range"));
            }

            #[test]
            fn prop_serialization_error_contains_message(
                msg_part in "[a-z ]{1,50}",
            ) {
                let err = ProposalError::Serialization(msg_part.clone());
                let display = err.to_string();
                prop_assert!(display.contains(&msg_part));
            }
        }
    }
}
