//! Configuration types for the SBIR proposal template system.
//!
//! All configuration structs follow the workspace convention:
//! - Derive `Clone, Debug, Serialize, Deserialize`
//! - Use `#[serde(default)]` for optional fields
//! - Implement `Default` using named constants from [`crate::constants`]

use serde::{Deserialize, Serialize};

use crate::constants;

/// Top-level proposal configuration.
///
/// Controls rendering, validation thresholds, and structural limits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProposalConfig {
    /// Rendering configuration.
    pub render: RenderConfig,
    /// Validation configuration.
    pub validation: ValidationConfig,
    /// Structural limits for proposal content.
    pub limits: LimitsConfig,
}

/// Configuration for Markdown rendering output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderConfig {
    /// Heading level offset (1 = `#` for top-level sections).
    pub heading_level_offset: u8,
    /// Whether to include a table of contents.
    pub include_toc: bool,
    /// Whether to include page break hints (`---`).
    pub include_page_breaks: bool,
    /// Date format string for `chrono` formatting.
    pub date_format: String,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            heading_level_offset: constants::DEFAULT_HEADING_LEVEL_OFFSET,
            include_toc: constants::DEFAULT_INCLUDE_TOC,
            include_page_breaks: constants::DEFAULT_INCLUDE_PAGE_BREAKS,
            date_format: constants::DEFAULT_DATE_FORMAT.to_string(),
        }
    }
}

/// Configuration for proposal validation behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationConfig {
    /// Estimated words per page for page count calculation.
    pub words_per_page: u32,
    /// Whether to enforce strict validation (fail on first error).
    pub strict: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            words_per_page: constants::DEFAULT_WORDS_PER_PAGE,
            strict: false,
        }
    }
}

/// Structural limits for proposal content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LimitsConfig {
    /// Maximum number of sections.
    pub max_sections: usize,
    /// Maximum milestones per work plan month.
    pub max_milestones_per_month: u8,
    /// Maximum objectives in a technical approach.
    pub max_objectives: u8,
    /// Maximum novelty claims in an innovation section.
    pub max_novelty_claims: u8,
    /// Maximum prior results in a merit section.
    pub max_prior_results: u8,
    /// Maximum labor categories in a cost volume.
    pub max_labor_categories: u8,
    /// Maximum deliverables across all work plan months.
    pub max_deliverables: u16,
    /// Maximum profit rate (fraction).
    pub profit_rate_max: f32,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_sections: constants::DEFAULT_MAX_SECTIONS,
            max_milestones_per_month: constants::DEFAULT_MAX_MILESTONES_PER_MONTH,
            max_objectives: constants::DEFAULT_MAX_OBJECTIVES,
            max_novelty_claims: constants::DEFAULT_MAX_NOVELTY_CLAIMS,
            max_prior_results: constants::DEFAULT_MAX_PRIOR_RESULTS,
            max_labor_categories: constants::DEFAULT_MAX_LABOR_CATEGORIES,
            max_deliverables: constants::DEFAULT_MAX_DELIVERABLES,
            profit_rate_max: constants::DEFAULT_PROFIT_RATE_MAX,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposal_config_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(ProposalConfig);
    }

    #[test]
    fn test_proposal_config_defaults_valid() {
        forge_types::assert_config_defaults_valid!(ProposalConfig);
    }

    #[test]
    fn test_render_config_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(RenderConfig);
    }

    #[test]
    fn test_render_config_defaults_valid() {
        forge_types::assert_config_defaults_valid!(RenderConfig);
    }

    #[test]
    fn test_validation_config_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(ValidationConfig);
    }

    #[test]
    fn test_validation_config_defaults_valid() {
        forge_types::assert_config_defaults_valid!(ValidationConfig);
    }

    #[test]
    fn test_limits_config_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(LimitsConfig);
    }

    #[test]
    fn test_limits_config_defaults_valid() {
        forge_types::assert_config_defaults_valid!(LimitsConfig);
    }

    #[test]
    fn test_render_config_defaults() {
        let config = RenderConfig::default();
        assert_eq!(
            config.heading_level_offset,
            constants::DEFAULT_HEADING_LEVEL_OFFSET
        );
        assert_eq!(config.include_toc, constants::DEFAULT_INCLUDE_TOC);
        assert_eq!(
            config.include_page_breaks,
            constants::DEFAULT_INCLUDE_PAGE_BREAKS
        );
        assert_eq!(config.date_format, constants::DEFAULT_DATE_FORMAT);
    }

    #[test]
    fn test_validation_config_defaults() {
        let config = ValidationConfig::default();
        assert_eq!(config.words_per_page, constants::DEFAULT_WORDS_PER_PAGE);
        assert!(!config.strict);
    }

    #[test]
    fn test_limits_config_defaults() {
        let config = LimitsConfig::default();
        assert_eq!(config.max_sections, constants::DEFAULT_MAX_SECTIONS);
        assert_eq!(config.max_objectives, constants::DEFAULT_MAX_OBJECTIVES);
        assert_eq!(config.profit_rate_max, constants::DEFAULT_PROFIT_RATE_MAX);
    }

    #[test]
    fn test_proposal_config_toml_roundtrip() {
        let config = ProposalConfig::default();
        let toml_str = toml::to_string_pretty(&config).expect("TOML serialization failed");
        let deser: ProposalConfig = toml::from_str(&toml_str).expect("TOML deserialization failed");
        let toml_str2 = toml::to_string_pretty(&deser).expect("TOML re-serialization failed");
        assert_eq!(toml_str, toml_str2);
    }
}
