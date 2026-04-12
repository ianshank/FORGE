//! Default constants for the SBIR proposal template system.
//!
//! All proposal configuration defaults are defined here as named constants.
//! Every constant is overridable via the corresponding config field.
//! No hard-coded values exist outside this module.

// ---------- DoD Phase I defaults ----------

/// Default technical page limit for DoD Phase I proposals.
pub const DEFAULT_DOD_PHASE_I_PAGE_LIMIT: u32 = 10;
/// Default minimum cost for DoD Phase I proposals (USD).
pub const DEFAULT_DOD_PHASE_I_COST_MIN: u64 = 150_000;
/// Default maximum cost for DoD Phase I proposals (USD).
pub const DEFAULT_DOD_PHASE_I_COST_MAX: u64 = 250_000;
/// Default duration for DoD Phase I proposals (months).
pub const DEFAULT_DOD_PHASE_I_DURATION_MONTHS: u32 = 6;

// ---------- NSF Phase I defaults ----------

/// Default technical page limit for NSF Phase I proposals.
pub const DEFAULT_NSF_PHASE_I_PAGE_LIMIT: u32 = 15;
/// Default minimum cost for NSF Phase I proposals (USD).
pub const DEFAULT_NSF_PHASE_I_COST_MIN: u64 = 305_000;
/// Default maximum cost for NSF Phase I proposals (USD).
pub const DEFAULT_NSF_PHASE_I_COST_MAX: u64 = 305_000;
/// Default duration for NSF Phase I proposals (months).
pub const DEFAULT_NSF_PHASE_I_DURATION_MONTHS: u32 = 12;

// ---------- AFWERX Open Topic defaults ----------

/// Default technical page limit for AFWERX Open Topic proposals.
pub const DEFAULT_AFWERX_PAGE_LIMIT: u32 = 5;
/// Default minimum cost for AFWERX Open Topic proposals (USD).
pub const DEFAULT_AFWERX_COST_MIN: u64 = 75_000;
/// Default maximum cost for AFWERX Open Topic proposals (USD).
pub const DEFAULT_AFWERX_COST_MAX: u64 = 75_000;
/// Default duration for AFWERX Open Topic proposals (months).
pub const DEFAULT_AFWERX_DURATION_MONTHS: u32 = 3;

// ---------- DARPA Direct-to-Phase-II defaults ----------

/// Default technical page limit for DARPA Direct-to-Phase-II proposals.
pub const DEFAULT_DARPA_D2P2_PAGE_LIMIT: u32 = 20;
/// Default feasibility addendum page limit for DARPA Direct-to-Phase-II.
pub const DEFAULT_DARPA_D2P2_FEASIBILITY_PAGE_LIMIT: u32 = 10;
/// Default minimum cost for DARPA Direct-to-Phase-II proposals (USD).
pub const DEFAULT_DARPA_D2P2_COST_MIN: u64 = 750_000;
/// Default maximum cost for DARPA Direct-to-Phase-II proposals (USD).
pub const DEFAULT_DARPA_D2P2_COST_MAX: u64 = 1_500_000;
/// Default duration for DARPA Direct-to-Phase-II proposals (months).
pub const DEFAULT_DARPA_D2P2_DURATION_MONTHS: u32 = 18;

// ---------- General proposal defaults ----------

/// Default NAICS code for R&D in Physical Sciences.
pub const DEFAULT_NAICS_CODE: &str = "541715";
/// Default minimum PI effort commitment (percentage).
pub const DEFAULT_PI_MIN_EFFORT_PERCENT: u8 = 51;
/// Default subcontract cost limit (percentage of total).
pub const DEFAULT_SUBCONTRACT_LIMIT_PERCENT: u8 = 33;
/// Default estimated words per page for page count estimation.
pub const DEFAULT_WORDS_PER_PAGE: u32 = 300;
/// Default heading level offset for Markdown rendering (1 = `#`).
pub const DEFAULT_HEADING_LEVEL_OFFSET: u8 = 1;
/// Default date format string (ISO 8601).
pub const DEFAULT_DATE_FORMAT: &str = "%Y-%m-%d";
/// Default maximum number of sections in a proposal.
pub const DEFAULT_MAX_SECTIONS: usize = 20;
/// Default maximum milestones per month block.
pub const DEFAULT_MAX_MILESTONES_PER_MONTH: u8 = 5;
/// Default maximum number of objectives in a technical approach.
pub const DEFAULT_MAX_OBJECTIVES: u8 = 10;
/// Default maximum number of novelty claims in an innovation section.
pub const DEFAULT_MAX_NOVELTY_CLAIMS: u8 = 10;
/// Default maximum number of prior results in a merit section.
pub const DEFAULT_MAX_PRIOR_RESULTS: u8 = 20;
/// Default maximum number of labor categories in a cost volume.
pub const DEFAULT_MAX_LABOR_CATEGORIES: u8 = 10;
/// Default maximum number of deliverables across all work plan months.
pub const DEFAULT_MAX_DELIVERABLES: u16 = 50;
/// Default maximum profit rate (fraction, e.g., 0.10 = 10%).
pub const DEFAULT_PROFIT_RATE_MAX: f32 = 0.10;
/// Default overhead rate (fraction, e.g., 0.40 = 40%).
pub const DEFAULT_OVERHEAD_RATE: f32 = 0.40;
/// Default G&A rate (fraction, e.g., 0.05 = 5%).
pub const DEFAULT_GA_RATE: f32 = 0.05;
/// Whether to include a table of contents by default.
pub const DEFAULT_INCLUDE_TOC: bool = true;
/// Whether to include page break hints by default.
pub const DEFAULT_INCLUDE_PAGE_BREAKS: bool = true;

// ---------- Page estimation defaults ----------

/// Default estimated pages for a cover page.
pub const DEFAULT_COVER_PAGE_PAGES: f32 = 1.0;
/// Default estimated table rows per page in cost volumes.
pub const DEFAULT_TABLE_ROWS_PER_PAGE: f32 = 40.0;
/// Default estimated pages per subcontract plan entry.
pub const DEFAULT_PAGES_PER_SUBCONTRACT_PLAN: f32 = 0.5;
/// Default base pages for supporting documentation (registration + PI commitment).
pub const DEFAULT_SUPPORTING_DOCS_BASE_PAGES: f32 = 1.0;
/// Default estimated pages for a data management plan when present.
pub const DEFAULT_DATA_MANAGEMENT_PLAN_PAGES: f32 = 1.0;

/// Default minimum duration for any proposal (months).
pub const DEFAULT_MIN_DURATION_MONTHS: u32 = 1;
/// Default maximum duration for any proposal (months).
pub const DEFAULT_MAX_DURATION_MONTHS: u32 = 24;

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_dod_cost_range_valid() {
        assert!(DEFAULT_DOD_PHASE_I_COST_MIN < DEFAULT_DOD_PHASE_I_COST_MAX);
    }

    #[test]
    fn test_nsf_cost_range_valid() {
        assert!(DEFAULT_NSF_PHASE_I_COST_MIN <= DEFAULT_NSF_PHASE_I_COST_MAX);
    }

    #[test]
    fn test_afwerx_cost_range_valid() {
        assert!(DEFAULT_AFWERX_COST_MIN <= DEFAULT_AFWERX_COST_MAX);
    }

    #[test]
    fn test_darpa_cost_range_valid() {
        assert!(DEFAULT_DARPA_D2P2_COST_MIN < DEFAULT_DARPA_D2P2_COST_MAX);
    }

    #[test]
    fn test_page_limits_positive() {
        assert!(DEFAULT_DOD_PHASE_I_PAGE_LIMIT > 0);
        assert!(DEFAULT_NSF_PHASE_I_PAGE_LIMIT > 0);
        assert!(DEFAULT_AFWERX_PAGE_LIMIT > 0);
        assert!(DEFAULT_DARPA_D2P2_PAGE_LIMIT > 0);
        assert!(DEFAULT_DARPA_D2P2_FEASIBILITY_PAGE_LIMIT > 0);
    }

    #[test]
    fn test_duration_months_positive() {
        assert!(DEFAULT_DOD_PHASE_I_DURATION_MONTHS > 0);
        assert!(DEFAULT_NSF_PHASE_I_DURATION_MONTHS > 0);
        assert!(DEFAULT_AFWERX_DURATION_MONTHS > 0);
        assert!(DEFAULT_DARPA_D2P2_DURATION_MONTHS > 0);
    }

    #[test]
    fn test_duration_range_valid() {
        assert!(DEFAULT_MIN_DURATION_MONTHS < DEFAULT_MAX_DURATION_MONTHS);
    }

    #[test]
    fn test_all_durations_within_range() {
        assert!(DEFAULT_DOD_PHASE_I_DURATION_MONTHS >= DEFAULT_MIN_DURATION_MONTHS);
        assert!(DEFAULT_DOD_PHASE_I_DURATION_MONTHS <= DEFAULT_MAX_DURATION_MONTHS);
        assert!(DEFAULT_NSF_PHASE_I_DURATION_MONTHS >= DEFAULT_MIN_DURATION_MONTHS);
        assert!(DEFAULT_NSF_PHASE_I_DURATION_MONTHS <= DEFAULT_MAX_DURATION_MONTHS);
        assert!(DEFAULT_AFWERX_DURATION_MONTHS >= DEFAULT_MIN_DURATION_MONTHS);
        assert!(DEFAULT_AFWERX_DURATION_MONTHS <= DEFAULT_MAX_DURATION_MONTHS);
        assert!(DEFAULT_DARPA_D2P2_DURATION_MONTHS >= DEFAULT_MIN_DURATION_MONTHS);
        assert!(DEFAULT_DARPA_D2P2_DURATION_MONTHS <= DEFAULT_MAX_DURATION_MONTHS);
    }

    #[test]
    fn test_pi_effort_percent_valid() {
        assert!(DEFAULT_PI_MIN_EFFORT_PERCENT > 0);
        assert!(DEFAULT_PI_MIN_EFFORT_PERCENT <= 100);
    }

    #[test]
    fn test_subcontract_limit_valid() {
        assert!(DEFAULT_SUBCONTRACT_LIMIT_PERCENT > 0);
        assert!(DEFAULT_SUBCONTRACT_LIMIT_PERCENT < 100);
    }

    #[test]
    fn test_words_per_page_positive() {
        assert!(DEFAULT_WORDS_PER_PAGE > 0);
    }

    #[test]
    fn test_rates_in_unit_range() {
        assert!(DEFAULT_PROFIT_RATE_MAX >= 0.0 && DEFAULT_PROFIT_RATE_MAX <= 1.0);
        assert!(DEFAULT_OVERHEAD_RATE >= 0.0 && DEFAULT_OVERHEAD_RATE <= 1.0);
        assert!(DEFAULT_GA_RATE >= 0.0 && DEFAULT_GA_RATE <= 1.0);
    }

    #[test]
    fn test_naics_code_not_empty() {
        assert!(!DEFAULT_NAICS_CODE.is_empty());
    }

    #[test]
    fn test_date_format_not_empty() {
        assert!(!DEFAULT_DATE_FORMAT.is_empty());
    }

    #[test]
    fn test_max_limits_positive() {
        assert!(DEFAULT_MAX_SECTIONS > 0);
        assert!(DEFAULT_MAX_MILESTONES_PER_MONTH > 0);
        assert!(DEFAULT_MAX_OBJECTIVES > 0);
        assert!(DEFAULT_MAX_NOVELTY_CLAIMS > 0);
        assert!(DEFAULT_MAX_PRIOR_RESULTS > 0);
        assert!(DEFAULT_MAX_LABOR_CATEGORIES > 0);
        assert!(DEFAULT_MAX_DELIVERABLES > 0);
    }

    #[test]
    fn test_heading_level_offset_nonzero() {
        assert!(DEFAULT_HEADING_LEVEL_OFFSET > 0);
    }

    #[test]
    fn test_dod_page_limit_within_bounds() {
        // DoD Phase I is shorter than DARPA D2P2
        assert!(DEFAULT_DOD_PHASE_I_PAGE_LIMIT < DEFAULT_DARPA_D2P2_PAGE_LIMIT);
    }

    #[test]
    fn test_page_estimation_constants_positive() {
        assert!(DEFAULT_COVER_PAGE_PAGES > 0.0);
        assert!(DEFAULT_TABLE_ROWS_PER_PAGE > 0.0);
        assert!(DEFAULT_PAGES_PER_SUBCONTRACT_PLAN > 0.0);
        assert!(DEFAULT_SUPPORTING_DOCS_BASE_PAGES > 0.0);
        assert!(DEFAULT_DATA_MANAGEMENT_PLAN_PAGES > 0.0);
    }

    #[test]
    fn test_afwerx_is_shortest_program() {
        assert!(DEFAULT_AFWERX_DURATION_MONTHS <= DEFAULT_DOD_PHASE_I_DURATION_MONTHS);
        assert!(DEFAULT_AFWERX_DURATION_MONTHS <= DEFAULT_NSF_PHASE_I_DURATION_MONTHS);
        assert!(DEFAULT_AFWERX_PAGE_LIMIT <= DEFAULT_DOD_PHASE_I_PAGE_LIMIT);
    }

    #[test]
    fn test_table_rows_per_page_reasonable() {
        // Should be at least 10 and at most 100
        assert!(DEFAULT_TABLE_ROWS_PER_PAGE >= 10.0);
        assert!(DEFAULT_TABLE_ROWS_PER_PAGE <= 100.0);
    }

    #[test]
    fn test_page_estimation_constants_less_than_one_page() {
        // Per-item page estimates should be less than a full page
        assert!(DEFAULT_PAGES_PER_SUBCONTRACT_PLAN <= 1.0);
    }
}
