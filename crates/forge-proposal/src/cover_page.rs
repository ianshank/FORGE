//! Cover page section of an SBIR proposal.
//!
//! Contains all fields required on the proposal cover page:
//! title, topic number, company details, PI information, etc.

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::constants;

/// Cover page information for an SBIR proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CoverPage {
    /// Proposal title.
    pub title: String,
    /// Agency topic number (e.g., "N252-088").
    pub topic_number: String,
    /// Company name.
    pub company_name: String,
    /// Principal Investigator name.
    pub pi_name: String,
    /// PI effort commitment percentage.
    pub pi_effort_percent: u8,
    /// Proposed duration in months.
    pub duration_months: u32,
    /// Proposed total cost (USD).
    pub proposed_cost: u64,
    /// NAICS code.
    pub naics_code: String,
    /// Unique Entity Identifier (from SAM.gov).
    pub uei: String,
    /// Company DUNS number (if applicable).
    pub duns: String,
    /// Proposal submission date (ISO 8601).
    pub submission_date: String,
    /// Company address.
    pub company_address: String,
    /// PI email address.
    pub pi_email: String,
    /// PI phone number.
    pub pi_phone: String,
}

impl Default for CoverPage {
    fn default() -> Self {
        Self {
            title: String::new(),
            topic_number: String::new(),
            company_name: String::new(),
            pi_name: String::new(),
            pi_effort_percent: constants::DEFAULT_PI_MIN_EFFORT_PERCENT,
            duration_months: constants::DEFAULT_DOD_PHASE_I_DURATION_MONTHS,
            proposed_cost: constants::DEFAULT_DOD_PHASE_I_COST_MIN,
            naics_code: constants::DEFAULT_NAICS_CODE.to_string(),
            uei: String::new(),
            duns: String::new(),
            submission_date: String::new(),
            company_address: String::new(),
            pi_email: String::new(),
            pi_phone: String::new(),
        }
    }
}

impl CoverPage {
    /// Returns a list of field names that are empty but should be filled.
    #[instrument(skip_all)]
    pub fn missing_fields(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.title.is_empty() {
            missing.push("title");
        }
        if self.topic_number.is_empty() {
            missing.push("topic_number");
        }
        if self.company_name.is_empty() {
            missing.push("company_name");
        }
        if self.pi_name.is_empty() {
            missing.push("pi_name");
        }
        if self.uei.is_empty() {
            missing.push("uei");
        }
        missing
    }

    /// Estimates the page count for this cover page.
    ///
    /// Cover pages are typically a single page.
    pub fn estimated_pages(&self) -> f32 {
        constants::DEFAULT_COVER_PAGE_PAGES
    }

    /// Renders this cover page to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Cover Page\n\n"));
        out.push_str(&format!("**Title:** {}\n\n", self.title));
        out.push_str(&format!("**Topic Number:** {}\n\n", self.topic_number));
        out.push_str(&format!("**Company:** {}\n\n", self.company_name));
        out.push_str(&format!(
            "**Principal Investigator:** {} ({}% effort)\n\n",
            self.pi_name, self.pi_effort_percent,
        ));
        out.push_str(&format!(
            "**Duration:** {} months\n\n",
            self.duration_months
        ));
        out.push_str(&format!("**Cost:** ${}\n\n", self.proposed_cost));
        out.push_str(&format!("**NAICS:** {}\n\n", self.naics_code));
        out.push_str(&format!("**UEI:** {}\n\n", self.uei));
        if !self.submission_date.is_empty() {
            out.push_str(&format!(
                "**Submission Date:** {}\n\n",
                self.submission_date
            ));
        }
        if !self.company_address.is_empty() {
            out.push_str(&format!("**Address:** {}\n\n", self.company_address));
        }
        out
    }
}

impl std::fmt::Display for CoverPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} — {} (PI: {})",
            self.title, self.company_name, self.pi_name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cover_page_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(CoverPage);
    }

    #[test]
    fn test_cover_page_defaults_valid() {
        forge_types::assert_config_defaults_valid!(CoverPage);
    }

    #[test]
    fn test_default_has_naics() {
        let cover = CoverPage::default();
        assert_eq!(cover.naics_code, constants::DEFAULT_NAICS_CODE);
    }

    #[test]
    fn test_missing_fields_on_default() {
        let cover = CoverPage::default();
        let missing = cover.missing_fields();
        assert!(missing.contains(&"title"));
        assert!(missing.contains(&"company_name"));
        assert!(missing.contains(&"pi_name"));
        assert!(missing.contains(&"topic_number"));
        assert!(missing.contains(&"uei"));
    }

    #[test]
    fn test_missing_fields_when_filled() {
        let cover = CoverPage {
            title: "Test".to_string(),
            topic_number: "N252-088".to_string(),
            company_name: "Acme".to_string(),
            pi_name: "Dr. Smith".to_string(),
            uei: "ABC123".to_string(),
            ..Default::default()
        };
        let missing = cover.missing_fields();
        assert!(missing.is_empty());
    }

    #[test]
    fn test_estimated_pages() {
        let cover = CoverPage::default();
        assert!((cover.estimated_pages() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_render_markdown_contains_fields() {
        let cover = CoverPage {
            title: "MCTS-Guided AMR".to_string(),
            topic_number: "N252-088".to_string(),
            company_name: "AlphaGalerkin Inc.".to_string(),
            pi_name: "Dr. Jane Smith".to_string(),
            pi_effort_percent: 60,
            ..Default::default()
        };
        let md = cover.render_markdown(1);
        assert!(md.contains("# Cover Page"));
        assert!(md.contains("MCTS-Guided AMR"));
        assert!(md.contains("N252-088"));
        assert!(md.contains("AlphaGalerkin Inc."));
        assert!(md.contains("Dr. Jane Smith"));
        assert!(md.contains("60%"));
    }

    #[test]
    fn test_render_markdown_optional_fields() {
        let cover = CoverPage {
            submission_date: "2026-04-12".to_string(),
            company_address: "123 Main St".to_string(),
            ..Default::default()
        };
        let md = cover.render_markdown(2);
        assert!(md.contains("## Cover Page"));
        assert!(md.contains("2026-04-12"));
        assert!(md.contains("123 Main St"));
    }

    #[test]
    fn test_render_markdown_no_optional_when_empty() {
        let cover = CoverPage::default();
        let md = cover.render_markdown(1);
        assert!(!md.contains("Submission Date"));
        assert!(!md.contains("Address"));
    }

    #[test]
    fn test_display() {
        let cover = CoverPage {
            title: "Test Title".to_string(),
            company_name: "TestCo".to_string(),
            pi_name: "PI".to_string(),
            ..Default::default()
        };
        let display = format!("{cover}");
        assert!(display.contains("Test Title"));
        assert!(display.contains("TestCo"));
        assert!(display.contains("PI"));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        prop_compose! {
            fn arb_cover_page()(
                title in "[a-zA-Z ]{0,50}",
                topic in "[A-Z0-9-]{0,10}",
                company in "[a-zA-Z ]{0,30}",
                pi in "[a-zA-Z. ]{0,20}",
                effort in 0u8..100,
                duration in 1u32..36,
                cost in 0u64..2_000_000,
            ) -> CoverPage {
                CoverPage {
                    title,
                    topic_number: topic,
                    company_name: company,
                    pi_name: pi,
                    pi_effort_percent: effort,
                    duration_months: duration,
                    proposed_cost: cost,
                    ..Default::default()
                }
            }
        }

        proptest! {
            #[test]
            fn prop_estimated_pages_positive(cover in arb_cover_page()) {
                prop_assert!(cover.estimated_pages() > 0.0);
            }

            #[test]
            fn prop_serde_roundtrip(cover in arb_cover_page()) {
                let json = serde_json::to_string(&cover).unwrap();
                let deser: CoverPage = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(deser.title, cover.title);
                prop_assert_eq!(deser.proposed_cost, cover.proposed_cost);
            }

            #[test]
            fn prop_render_non_empty(cover in arb_cover_page()) {
                let md = cover.render_markdown(1);
                prop_assert!(!md.is_empty());
                prop_assert!(md.contains("Cover Page"));
            }

            #[test]
            fn prop_missing_fields_count_bounded(cover in arb_cover_page()) {
                let missing = cover.missing_fields();
                prop_assert!(missing.len() <= 5);
            }

            #[test]
            fn prop_display_non_empty(cover in arb_cover_page()) {
                let display = format!("{cover}");
                prop_assert!(!display.is_empty());
            }
        }
    }
}
