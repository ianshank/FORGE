//! Markdown rendering for proposals.
//!
//! Converts structured [`Proposal`] data into Markdown output.
//! Uses structured traversal rather than string templates.

use tracing::instrument;

use crate::config::RenderConfig;
use crate::proposal::Proposal;

/// Renders a complete proposal to Markdown.
///
/// Walks the proposal structure, calling each section's render method
/// and joining with appropriate headings and page breaks.
#[instrument(skip_all)]
pub fn render_to_markdown(proposal: &Proposal, config: &RenderConfig) -> String {
    let h = config.heading_level_offset;
    let mut out = String::new();

    // Title header
    let title_hashes = "#".repeat(h as usize);
    out.push_str(&format!(
        "{title_hashes} {}\n\n",
        if proposal.cover_page.title.is_empty() {
            "SBIR Proposal"
        } else {
            &proposal.cover_page.title
        },
    ));

    // Table of contents
    if config.include_toc {
        out.push_str(&render_toc(h));
    }

    // Page break
    if config.include_page_breaks {
        out.push_str("---\n\n");
    }

    // A. Cover Page
    out.push_str(&proposal.cover_page.render_markdown(h + 1));

    if config.include_page_breaks {
        out.push_str("---\n\n");
    }

    // B. Technical Volume
    out.push_str(&proposal.technical.render_markdown(h + 1));

    if config.include_page_breaks {
        out.push_str("---\n\n");
    }

    // C. Cost Volume
    out.push_str(&proposal.cost.render_markdown(h + 1));

    if config.include_page_breaks {
        out.push_str("---\n\n");
    }

    // D. Supporting Documentation
    out.push_str(&proposal.supporting.render_markdown(h + 1));

    out
}

/// Renders a table of contents for the standard SBIR proposal structure.
fn render_toc(heading_level: u8) -> String {
    let h = "#".repeat((heading_level + 1) as usize);
    let mut out = String::new();
    out.push_str(&format!("{h} Table of Contents\n\n"));
    out.push_str("- A. Cover Page\n");
    out.push_str("- B. Technical Volume\n");
    out.push_str("  - B.1 Identification and Significance of the Problem\n");
    out.push_str("  - B.2 Technical Approach (Phase I Objectives)\n");
    out.push_str("  - B.3 Key Innovation (Verified Novelty)\n");
    out.push_str("  - B.4 Technical Merit (Prior Results)\n");
    out.push_str("  - B.5 Phase I Work Plan\n");
    out.push_str("  - B.6 Related Work / Principal Investigator\n");
    out.push_str("- C. Cost Volume\n");
    out.push_str("- D. Supporting Documentation\n");
    out.push('\n');
    out
}

/// Renders a summary metadata block for the proposal.
#[instrument(skip_all)]
pub fn render_summary(proposal: &Proposal) -> String {
    let mut out = String::new();
    out.push_str("## Proposal Summary\n\n");
    out.push_str(&format!("- **Agency:** {}\n", proposal.agency.name));
    out.push_str(&format!("- **Title:** {}\n", proposal.cover_page.title));
    out.push_str(&format!("- **PI:** {}\n", proposal.cover_page.pi_name));
    out.push_str(&format!(
        "- **Duration:** {} months\n",
        proposal.cover_page.duration_months
    ));
    out.push_str(&format!(
        "- **Proposed Cost:** ${}\n",
        proposal.cover_page.proposed_cost
    ));
    out.push_str(&format!(
        "- **Total Calculated Cost:** ${}\n",
        proposal.total_cost()
    ));
    out.push_str(&format!(
        "- **Estimated Pages:** {:.1}\n",
        proposal.total_estimated_pages(),
    ));
    out.push_str(&format!(
        "- **Page Limit:** {}\n",
        proposal.agency.technical_page_limit,
    ));
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agency::AgencyProfile;
    use crate::config::ProposalConfig;
    use crate::cost::{CostVolume, LaborItem};
    use crate::cover_page::CoverPage;
    use crate::supporting::SupportingDocumentation;
    use crate::technical::{
        Deliverable, MonthBlock, Objective, ProblemSection, TechnicalApproachSection,
        TechnicalVolume, WorkPlanSection,
    };

    fn make_test_proposal() -> Proposal {
        Proposal {
            agency: AgencyProfile::dod_phase_i(),
            cover_page: CoverPage {
                title: "MCTS-Guided AMR for PDE Applications".to_string(),
                topic_number: "N252-088".to_string(),
                company_name: "AlphaGalerkin Inc.".to_string(),
                pi_name: "Dr. Jane Smith".to_string(),
                pi_effort_percent: 60,
                duration_months: 6,
                proposed_cost: 200_000,
                uei: "ABC123".to_string(),
                ..Default::default()
            },
            technical: TechnicalVolume {
                problem: ProblemSection {
                    description: "AMR bottleneck in PDE simulations.".to_string(),
                    key_insight: "MCTS for mesh refinement.".to_string(),
                    ..Default::default()
                },
                approach: TechnicalApproachSection {
                    objectives: vec![Objective {
                        id: "Objective 1".to_string(),
                        description: "Validate MCTS-guided AMR".to_string(),
                        benchmarks: vec!["L-shaped Poisson".to_string()],
                    }],
                    ..Default::default()
                },
                work_plan: WorkPlanSection {
                    months: vec![MonthBlock {
                        start_month: 1,
                        end_month: 2,
                        milestone: "Implementation".to_string(),
                        deliverables: vec![Deliverable {
                            name: "Report".to_string(),
                            description: "Technical report".to_string(),
                        }],
                    }],
                },
                ..Default::default()
            },
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
    fn test_render_to_markdown_non_empty() {
        let proposal = make_test_proposal();
        let md = render_to_markdown(&proposal, &proposal.config.render);
        assert!(!md.is_empty());
    }

    #[test]
    fn test_render_contains_all_sections() {
        let proposal = make_test_proposal();
        let md = render_to_markdown(&proposal, &proposal.config.render);
        assert!(md.contains("Cover Page"), "missing Cover Page");
        assert!(md.contains("Technical Volume"), "missing Technical Volume");
        assert!(md.contains("Cost Volume"), "missing Cost Volume");
        assert!(
            md.contains("Supporting Documentation"),
            "missing Supporting Docs"
        );
    }

    #[test]
    fn test_render_contains_proposal_content() {
        let proposal = make_test_proposal();
        let md = render_to_markdown(&proposal, &proposal.config.render);
        assert!(md.contains("MCTS-Guided AMR"));
        assert!(md.contains("N252-088"));
        assert!(md.contains("Dr. Jane Smith"));
        assert!(md.contains("L-shaped Poisson"));
    }

    #[test]
    fn test_render_with_toc() {
        let proposal = make_test_proposal();
        let config = RenderConfig {
            include_toc: true,
            ..Default::default()
        };
        let md = render_to_markdown(&proposal, &config);
        assert!(md.contains("Table of Contents"));
        assert!(md.contains("- A. Cover Page"));
    }

    #[test]
    fn test_render_without_toc() {
        let proposal = make_test_proposal();
        let config = RenderConfig {
            include_toc: false,
            ..Default::default()
        };
        let md = render_to_markdown(&proposal, &config);
        assert!(!md.contains("Table of Contents"));
    }

    #[test]
    fn test_render_with_page_breaks() {
        let proposal = make_test_proposal();
        let config = RenderConfig {
            include_page_breaks: true,
            ..Default::default()
        };
        let md = render_to_markdown(&proposal, &config);
        assert!(md.contains("---"));
    }

    #[test]
    fn test_render_without_page_breaks() {
        let proposal = make_test_proposal();
        let config = RenderConfig {
            include_page_breaks: false,
            include_toc: false,
            ..Default::default()
        };
        let md = render_to_markdown(&proposal, &config);
        // Page breaks are standalone "---\n\n" lines, not table separators like "|---|"
        assert!(
            !md.contains("\n---\n\n"),
            "should not contain standalone page break markers",
        );
    }

    #[test]
    fn test_render_heading_level_offset() {
        let proposal = make_test_proposal();
        let config = RenderConfig {
            heading_level_offset: 2,
            include_toc: false,
            include_page_breaks: false,
            ..Default::default()
        };
        let md = render_to_markdown(&proposal, &config);
        // Top-level should use ## (level 2)
        assert!(md.contains("## MCTS-Guided AMR"));
        // Sections should use ### (level 3)
        assert!(md.contains("### Cover Page"));
    }

    #[test]
    fn test_render_summary() {
        let proposal = make_test_proposal();
        let summary = render_summary(&proposal);
        assert!(summary.contains("Proposal Summary"));
        assert!(summary.contains("DoD"));
        assert!(summary.contains("Dr. Jane Smith"));
        assert!(summary.contains("6 months"));
    }

    #[test]
    fn test_render_empty_title_uses_default() {
        let mut proposal = make_test_proposal();
        proposal.cover_page.title = String::new();
        let md = render_to_markdown(&proposal, &proposal.config.render);
        assert!(md.contains("SBIR Proposal"));
    }

    #[test]
    fn test_toc_structure() {
        let toc = render_toc(1);
        assert!(toc.contains("A. Cover Page"));
        assert!(toc.contains("B. Technical Volume"));
        assert!(toc.contains("B.1 Identification and Significance"));
        assert!(toc.contains("B.5 Phase I Work Plan"));
        assert!(toc.contains("C. Cost Volume"));
        assert!(toc.contains("D. Supporting Documentation"));
    }

    #[test]
    fn test_render_summary_total_cost() {
        let proposal = make_test_proposal();
        let summary = render_summary(&proposal);
        assert!(summary.contains("Total Calculated Cost"));
    }

    #[test]
    fn test_render_summary_page_info() {
        let proposal = make_test_proposal();
        let summary = render_summary(&proposal);
        assert!(summary.contains("Estimated Pages"));
        assert!(summary.contains("Page Limit"));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_render_always_non_empty(
                heading in 1u8..4,
                toc in proptest::bool::ANY,
                breaks in proptest::bool::ANY,
            ) {
                let proposal = make_test_proposal();
                let config = RenderConfig {
                    heading_level_offset: heading,
                    include_toc: toc,
                    include_page_breaks: breaks,
                    ..Default::default()
                };
                let md = render_to_markdown(&proposal, &config);
                prop_assert!(!md.is_empty());
            }

            #[test]
            fn prop_render_contains_required_sections(
                heading in 1u8..4,
            ) {
                let proposal = make_test_proposal();
                let config = RenderConfig {
                    heading_level_offset: heading,
                    include_toc: false,
                    include_page_breaks: false,
                    ..Default::default()
                };
                let md = render_to_markdown(&proposal, &config);
                prop_assert!(md.contains("Cover Page"));
                prop_assert!(md.contains("Technical Volume"));
                prop_assert!(md.contains("Cost Volume"));
                prop_assert!(md.contains("Supporting Documentation"));
            }

            #[test]
            fn prop_summary_always_non_empty(_ in 0u8..1) {
                let proposal = make_test_proposal();
                let summary = render_summary(&proposal);
                prop_assert!(!summary.is_empty());
                prop_assert!(summary.contains("Proposal Summary"));
            }
        }
    }
}
