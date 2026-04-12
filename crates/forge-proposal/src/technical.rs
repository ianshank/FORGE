//! Technical volume sections for an SBIR proposal.
//!
//! The technical volume contains six sections:
//! 1. Problem identification and significance
//! 2. Technical approach (Phase I objectives)
//! 3. Key innovation (novelty claims)
//! 4. Technical merit (prior results)
//! 5. Phase I work plan (month-by-month milestones)
//! 6. Related work / PI qualifications

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::constants;

/// Complete technical volume containing all six sections.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TechnicalVolume {
    /// Section 1: Problem identification and significance.
    pub problem: ProblemSection,
    /// Section 2: Technical approach with objectives and milestones.
    pub approach: TechnicalApproachSection,
    /// Section 3: Key innovation and novelty claims.
    pub innovation: InnovationSection,
    /// Section 4: Technical merit with prior results.
    pub merit: TechnicalMeritSection,
    /// Section 5: Phase I work plan.
    pub work_plan: WorkPlanSection,
    /// Section 6: Related work and PI qualifications.
    pub related_work: RelatedWorkSection,
}

impl TechnicalVolume {
    /// Estimates the total page count across all sections.
    #[instrument(skip_all)]
    pub fn estimated_pages(&self, words_per_page: u32) -> f32 {
        self.problem.estimated_pages(words_per_page)
            + self.approach.estimated_pages(words_per_page)
            + self.innovation.estimated_pages(words_per_page)
            + self.merit.estimated_pages(words_per_page)
            + self.work_plan.estimated_pages(words_per_page)
            + self.related_work.estimated_pages(words_per_page)
    }

    /// Renders the entire technical volume to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let sub = heading_level + 1;
        let mut out = String::new();
        out.push_str(&format!("{h} Technical Volume\n\n"));
        out.push_str(&self.problem.render_markdown(sub));
        out.push_str(&self.approach.render_markdown(sub));
        out.push_str(&self.innovation.render_markdown(sub));
        out.push_str(&self.merit.render_markdown(sub));
        out.push_str(&self.work_plan.render_markdown(sub));
        out.push_str(&self.related_work.render_markdown(sub));
        out
    }
}

// ---------- Section 1: Problem ----------

/// Problem identification and significance section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProblemSection {
    /// Description of the problem being addressed.
    pub description: String,
    /// Significance and impact of the problem.
    pub significance: String,
    /// Key bottlenecks or pain points.
    pub bottlenecks: Vec<String>,
    /// Key insight or unique angle.
    pub key_insight: String,
}

impl ProblemSection {
    /// Estimates page count based on word count.
    #[instrument(skip_all)]
    pub fn estimated_pages(&self, words_per_page: u32) -> f32 {
        let words = self.description.split_whitespace().count()
            + self.significance.split_whitespace().count()
            + self
                .bottlenecks
                .iter()
                .map(|b| b.split_whitespace().count())
                .sum::<usize>()
            + self.key_insight.split_whitespace().count();
        words as f32 / words_per_page as f32
    }

    /// Renders this section to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!(
            "{h} Identification and Significance of the Problem\n\n"
        ));
        if !self.description.is_empty() {
            out.push_str(&self.description);
            out.push_str("\n\n");
        }
        if !self.significance.is_empty() {
            out.push_str(&self.significance);
            out.push_str("\n\n");
        }
        if !self.bottlenecks.is_empty() {
            for bottleneck in &self.bottlenecks {
                out.push_str(&format!("- {bottleneck}\n"));
            }
            out.push('\n');
        }
        if !self.key_insight.is_empty() {
            out.push_str(&format!("**Key insight:** {}\n\n", self.key_insight));
        }
        out
    }
}

// ---------- Section 2: Technical Approach ----------

/// A single Phase I objective with associated milestones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    /// Objective identifier (e.g., "Objective 1").
    pub id: String,
    /// Description of the objective.
    pub description: String,
    /// Specific benchmarks or tests.
    pub benchmarks: Vec<String>,
}

/// A milestone within an objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    /// Milestone identifier.
    pub id: String,
    /// Description of the milestone.
    pub description: String,
    /// Month(s) when this milestone should be achieved.
    pub target_month: u32,
}

/// Technical approach section with Phase I objectives.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TechnicalApproachSection {
    /// Overview of the technical approach.
    pub overview: String,
    /// Phase I objectives.
    pub objectives: Vec<Objective>,
    /// Key milestones.
    pub milestones: Vec<Milestone>,
}

impl TechnicalApproachSection {
    /// Estimates page count based on word count.
    #[instrument(skip_all)]
    pub fn estimated_pages(&self, words_per_page: u32) -> f32 {
        let words = self.overview.split_whitespace().count()
            + self
                .objectives
                .iter()
                .map(|o| {
                    o.description.split_whitespace().count()
                        + o.benchmarks
                            .iter()
                            .map(|b| b.split_whitespace().count())
                            .sum::<usize>()
                })
                .sum::<usize>()
            + self
                .milestones
                .iter()
                .map(|m| m.description.split_whitespace().count())
                .sum::<usize>();
        words as f32 / words_per_page as f32
    }

    /// Renders this section to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Technical Approach (Phase I Objectives)\n\n"));
        if !self.overview.is_empty() {
            out.push_str(&self.overview);
            out.push_str("\n\n");
        }
        for obj in &self.objectives {
            out.push_str(&format!("**{}:** {}\n\n", obj.id, obj.description));
            for bench in &obj.benchmarks {
                out.push_str(&format!("- {bench}\n"));
            }
            if !obj.benchmarks.is_empty() {
                out.push('\n');
            }
        }
        if !self.milestones.is_empty() {
            out.push_str("**Milestones:**\n\n");
            for ms in &self.milestones {
                out.push_str(&format!(
                    "- **{}** (Month {}): {}\n",
                    ms.id, ms.target_month, ms.description,
                ));
            }
            out.push('\n');
        }
        out
    }
}

// ---------- Section 3: Innovation ----------

/// A single novelty claim with supporting evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyClaim {
    /// Claim statement.
    pub claim: String,
    /// Supporting evidence or rationale.
    pub evidence: String,
}

/// Key innovation and novelty section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InnovationSection {
    /// Summary of the key innovation.
    pub summary: String,
    /// Specific novelty claims.
    pub claims: Vec<NoveltyClaim>,
    /// Gaps in existing literature or approaches.
    pub literature_gaps: Vec<String>,
}

impl InnovationSection {
    /// Estimates page count based on word count.
    #[instrument(skip_all)]
    pub fn estimated_pages(&self, words_per_page: u32) -> f32 {
        let words = self.summary.split_whitespace().count()
            + self
                .claims
                .iter()
                .map(|c| c.claim.split_whitespace().count() + c.evidence.split_whitespace().count())
                .sum::<usize>()
            + self
                .literature_gaps
                .iter()
                .map(|g| g.split_whitespace().count())
                .sum::<usize>();
        words as f32 / words_per_page as f32
    }

    /// Renders this section to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Key Innovation (Verified Novelty)\n\n"));
        if !self.summary.is_empty() {
            out.push_str(&self.summary);
            out.push_str("\n\n");
        }
        for claim in &self.claims {
            out.push_str(&format!("- **{}** {}\n", claim.claim, claim.evidence));
        }
        if !self.claims.is_empty() {
            out.push('\n');
        }
        if !self.literature_gaps.is_empty() {
            out.push_str("**Literature gaps:**\n\n");
            for gap in &self.literature_gaps {
                out.push_str(&format!("- {gap}\n"));
            }
            out.push('\n');
        }
        out
    }
}

// ---------- Section 4: Technical Merit ----------

/// A prior result demonstrating technical capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorResult {
    /// Description of the result.
    pub description: String,
    /// Quantitative metrics (e.g., "MSE = 0.000209").
    pub metrics: Vec<String>,
}

/// Technical merit and prior results section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TechnicalMeritSection {
    /// Overview of technical merit.
    pub overview: String,
    /// Prior results demonstrating feasibility.
    pub prior_results: Vec<PriorResult>,
}

impl TechnicalMeritSection {
    /// Estimates page count based on word count.
    #[instrument(skip_all)]
    pub fn estimated_pages(&self, words_per_page: u32) -> f32 {
        let words = self.overview.split_whitespace().count()
            + self
                .prior_results
                .iter()
                .map(|r| {
                    r.description.split_whitespace().count()
                        + r.metrics
                            .iter()
                            .map(|m| m.split_whitespace().count())
                            .sum::<usize>()
                })
                .sum::<usize>();
        words as f32 / words_per_page as f32
    }

    /// Renders this section to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Technical Merit (Prior Results)\n\n"));
        if !self.overview.is_empty() {
            out.push_str(&self.overview);
            out.push_str("\n\n");
        }
        for result in &self.prior_results {
            out.push_str(&format!("**{}**\n\n", result.description));
            for metric in &result.metrics {
                out.push_str(&format!("- {metric}\n"));
            }
            if !result.metrics.is_empty() {
                out.push('\n');
            }
        }
        out
    }
}

// ---------- Section 5: Work Plan ----------

/// A deliverable produced during a work plan month.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deliverable {
    /// Deliverable name.
    pub name: String,
    /// Deliverable description.
    pub description: String,
}

/// A single month block in the work plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthBlock {
    /// Starting month number (1-indexed).
    pub start_month: u32,
    /// Ending month number (1-indexed, inclusive).
    pub end_month: u32,
    /// Milestone for this period.
    pub milestone: String,
    /// Deliverables produced.
    pub deliverables: Vec<Deliverable>,
}

/// Phase I work plan section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkPlanSection {
    /// Work plan month blocks.
    pub months: Vec<MonthBlock>,
}

impl WorkPlanSection {
    /// Returns the total number of months covered by the work plan.
    #[instrument(skip_all)]
    pub fn total_months(&self) -> u32 {
        self.months.iter().map(|m| m.end_month).max().unwrap_or(0)
    }

    /// Returns the total number of deliverables across all months.
    #[instrument(skip_all)]
    pub fn total_deliverables(&self) -> usize {
        self.months.iter().map(|m| m.deliverables.len()).sum()
    }

    /// Estimates page count based on word count.
    #[instrument(skip_all)]
    pub fn estimated_pages(&self, words_per_page: u32) -> f32 {
        let words: usize = self
            .months
            .iter()
            .map(|m| {
                m.milestone.split_whitespace().count()
                    + m.deliverables
                        .iter()
                        .map(|d| {
                            d.name.split_whitespace().count()
                                + d.description.split_whitespace().count()
                        })
                        .sum::<usize>()
            })
            .sum();
        // Add overhead for table formatting
        let table_overhead =
            self.months.len() * constants::DEFAULT_MAX_MILESTONES_PER_MONTH as usize;
        (words + table_overhead) as f32 / words_per_page as f32
    }

    /// Renders this section to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Phase I Work Plan\n\n"));
        if self.months.is_empty() {
            out.push_str("*Work plan to be defined.*\n\n");
            return out;
        }
        out.push_str("| Month | Milestone | Deliverable |\n");
        out.push_str("|-------|-----------|-------------|\n");
        for block in &self.months {
            let month_range = if block.start_month == block.end_month {
                format!("{}", block.start_month)
            } else {
                format!("{}-{}", block.start_month, block.end_month)
            };
            let deliverable_names: Vec<&str> =
                block.deliverables.iter().map(|d| d.name.as_str()).collect();
            out.push_str(&format!(
                "| {} | {} | {} |\n",
                month_range,
                block.milestone,
                deliverable_names.join(", "),
            ));
        }
        out.push('\n');
        out
    }
}

// ---------- Section 6: Related Work ----------

/// PI qualification and publication record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PIQualification {
    /// PI name.
    pub name: String,
    /// PI title/position.
    pub title: String,
    /// Relevant qualifications summary.
    pub qualifications: String,
    /// Relevant publications.
    pub publications: Vec<String>,
}

/// Related work and PI qualifications section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RelatedWorkSection {
    /// Overview of related work.
    pub overview: String,
    /// PI qualifications.
    pub pi: PIQualification,
    /// Company capabilities.
    pub company_capabilities: String,
}

impl RelatedWorkSection {
    /// Estimates page count based on word count.
    #[instrument(skip_all)]
    pub fn estimated_pages(&self, words_per_page: u32) -> f32 {
        let words = self.overview.split_whitespace().count()
            + self.pi.qualifications.split_whitespace().count()
            + self
                .pi
                .publications
                .iter()
                .map(|p| p.split_whitespace().count())
                .sum::<usize>()
            + self.company_capabilities.split_whitespace().count();
        words as f32 / words_per_page as f32
    }

    /// Renders this section to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Related Work / Principal Investigator\n\n"));
        if !self.overview.is_empty() {
            out.push_str(&self.overview);
            out.push_str("\n\n");
        }
        if !self.pi.name.is_empty() {
            out.push_str(&format!("**PI:** {} — {}\n\n", self.pi.name, self.pi.title));
        }
        if !self.pi.qualifications.is_empty() {
            out.push_str(&self.pi.qualifications);
            out.push_str("\n\n");
        }
        if !self.pi.publications.is_empty() {
            out.push_str("**Publications:**\n\n");
            for pub_ref in &self.pi.publications {
                out.push_str(&format!("- {pub_ref}\n"));
            }
            out.push('\n');
        }
        if !self.company_capabilities.is_empty() {
            out.push_str(&format!(
                "**Company Capabilities:** {}\n\n",
                self.company_capabilities
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_technical_volume_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(TechnicalVolume);
    }

    #[test]
    fn test_technical_volume_defaults_valid() {
        forge_types::assert_config_defaults_valid!(TechnicalVolume);
    }

    #[test]
    fn test_problem_section_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(ProblemSection);
    }

    #[test]
    fn test_approach_section_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(TechnicalApproachSection);
    }

    #[test]
    fn test_innovation_section_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(InnovationSection);
    }

    #[test]
    fn test_merit_section_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(TechnicalMeritSection);
    }

    #[test]
    fn test_work_plan_section_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(WorkPlanSection);
    }

    #[test]
    fn test_related_work_section_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(RelatedWorkSection);
    }

    #[test]
    fn test_default_volume_zero_pages() {
        let vol = TechnicalVolume::default();
        assert!((vol.estimated_pages(constants::DEFAULT_WORDS_PER_PAGE)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_problem_section_estimated_pages() {
        let section = ProblemSection {
            description: "word ".repeat(150),
            significance: "word ".repeat(150),
            ..Default::default()
        };
        let pages = section.estimated_pages(constants::DEFAULT_WORDS_PER_PAGE);
        assert!(pages > 0.9 && pages < 1.1, "expected ~1 page, got {pages}");
    }

    #[test]
    fn test_work_plan_total_months() {
        let plan = WorkPlanSection {
            months: vec![
                MonthBlock {
                    start_month: 1,
                    end_month: 2,
                    milestone: "Setup".to_string(),
                    deliverables: vec![],
                },
                MonthBlock {
                    start_month: 3,
                    end_month: 6,
                    milestone: "Implementation".to_string(),
                    deliverables: vec![],
                },
            ],
        };
        assert_eq!(plan.total_months(), 6);
    }

    #[test]
    fn test_work_plan_total_deliverables() {
        let plan = WorkPlanSection {
            months: vec![MonthBlock {
                start_month: 1,
                end_month: 2,
                milestone: "M1".to_string(),
                deliverables: vec![
                    Deliverable {
                        name: "D1".to_string(),
                        description: "desc".to_string(),
                    },
                    Deliverable {
                        name: "D2".to_string(),
                        description: "desc".to_string(),
                    },
                ],
            }],
        };
        assert_eq!(plan.total_deliverables(), 2);
    }

    #[test]
    fn test_empty_work_plan_total_months() {
        let plan = WorkPlanSection::default();
        assert_eq!(plan.total_months(), 0);
    }

    #[test]
    fn test_render_problem_section() {
        let section = ProblemSection {
            description: "The problem is significant.".to_string(),
            bottlenecks: vec!["Bottleneck 1".to_string(), "Bottleneck 2".to_string()],
            key_insight: "Our insight is novel.".to_string(),
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("## Identification and Significance"));
        assert!(md.contains("The problem is significant."));
        assert!(md.contains("- Bottleneck 1"));
        assert!(md.contains("- Bottleneck 2"));
        assert!(md.contains("**Key insight:**"));
    }

    #[test]
    fn test_render_approach_section() {
        let section = TechnicalApproachSection {
            overview: "Our approach uses MCTS.".to_string(),
            objectives: vec![Objective {
                id: "Objective 1".to_string(),
                description: "Validate MCTS-guided AMR".to_string(),
                benchmarks: vec!["L-shaped Poisson".to_string()],
            }],
            milestones: vec![Milestone {
                id: "M1".to_string(),
                description: "Benchmark implementation".to_string(),
                target_month: 2,
            }],
        };
        let md = section.render_markdown(2);
        assert!(md.contains("## Technical Approach"));
        assert!(md.contains("Objective 1"));
        assert!(md.contains("L-shaped Poisson"));
        assert!(md.contains("M1"));
        assert!(md.contains("Month 2"));
    }

    #[test]
    fn test_render_innovation_section() {
        let section = InnovationSection {
            summary: "No published papers combine MCTS with Galerkin.".to_string(),
            claims: vec![NoveltyClaim {
                claim: "Multi-step look-ahead".to_string(),
                evidence: "vs. myopic RL policies".to_string(),
            }],
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("## Key Innovation"));
        assert!(md.contains("Multi-step look-ahead"));
    }

    #[test]
    fn test_render_work_plan_empty() {
        let plan = WorkPlanSection::default();
        let md = plan.render_markdown(2);
        assert!(md.contains("Work plan to be defined"));
    }

    #[test]
    fn test_render_work_plan_with_data() {
        let plan = WorkPlanSection {
            months: vec![MonthBlock {
                start_month: 1,
                end_month: 2,
                milestone: "Setup".to_string(),
                deliverables: vec![Deliverable {
                    name: "Report".to_string(),
                    description: "Technical report".to_string(),
                }],
            }],
        };
        let md = plan.render_markdown(2);
        assert!(md.contains("| 1-2 |"));
        assert!(md.contains("Setup"));
        assert!(md.contains("Report"));
    }

    #[test]
    fn test_render_technical_volume() {
        let vol = TechnicalVolume {
            problem: ProblemSection {
                description: "A significant problem.".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let md = vol.render_markdown(1);
        assert!(md.contains("# Technical Volume"));
        assert!(md.contains("## Identification and Significance"));
        assert!(md.contains("A significant problem."));
    }

    #[test]
    fn test_pi_qualification_defaults() {
        let pi = PIQualification::default();
        assert!(pi.name.is_empty());
        assert!(pi.publications.is_empty());
    }

    #[test]
    fn test_render_related_work() {
        let section = RelatedWorkSection {
            pi: PIQualification {
                name: "Dr. Smith".to_string(),
                title: "Professor".to_string(),
                qualifications: "Expert in numerical methods.".to_string(),
                publications: vec!["Smith et al. 2025".to_string()],
            },
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("Dr. Smith"));
        assert!(md.contains("Professor"));
        assert!(md.contains("Smith et al. 2025"));
    }

    #[test]
    fn test_render_problem_with_significance() {
        let section = ProblemSection {
            description: "A problem.".to_string(),
            significance: "Highly significant impact.".to_string(),
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("Highly significant impact."));
    }

    #[test]
    fn test_render_merit_overview() {
        let section = TechnicalMeritSection {
            overview: "Strong track record in numerical methods.".to_string(),
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("Strong track record"));
    }

    #[test]
    fn test_render_related_work_overview() {
        let section = RelatedWorkSection {
            overview: "Extensive related work in HPC.".to_string(),
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("Extensive related work in HPC."));
    }

    #[test]
    fn test_render_related_work_pi_qualifications() {
        let section = RelatedWorkSection {
            pi: PIQualification {
                name: "Dr. Jones".to_string(),
                title: "CTO".to_string(),
                qualifications: "20 years in simulation.".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("Dr. Jones"));
        assert!(md.contains("CTO"));
        assert!(md.contains("20 years in simulation."));
    }

    #[test]
    fn test_render_innovation_with_gaps() {
        let section = InnovationSection {
            literature_gaps: vec!["Gap 1".to_string(), "Gap 2".to_string()],
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("Literature gaps"));
        assert!(md.contains("Gap 1"));
        assert!(md.contains("Gap 2"));
    }

    #[test]
    fn test_render_merit_with_results() {
        let section = TechnicalMeritSection {
            prior_results: vec![PriorResult {
                description: "Benchmark result".to_string(),
                metrics: vec!["MSE = 0.001".to_string()],
            }],
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("Benchmark result"));
        assert!(md.contains("MSE = 0.001"));
    }

    #[test]
    fn test_render_related_work_company_capabilities() {
        let section = RelatedWorkSection {
            company_capabilities: "Expert in HPC.".to_string(),
            ..Default::default()
        };
        let md = section.render_markdown(2);
        assert!(md.contains("Company Capabilities"));
        assert!(md.contains("Expert in HPC"));
    }

    #[test]
    fn test_work_plan_single_month() {
        let plan = WorkPlanSection {
            months: vec![MonthBlock {
                start_month: 3,
                end_month: 3,
                milestone: "Single month".to_string(),
                deliverables: vec![],
            }],
        };
        let md = plan.render_markdown(2);
        assert!(md.contains("| 3 |"));
        assert!(!md.contains("3-3"));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_problem_pages_non_negative(
                word_count in 0usize..500,
                wpp in 1u32..1000,
            ) {
                let text = "word ".repeat(word_count);
                let section = ProblemSection {
                    description: text,
                    ..Default::default()
                };
                prop_assert!(section.estimated_pages(wpp) >= 0.0);
            }

            #[test]
            fn prop_work_plan_deliverables_count(
                block_count in 0usize..5,
                del_per_block in 0usize..4,
            ) {
                let plan = WorkPlanSection {
                    months: (0..block_count).map(|i| MonthBlock {
                        start_month: (i as u32) + 1,
                        end_month: (i as u32) + 1,
                        milestone: "M".to_string(),
                        deliverables: (0..del_per_block).map(|j| Deliverable {
                            name: format!("D{j}"),
                            description: "desc".to_string(),
                        }).collect(),
                    }).collect(),
                };
                prop_assert_eq!(plan.total_deliverables(), block_count * del_per_block);
            }

            #[test]
            fn prop_technical_volume_render_non_empty(
                desc in "[a-z ]{0,100}",
            ) {
                let vol = TechnicalVolume {
                    problem: ProblemSection {
                        description: desc,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let md = vol.render_markdown(1);
                prop_assert!(!md.is_empty());
                prop_assert!(md.contains("Technical Volume"));
            }

            #[test]
            fn prop_volume_pages_non_negative(
                wpp in 1u32..1000,
            ) {
                let vol = TechnicalVolume::default();
                prop_assert!(vol.estimated_pages(wpp) >= 0.0);
            }
        }
    }
}
