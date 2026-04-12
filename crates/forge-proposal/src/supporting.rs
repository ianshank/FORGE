//! Supporting documentation section of an SBIR proposal.
//!
//! Contains registration details, PI commitment, subcontract plans,
//! and data management plan.

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Complete supporting documentation for a proposal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SupportingDocumentation {
    /// Company registration details.
    pub registration: Registration,
    /// PI commitment letter details.
    pub pi_commitment: PICommitment,
    /// Subcontractor agreements.
    pub subcontract_plans: Vec<SubcontractPlan>,
    /// Data management plan.
    pub data_management: DataManagementPlan,
}

impl SupportingDocumentation {
    /// Estimates page count for supporting documentation.
    #[instrument(skip_all)]
    pub fn estimated_pages(&self) -> f32 {
        let base = crate::constants::DEFAULT_SUPPORTING_DOCS_BASE_PAGES;
        let sub_pages = self.subcontract_plans.len() as f32
            * crate::constants::DEFAULT_PAGES_PER_SUBCONTRACT_PLAN;
        let dmp_pages = if self.data_management.plan_text.is_empty() {
            0.0
        } else {
            crate::constants::DEFAULT_DATA_MANAGEMENT_PLAN_PAGES
        };
        base + sub_pages + dmp_pages
    }

    /// Renders this section to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let sub = heading_level + 1;
        let mut out = String::new();
        out.push_str(&format!("{h} Supporting Documentation\n\n"));
        out.push_str(&self.registration.render_markdown(sub));
        out.push_str(&self.pi_commitment.render_markdown(sub));
        if !self.subcontract_plans.is_empty() {
            let sh = "#".repeat(sub as usize);
            out.push_str(&format!("{sh} Subcontractor Agreements\n\n"));
            for plan in &self.subcontract_plans {
                out.push_str(&format!("- **{}**: {}\n", plan.organization, plan.scope,));
            }
            out.push('\n');
        }
        out.push_str(&self.data_management.render_markdown(sub));
        out
    }
}

/// Company registration details.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Registration {
    /// Whether the company is registered on SAM.gov.
    pub sam_registered: bool,
    /// Whether the company is registered on SBIR.gov.
    pub sbir_registered: bool,
    /// SAM.gov registration date.
    pub sam_registration_date: String,
    /// CAGE code.
    pub cage_code: String,
}

impl Registration {
    /// Renders registration info to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Company Registration\n\n"));
        out.push_str(&format!(
            "- SAM.gov: {}\n",
            if self.sam_registered {
                "Registered"
            } else {
                "Not registered"
            },
        ));
        out.push_str(&format!(
            "- SBIR.gov: {}\n",
            if self.sbir_registered {
                "Registered"
            } else {
                "Not registered"
            },
        ));
        if !self.cage_code.is_empty() {
            out.push_str(&format!("- CAGE Code: {}\n", self.cage_code));
        }
        out.push('\n');
        out
    }
}

/// PI commitment letter details.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PICommitment {
    /// PI name.
    pub pi_name: String,
    /// Effort percentage committed.
    pub effort_percent: u8,
    /// Whether the commitment letter is attached.
    pub letter_attached: bool,
}

impl Default for PICommitment {
    fn default() -> Self {
        Self {
            pi_name: String::new(),
            effort_percent: crate::constants::DEFAULT_PI_MIN_EFFORT_PERCENT,
            letter_attached: false,
        }
    }
}

impl PICommitment {
    /// Renders PI commitment info to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} PI Commitment\n\n"));
        if !self.pi_name.is_empty() {
            out.push_str(&format!(
                "**PI:** {} ({}% effort)\n\n",
                self.pi_name, self.effort_percent
            ));
        }
        out.push_str(&format!(
            "Commitment letter: {}\n\n",
            if self.letter_attached {
                "Attached"
            } else {
                "Not attached"
            },
        ));
        out
    }
}

/// A subcontractor agreement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubcontractPlan {
    /// Subcontractor organization name.
    pub organization: String,
    /// Scope of work.
    pub scope: String,
    /// Estimated cost (USD).
    pub estimated_cost: u64,
    /// Whether the agreement is signed.
    pub agreement_signed: bool,
}

/// Data management plan.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DataManagementPlan {
    /// Full text of the data management plan.
    pub plan_text: String,
    /// Data retention period description.
    pub retention_period: String,
    /// Data sharing approach.
    pub sharing_approach: String,
}

impl DataManagementPlan {
    /// Renders data management plan to Markdown.
    #[instrument(skip_all)]
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Data Management Plan\n\n"));
        if !self.plan_text.is_empty() {
            out.push_str(&self.plan_text);
            out.push_str("\n\n");
        }
        if !self.retention_period.is_empty() {
            out.push_str(&format!("**Retention:** {}\n\n", self.retention_period));
        }
        if !self.sharing_approach.is_empty() {
            out.push_str(&format!("**Sharing:** {}\n\n", self.sharing_approach));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supporting_docs_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(SupportingDocumentation);
    }

    #[test]
    fn test_supporting_docs_defaults_valid() {
        forge_types::assert_config_defaults_valid!(SupportingDocumentation);
    }

    #[test]
    fn test_registration_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(Registration);
    }

    #[test]
    fn test_pi_commitment_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(PICommitment);
    }

    #[test]
    fn test_data_management_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(DataManagementPlan);
    }

    #[test]
    fn test_estimated_pages_minimal() {
        let docs = SupportingDocumentation::default();
        assert!(docs.estimated_pages() >= 1.0);
    }

    #[test]
    fn test_estimated_pages_with_subcontracts() {
        let docs = SupportingDocumentation {
            subcontract_plans: vec![
                SubcontractPlan {
                    organization: "Uni".to_string(),
                    scope: "Research".to_string(),
                    estimated_cost: 50_000,
                    agreement_signed: true,
                },
                SubcontractPlan {
                    organization: "Lab".to_string(),
                    scope: "Testing".to_string(),
                    estimated_cost: 25_000,
                    agreement_signed: false,
                },
            ],
            ..Default::default()
        };
        assert!(docs.estimated_pages() > 1.5);
    }

    #[test]
    fn test_estimated_pages_with_dmp() {
        let docs = SupportingDocumentation {
            data_management: DataManagementPlan {
                plan_text: "We will manage data.".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(docs.estimated_pages() >= 2.0);
    }

    #[test]
    fn test_render_registration() {
        let reg = Registration {
            sam_registered: true,
            sbir_registered: false,
            cage_code: "ABC12".to_string(),
            ..Default::default()
        };
        let md = reg.render_markdown(3);
        assert!(md.contains("### Company Registration"));
        assert!(md.contains("Registered"));
        assert!(md.contains("Not registered"));
        assert!(md.contains("ABC12"));
    }

    #[test]
    fn test_render_pi_commitment() {
        let pi = PICommitment {
            pi_name: "Dr. Smith".to_string(),
            effort_percent: 60,
            letter_attached: true,
        };
        let md = pi.render_markdown(3);
        assert!(md.contains("Dr. Smith"));
        assert!(md.contains("60%"));
        assert!(md.contains("Attached"));
    }

    #[test]
    fn test_render_data_management_plan() {
        let dmp = DataManagementPlan {
            plan_text: "All data will be archived.".to_string(),
            retention_period: "5 years".to_string(),
            sharing_approach: "Open access after embargo.".to_string(),
        };
        let md = dmp.render_markdown(3);
        assert!(md.contains("All data will be archived."));
        assert!(md.contains("5 years"));
        assert!(md.contains("Open access"));
    }

    #[test]
    fn test_render_empty_supporting_docs() {
        let docs = SupportingDocumentation::default();
        let md = docs.render_markdown(1);
        assert!(md.contains("# Supporting Documentation"));
        assert!(md.contains("Company Registration"));
        assert!(md.contains("PI Commitment"));
        assert!(md.contains("Data Management Plan"));
    }

    #[test]
    fn test_render_supporting_docs_full() {
        let docs = SupportingDocumentation {
            registration: Registration {
                sam_registered: true,
                ..Default::default()
            },
            pi_commitment: PICommitment {
                pi_name: "PI".to_string(),
                effort_percent: 55,
                letter_attached: true,
            },
            subcontract_plans: vec![SubcontractPlan {
                organization: "SubCo".to_string(),
                scope: "Analysis".to_string(),
                estimated_cost: 10_000,
                agreement_signed: true,
            }],
            data_management: DataManagementPlan {
                plan_text: "DMP text.".to_string(),
                ..Default::default()
            },
        };
        let md = docs.render_markdown(1);
        assert!(md.contains("# Supporting Documentation"));
        assert!(md.contains("Company Registration"));
        assert!(md.contains("PI Commitment"));
        assert!(md.contains("SubCo"));
        assert!(md.contains("Data Management Plan"));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        prop_compose! {
            fn arb_supporting_docs()(
                sub_count in 0usize..5,
                has_dmp in proptest::bool::ANY,
                sam_reg in proptest::bool::ANY,
                sbir_reg in proptest::bool::ANY,
            ) -> SupportingDocumentation {
                SupportingDocumentation {
                    registration: Registration {
                        sam_registered: sam_reg,
                        sbir_registered: sbir_reg,
                        ..Default::default()
                    },
                    pi_commitment: PICommitment::default(),
                    subcontract_plans: (0..sub_count).map(|i| SubcontractPlan {
                        organization: format!("Org-{i}"),
                        scope: "Research".to_string(),
                        estimated_cost: 10_000,
                        agreement_signed: true,
                    }).collect(),
                    data_management: if has_dmp {
                        DataManagementPlan {
                            plan_text: "Data management plan.".to_string(),
                            retention_period: "5 years".to_string(),
                            sharing_approach: "Open access".to_string(),
                        }
                    } else {
                        DataManagementPlan::default()
                    },
                }
            }
        }

        proptest! {
            #[test]
            fn prop_estimated_pages_non_negative(docs in arb_supporting_docs()) {
                prop_assert!(docs.estimated_pages() >= 0.0);
            }

            #[test]
            fn prop_estimated_pages_grows_with_subcontracts(sub_count in 0usize..10) {
                let docs = SupportingDocumentation {
                    subcontract_plans: (0..sub_count).map(|i| SubcontractPlan {
                        organization: format!("Org-{i}"),
                        scope: "Work".to_string(),
                        estimated_cost: 10_000,
                        agreement_signed: true,
                    }).collect(),
                    ..Default::default()
                };
                let pages = docs.estimated_pages();
                prop_assert!(pages >= crate::constants::DEFAULT_SUPPORTING_DOCS_BASE_PAGES);
            }

            #[test]
            fn prop_render_non_empty(docs in arb_supporting_docs()) {
                let md = docs.render_markdown(1);
                prop_assert!(!md.is_empty());
                prop_assert!(md.contains("Supporting Documentation"));
            }

            #[test]
            fn prop_serde_roundtrip(docs in arb_supporting_docs()) {
                let json = serde_json::to_string(&docs).unwrap();
                let deser: SupportingDocumentation = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(deser.subcontract_plans.len(), docs.subcontract_plans.len());
            }
        }
    }
}
