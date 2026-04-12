//! Integration tests for the forge-proposal crate.
//!
//! Exercises the full proposal workflow per agency:
//! Build → Validate → Render → TOML roundtrip.
//!
//! To run: `cargo test --test integration_proposal`

use forge_proposal::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Creates a fully-populated proposal for a given agency profile.
fn make_full_proposal(agency: AgencyProfile) -> Proposal {
    let duration = agency.duration_months;
    let cost_mid = (agency.cost_range_min + agency.cost_range_max) / 2;

    ProposalBuilder::new()
        .agency(agency)
        .cover_page(CoverPage {
            title: "MCTS-Guided Adaptive Mesh Refinement for PDE Applications".to_string(),
            topic_number: "N252-088".to_string(),
            company_name: "AlphaGalerkin Inc.".to_string(),
            pi_name: "Dr. Jane Smith".to_string(),
            pi_effort_percent: 60,
            duration_months: duration,
            proposed_cost: cost_mid,
            uei: "ABC123DEF456".to_string(),
            ..Default::default()
        })
        .technical(TechnicalVolume {
            problem: ProblemSection {
                description: "Adaptive mesh refinement is a key bottleneck.".to_string(),
                significance: "Significant impact on simulation accuracy.".to_string(),
                bottlenecks: vec!["Current methods are heuristic.".to_string()],
                key_insight: "MCTS for multi-step lookahead.".to_string(),
            },
            approach: TechnicalApproachSection {
                overview: "We propose an MCTS-guided AMR framework.".to_string(),
                objectives: vec![Objective {
                    id: "Obj-1".to_string(),
                    description: "Validate MCTS-guided AMR on canonical benchmarks".to_string(),
                    benchmarks: vec!["L-shaped Poisson".to_string(), "Fichera corner".to_string()],
                }],
                milestones: vec![Milestone {
                    id: "M1".to_string(),
                    description: "Complete benchmark implementation".to_string(),
                    target_month: 2,
                }],
            },
            innovation: InnovationSection {
                summary: "First application of MCTS to Galerkin AMR.".to_string(),
                claims: vec![NoveltyClaim {
                    claim: "Multi-step lookahead".to_string(),
                    evidence: "vs. myopic RL policies".to_string(),
                }],
                literature_gaps: vec!["No published work combines MCTS with Galerkin.".to_string()],
            },
            merit: TechnicalMeritSection {
                overview: "Strong prior results in numerical methods.".to_string(),
                prior_results: vec![PriorResult {
                    description: "L-shaped domain convergence".to_string(),
                    metrics: vec!["MSE = 0.000209".to_string()],
                }],
            },
            work_plan: WorkPlanSection {
                months: vec![
                    MonthBlock {
                        start_month: 1,
                        end_month: duration / 2,
                        milestone: "Phase 1: Implementation".to_string(),
                        deliverables: vec![Deliverable {
                            name: "Prototype".to_string(),
                            description: "Working MCTS-AMR prototype".to_string(),
                        }],
                    },
                    MonthBlock {
                        start_month: duration / 2 + 1,
                        end_month: duration,
                        milestone: "Phase 2: Evaluation".to_string(),
                        deliverables: vec![Deliverable {
                            name: "Final Report".to_string(),
                            description: "Technical report with results".to_string(),
                        }],
                    },
                ],
            },
            related_work: RelatedWorkSection {
                overview: "Extensive related work in AMR and MCTS.".to_string(),
                pi: PIQualification {
                    name: "Dr. Jane Smith".to_string(),
                    title: "Principal Research Scientist".to_string(),
                    qualifications: "10 years in numerical methods.".to_string(),
                    publications: vec!["Smith et al. 2025, J. Comp. Physics".to_string()],
                },
                company_capabilities: "Expertise in HPC and simulation.".to_string(),
            },
        })
        .cost(CostVolume {
            labor: vec![
                LaborItem {
                    category: "Principal Investigator".to_string(),
                    hours: 600,
                    hourly_rate: 100,
                },
                LaborItem {
                    category: "Research Engineer".to_string(),
                    hours: 400,
                    hourly_rate: 80,
                },
            ],
            materials: vec![MaterialItem {
                description: "Cloud compute credits".to_string(),
                cost: 5_000,
            }],
            travel: vec![TravelItem {
                description: "Conference attendance".to_string(),
                cost: 3_000,
            }],
            ..Default::default()
        })
        .supporting(SupportingDocumentation {
            registration: Registration {
                sam_registered: true,
                sbir_registered: true,
                cage_code: "1A2B3".to_string(),
                ..Default::default()
            },
            pi_commitment: PICommitment {
                pi_name: "Dr. Jane Smith".to_string(),
                effort_percent: 60,
                letter_attached: true,
            },
            data_management: DataManagementPlan {
                plan_text: "All data will be archived in accordance with agency policy."
                    .to_string(),
                retention_period: "5 years after project completion".to_string(),
                sharing_approach: "Open access after embargo period.".to_string(),
            },
            ..Default::default()
        })
        .build()
        .expect("full proposal build should succeed")
}

// ---------------------------------------------------------------------------
// Per-Agency Workflow Tests
// ---------------------------------------------------------------------------

#[test]
fn test_dod_phase_i_full_workflow() {
    let proposal = make_full_proposal(AgencyProfile::dod_phase_i());
    assert_eq!(proposal.agency.id, AgencyId::DodPhaseI);

    // Validate
    let report = proposal.validate();
    for issue in &report.issues {
        if issue.is_blocking {
            panic!("unexpected blocking issue for DoD: {:?}", issue.error);
        }
    }

    // Render
    let md = proposal.render_markdown();
    assert!(md.contains("Cover Page"));
    assert!(md.contains("Technical Volume"));
    assert!(md.contains("Cost Volume"));
    assert!(md.contains("Supporting Documentation"));
    assert!(md.contains("MCTS-Guided"));

    // TOML roundtrip
    let toml_str = proposal.to_toml().expect("TOML serialize");
    let restored = Proposal::from_toml_str(&toml_str).expect("TOML deserialize");
    assert_eq!(restored.cover_page.title, proposal.cover_page.title);
    assert_eq!(restored.agency.id, proposal.agency.id);
    assert_eq!(restored.total_cost(), proposal.total_cost());
}

#[test]
fn test_nsf_phase_i_full_workflow() {
    let proposal = make_full_proposal(AgencyProfile::nsf_phase_i());
    assert_eq!(proposal.agency.id, AgencyId::NsfPhaseI);
    assert_eq!(proposal.cover_page.duration_months, 12);

    let report = proposal.validate();
    let md = proposal.render_markdown();
    assert!(!md.is_empty());

    let toml_str = proposal.to_toml().expect("TOML serialize");
    let restored = Proposal::from_toml_str(&toml_str).expect("TOML deserialize");
    assert_eq!(restored.agency.id, AgencyId::NsfPhaseI);
    assert_eq!(
        report.total_count(),
        report.warning_count() + report.blocking_count()
    );
}

#[test]
fn test_afwerx_full_workflow() {
    let proposal = make_full_proposal(AgencyProfile::afwerx_open_topic());
    assert_eq!(proposal.agency.id, AgencyId::AfwerxOpenTopic);
    assert_eq!(proposal.cover_page.duration_months, 3);

    let report = proposal.validate();
    let md = proposal.render_markdown();
    assert!(md.contains("Cost Volume"));

    let toml_str = proposal.to_toml().expect("TOML serialize");
    let restored = Proposal::from_toml_str(&toml_str).expect("TOML deserialize");
    assert_eq!(restored.agency.id, AgencyId::AfwerxOpenTopic);
    assert_eq!(
        report.total_count(),
        report.warning_count() + report.blocking_count()
    );
}

#[test]
fn test_darpa_d2p2_full_workflow() {
    let proposal = make_full_proposal(AgencyProfile::darpa_direct_to_phase_ii());
    assert_eq!(proposal.agency.id, AgencyId::DarpaDirectToPhaseII);
    assert_eq!(proposal.cover_page.duration_months, 18);

    let report = proposal.validate();
    let md = proposal.render_markdown();
    assert!(md.contains("Technical Volume"));

    let toml_str = proposal.to_toml().expect("TOML serialize");
    let restored = Proposal::from_toml_str(&toml_str).expect("TOML deserialize");
    assert_eq!(restored.agency.id, AgencyId::DarpaDirectToPhaseII);
    assert_eq!(
        report.total_count(),
        report.warning_count() + report.blocking_count()
    );
}

// ---------------------------------------------------------------------------
// Cross-Agency Tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_agencies_render_non_empty() {
    for profile in AgencyProfile::builtin_profiles() {
        let proposal = make_full_proposal(profile.clone());
        let md = proposal.render_markdown();
        assert!(
            !md.is_empty(),
            "render should not be empty for agency {:?}",
            profile.id,
        );
    }
}

#[test]
fn test_all_agencies_toml_roundtrip() {
    for profile in AgencyProfile::builtin_profiles() {
        let proposal = make_full_proposal(profile.clone());
        let toml_str = proposal.to_toml().expect("TOML serialize");
        let restored = Proposal::from_toml_str(&toml_str).expect("TOML deserialize");
        assert_eq!(
            restored.cover_page.title, proposal.cover_page.title,
            "title mismatch for agency {:?}",
            proposal.agency.id,
        );
        assert_eq!(
            restored.total_cost(),
            proposal.total_cost(),
            "cost mismatch for agency {:?}",
            proposal.agency.id,
        );
    }
}

#[test]
fn test_summary_render_all_agencies() {
    for profile in AgencyProfile::builtin_profiles() {
        let proposal = make_full_proposal(profile.clone());
        let summary = render_summary(&proposal);
        assert!(
            summary.contains("Proposal Summary"),
            "missing summary header for {:?}",
            profile.id,
        );
        assert!(
            summary.contains(&proposal.agency.name),
            "missing agency name in summary for {:?}",
            profile.id,
        );
    }
}

#[test]
fn test_section_composition_with_agency_constraints() {
    let profile = AgencyProfile::dod_phase_i();

    let tree = SectionComposition::sequence(vec![
        SectionComposition::constrained(
            SectionComposition::leaf("problem", "Problem", "Content here.", 2.0),
            Some(profile.technical_page_limit),
            true,
        ),
        SectionComposition::constrained(
            SectionComposition::leaf("approach", "Approach", "More content.", 3.0),
            Some(profile.technical_page_limit),
            true,
        ),
        SectionComposition::optional(SectionComposition::leaf(
            "addendum",
            "Addendum",
            "Extra info.",
            1.0,
        )),
    ]);

    assert_eq!(tree.leaf_count(), 3);
    let ids = tree.leaf_ids();
    assert_eq!(ids, vec!["problem", "approach", "addendum"]);
    assert!((tree.total_estimated_pages() - 6.0).abs() < f32::EPSILON);

    let constraints = tree.page_constraints();
    assert_eq!(constraints.len(), 2);
}

#[test]
fn test_validation_catches_over_budget_proposal() {
    let agency = AgencyProfile::dod_phase_i();
    let proposal = ProposalBuilder::new()
        .agency(agency)
        .cover_page(CoverPage {
            title: "Expensive Project".to_string(),
            topic_number: "T001".to_string(),
            company_name: "BigCo".to_string(),
            pi_name: "Dr. Big".to_string(),
            uei: "UEI999".to_string(),
            ..Default::default()
        })
        .cost(CostVolume {
            labor: vec![LaborItem {
                category: "PI".to_string(),
                hours: 5000,
                hourly_rate: 200,
            }],
            ..Default::default()
        })
        .build()
        .unwrap();

    let report = proposal.validate();
    let has_cost_error = report
        .issues
        .iter()
        .any(|i| matches!(&i.error, ValidationError::CostOutOfRange { .. }));
    assert!(has_cost_error, "should catch over-budget proposal");
}

#[test]
fn test_validation_catches_low_pi_effort() {
    let agency = AgencyProfile::dod_phase_i();
    let proposal = ProposalBuilder::new()
        .agency(agency)
        .cover_page(CoverPage {
            title: "Low Effort".to_string(),
            topic_number: "T002".to_string(),
            company_name: "LazyCo".to_string(),
            pi_name: "Dr. Lazy".to_string(),
            pi_effort_percent: 20,
            uei: "UEI888".to_string(),
            ..Default::default()
        })
        .build()
        .unwrap();

    let report = proposal.validate();
    let has_pi_error = report
        .issues
        .iter()
        .any(|i| matches!(&i.error, ValidationError::InsufficientPIEffort { .. }));
    assert!(has_pi_error, "should catch low PI effort");
}

#[test]
fn test_builder_default_technical_and_cost() {
    let proposal = ProposalBuilder::new()
        .agency(AgencyProfile::dod_phase_i())
        .cover_page(CoverPage {
            title: "Minimal".to_string(),
            topic_number: "T003".to_string(),
            company_name: "MinCo".to_string(),
            pi_name: "PI".to_string(),
            uei: "UEI777".to_string(),
            ..Default::default()
        })
        .build()
        .unwrap();

    // Should have default (empty) technical and cost volumes
    assert_eq!(proposal.cost.total_cost(), 0);
    assert!(proposal.total_estimated_pages() >= 1.0);
}

#[test]
fn test_estimated_pages_non_negative_all_agencies() {
    for profile in AgencyProfile::builtin_profiles() {
        let proposal = make_full_proposal(profile.clone());
        let pages = proposal.total_estimated_pages();
        assert!(
            pages >= 0.0,
            "negative page estimate for {:?}: {}",
            profile.id,
            pages,
        );
    }
}
