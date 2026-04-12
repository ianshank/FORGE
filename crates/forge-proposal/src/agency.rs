//! Agency profile definitions and built-in registry.
//!
//! Each SBIR soliciting agency has specific constraints (page limits, cost ranges,
//! duration) that proposals must satisfy. [`AgencyProfile`] captures these constraints
//! and [`AgencyId`] identifies built-in profiles.

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::constants;

/// Identifies a known SBIR soliciting agency program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgencyId {
    /// Department of Defense Phase I SBIR.
    DodPhaseI,
    /// National Science Foundation Phase I SBIR.
    NsfPhaseI,
    /// Air Force AFWERX Open Topic.
    AfwerxOpenTopic,
    /// DARPA Direct-to-Phase-II.
    DarpaDirectToPhaseII,
    /// User-defined custom agency profile.
    Custom,
}

/// Agency-specific constraints for an SBIR proposal.
///
/// All numeric values are sourced from named constants when using built-in profiles.
/// Custom profiles can be loaded from TOML with arbitrary values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgencyProfile {
    /// Agency identifier.
    pub id: AgencyId,
    /// Human-readable agency name.
    pub name: String,
    /// Maximum pages for the technical volume.
    pub technical_page_limit: u32,
    /// Maximum pages for supplemental material (e.g., DARPA feasibility addendum).
    pub supplemental_page_limit: Option<u32>,
    /// Minimum allowed proposal cost (USD).
    pub cost_range_min: u64,
    /// Maximum allowed proposal cost (USD).
    pub cost_range_max: u64,
    /// Standard proposal duration (months).
    pub duration_months: u32,
    /// Applicable NAICS codes.
    pub naics_codes: Vec<String>,
    /// Names of sections required by this agency.
    pub required_sections: Vec<String>,
    /// Whether subcontracting is allowed.
    pub allows_subcontracts: bool,
    /// Whether cost sharing is required.
    pub cost_sharing_required: bool,
    /// Minimum PI effort commitment (percentage).
    pub pi_min_effort_percent: u8,
    /// Maximum subcontract cost as percentage of total.
    pub subcontract_limit_percent: u8,
}

impl Default for AgencyProfile {
    fn default() -> Self {
        Self::dod_phase_i()
    }
}

impl AgencyProfile {
    /// Returns the DoD Phase I SBIR profile.
    #[instrument]
    pub fn dod_phase_i() -> Self {
        Self {
            id: AgencyId::DodPhaseI,
            name: "DoD Phase I SBIR".to_string(),
            technical_page_limit: constants::DEFAULT_DOD_PHASE_I_PAGE_LIMIT,
            supplemental_page_limit: None,
            cost_range_min: constants::DEFAULT_DOD_PHASE_I_COST_MIN,
            cost_range_max: constants::DEFAULT_DOD_PHASE_I_COST_MAX,
            duration_months: constants::DEFAULT_DOD_PHASE_I_DURATION_MONTHS,
            naics_codes: vec![constants::DEFAULT_NAICS_CODE.to_string()],
            required_sections: vec![
                "cover_page".to_string(),
                "problem".to_string(),
                "approach".to_string(),
                "innovation".to_string(),
                "merit".to_string(),
                "work_plan".to_string(),
                "related_work".to_string(),
                "cost_volume".to_string(),
            ],
            allows_subcontracts: true,
            cost_sharing_required: false,
            pi_min_effort_percent: constants::DEFAULT_PI_MIN_EFFORT_PERCENT,
            subcontract_limit_percent: constants::DEFAULT_SUBCONTRACT_LIMIT_PERCENT,
        }
    }

    /// Returns the NSF Phase I SBIR profile.
    #[instrument]
    pub fn nsf_phase_i() -> Self {
        Self {
            id: AgencyId::NsfPhaseI,
            name: "NSF Phase I SBIR".to_string(),
            technical_page_limit: constants::DEFAULT_NSF_PHASE_I_PAGE_LIMIT,
            supplemental_page_limit: None,
            cost_range_min: constants::DEFAULT_NSF_PHASE_I_COST_MIN,
            cost_range_max: constants::DEFAULT_NSF_PHASE_I_COST_MAX,
            duration_months: constants::DEFAULT_NSF_PHASE_I_DURATION_MONTHS,
            naics_codes: vec![constants::DEFAULT_NAICS_CODE.to_string()],
            required_sections: vec![
                "cover_page".to_string(),
                "problem".to_string(),
                "approach".to_string(),
                "innovation".to_string(),
                "merit".to_string(),
                "work_plan".to_string(),
                "related_work".to_string(),
                "cost_volume".to_string(),
            ],
            allows_subcontracts: true,
            cost_sharing_required: false,
            pi_min_effort_percent: constants::DEFAULT_PI_MIN_EFFORT_PERCENT,
            subcontract_limit_percent: constants::DEFAULT_SUBCONTRACT_LIMIT_PERCENT,
        }
    }

    /// Returns the AFWERX Open Topic profile.
    #[instrument]
    pub fn afwerx_open_topic() -> Self {
        Self {
            id: AgencyId::AfwerxOpenTopic,
            name: "AFWERX Open Topic".to_string(),
            technical_page_limit: constants::DEFAULT_AFWERX_PAGE_LIMIT,
            supplemental_page_limit: None,
            cost_range_min: constants::DEFAULT_AFWERX_COST_MIN,
            cost_range_max: constants::DEFAULT_AFWERX_COST_MAX,
            duration_months: constants::DEFAULT_AFWERX_DURATION_MONTHS,
            naics_codes: vec![constants::DEFAULT_NAICS_CODE.to_string()],
            required_sections: vec![
                "cover_page".to_string(),
                "problem".to_string(),
                "approach".to_string(),
                "work_plan".to_string(),
                "cost_volume".to_string(),
            ],
            allows_subcontracts: true,
            cost_sharing_required: false,
            pi_min_effort_percent: constants::DEFAULT_PI_MIN_EFFORT_PERCENT,
            subcontract_limit_percent: constants::DEFAULT_SUBCONTRACT_LIMIT_PERCENT,
        }
    }

    /// Returns the DARPA Direct-to-Phase-II profile.
    #[instrument]
    pub fn darpa_direct_to_phase_ii() -> Self {
        Self {
            id: AgencyId::DarpaDirectToPhaseII,
            name: "DARPA Direct-to-Phase-II".to_string(),
            technical_page_limit: constants::DEFAULT_DARPA_D2P2_PAGE_LIMIT,
            supplemental_page_limit: Some(constants::DEFAULT_DARPA_D2P2_FEASIBILITY_PAGE_LIMIT),
            cost_range_min: constants::DEFAULT_DARPA_D2P2_COST_MIN,
            cost_range_max: constants::DEFAULT_DARPA_D2P2_COST_MAX,
            duration_months: constants::DEFAULT_DARPA_D2P2_DURATION_MONTHS,
            naics_codes: vec![constants::DEFAULT_NAICS_CODE.to_string()],
            required_sections: vec![
                "cover_page".to_string(),
                "problem".to_string(),
                "approach".to_string(),
                "innovation".to_string(),
                "merit".to_string(),
                "work_plan".to_string(),
                "related_work".to_string(),
                "cost_volume".to_string(),
                "feasibility".to_string(),
            ],
            allows_subcontracts: true,
            cost_sharing_required: false,
            pi_min_effort_percent: constants::DEFAULT_PI_MIN_EFFORT_PERCENT,
            subcontract_limit_percent: constants::DEFAULT_SUBCONTRACT_LIMIT_PERCENT,
        }
    }

    /// Returns all built-in agency profiles.
    #[instrument]
    pub fn builtin_profiles() -> Vec<AgencyProfile> {
        vec![
            Self::dod_phase_i(),
            Self::nsf_phase_i(),
            Self::afwerx_open_topic(),
            Self::darpa_direct_to_phase_ii(),
        ]
    }

    /// Looks up a built-in profile by agency ID.
    ///
    /// Returns `None` for [`AgencyId::Custom`] since custom profiles
    /// must be constructed directly.
    #[instrument]
    pub fn by_id(id: AgencyId) -> Option<AgencyProfile> {
        match id {
            AgencyId::DodPhaseI => Some(Self::dod_phase_i()),
            AgencyId::NsfPhaseI => Some(Self::nsf_phase_i()),
            AgencyId::AfwerxOpenTopic => Some(Self::afwerx_open_topic()),
            AgencyId::DarpaDirectToPhaseII => Some(Self::darpa_direct_to_phase_ii()),
            AgencyId::Custom => None,
        }
    }

    /// Parses an agency profile from a TOML string.
    #[instrument(skip_all)]
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        toml::from_str(toml_str).map_err(|e| format!("TOML parse error: {e}"))
    }

    /// Serializes this agency profile to a TOML string.
    #[instrument(skip_all)]
    pub fn to_toml(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("TOML serialization error: {e}"))
    }
}

impl std::fmt::Display for AgencyProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({} pages, ${}-${}, {} months)",
            self.name,
            self.technical_page_limit,
            self.cost_range_min,
            self.cost_range_max,
            self.duration_months,
        )
    }
}

impl std::fmt::Display for AgencyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DodPhaseI => write!(f, "DoD Phase I"),
            Self::NsfPhaseI => write!(f, "NSF Phase I"),
            Self::AfwerxOpenTopic => write!(f, "AFWERX Open Topic"),
            Self::DarpaDirectToPhaseII => write!(f, "DARPA Direct-to-Phase-II"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agency_profile_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(AgencyProfile);
    }

    #[test]
    fn test_agency_profile_defaults_valid() {
        forge_types::assert_config_defaults_valid!(AgencyProfile);
    }

    #[test]
    fn test_default_is_dod() {
        let profile = AgencyProfile::default();
        assert_eq!(profile.id, AgencyId::DodPhaseI);
    }

    #[test]
    fn test_builtin_profiles_count() {
        let profiles = AgencyProfile::builtin_profiles();
        assert_eq!(profiles.len(), 4);
    }

    #[test]
    fn test_builtin_profiles_unique_ids() {
        let profiles = AgencyProfile::builtin_profiles();
        let ids: Vec<AgencyId> = profiles.iter().map(|p| p.id).collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "duplicate agency ID at indices {i} and {j}");
            }
        }
    }

    #[test]
    fn test_by_id_dod() {
        let profile = AgencyProfile::by_id(AgencyId::DodPhaseI).unwrap();
        assert_eq!(profile.id, AgencyId::DodPhaseI);
        assert_eq!(
            profile.technical_page_limit,
            constants::DEFAULT_DOD_PHASE_I_PAGE_LIMIT
        );
        assert_eq!(
            profile.cost_range_min,
            constants::DEFAULT_DOD_PHASE_I_COST_MIN
        );
        assert_eq!(
            profile.cost_range_max,
            constants::DEFAULT_DOD_PHASE_I_COST_MAX
        );
        assert_eq!(
            profile.duration_months,
            constants::DEFAULT_DOD_PHASE_I_DURATION_MONTHS
        );
    }

    #[test]
    fn test_by_id_nsf() {
        let profile = AgencyProfile::by_id(AgencyId::NsfPhaseI).unwrap();
        assert_eq!(profile.id, AgencyId::NsfPhaseI);
        assert_eq!(
            profile.technical_page_limit,
            constants::DEFAULT_NSF_PHASE_I_PAGE_LIMIT
        );
        assert_eq!(
            profile.duration_months,
            constants::DEFAULT_NSF_PHASE_I_DURATION_MONTHS
        );
    }

    #[test]
    fn test_by_id_afwerx() {
        let profile = AgencyProfile::by_id(AgencyId::AfwerxOpenTopic).unwrap();
        assert_eq!(profile.id, AgencyId::AfwerxOpenTopic);
        assert_eq!(
            profile.technical_page_limit,
            constants::DEFAULT_AFWERX_PAGE_LIMIT
        );
        assert_eq!(
            profile.duration_months,
            constants::DEFAULT_AFWERX_DURATION_MONTHS
        );
    }

    #[test]
    fn test_by_id_darpa() {
        let profile = AgencyProfile::by_id(AgencyId::DarpaDirectToPhaseII).unwrap();
        assert_eq!(profile.id, AgencyId::DarpaDirectToPhaseII);
        assert_eq!(
            profile.technical_page_limit,
            constants::DEFAULT_DARPA_D2P2_PAGE_LIMIT
        );
        assert_eq!(
            profile.supplemental_page_limit,
            Some(constants::DEFAULT_DARPA_D2P2_FEASIBILITY_PAGE_LIMIT)
        );
    }

    #[test]
    fn test_by_id_custom_returns_none() {
        assert!(AgencyProfile::by_id(AgencyId::Custom).is_none());
    }

    #[test]
    fn test_all_builtins_have_naics() {
        for profile in AgencyProfile::builtin_profiles() {
            assert!(
                !profile.naics_codes.is_empty(),
                "profile {} missing NAICS",
                profile.name
            );
        }
    }

    #[test]
    fn test_all_builtins_have_required_sections() {
        for profile in AgencyProfile::builtin_profiles() {
            assert!(
                !profile.required_sections.is_empty(),
                "profile {} has no required sections",
                profile.name,
            );
        }
    }

    #[test]
    fn test_all_builtins_cost_range_valid() {
        for profile in AgencyProfile::builtin_profiles() {
            assert!(
                profile.cost_range_min <= profile.cost_range_max,
                "profile {} has invalid cost range: {} > {}",
                profile.name,
                profile.cost_range_min,
                profile.cost_range_max,
            );
        }
    }

    #[test]
    fn test_agency_profile_toml_roundtrip() {
        let profile = AgencyProfile::dod_phase_i();
        let toml_str = profile.to_toml().unwrap();
        let deser = AgencyProfile::from_toml(&toml_str).unwrap();
        assert_eq!(deser.id, profile.id);
        assert_eq!(deser.technical_page_limit, profile.technical_page_limit);
        assert_eq!(deser.cost_range_min, profile.cost_range_min);
    }

    #[test]
    fn test_agency_profile_display() {
        let profile = AgencyProfile::dod_phase_i();
        let display = format!("{profile}");
        assert!(display.contains("DoD"));
        assert!(display.contains("10 pages"));
    }

    #[test]
    fn test_agency_id_display() {
        assert_eq!(format!("{}", AgencyId::DodPhaseI), "DoD Phase I");
        assert_eq!(format!("{}", AgencyId::NsfPhaseI), "NSF Phase I");
        assert_eq!(
            format!("{}", AgencyId::AfwerxOpenTopic),
            "AFWERX Open Topic"
        );
        assert_eq!(format!("{}", AgencyId::Custom), "Custom");
    }

    #[test]
    fn test_darpa_has_feasibility_section() {
        let profile = AgencyProfile::darpa_direct_to_phase_ii();
        assert!(profile
            .required_sections
            .contains(&"feasibility".to_string()));
    }

    #[test]
    fn test_afwerx_fewer_required_sections() {
        let afwerx = AgencyProfile::afwerx_open_topic();
        let dod = AgencyProfile::dod_phase_i();
        assert!(
            afwerx.required_sections.len() < dod.required_sections.len(),
            "AFWERX should have fewer required sections than DoD",
        );
    }
}
