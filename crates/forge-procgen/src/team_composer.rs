//! Team composition generation.
//!
//! Generates randomized team compositions with configurable unit types,
//! team sizes, and mixing policies.

use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Types of combat units available for team composition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnitType {
    /// Fast reconnaissance unit with high vision range.
    Scout,
    /// Standard combat unit, balanced stats.
    #[default]
    Infantry,
    /// Healing/buff support unit.
    Support,
    /// Slow, high-damage heavy unit.
    Heavy,
}

impl UnitType {
    /// Returns all available unit types.
    pub fn all() -> &'static [UnitType] {
        &[
            UnitType::Scout,
            UnitType::Infantry,
            UnitType::Support,
            UnitType::Heavy,
        ]
    }
}

/// A single slot in a team composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamSlot {
    /// The type of unit in this slot.
    pub unit_type: UnitType,
    /// Unique agent identifier.
    pub agent_id: u32,
}

/// Configuration for team composition generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamConfig {
    /// Number of units per team.
    pub team_size: u32,
    /// Number of teams to generate.
    pub num_teams: u32,
    /// Whether teams can have mixed unit types.
    pub allow_mixed: bool,
}

impl Default for TeamConfig {
    fn default() -> Self {
        Self {
            team_size: 4,
            num_teams: 2,
            allow_mixed: true,
        }
    }
}

/// Generates team compositions based on configuration and a seed.
///
/// Each team is a `Vec<TeamSlot>` with unique, sequential agent IDs.
/// When `allow_mixed` is true, each slot gets a random unit type.
/// When false, all units in a team share one randomly chosen type.
#[instrument(skip(config))]
pub fn compose_teams(config: &TeamConfig, seed: u64) -> Vec<Vec<TeamSlot>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut agent_id_counter: u32 = 0;

    (0..config.num_teams)
        .map(|_| {
            let team_type = if config.allow_mixed {
                None
            } else {
                let types = UnitType::all();
                Some(types[rng.gen_range(0..types.len())])
            };

            (0..config.team_size)
                .map(|_| {
                    let unit_type = match team_type {
                        Some(t) => t,
                        None => {
                            let types = UnitType::all();
                            types[rng.gen_range(0..types.len())]
                        }
                    };
                    let slot = TeamSlot {
                        unit_type,
                        agent_id: agent_id_counter,
                    };
                    agent_id_counter += 1;
                    slot
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_teams_correct_counts() {
        let config = TeamConfig {
            team_size: 3,
            num_teams: 4,
            ..Default::default()
        };
        let teams = compose_teams(&config, 42);
        assert_eq!(teams.len(), 4);
        for team in &teams {
            assert_eq!(team.len(), 3);
        }
    }

    #[test]
    fn test_compose_teams_deterministic() {
        let config = TeamConfig::default();
        let a = compose_teams(&config, 42);
        let b = compose_teams(&config, 42);
        for (team_a, team_b) in a.iter().zip(b.iter()) {
            for (slot_a, slot_b) in team_a.iter().zip(team_b.iter()) {
                assert_eq!(slot_a.unit_type, slot_b.unit_type);
                assert_eq!(slot_a.agent_id, slot_b.agent_id);
            }
        }
    }

    #[test]
    fn test_compose_teams_unique_agent_ids() {
        let config = TeamConfig {
            team_size: 5,
            num_teams: 3,
            ..Default::default()
        };
        let teams = compose_teams(&config, 99);
        let ids: Vec<u32> = teams
            .iter()
            .flat_map(|t| t.iter().map(|s| s.agent_id))
            .collect();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "All agent IDs must be unique");
    }

    #[test]
    fn test_compose_teams_unmixed() {
        let config = TeamConfig {
            team_size: 4,
            num_teams: 2,
            allow_mixed: false,
        };
        let teams = compose_teams(&config, 42);
        for team in &teams {
            let first_type = team[0].unit_type;
            for slot in team {
                assert_eq!(
                    slot.unit_type, first_type,
                    "Unmixed teams should have uniform types"
                );
            }
        }
    }

    #[test]
    fn test_unit_type_default() {
        assert_eq!(UnitType::default(), UnitType::Infantry);
    }
}
