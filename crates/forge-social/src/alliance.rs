//! Alliance system: dynamic team formation and betrayal detection.
//!
//! Alliances are formed when mutual trust exceeds a threshold and broken
//! when trust drops below it. This enables emergent cooperation and
//! the kind of social dynamics that shape Data's moral reasoning.

use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};

use crate::config::SocialConfig;
use crate::trust::TrustMatrix;

/// An alliance between two or more agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alliance {
    /// Unique alliance identifier.
    pub id: u32,
    /// Member agent indices.
    pub members: Vec<usize>,
    /// Tick when the alliance was formed.
    pub formed_at: u64,
}

/// Manages dynamic alliances between agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllianceSystem {
    /// Active alliances.
    alliances: Vec<Alliance>,
    /// Per-agent alliance membership (agent_idx → alliance_id).
    membership: Vec<Option<u32>>,
    /// Next alliance ID.
    next_id: u32,
}

impl AllianceSystem {
    /// Creates a new alliance system for `n` agents.
    #[instrument(skip_all)]
    pub fn new(num_agents: usize) -> Self {
        Self {
            alliances: Vec::new(),
            membership: vec![None; num_agents],
            next_id: 0,
        }
    }

    /// Returns the alliance ID for the given agent, if any.
    #[instrument(skip_all)]
    pub fn alliance_of(&self, agent: usize) -> Option<u32> {
        self.membership.get(agent).copied().flatten()
    }

    /// Returns the members of the given alliance.
    #[instrument(skip_all)]
    pub fn alliance_members(&self, alliance_id: u32) -> Vec<usize> {
        self.alliances
            .iter()
            .find(|a| a.id == alliance_id)
            .map(|a| a.members.clone())
            .unwrap_or_default()
    }

    /// Returns `true` if two agents are in the same alliance.
    #[instrument(skip_all)]
    pub fn are_allied(&self, a: usize, b: usize) -> bool {
        match (self.alliance_of(a), self.alliance_of(b)) {
            (Some(aa), Some(ab)) => aa == ab,
            _ => false,
        }
    }

    /// Evaluates trust matrix and forms/breaks alliances based on thresholds.
    #[instrument(skip_all)]
    pub fn update(&mut self, trust: &TrustMatrix, config: &SocialConfig, current_tick: u64) {
        let n = trust.num_agents();
        let threshold = config.alliance_threshold;

        // Check for new alliance opportunities
        for i in 0..n {
            for j in (i + 1)..n {
                let mutual_trust = trust.trust(i, j).min(trust.trust(j, i));
                let both_unallied = self.alliance_of(i).is_none() && self.alliance_of(j).is_none();

                if mutual_trust >= threshold && both_unallied {
                    let i_count = self.alliance_count(i);
                    let j_count = self.alliance_count(j);
                    if i_count < config.max_alliances && j_count < config.max_alliances {
                        self.form_alliance(vec![i, j], current_tick);
                        trace!(agent_a = i, agent_b = j, "alliance formed");
                    }
                }
            }
        }

        // Check for alliance breaks
        let mut to_remove = Vec::new();
        for alliance in &self.alliances {
            let should_break = alliance.members.iter().any(|&m| {
                alliance
                    .members
                    .iter()
                    .any(|&other| m != other && trust.trust(m, other) < threshold * 0.5)
            });
            if should_break {
                to_remove.push(alliance.id);
                trace!(alliance_id = alliance.id, "alliance broken");
            }
        }

        for id in to_remove {
            self.dissolve_alliance(id);
        }
    }

    /// Returns the number of alliances the given agent belongs to.
    fn alliance_count(&self, agent: usize) -> u32 {
        if agent < self.membership.len() && self.membership[agent].is_some() {
            1
        } else {
            0
        }
    }

    /// Forms a new alliance between the given agents.
    fn form_alliance(&mut self, members: Vec<usize>, current_tick: u64) {
        let id = self.next_id;
        self.next_id += 1;
        for &m in &members {
            if m < self.membership.len() {
                self.membership[m] = Some(id);
            }
        }
        self.alliances.push(Alliance {
            id,
            members,
            formed_at: current_tick,
        });
    }

    /// Dissolves an alliance by ID.
    fn dissolve_alliance(&mut self, alliance_id: u32) {
        if let Some(pos) = self.alliances.iter().position(|a| a.id == alliance_id) {
            let alliance = self.alliances.swap_remove(pos);
            for &m in &alliance.members {
                if m < self.membership.len() {
                    self.membership[m] = None;
                }
            }
        }
    }

    /// Returns the number of active alliances.
    #[instrument(skip_all)]
    pub fn num_alliances(&self) -> usize {
        self.alliances.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SocialConfig {
        SocialConfig {
            alliance_threshold: 0.7,
            trust_update_rate: 0.1,
            ..SocialConfig::default()
        }
    }

    #[test]
    fn test_no_initial_alliances() {
        let sys = AllianceSystem::new(4);
        assert_eq!(sys.num_alliances(), 0);
        assert!(sys.alliance_of(0).is_none());
    }

    #[test]
    fn test_alliance_formation() {
        let mut sys = AllianceSystem::new(4);
        let trust = TrustMatrix::new(4, 0.8); // all above threshold
        let config = test_config();
        sys.update(&trust, &config, 100);
        // Agents 0 and 1 should form an alliance (first pair found)
        assert!(sys.num_alliances() > 0);
    }

    #[test]
    fn test_alliance_break() {
        let mut sys = AllianceSystem::new(4);
        let mut trust = TrustMatrix::new(4, 0.8);
        let config = test_config();
        sys.update(&trust, &config, 100);
        let initial = sys.num_alliances();

        // Drop trust between allied agents below break threshold
        for _ in 0..20 {
            trust.record_hostility(0, 1, &config);
        }
        sys.update(&trust, &config, 200);
        assert!(sys.num_alliances() <= initial);
    }

    #[test]
    fn test_are_allied() {
        let mut sys = AllianceSystem::new(3);
        let trust = TrustMatrix::new(3, 0.9);
        let config = test_config();
        sys.update(&trust, &config, 0);
        // At least some agents should be allied
        let has_alliance = (0..3).any(|i| sys.alliance_of(i).is_some());
        assert!(has_alliance);
    }

    #[test]
    fn test_alliance_members_returns_correct_set() {
        let mut sys = AllianceSystem::new(4);
        let trust = TrustMatrix::new(4, 0.9);
        let config = test_config();
        sys.update(&trust, &config, 0);
        if let Some(aid) = sys.alliance_of(0) {
            let members = sys.alliance_members(aid);
            assert!(members.contains(&0));
        }
    }

    #[test]
    fn test_nonexistent_alliance_returns_empty() {
        let sys = AllianceSystem::new(4);
        assert!(sys.alliance_members(999).is_empty());
    }

    #[test]
    fn test_multi_agent_alliance_formations() {
        // With 6 agents and high trust, multiple alliances should form
        let mut sys = AllianceSystem::new(6);
        let trust = TrustMatrix::new(6, 0.9);
        let config = test_config();
        sys.update(&trust, &config, 0);

        // With all trust at 0.9 (above 0.7 threshold), pairs form alliances
        assert!(sys.num_alliances() > 0);
        // Since agents can only be in one alliance, at most 3 alliances (6/2)
        assert!(sys.num_alliances() <= 3);

        // Verify every allied agent has a valid membership
        for i in 0..6 {
            if let Some(aid) = sys.alliance_of(i) {
                let members = sys.alliance_members(aid);
                assert!(members.contains(&i));
                assert!(members.len() >= 2);
            }
        }
    }

    #[test]
    fn test_alliance_dissolution_clears_membership() {
        let mut sys = AllianceSystem::new(4);
        let mut trust = TrustMatrix::new(4, 0.9);
        let config = test_config();
        sys.update(&trust, &config, 0);

        // Identify an allied pair
        let allied_agent = (0..4).find(|&i| sys.alliance_of(i).is_some());
        assert!(allied_agent.is_some(), "should have at least one alliance");
        let agent = allied_agent.unwrap();
        let aid = sys.alliance_of(agent).unwrap();
        let partner = sys
            .alliance_members(aid)
            .into_iter()
            .find(|&m| m != agent)
            .unwrap();

        // Destroy trust between the pair (drop below threshold * 0.5 = 0.35)
        for _ in 0..20 {
            trust.record_hostility(agent, partner, &config);
        }
        sys.update(&trust, &config, 100);

        // Alliance should be dissolved; both agents should have no alliance
        assert!(
            sys.alliance_of(agent).is_none() || sys.alliance_of(partner).is_none(),
            "at least one of the pair should lose alliance membership"
        );
    }

    #[test]
    fn test_alliance_members_nonexistent_id() {
        let sys = AllianceSystem::new(10);
        assert!(sys.alliance_members(0).is_empty());
        assert!(sys.alliance_members(42).is_empty());
        assert!(sys.alliance_members(u32::MAX).is_empty());
    }

    #[test]
    fn test_zero_threshold_all_form() {
        let mut sys = AllianceSystem::new(4);
        let trust = TrustMatrix::new(4, 0.0); // all trust at 0.0
        let config = SocialConfig {
            alliance_threshold: 0.0,
            max_alliances: 3,
            trust_update_rate: 0.1,
            ..SocialConfig::default()
        };
        sys.update(&trust, &config, 0);

        // With threshold 0.0, mutual trust 0.0 >= 0.0 so alliances should form
        assert!(sys.num_alliances() > 0, "alliances should form at threshold 0.0");
        // With 4 agents forming pairs, expect 2 alliances
        assert_eq!(sys.num_alliances(), 2);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn membership_consistent_with_alliances(
            n in 2_usize..8,
            initial_trust in 0.0_f32..=1.0
        ) {
            let mut sys = AllianceSystem::new(n);
            let trust = TrustMatrix::new(n, initial_trust);
            let config = SocialConfig {
                alliance_threshold: 0.7,
                ..SocialConfig::default()
            };
            sys.update(&trust, &config, 0);

            // Every agent in an alliance should appear in that alliance's member list
            for i in 0..n {
                if let Some(aid) = sys.alliance_of(i) {
                    let members = sys.alliance_members(aid);
                    prop_assert!(
                        members.contains(&i),
                        "agent {i} claims alliance {aid} but is not in member list"
                    );
                }
            }

            // are_allied should be symmetric
            for i in 0..n {
                for j in 0..n {
                    prop_assert_eq!(
                        sys.are_allied(i, j),
                        sys.are_allied(j, i),
                        "are_allied not symmetric for ({}, {})", i, j
                    );
                }
            }
        }

        /// max_alliances=0 prevents any alliance from forming.
        #[test]
        fn max_alliances_zero_prevents_formation(
            n in 2_usize..6,
            initial_trust in 0.8_f32..=1.0
        ) {
            let mut sys = AllianceSystem::new(n);
            let trust = TrustMatrix::new(n, initial_trust);
            let config = SocialConfig {
                alliance_threshold: 0.5,
                max_alliances: 0,
                ..SocialConfig::default()
            };
            sys.update(&trust, &config, 0);
            prop_assert_eq!(sys.num_alliances(), 0, "no alliances should form when max_alliances=0");
        }
    }
}
