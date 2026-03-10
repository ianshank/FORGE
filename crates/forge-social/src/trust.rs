//! Trust matrix for pairwise agent trust tracking.
//!
//! Trust is a fundamental social primitive that determines how agents
//! interact: high trust enables cooperation, low trust triggers caution.
//! Trust is updated based on observed behavior (cooperation, combat, betrayal).

use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};

use crate::config::SocialConfig;

/// Pairwise trust matrix between agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustMatrix {
    /// Trust scores: `scores[i][j]` = agent i's trust toward agent j.
    scores: Vec<Vec<f32>>,
    /// Number of interactions between each pair.
    interaction_count: Vec<Vec<u32>>,
    /// Number of agents tracked.
    num_agents: usize,
}

impl TrustMatrix {
    /// Creates a new trust matrix for `n` agents with initial trust values.
    #[instrument(skip_all)]
    pub fn new(num_agents: usize, initial_trust: f32) -> Self {
        Self {
            scores: vec![vec![initial_trust; num_agents]; num_agents],
            interaction_count: vec![vec![0; num_agents]; num_agents],
            num_agents,
        }
    }

    /// Returns the trust that agent `from` has toward agent `to`.
    pub fn trust(&self, from: usize, to: usize) -> f32 {
        if from < self.num_agents && to < self.num_agents {
            self.scores[from][to]
        } else {
            0.0
        }
    }

    /// Returns the trust scores that a given agent has toward all others.
    pub fn trust_vector(&self, agent: usize) -> &[f32] {
        if agent < self.num_agents {
            &self.scores[agent]
        } else {
            &[]
        }
    }

    /// Updates trust based on a cooperative interaction.
    #[instrument(skip_all)]
    pub fn record_cooperation(&mut self, agent_a: usize, agent_b: usize, config: &SocialConfig) {
        if agent_a >= self.num_agents || agent_b >= self.num_agents || agent_a == agent_b {
            return;
        }
        let lr = config.trust_update_rate;
        self.scores[agent_a][agent_b] = (self.scores[agent_a][agent_b] + lr).min(1.0);
        self.scores[agent_b][agent_a] = (self.scores[agent_b][agent_a] + lr).min(1.0);
        self.interaction_count[agent_a][agent_b] += 1;
        self.interaction_count[agent_b][agent_a] += 1;
        trace!(agent_a, agent_b, "cooperation recorded");
    }

    /// Updates trust based on a hostile interaction (combat, betrayal).
    #[instrument(skip_all)]
    pub fn record_hostility(&mut self, agent_a: usize, agent_b: usize, config: &SocialConfig) {
        if agent_a >= self.num_agents || agent_b >= self.num_agents || agent_a == agent_b {
            return;
        }
        let lr = config.trust_update_rate;
        self.scores[agent_a][agent_b] = (self.scores[agent_a][agent_b] - lr).max(0.0);
        self.scores[agent_b][agent_a] = (self.scores[agent_b][agent_a] - lr).max(0.0);
        self.interaction_count[agent_a][agent_b] += 1;
        self.interaction_count[agent_b][agent_a] += 1;
        trace!(agent_a, agent_b, "hostility recorded");
    }

    /// Returns the number of interactions between two agents.
    pub fn interactions(&self, agent_a: usize, agent_b: usize) -> u32 {
        if agent_a < self.num_agents && agent_b < self.num_agents {
            self.interaction_count[agent_a][agent_b]
        } else {
            0
        }
    }

    /// Returns the number of agents in the matrix.
    pub fn num_agents(&self) -> usize {
        self.num_agents
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SocialConfig {
        SocialConfig {
            trust_update_rate: 0.1,
            ..SocialConfig::default()
        }
    }

    #[test]
    fn test_initial_trust() {
        let tm = TrustMatrix::new(3, 0.5);
        assert_eq!(tm.trust(0, 1), 0.5);
        assert_eq!(tm.trust(1, 2), 0.5);
    }

    #[test]
    fn test_cooperation_increases_trust() {
        let mut tm = TrustMatrix::new(3, 0.5);
        let config = test_config();
        tm.record_cooperation(0, 1, &config);
        assert!(tm.trust(0, 1) > 0.5);
        assert!(tm.trust(1, 0) > 0.5);
        assert_eq!(tm.interactions(0, 1), 1);
    }

    #[test]
    fn test_hostility_decreases_trust() {
        let mut tm = TrustMatrix::new(3, 0.5);
        let config = test_config();
        tm.record_hostility(0, 1, &config);
        assert!(tm.trust(0, 1) < 0.5);
    }

    #[test]
    fn test_trust_bounds() {
        let mut tm = TrustMatrix::new(2, 0.95);
        let config = test_config();
        tm.record_cooperation(0, 1, &config);
        assert!(tm.trust(0, 1) <= 1.0);

        let mut tm2 = TrustMatrix::new(2, 0.05);
        tm2.record_hostility(0, 1, &config);
        assert!(tm2.trust(0, 1) >= 0.0);
    }

    #[test]
    fn test_out_of_bounds() {
        let tm = TrustMatrix::new(2, 0.5);
        assert_eq!(tm.trust(5, 0), 0.0);
    }
}
