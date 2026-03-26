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
    #[instrument(skip_all)]
    pub fn trust(&self, from: usize, to: usize) -> f32 {
        if from < self.num_agents && to < self.num_agents {
            self.scores[from][to]
        } else {
            0.0
        }
    }

    /// Returns the trust scores that a given agent has toward all others.
    #[instrument(skip_all)]
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
    #[instrument(skip_all)]
    pub fn interactions(&self, agent_a: usize, agent_b: usize) -> u32 {
        if agent_a < self.num_agents && agent_b < self.num_agents {
            self.interaction_count[agent_a][agent_b]
        } else {
            0
        }
    }

    /// Returns the number of agents in the matrix.
    #[instrument(skip_all)]
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

    #[test]
    fn test_self_interaction_is_noop() {
        let mut tm = TrustMatrix::new(3, 0.5);
        let config = test_config();
        tm.record_cooperation(1, 1, &config);
        assert_eq!(tm.trust(1, 1), 0.5); // unchanged
        assert_eq!(tm.interactions(1, 1), 0);
    }

    #[test]
    fn test_trust_vector() {
        let tm = TrustMatrix::new(3, 0.5);
        let vec = tm.trust_vector(0);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[1], 0.5);
    }

    #[test]
    fn test_multiple_updates_accumulate() {
        let mut tm = TrustMatrix::new(3, 0.5);
        let config = test_config();
        tm.record_cooperation(0, 1, &config);
        tm.record_cooperation(0, 1, &config);
        tm.record_cooperation(0, 1, &config);
        // Three cooperations at lr=0.1 from 0.5 → 0.8
        let expected = 0.5 + 3.0 * 0.1;
        assert!((tm.trust(0, 1) - expected).abs() < 1e-6);
        assert_eq!(tm.interactions(0, 1), 3);
    }

    #[test]
    fn test_large_agent_count() {
        let n = 10;
        let mut tm = TrustMatrix::new(n, 0.5);
        let config = test_config();
        assert_eq!(tm.num_agents(), n);

        // Record interactions across many pairs
        for i in 0..n {
            for j in (i + 1)..n {
                tm.record_cooperation(i, j, &config);
            }
        }

        // Verify all pairwise trust increased and stays in bounds
        for i in 0..n {
            for j in 0..n {
                let t = tm.trust(i, j);
                assert!(t >= 0.0 && t <= 1.0);
                if i != j {
                    assert!(t > 0.5, "trust({i},{j}) should have increased");
                }
            }
        }
    }

    #[test]
    fn test_trust_vector_length() {
        for n in [1, 5, 8] {
            let tm = TrustMatrix::new(n, 0.5);
            for agent in 0..n {
                assert_eq!(tm.trust_vector(agent).len(), n);
            }
            // Out-of-bounds agent returns empty slice
            assert_eq!(tm.trust_vector(n + 10).len(), 0);
        }
    }

    #[test]
    fn test_interactions_symmetric_after_mixed() {
        let mut tm = TrustMatrix::new(4, 0.5);
        let config = test_config();
        tm.record_cooperation(1, 2, &config);
        tm.record_cooperation(1, 2, &config);
        tm.record_hostility(1, 2, &config);

        assert_eq!(tm.interactions(1, 2), 3);
        assert_eq!(tm.interactions(2, 1), 3);
        // Trust values are also symmetric since both cooperation and hostility
        // update symmetrically
        assert!((tm.trust(1, 2) - tm.trust(2, 1)).abs() < 1e-6);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::test_util::lcg_next;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn trust_always_in_0_1(
            n in 2_usize..10,
            initial in 0.0_f32..=1.0,
            num_ops in 0_usize..50,
            seed in any::<u64>()
        ) {
            let config = SocialConfig {
                trust_update_rate: 0.1,
                ..SocialConfig::default()
            };
            let mut tm = TrustMatrix::new(n, initial);
            let mut rng_val = seed;
            for _ in 0..num_ops {
                rng_val = lcg_next(rng_val);
                let a = (rng_val as usize) % n;
                let b = ((rng_val >> 16) as usize) % n;
                if rng_val % 2 == 0 {
                    tm.record_cooperation(a, b, &config);
                } else {
                    tm.record_hostility(a, b, &config);
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let t = tm.trust(i, j);
                    prop_assert!(t >= 0.0, "trust({i},{j}) = {t} < 0");
                    prop_assert!(t <= 1.0, "trust({i},{j}) = {t} > 1");
                }
            }
        }

        #[test]
        fn interaction_count_symmetric(
            n in 2_usize..10,
            a in 0_usize..10,
            b in 0_usize..10
        ) {
            let a = a % n;
            let b = b % n;
            let config = SocialConfig::default();
            let mut tm = TrustMatrix::new(n, 0.5);
            tm.record_cooperation(a, b, &config);
            if a != b {
                prop_assert_eq!(tm.interactions(a, b), tm.interactions(b, a));
            }
        }
    }
}
