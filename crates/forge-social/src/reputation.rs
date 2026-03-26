//! Reputation tracking: aggregate reputation scores from observed behavior.
//!
//! Reputation is a public score that all agents can observe, computed from
//! an agent's history of cooperative vs. hostile actions.

use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};

/// Tracks reputation scores for each agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationTracker {
    /// Per-agent reputation scores in [-1.0, 1.0].
    scores: Vec<f32>,
    /// Total cooperative actions per agent.
    cooperative_actions: Vec<u32>,
    /// Total hostile actions per agent.
    hostile_actions: Vec<u32>,
}

impl ReputationTracker {
    /// Creates a new reputation tracker for `n` agents at neutral reputation.
    #[instrument(skip_all)]
    pub fn new(num_agents: usize) -> Self {
        Self {
            scores: vec![0.0; num_agents],
            cooperative_actions: vec![0; num_agents],
            hostile_actions: vec![0; num_agents],
        }
    }

    /// Returns the reputation of the given agent.
    #[instrument(skip_all)]
    pub fn reputation(&self, agent: usize) -> f32 {
        self.scores.get(agent).copied().unwrap_or(0.0)
    }

    /// Returns all reputation scores.
    #[instrument(skip_all)]
    pub fn all_scores(&self) -> &[f32] {
        &self.scores
    }

    /// Records a cooperative action by the given agent.
    #[instrument(skip_all)]
    pub fn record_cooperation(&mut self, agent: usize) {
        if agent < self.scores.len() {
            self.cooperative_actions[agent] += 1;
            self.recompute(agent);
            trace!(
                agent,
                reputation = self.scores[agent],
                "cooperation recorded"
            );
        }
    }

    /// Records a hostile action by the given agent.
    #[instrument(skip_all)]
    pub fn record_hostility(&mut self, agent: usize) {
        if agent < self.scores.len() {
            self.hostile_actions[agent] += 1;
            self.recompute(agent);
            trace!(agent, reputation = self.scores[agent], "hostility recorded");
        }
    }

    /// Recomputes reputation as (cooperative - hostile) / total.
    fn recompute(&mut self, agent: usize) {
        let coop = self.cooperative_actions[agent] as f32;
        let hostile = self.hostile_actions[agent] as f32;
        let total = coop + hostile;
        if total > 0.0 {
            self.scores[agent] = (coop - hostile) / total;
        }
    }

    /// Returns the number of agents tracked.
    #[instrument(skip_all)]
    pub fn num_agents(&self) -> usize {
        self.scores.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_reputation() {
        let rt = ReputationTracker::new(3);
        assert_eq!(rt.reputation(0), 0.0);
        assert_eq!(rt.reputation(1), 0.0);
    }

    #[test]
    fn test_cooperation_increases_reputation() {
        let mut rt = ReputationTracker::new(3);
        rt.record_cooperation(0);
        assert_eq!(rt.reputation(0), 1.0); // 1 coop, 0 hostile → 1.0
    }

    #[test]
    fn test_hostility_decreases_reputation() {
        let mut rt = ReputationTracker::new(3);
        rt.record_hostility(0);
        assert_eq!(rt.reputation(0), -1.0);
    }

    #[test]
    fn test_mixed_reputation() {
        let mut rt = ReputationTracker::new(3);
        rt.record_cooperation(0);
        rt.record_cooperation(0);
        rt.record_hostility(0);
        // (2 - 1) / 3 = 0.333...
        let rep = rt.reputation(0);
        assert!((rep - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_out_of_bounds() {
        let rt = ReputationTracker::new(2);
        assert_eq!(rt.reputation(99), 0.0);
    }

    #[test]
    fn test_all_scores() {
        let mut rt = ReputationTracker::new(3);
        rt.record_cooperation(0);
        let scores = rt.all_scores();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 0.0);
    }

    #[test]
    fn test_mixed_cooperation_hostility_ratio() {
        let mut rt = ReputationTracker::new(1);
        // 3 cooperative, 1 hostile → (3-1)/4 = 0.5
        rt.record_cooperation(0);
        rt.record_cooperation(0);
        rt.record_cooperation(0);
        rt.record_hostility(0);
        assert!((rt.reputation(0) - 0.5).abs() < 1e-6);

        // Add 1 more hostile → (3-2)/5 = 0.2
        rt.record_hostility(0);
        assert!((rt.reputation(0) - 0.2).abs() < 1e-6);

        // Equal coop and hostile → 0.0
        let mut rt2 = ReputationTracker::new(1);
        rt2.record_cooperation(0);
        rt2.record_hostility(0);
        assert!((rt2.reputation(0)).abs() < 1e-6);
    }

    #[test]
    fn test_out_of_bounds_agent_reputation() {
        let rt = ReputationTracker::new(3);
        assert_eq!(rt.reputation(3), 0.0);
        assert_eq!(rt.reputation(100), 0.0);
        assert_eq!(rt.reputation(usize::MAX), 0.0);
    }

    #[test]
    fn test_many_agents_bounds() {
        let n = 20;
        let mut rt = ReputationTracker::new(n);
        // Mix of operations across many agents
        for i in 0..n {
            for _ in 0..(i + 1) {
                rt.record_cooperation(i);
            }
            for _ in 0..i {
                rt.record_hostility(i);
            }
        }
        for i in 0..n {
            let r = rt.reputation(i);
            assert!(r >= -1.0, "reputation({i}) = {r} below -1.0");
            assert!(r <= 1.0, "reputation({i}) = {r} above 1.0");
        }

        // Agent 0 had 1 coop, 0 hostile → 1.0
        assert_eq!(rt.reputation(0), 1.0);
        // Agent 1 had 2 coop, 1 hostile → (2-1)/3 = 0.333...
        assert!((rt.reputation(1) - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_out_of_bounds_record_is_noop() {
        let mut rt = ReputationTracker::new(2);
        rt.record_cooperation(5); // should not panic
        rt.record_hostility(5);   // should not panic
        assert_eq!(rt.reputation(0), 0.0);
        assert_eq!(rt.reputation(1), 0.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::test_util::lcg_next;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn reputation_always_in_neg1_to_1(
            n in 2_usize..10,
            num_ops in 0_usize..100,
            seed in any::<u64>()
        ) {
            let mut rt = ReputationTracker::new(n);
            let mut rng_val = seed;
            for _ in 0..num_ops {
                rng_val = lcg_next(rng_val);
                let agent = (rng_val as usize) % n;
                if rng_val % 2 == 0 {
                    rt.record_cooperation(agent);
                } else {
                    rt.record_hostility(agent);
                }
            }
            for i in 0..n {
                let r = rt.reputation(i);
                prop_assert!(r >= -1.0, "reputation({i}) = {r} < -1.0");
                prop_assert!(r <= 1.0, "reputation({i}) = {r} > 1.0");
            }
        }
    }
}
