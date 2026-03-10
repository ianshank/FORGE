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
    pub fn new(num_agents: usize) -> Self {
        Self {
            scores: vec![0.0; num_agents],
            cooperative_actions: vec![0; num_agents],
            hostile_actions: vec![0; num_agents],
        }
    }

    /// Returns the reputation of the given agent.
    pub fn reputation(&self, agent: usize) -> f32 {
        self.scores.get(agent).copied().unwrap_or(0.0)
    }

    /// Returns all reputation scores.
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
}
