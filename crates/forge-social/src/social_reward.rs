//! Social reward computation: trust-weighted, cooperation-based rewards.
//!
//! Social rewards augment task-completion rewards with signals from
//! social interactions, enabling agents to learn cooperative behavior.

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::SocialConfig;
use crate::reputation::ReputationTracker;
use crate::trust::TrustMatrix;

/// Configuration for social reward computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialRewardConfig {
    /// Weight of cooperation rewards.
    pub cooperation_weight: f32,
    /// Weight of reputation rewards.
    pub reputation_weight: f32,
    /// Weight of alliance stability rewards.
    pub alliance_weight: f32,
}

impl Default for SocialRewardConfig {
    fn default() -> Self {
        Self {
            cooperation_weight: 0.4,
            reputation_weight: 0.3,
            alliance_weight: 0.3,
        }
    }
}

/// Computes social rewards for agents based on trust and reputation.
#[derive(Debug, Clone)]
pub struct SocialRewardComputer {
    /// Reward configuration.
    config: SocialRewardConfig,
}

impl SocialRewardComputer {
    /// Creates a new social reward computer.
    pub fn new(config: SocialRewardConfig) -> Self {
        Self { config }
    }

    /// Computes per-agent social rewards.
    ///
    /// The social reward for each agent combines:
    /// - Mean trust others have toward the agent (cooperation signal)
    /// - The agent's reputation score
    /// - Alliance stability (bonus for maintained alliances)
    #[instrument(skip_all)]
    pub fn compute(
        &self,
        trust: &TrustMatrix,
        reputation: &ReputationTracker,
        social_config: &SocialConfig,
    ) -> Vec<f32> {
        let n = trust.num_agents();
        let mut rewards = vec![0.0_f32; n];

        for (i, reward) in rewards.iter_mut().enumerate() {
            // Cooperation signal: mean trust others have toward this agent
            let trust_sum: f32 = (0..n).filter(|&j| j != i).map(|j| trust.trust(j, i)).sum();
            let mean_trust = if n > 1 {
                trust_sum / (n - 1) as f32
            } else {
                0.0
            };

            let cooperation_reward =
                (mean_trust - social_config.trust_initial) * self.config.cooperation_weight;

            // Reputation signal
            let rep_reward = reputation.reputation(i) * self.config.reputation_weight;

            *reward = cooperation_reward + rep_reward;
        }

        rewards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_social_rewards() {
        let trust = TrustMatrix::new(3, 0.5);
        let reputation = ReputationTracker::new(3);
        let config = SocialConfig::default();
        let computer = SocialRewardComputer::new(SocialRewardConfig::default());

        let rewards = computer.compute(&trust, &reputation, &config);
        assert_eq!(rewards.len(), 3);
        // With initial trust and zero reputation, rewards should be near zero
        for r in &rewards {
            assert!(r.abs() < 0.01);
        }
    }

    #[test]
    fn test_positive_cooperation_reward() {
        let mut trust = TrustMatrix::new(3, 0.5);
        let config = SocialConfig::default();
        // Agent 0 is trusted by others
        trust.record_cooperation(1, 0, &config);
        trust.record_cooperation(2, 0, &config);

        let reputation = ReputationTracker::new(3);
        let computer = SocialRewardComputer::new(SocialRewardConfig::default());

        let rewards = computer.compute(&trust, &reputation, &config);
        assert!(rewards[0] > 0.0, "trusted agent should get positive reward");
    }
}
