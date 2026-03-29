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
#[serde(default)]
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
    #[instrument(skip_all)]
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

    #[test]
    fn test_different_weights() {
        let trust = TrustMatrix::new(2, 0.5);
        let mut reputation = ReputationTracker::new(2);
        reputation.record_cooperation(0); // agent 0 reputation = 1.0
        let config = SocialConfig::default();

        // Heavy reputation weight
        let computer_rep = SocialRewardComputer::new(SocialRewardConfig {
            cooperation_weight: 0.0,
            reputation_weight: 1.0,
            alliance_weight: 0.0,
        });
        let rewards_rep = computer_rep.compute(&trust, &reputation, &config);
        // Agent 0 should get reward = 1.0 * 1.0 = 1.0 (reputation only)
        assert!((rewards_rep[0] - 1.0).abs() < 1e-6);
        // Agent 1 has 0 reputation
        assert!(rewards_rep[1].abs() < 1e-6);

        // Heavy cooperation weight, zero reputation weight
        let computer_coop = SocialRewardComputer::new(SocialRewardConfig {
            cooperation_weight: 1.0,
            reputation_weight: 0.0,
            alliance_weight: 0.0,
        });
        let rewards_coop = computer_coop.compute(&trust, &reputation, &config);
        // Trust is all at initial 0.5, so mean_trust - initial = 0.0
        assert!(rewards_coop[0].abs() < 1e-6);
    }

    #[test]
    fn test_large_agent_count() {
        let n = 50;
        let mut trust = TrustMatrix::new(n, 0.5);
        let mut reputation = ReputationTracker::new(n);
        let config = SocialConfig::default();

        // Give some agents cooperation and hostility
        for i in 0..n / 2 {
            trust.record_cooperation(i, i + 1, &config);
            reputation.record_cooperation(i);
        }
        for i in n / 2..n {
            reputation.record_hostility(i);
        }

        let computer = SocialRewardComputer::new(SocialRewardConfig::default());
        let rewards = computer.compute(&trust, &reputation, &config);
        assert_eq!(rewards.len(), n);

        // All rewards should be finite
        for (i, r) in rewards.iter().enumerate() {
            assert!(r.is_finite(), "reward[{i}] is not finite: {r}");
        }
    }

    #[test]
    fn test_all_neutral_zero_rewards() {
        // Zero trust, zero reputation => all rewards should be near-zero or negative
        let trust = TrustMatrix::new(5, 0.0);
        let reputation = ReputationTracker::new(5);
        let config = SocialConfig {
            trust_initial: 0.0,
            ..SocialConfig::default()
        };
        let computer = SocialRewardComputer::new(SocialRewardConfig::default());

        let rewards = computer.compute(&trust, &reputation, &config);
        assert_eq!(rewards.len(), 5);
        for (i, r) in rewards.iter().enumerate() {
            assert!(
                r.abs() < 1e-6,
                "reward[{i}] should be ~0 with zero trust and zero reputation, got {r}"
            );
        }
    }
}
