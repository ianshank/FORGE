//! Policy and value network interface for MCTS.
//!
//! Defines the trait that policy/value networks must implement
//! to guide the MCTS search. Provides a uniform random policy
//! as a default.

use forge_core::WorldState;

/// A policy/value function that guides MCTS search.
///
/// Implementations can be neural networks, heuristics, or random.
pub trait PolicyValue: Send + Sync {
    /// Evaluates a state, returning:
    /// - `prior`: Action probability distribution (unnormalized or normalized).
    /// - `value`: Estimated state value for the current agent.
    fn evaluate(&self, state: &WorldState, agent_idx: usize) -> PolicyValueOutput;
}

/// Output of a policy-value evaluation.
#[derive(Debug, Clone)]
pub struct PolicyValueOutput {
    /// Prior probability for each action (indexed by discrete action id).
    pub priors: Vec<f32>,
    /// Estimated state value for the agent.
    pub value: f32,
}

/// Uniform random policy: all actions get equal probability.
#[derive(Debug, Default)]
pub struct UniformPolicy {
    action_space: u32,
}

impl UniformPolicy {
    /// Creates a new uniform policy for the given action space.
    pub fn new(action_space: u32) -> Self {
        Self { action_space }
    }
}

impl PolicyValue for UniformPolicy {
    fn evaluate(&self, _state: &WorldState, _agent_idx: usize) -> PolicyValueOutput {
        let prior = 1.0 / self.action_space.max(1) as f32;
        PolicyValueOutput {
            priors: vec![prior; self.action_space as usize],
            value: 0.0,
        }
    }
}

/// Heuristic policy: gives higher probability to movement actions.
#[derive(Debug)]
pub struct HeuristicPolicy {
    action_space: u32,
}

impl HeuristicPolicy {
    /// Creates a new heuristic policy.
    pub fn new(action_space: u32) -> Self {
        Self { action_space }
    }
}

impl PolicyValue for HeuristicPolicy {
    fn evaluate(&self, _state: &WorldState, _agent_idx: usize) -> PolicyValueOutput {
        let mut priors = vec![0.05; self.action_space as usize];

        // Boost movement actions (indices 1-4)
        for i in 1..=4.min(self.action_space.saturating_sub(1)) {
            if (i as usize) < priors.len() {
                priors[i as usize] = 0.2;
            }
        }

        // Boost PickUp (index 5)
        if priors.len() > 5 {
            priors[5] = 0.15;
        }

        // Normalize
        let total: f32 = priors.iter().sum();
        if total > 0.0 {
            for p in &mut priors {
                *p /= total;
            }
        }

        PolicyValueOutput { priors, value: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;

    fn make_state() -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.agents.num_agents = 1;
        config.agents.default_vision_radius = 3;
        WorldState::new(config).unwrap()
    }

    #[test]
    fn test_uniform_policy() {
        let policy = UniformPolicy::new(32);
        let state = make_state();
        let output = policy.evaluate(&state, 0);

        assert_eq!(output.priors.len(), 32);
        let sum: f32 = output.priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(output.value, 0.0);
    }

    #[test]
    fn test_heuristic_policy() {
        let policy = HeuristicPolicy::new(32);
        let state = make_state();
        let output = policy.evaluate(&state, 0);

        assert_eq!(output.priors.len(), 32);
        let sum: f32 = output.priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);

        // Movement actions should have higher prior than noop
        assert!(output.priors[1] > output.priors[0]);
    }

    #[test]
    fn test_uniform_policy_normalization() {
        let policy = UniformPolicy::new(4);
        let state = make_state();
        let output = policy.evaluate(&state, 0);

        assert_eq!(output.priors.len(), 4);
        for p in &output.priors {
            assert!((p - 0.25).abs() < 1e-5);
        }
    }
}
