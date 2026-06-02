//! Joint value estimation for cooperative multi-agent planning.
//!
//! Extends the single-agent [`PolicyValue`](forge_agent::mcts::policy::PolicyValue)
//! interface with a *joint* value estimate over the whole swarm. Two
//! Centralized-Training / Decentralized-Execution (CTDE) flavours are provided:
//!
//! * [`CentralizedCritic`] — the joint value aggregates every agent's value
//!   estimate (cooperative: agents are rewarded for the team outcome).
//! * [`IndependentCritic`] — the joint value is a single agent's value only
//!   (decentralized baseline: no cross-agent credit assignment).
//!
//! Both wrap any inner [`PolicyValue`] (default: a zero-dependency
//! [`UniformPolicy`](forge_agent::mcts::policy::UniformPolicy)) so the planner
//! stays deterministic and free of ML runtime dependencies.

use forge_agent::mcts::policy::{PolicyValue, PolicyValueOutput};
use forge_core::WorldState;

/// A policy/value function that can also estimate the *joint* value of a state
/// across all swarm agents.
///
/// Implementors are also [`PolicyValue`], so they can drive the existing
/// single-agent MCTS search per agent while additionally exposing a joint
/// value for joint-action scoring.
pub trait JointPolicyValue: PolicyValue {
    /// Estimates the joint (team) value of `state` over `num_agents` agents.
    fn joint_value(&self, state: &WorldState, num_agents: usize) -> f32;
}

/// Centralized critic: the joint value is the sum of every agent's value
/// estimate. This couples the agents so the planner prefers joint actions that
/// are good for the team as a whole (cooperative CTDE).
#[derive(Debug, Clone)]
pub struct CentralizedCritic<P: PolicyValue + Clone> {
    inner: P,
}

impl<P: PolicyValue + Clone> CentralizedCritic<P> {
    /// Wraps an inner policy/value function as a centralized critic.
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<P: PolicyValue + Clone> PolicyValue for CentralizedCritic<P> {
    fn evaluate(&self, state: &WorldState, agent_idx: usize) -> PolicyValueOutput {
        self.inner.evaluate(state, agent_idx)
    }
}

impl<P: PolicyValue + Clone> JointPolicyValue for CentralizedCritic<P> {
    fn joint_value(&self, state: &WorldState, num_agents: usize) -> f32 {
        (0..num_agents)
            .map(|i| self.inner.evaluate(state, i).value)
            .sum()
    }
}

/// Independent critic: the joint value is agent 0's value only. A decentralized
/// baseline with no cross-agent credit assignment, useful for ablations against
/// [`CentralizedCritic`].
#[derive(Debug, Clone)]
pub struct IndependentCritic<P: PolicyValue + Clone> {
    inner: P,
}

impl<P: PolicyValue + Clone> IndependentCritic<P> {
    /// Wraps an inner policy/value function as an independent critic.
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<P: PolicyValue + Clone> PolicyValue for IndependentCritic<P> {
    fn evaluate(&self, state: &WorldState, agent_idx: usize) -> PolicyValueOutput {
        self.inner.evaluate(state, agent_idx)
    }
}

impl<P: PolicyValue + Clone> JointPolicyValue for IndependentCritic<P> {
    fn joint_value(&self, state: &WorldState, num_agents: usize) -> f32 {
        // Decentralized baseline: only the first agent's value is considered.
        if num_agents == 0 {
            return 0.0;
        }
        self.inner.evaluate(state, 0).value
    }
}

/// Runtime-selectable critic so a single planner type can switch between
/// [`CentralizedCritic`] and [`IndependentCritic`] based on config, without a
/// generic type parameter leaking up to callers.
#[derive(Debug, Clone)]
pub enum Critic<P: PolicyValue + Clone> {
    /// Cooperative critic — joint value aggregates all agents.
    Centralized(CentralizedCritic<P>),
    /// Decentralized baseline — joint value uses a single agent.
    Independent(IndependentCritic<P>),
}

impl<P: PolicyValue + Clone> Critic<P> {
    /// Builds a critic from an inner policy and the `centralized` config flag.
    pub fn from_config(inner: P, centralized: bool) -> Self {
        if centralized {
            Critic::Centralized(CentralizedCritic::new(inner))
        } else {
            Critic::Independent(IndependentCritic::new(inner))
        }
    }
}

impl<P: PolicyValue + Clone> PolicyValue for Critic<P> {
    fn evaluate(&self, state: &WorldState, agent_idx: usize) -> PolicyValueOutput {
        match self {
            Critic::Centralized(c) => c.evaluate(state, agent_idx),
            Critic::Independent(c) => c.evaluate(state, agent_idx),
        }
    }
}

impl<P: PolicyValue + Clone> JointPolicyValue for Critic<P> {
    fn joint_value(&self, state: &WorldState, num_agents: usize) -> f32 {
        match self {
            Critic::Centralized(c) => c.joint_value(state, num_agents),
            Critic::Independent(c) => c.joint_value(state, num_agents),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_agent::mcts::policy::UniformPolicy;
    use forge_types::config::ForgeConfig;

    /// Deterministic policy returning a constant value for coverage of the
    /// aggregation logic (UniformPolicy returns value 0.0 which can't
    /// distinguish centralized from independent).
    #[derive(Debug, Clone)]
    struct ConstValuePolicy {
        value: f32,
        action_space: u32,
    }

    impl PolicyValue for ConstValuePolicy {
        fn evaluate(&self, _state: &WorldState, _agent_idx: usize) -> PolicyValueOutput {
            PolicyValueOutput {
                priors: vec![1.0 / self.action_space as f32; self.action_space as usize],
                value: self.value,
            }
        }
    }

    fn make_state(num_agents: usize) -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.world.seed = 7;
        config.agents.num_agents = num_agents as u32;
        config.agents.default_vision_radius = 3;
        WorldState::new(config).unwrap()
    }

    #[test]
    fn test_centralized_sums_agent_values() {
        let critic = CentralizedCritic::new(ConstValuePolicy {
            value: 2.0,
            action_space: 4,
        });
        let state = make_state(3);
        // 3 agents * 2.0 each = 6.0
        assert!((critic.joint_value(&state, 3) - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_independent_uses_single_agent() {
        let critic = IndependentCritic::new(ConstValuePolicy {
            value: 2.0,
            action_space: 4,
        });
        let state = make_state(3);
        // Only agent 0 considered: 2.0 regardless of agent count.
        assert!((critic.joint_value(&state, 3) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_independent_zero_agents() {
        let critic = IndependentCritic::new(ConstValuePolicy {
            value: 2.0,
            action_space: 4,
        });
        let state = make_state(1);
        assert_eq!(critic.joint_value(&state, 0), 0.0);
    }

    #[test]
    fn test_centralized_vs_independent_differ() {
        let inner = ConstValuePolicy {
            value: 1.5,
            action_space: 4,
        };
        let state = make_state(2);
        let central = CentralizedCritic::new(inner.clone());
        let indep = IndependentCritic::new(inner);
        assert!(central.joint_value(&state, 2) > indep.joint_value(&state, 2));
    }

    #[test]
    fn test_critic_enum_dispatch() {
        let state = make_state(2);
        let central = Critic::from_config(UniformPolicy::new(4), true);
        let indep = Critic::from_config(UniformPolicy::new(4), false);
        // UniformPolicy yields value 0.0, so both joint values are 0.0, but the
        // dispatch must not panic and priors must be populated.
        assert_eq!(central.evaluate(&state, 0).priors.len(), 4);
        assert_eq!(indep.evaluate(&state, 0).priors.len(), 4);
        assert_eq!(central.joint_value(&state, 2), 0.0);
        assert_eq!(indep.joint_value(&state, 2), 0.0);
    }
}
