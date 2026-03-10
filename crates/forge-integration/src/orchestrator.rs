//! Integration orchestrator: wires memory, cognition, and social layers together.
//!
//! The orchestrator manages the lifecycle of all proto-Data subsystems per agent,
//! coordinating memory writes, social updates, and cognitive processing.

use tracing::{info, instrument};

use forge_memory::store::InMemoryStore;
use forge_social::reputation::ReputationTracker;
use forge_social::social_reward::{SocialRewardComputer, SocialRewardConfig};
use forge_social::trust::TrustMatrix;

use crate::config::IntegrationConfig;

/// Per-agent integration state combining memory, social, and cognitive layers.
pub struct AgentIntegrationState {
    /// Agent's persistent memory store.
    pub memory: InMemoryStore,
    /// Agent index in the simulation.
    pub agent_idx: usize,
}

/// Orchestrator that manages all integration layers for a simulation.
pub struct IntegrationOrchestrator {
    /// Configuration.
    config: IntegrationConfig,
    /// Per-agent integration state.
    agent_states: Vec<AgentIntegrationState>,
    /// Shared trust matrix between all agents.
    pub trust: TrustMatrix,
    /// Shared reputation tracker.
    pub reputation: ReputationTracker,
    /// Cached social reward computer (avoids re-creation per tick).
    social_computer: SocialRewardComputer,
    /// Current tick (synchronized with simulation).
    tick: u64,
}

impl IntegrationOrchestrator {
    /// Creates a new orchestrator for the given number of agents.
    #[instrument(skip_all)]
    pub fn new(num_agents: usize, config: IntegrationConfig) -> Self {
        let memory_config = &config.memory;
        let social_config = &config.social;

        let agent_states = (0..num_agents)
            .map(|i| AgentIntegrationState {
                memory: InMemoryStore::new(i as u32, memory_config),
                agent_idx: i,
            })
            .collect();

        let trust = TrustMatrix::new(num_agents, social_config.trust_initial);
        let reputation = ReputationTracker::new(num_agents);

        let social_computer = SocialRewardComputer::new(SocialRewardConfig::default());

        info!(
            num_agents,
            memory_enabled = memory_config.enabled,
            social_enabled = social_config.enabled,
            "integration orchestrator created"
        );

        Self {
            config,
            agent_states,
            trust,
            reputation,
            social_computer,
            tick: 0,
        }
    }

    /// Advances the integration state by one tick.
    #[instrument(skip_all)]
    pub fn tick(&mut self) {
        self.tick += 1;

        let memory_config = &self.config.memory;
        // Apply memory decay at configured intervals when memory is enabled
        if memory_config.enabled && self.tick % self.config.memory_write_interval == 0 {
            for state in &mut self.agent_states {
                state.memory.tick_decay(memory_config);
            }
        }
    }

    /// Records a cooperative interaction between two agents.
    #[instrument(skip_all)]
    pub fn record_cooperation(&mut self, agent_a: usize, agent_b: usize) {
        self.trust
            .record_cooperation(agent_a, agent_b, &self.config.social);
        self.reputation.record_cooperation(agent_a);
        self.reputation.record_cooperation(agent_b);
    }

    /// Records a hostile interaction between two agents.
    #[instrument(skip_all)]
    pub fn record_hostility(&mut self, agent_a: usize, agent_b: usize) {
        self.trust
            .record_hostility(agent_a, agent_b, &self.config.social);
        self.reputation.record_hostility(agent_a);
    }

    /// Returns a reference to the agent's memory store.
    pub fn agent_memory(&self, agent_idx: usize) -> Option<&InMemoryStore> {
        self.agent_states.get(agent_idx).map(|s| &s.memory)
    }

    /// Returns a mutable reference to the agent's memory store.
    pub fn agent_memory_mut(&mut self, agent_idx: usize) -> Option<&mut InMemoryStore> {
        self.agent_states.get_mut(agent_idx).map(|s| &mut s.memory)
    }

    /// Computes blended rewards (task + social).
    #[instrument(skip_all)]
    pub fn blend_rewards(&self, task_rewards: &[f32]) -> Vec<f32> {
        let w = self.config.social_reward_weight;

        // If social integration is disabled or its weight is zero, return task rewards unchanged.
        if !self.config.social.enabled || w == 0.0 {
            return task_rewards.to_vec();
        }

        let social_rewards =
            self.social_computer
                .compute(&self.trust, &self.reputation, &self.config.social);
        task_rewards
            .iter()
            .zip(social_rewards.iter())
            .map(|(task, social)| task * (1.0 - w) + social * w)
            .collect()
    }

    /// Returns the current tick.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Returns the number of agents managed.
    pub fn num_agents(&self) -> usize {
        self.agent_states.len()
    }

    /// Returns the configuration.
    pub fn config(&self) -> &IntegrationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> IntegrationConfig {
        IntegrationConfig {
            enabled: true,
            memory_write_interval: 5,
            ..IntegrationConfig::default()
        }
    }

    #[test]
    fn test_orchestrator_creation() {
        let orch = IntegrationOrchestrator::new(4, test_config());
        assert_eq!(orch.num_agents(), 4);
        assert_eq!(orch.current_tick(), 0);
    }

    #[test]
    fn test_tick_advances() {
        let mut orch = IntegrationOrchestrator::new(2, test_config());
        orch.tick();
        orch.tick();
        assert_eq!(orch.current_tick(), 2);
    }

    #[test]
    fn test_cooperation_updates_trust() {
        let mut orch = IntegrationOrchestrator::new(3, test_config());
        let initial_trust = orch.trust.trust(0, 1);
        orch.record_cooperation(0, 1);
        assert!(orch.trust.trust(0, 1) > initial_trust);
    }

    #[test]
    fn test_blend_rewards() {
        let orch = IntegrationOrchestrator::new(2, test_config());
        let task_rewards = vec![1.0, 0.5];
        let blended = orch.blend_rewards(&task_rewards);
        assert_eq!(blended.len(), 2);
    }

    #[test]
    fn test_agent_memory_access() {
        let mut orch = IntegrationOrchestrator::new(2, test_config());
        assert!(orch.agent_memory(0).is_some());
        assert!(orch.agent_memory(99).is_none());
        assert!(orch.agent_memory_mut(0).is_some());
    }
}
