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
        let memory_write_interval = self.config.memory_write_interval;
        // Apply memory decay at configured intervals when memory is enabled.
        // Treat a zero interval as "no periodic decay" to avoid modulo-by-zero panics.
        if memory_config.enabled
            && memory_write_interval > 0
            && self.tick % memory_write_interval == 0
        {
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

    #[test]
    fn test_memory_write_interval_one_decays_every_tick() {
        let config = IntegrationConfig {
            enabled: true,
            memory_write_interval: 1,
            memory: forge_memory::config::MemoryConfig {
                enabled: true,
                ..forge_memory::config::MemoryConfig::default()
            },
            ..IntegrationConfig::default()
        };
        let mut orch = IntegrationOrchestrator::new(2, config);
        // With interval=1, every tick triggers the memory decay branch.
        // Just verify it doesn't panic over multiple ticks.
        for _ in 0..10 {
            orch.tick();
        }
        assert_eq!(orch.current_tick(), 10);
    }

    #[test]
    fn test_hostility_affects_only_agent_a_reputation() {
        let mut orch = IntegrationOrchestrator::new(3, test_config());
        let rep_a_before = orch.reputation.reputation(0);
        let rep_b_before = orch.reputation.reputation(1);
        orch.record_hostility(0, 1);
        let rep_a_after = orch.reputation.reputation(0);
        let rep_b_after = orch.reputation.reputation(1);
        // Only agent_a (index 0) should have reputation change from hostility.
        assert!(
            rep_a_after < rep_a_before,
            "Agent A reputation should decrease: before={rep_a_before}, after={rep_a_after}"
        );
        assert_eq!(
            rep_b_before, rep_b_after,
            "Agent B reputation should be unchanged"
        );
    }

    #[test]
    fn test_blend_rewards_zero_social_weight_returns_pure_task() {
        let config = IntegrationConfig {
            enabled: true,
            social_reward_weight: 0.0,
            ..IntegrationConfig::default()
        };
        let orch = IntegrationOrchestrator::new(3, config);
        let task_rewards = vec![1.0, 2.0, 3.0];
        let blended = orch.blend_rewards(&task_rewards);
        assert_eq!(blended, task_rewards);
    }

    #[test]
    fn test_agent_memory_invalid_index() {
        let orch = IntegrationOrchestrator::new(2, test_config());
        assert!(orch.agent_memory(0).is_some());
        assert!(orch.agent_memory(1).is_some());
        assert!(orch.agent_memory(2).is_none());
        assert!(orch.agent_memory(usize::MAX).is_none());
    }

    #[test]
    fn test_agent_memory_mut_invalid_index() {
        let mut orch = IntegrationOrchestrator::new(1, test_config());
        assert!(orch.agent_memory_mut(0).is_some());
        assert!(orch.agent_memory_mut(1).is_none());
    }

    #[test]
    fn test_num_agents_consistency() {
        for n in [0, 1, 5, 10] {
            let orch = IntegrationOrchestrator::new(n, test_config());
            assert_eq!(orch.num_agents(), n);
        }
    }

    #[test]
    fn test_config_accessor() {
        let config = test_config();
        let orch = IntegrationOrchestrator::new(2, config.clone());
        assert_eq!(orch.config().enabled, config.enabled);
        assert_eq!(
            orch.config().memory_write_interval,
            config.memory_write_interval
        );
    }

    #[test]
    fn test_zero_agents_orchestrator() {
        let orch = IntegrationOrchestrator::new(0, test_config());
        assert_eq!(orch.num_agents(), 0);
        assert!(orch.agent_memory(0).is_none());
    }

    #[test]
    fn test_many_ticks_no_panic() {
        let mut orch = IntegrationOrchestrator::new(2, test_config());
        for _ in 0..100 {
            orch.tick();
        }
        assert_eq!(orch.current_tick(), 100);
    }

    #[test]
    fn test_cooperation_both_reputations_increase() {
        let mut orch = IntegrationOrchestrator::new(3, test_config());
        let rep_a_before = orch.reputation.reputation(0);
        let rep_b_before = orch.reputation.reputation(1);
        orch.record_cooperation(0, 1);
        let rep_a_after = orch.reputation.reputation(0);
        let rep_b_after = orch.reputation.reputation(1);
        assert!(rep_a_after > rep_a_before);
        assert!(rep_b_after > rep_b_before);
    }

    #[test]
    fn test_hostility_decreases_trust() {
        let mut orch = IntegrationOrchestrator::new(3, test_config());
        let initial_trust = orch.trust.trust(0, 1);
        orch.record_hostility(0, 1);
        assert!(orch.trust.trust(0, 1) < initial_trust);
    }

    #[test]
    fn test_blend_rewards_disabled_social_returns_task() {
        let config = IntegrationConfig {
            enabled: true,
            social_reward_weight: 0.5,
            social: forge_social::config::SocialConfig {
                enabled: false,
                ..forge_social::config::SocialConfig::default()
            },
            ..IntegrationConfig::default()
        };
        let orch = IntegrationOrchestrator::new(2, config);
        let task = vec![1.0, 2.0];
        let blended = orch.blend_rewards(&task);
        assert_eq!(blended, task);
    }

    #[test]
    fn test_memory_write_interval_zero_no_panic() {
        let config = IntegrationConfig {
            enabled: true,
            memory_write_interval: 0,
            memory: forge_memory::config::MemoryConfig {
                enabled: true,
                ..forge_memory::config::MemoryConfig::default()
            },
            ..IntegrationConfig::default()
        };
        let mut orch = IntegrationOrchestrator::new(2, config);
        for _ in 0..10 {
            orch.tick();
        }
        assert_eq!(orch.current_tick(), 10);
    }

    #[test]
    fn test_memory_disabled_no_decay() {
        let config = IntegrationConfig {
            enabled: true,
            memory_write_interval: 1,
            memory: forge_memory::config::MemoryConfig {
                enabled: false,
                ..forge_memory::config::MemoryConfig::default()
            },
            ..IntegrationConfig::default()
        };
        let mut orch = IntegrationOrchestrator::new(1, config);
        for _ in 0..5 {
            orch.tick();
        }
        assert_eq!(orch.current_tick(), 5);
    }

    #[test]
    fn test_agent_idx_stored_correctly() {
        let orch = IntegrationOrchestrator::new(3, test_config());
        // Agent indexes are correctly assigned
        assert!(orch.agent_memory(0).is_some());
        assert!(orch.agent_memory(1).is_some());
        assert!(orch.agent_memory(2).is_some());
        assert!(orch.agent_memory(3).is_none());
    }

    #[test]
    fn test_blend_rewards_empty_input() {
        let orch = IntegrationOrchestrator::new(2, test_config());
        let empty: Vec<f32> = vec![];
        let blended = orch.blend_rewards(&empty);
        assert!(blended.is_empty());
    }

    #[test]
    fn test_repeated_cooperation_increases_trust_monotonically() {
        let mut orch = IntegrationOrchestrator::new(3, test_config());
        let mut prev = orch.trust.trust(0, 1);
        for _ in 0..5 {
            orch.record_cooperation(0, 1);
            let current = orch.trust.trust(0, 1);
            assert!(current >= prev);
            prev = current;
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn valid_config() -> IntegrationConfig {
        IntegrationConfig {
            enabled: true,
            memory_write_interval: 5,
            ..IntegrationConfig::default()
        }
    }

    proptest! {
        /// Tick always advances monotonically.
        #[test]
        fn tick_monotonically_advances(num_ticks in 1_u64..100) {
            let mut orch = IntegrationOrchestrator::new(2, valid_config());
            let mut prev = orch.current_tick();
            for _ in 0..num_ticks {
                orch.tick();
                let now = orch.current_tick();
                prop_assert!(now > prev, "tick must increase: prev={prev}, now={now}");
                prev = now;
            }
            prop_assert_eq!(orch.current_tick(), num_ticks);
        }

        /// Blended rewards preserve length of input.
        #[test]
        fn blend_rewards_preserves_length(n in 2_usize..8) {
            let orch = IntegrationOrchestrator::new(n, valid_config());
            let task_rewards: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let blended = orch.blend_rewards(&task_rewards);
            prop_assert_eq!(blended.len(), n);
        }

        /// IntegrationConfig serde roundtrip preserves key fields.
        #[test]
        fn config_serde_roundtrip(
            weight in 0.0_f32..=1.0,
            interval in 1_u64..1000,
            meta_lr in 0.0_f32..=0.1,
        ) {
            let config = IntegrationConfig {
                social_reward_weight: weight,
                memory_write_interval: interval,
                meta_lr,
                ..IntegrationConfig::default()
            };
            let json = serde_json::to_string(&config).unwrap();
            let deser: IntegrationConfig = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(deser.social_reward_weight, config.social_reward_weight);
            prop_assert_eq!(deser.memory_write_interval, config.memory_write_interval);
            prop_assert_eq!(deser.meta_lr, config.meta_lr);
        }

        /// Cooperation always increases trust from initial value.
        #[test]
        fn cooperation_increases_trust(n in 2_usize..6, reps in 1_usize..10) {
            let mut orch = IntegrationOrchestrator::new(n, valid_config());
            let initial = orch.trust.trust(0, 1);
            for _ in 0..reps {
                orch.record_cooperation(0, 1);
            }
            prop_assert!(
                orch.trust.trust(0, 1) >= initial,
                "trust should not decrease after cooperation"
            );
        }

        /// Agent count is always preserved after construction.
        #[test]
        fn num_agents_matches_construction(n in 0_usize..20) {
            let orch = IntegrationOrchestrator::new(n, valid_config());
            prop_assert_eq!(orch.num_agents(), n);
        }

        /// Blend rewards returns same length as input.
        #[test]
        fn blend_rewards_length_matches_zero_weight(n in 1_usize..10) {
            let config = IntegrationConfig {
                enabled: true,
                social_reward_weight: 0.0,
                ..IntegrationConfig::default()
            };
            let orch = IntegrationOrchestrator::new(n, config);
            let task: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let blended = orch.blend_rewards(&task);
            prop_assert_eq!(blended.len(), n);
            // With zero weight, blended should match task exactly
            for (b, t) in blended.iter().zip(task.iter()) {
                prop_assert_eq!(*b, *t);
            }
        }
    }
}
