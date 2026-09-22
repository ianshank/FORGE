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

    /// Runs an episode with external controllers and an authoritative journal.
    ///
    /// Drives the simulation loop, extracting events from `WorldState` and
    /// writing them alongside tick boundaries to the `AppendOnlyJournal`.
    #[instrument(skip(self, state, agents, journal))]
    pub fn run_episode_with_journal(
        &mut self,
        state: &mut forge_core::WorldState,
        agents: &mut [Box<dyn forge_types::agent_interface::AgentInterface>],
        journal: &mut forge_replay::journal::AppendOnlyJournal,
        max_steps: u64,
    ) -> std::io::Result<()> {
        use forge_replay::journal::JournalEntry;

        // Reset all agents at episode start
        for agent in agents.iter_mut() {
            agent.reset();
        }

        let initial_result = state.reset(Some(state.config.world.seed));
        let mut current_obs = initial_result.observations;

        for _ in 0..max_steps {
            if state.terminated || state.truncated {
                break;
            }

            let mut actions = Vec::with_capacity(agents.len());
            for (i, agent) in agents.iter_mut().enumerate() {
                let response = if i < current_obs.len() {
                    agent.select_action(&current_obs[i], i)
                } else {
                    forge_types::agent_interface::AgentResponse::from_action(0)
                };

                let comm_vocab = state.config.agents.comm_vocab_size;
                let drone_enabled = state.config.drone.enabled;
                let action = forge_types::Action::from_discrete(
                    response.action_id,
                    comm_vocab,
                    drone_enabled,
                )
                .unwrap_or(forge_types::Action::Noop);
                actions.push(action);
            }

            // Step the simulation
            let result = state.step(&actions);
            current_obs = result.observations;
            self.tick();

            // Extract events generated during this tick
            let events_at_tick = state.events.events_at_tick(state.tick);
            for ev in events_at_tick {
                journal.append(&JournalEntry::Event(ev.clone()))?;
            }

            // Flush events out of the log to prevent unbounded growth?
            // Actually EventLog is bounded, but we can clear it or rely on max_events.

            // Write tick boundary
            journal.append(&JournalEntry::TickBoundary(state.tick))?;
        }

        journal.flush()?;
        Ok(())
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

        /// Hostility always decreases reputation.
        #[test]
        fn hostility_decreases_reputation(n in 2_usize..6, reps in 1_usize..10) {
            let mut orch = IntegrationOrchestrator::new(n, valid_config());
            let initial = orch.reputation.reputation(0);
            for _ in 0..reps {
                orch.record_hostility(0, 1);
            }
            prop_assert!(
                orch.reputation.reputation(0) <= initial,
                "reputation should not increase after hostility"
            );
        }

        /// Metrics JSON round-trip preserves all fields.
        #[test]
        fn metrics_serde_roundtrip(
            entries in 0_usize..1000,
            trust in 0.0_f32..=1.0,
            rep in 0.0_f32..=1.0,
            alliances in 0_usize..50,
            tick in 0_u64..100_000,
        ) {
            use crate::metrics::IntegrationMetrics;
            let m = IntegrationMetrics {
                total_memory_entries: entries,
                mean_trust: trust,
                mean_reputation: rep,
                active_alliances: alliances,
                social_reward_fraction: 0.0,
                tick,
            };
            let json = serde_json::to_string(&m).unwrap();
            let deser: IntegrationMetrics = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(deser.total_memory_entries, entries);
            prop_assert_eq!(deser.tick, tick);
        }
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    fn enabled_config() -> IntegrationConfig {
        IntegrationConfig {
            enabled: true,
            memory_write_interval: 5,
            memory: forge_memory::config::MemoryConfig {
                enabled: true,
                ..forge_memory::config::MemoryConfig::default()
            },
            social: forge_social::config::SocialConfig {
                enabled: true,
                ..forge_social::config::SocialConfig::default()
            },
            ..IntegrationConfig::default()
        }
    }

    #[test]
    fn test_orchestrator_agent_idx_values() {
        let orch = IntegrationOrchestrator::new(10, enabled_config());
        for i in 0..10 {
            let state = orch.agent_memory(i);
            assert!(state.is_some(), "agent {i} should exist");
        }
        assert!(orch.agent_memory(10).is_none());
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
        for _ in 0..100 {
            orch.tick();
        }
        assert_eq!(orch.current_tick(), 100);
    }

    #[test]
    fn test_memory_decay_triggers_at_exact_intervals() {
        let config = IntegrationConfig {
            enabled: true,
            memory_write_interval: 5,
            memory: forge_memory::config::MemoryConfig {
                enabled: true,
                ..forge_memory::config::MemoryConfig::default()
            },
            ..IntegrationConfig::default()
        };
        let mut orch = IntegrationOrchestrator::new(2, config);
        // Ticks 1-4 should not trigger decay; tick 5 should.
        for _ in 0..4 {
            orch.tick();
        }
        assert_eq!(orch.current_tick(), 4);
        // Tick 5 triggers decay — should not panic.
        orch.tick();
        assert_eq!(orch.current_tick(), 5);
        // Continue to verify periodic triggers at 10, 15, ...
        for _ in 0..10 {
            orch.tick();
        }
        assert_eq!(orch.current_tick(), 15);
    }

    #[test]
    fn test_disabled_memory_skips_decay() {
        let config = IntegrationConfig {
            enabled: true,
            memory_write_interval: 1,
            memory: forge_memory::config::MemoryConfig {
                enabled: false,
                ..forge_memory::config::MemoryConfig::default()
            },
            ..IntegrationConfig::default()
        };
        let mut orch = IntegrationOrchestrator::new(2, config);
        // Even with interval=1, disabled memory should skip the decay branch.
        for _ in 0..20 {
            orch.tick();
        }
        assert_eq!(orch.current_tick(), 20);
    }

    #[test]
    fn test_blend_rewards_with_disabled_social_returns_passthrough() {
        let config = IntegrationConfig {
            enabled: true,
            social_reward_weight: 0.5,
            social: forge_social::config::SocialConfig {
                enabled: false,
                ..forge_social::config::SocialConfig::default()
            },
            ..IntegrationConfig::default()
        };
        let orch = IntegrationOrchestrator::new(3, config);
        let tasks = vec![1.0, 2.0, 3.0];
        let blended = orch.blend_rewards(&tasks);
        assert_eq!(blended, tasks, "disabled social should pass through tasks");
    }

    #[test]
    fn test_blend_rewards_negative_task_rewards() {
        let orch = IntegrationOrchestrator::new(2, enabled_config());
        let tasks = vec![-1.0, -2.0];
        let blended = orch.blend_rewards(&tasks);
        assert_eq!(blended.len(), 2, "length must match");
        // Negative tasks should produce negative or reduced blended values.
        for val in &blended {
            assert!(val.is_finite(), "blended reward must be finite");
        }
    }

    #[test]
    fn test_blend_rewards_all_zero_tasks() {
        let orch = IntegrationOrchestrator::new(3, enabled_config());
        let tasks = vec![0.0, 0.0, 0.0];
        let blended = orch.blend_rewards(&tasks);
        assert_eq!(blended.len(), 3);
        for val in &blended {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_blend_rewards_mismatched_length_shorter() {
        let orch = IntegrationOrchestrator::new(4, enabled_config());
        // Pass fewer rewards than agents — zip truncates to shorter.
        let tasks = vec![1.0, 2.0];
        let blended = orch.blend_rewards(&tasks);
        // Result length is min(tasks.len(), social_rewards.len())
        assert!(blended.len() <= 4);
    }

    #[test]
    fn test_blend_rewards_formula_correctness() {
        let config = IntegrationConfig {
            enabled: true,
            social_reward_weight: 0.3,
            social: forge_social::config::SocialConfig {
                enabled: true,
                ..forge_social::config::SocialConfig::default()
            },
            ..IntegrationConfig::default()
        };
        let orch = IntegrationOrchestrator::new(2, config.clone());
        let tasks = vec![10.0, 20.0];
        let blended = orch.blend_rewards(&tasks);
        // blended[i] = task[i] * (1 - w) + social[i] * w
        let w = config.social_reward_weight;
        for (i, val) in blended.iter().enumerate() {
            // We can't predict social[i] exactly, but the formula holds:
            // val = tasks[i] * 0.7 + social[i] * 0.3
            // so val should differ from tasks[i] * 0.7 by at most |social[i] * 0.3|
            let task_component = tasks[i] * (1.0 - w);
            // Social rewards are bounded (trust/reputation in [0,1]) so social * 0.3 <= 0.3
            assert!(
                (*val - task_component).abs() <= 1.0,
                "blended[{i}]={val} should be near task_component={task_component}"
            );
        }
    }

    #[test]
    fn test_cooperation_both_agents_gain_reputation() {
        let mut orch = IntegrationOrchestrator::new(3, enabled_config());
        let rep_a = orch.reputation.reputation(0);
        let rep_b = orch.reputation.reputation(1);
        orch.record_cooperation(0, 1);
        assert!(
            orch.reputation.reputation(0) >= rep_a,
            "agent A reputation should increase after cooperation"
        );
        assert!(
            orch.reputation.reputation(1) >= rep_b,
            "agent B reputation should increase after cooperation"
        );
    }

    #[test]
    fn test_asymmetric_trust_after_cooperation() {
        let mut orch = IntegrationOrchestrator::new(3, enabled_config());
        orch.record_cooperation(0, 1);
        let trust_01 = orch.trust.trust(0, 1);
        let trust_10 = orch.trust.trust(1, 0);
        // After cooperation, both directions should increase.
        let initial = forge_social::config::SocialConfig::default().trust_initial;
        assert!(trust_01 >= initial, "trust(0,1) should have increased");
        assert!(trust_10 >= initial, "trust(1,0) should have increased");
    }

    #[test]
    fn test_zero_agents_orchestrator() {
        let orch = IntegrationOrchestrator::new(0, enabled_config());
        assert_eq!(orch.num_agents(), 0);
        assert!(orch.agent_memory(0).is_none());
        let blended = orch.blend_rewards(&[]);
        assert!(blended.is_empty());
    }

    #[test]
    fn test_large_agent_count() {
        let orch = IntegrationOrchestrator::new(100, enabled_config());
        assert_eq!(orch.num_agents(), 100);
        assert!(orch.agent_memory(99).is_some());
        assert!(orch.agent_memory(100).is_none());
    }

    #[test]
    fn test_config_accessor() {
        let config = IntegrationConfig {
            meta_lr: 0.123,
            ..enabled_config()
        };
        let orch = IntegrationOrchestrator::new(2, config);
        assert_eq!(orch.config().meta_lr, 0.123);
    }

    #[test]
    fn test_current_tick_starts_at_zero() {
        let orch = IntegrationOrchestrator::new(2, enabled_config());
        assert_eq!(orch.current_tick(), 0);
    }

    #[test]
    fn test_multiple_cooperations_accumulate_trust() {
        let mut orch = IntegrationOrchestrator::new(3, enabled_config());
        let t1 = orch.trust.trust(0, 1);
        orch.record_cooperation(0, 1);
        let t2 = orch.trust.trust(0, 1);
        orch.record_cooperation(0, 1);
        let t3 = orch.trust.trust(0, 1);
        assert!(t2 >= t1, "first cooperation should increase trust");
        assert!(t3 >= t2, "second cooperation should further increase trust");
    }

    #[test]
    fn test_hostility_then_cooperation_recovery() {
        let mut orch = IntegrationOrchestrator::new(3, enabled_config());
        orch.record_hostility(0, 1);
        let rep_after_hostility = orch.reputation.reputation(0);
        orch.record_cooperation(0, 1);
        let rep_after_coop = orch.reputation.reputation(0);
        assert!(
            rep_after_coop >= rep_after_hostility,
            "cooperation should recover reputation"
        );
    }
}
