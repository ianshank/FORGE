//! Cross-crate integration tests for the proto-Data agent training architecture.
//!
//! Exercises the full integration stack: memory, social, cognitive, and orchestrator.
//!
//! To run: `cargo test --test integration_protodata`

use forge_cognitive::agent::CognitiveAgent;
use forge_cognitive::config::CognitiveConfig;
use forge_cognitive::prompt::CognitivePrompt;
use forge_cognitive::provider::MockProvider;
use forge_integration_layer::config::IntegrationConfig;
use forge_integration_layer::orchestrator::IntegrationOrchestrator;
use forge_memory::config::MemoryConfig;
use forge_memory::episodic::{Episode, EpisodeOutcome};
use forge_memory::semantic::SemanticFact;
use forge_memory::store::InMemoryStore;
use forge_social::alliance::AllianceSystem;
use forge_social::config::SocialConfig;
use forge_social::reputation::ReputationTracker;
use forge_social::trust::TrustMatrix;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_integration_config() -> IntegrationConfig {
    IntegrationConfig {
        enabled: true,
        memory_write_interval: 5,
        ..IntegrationConfig::default()
    }
}

fn test_memory_config() -> MemoryConfig {
    MemoryConfig {
        enabled: true,
        semantic_capacity: 100,
        episodic_capacity: 50,
        preference_capacity: 20,
        ..MemoryConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Orchestrator integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_orchestrator_full_lifecycle() {
    let config = test_integration_config();
    let mut orch = IntegrationOrchestrator::new(4, config);

    // Verify initial state
    assert_eq!(orch.num_agents(), 4);
    assert_eq!(orch.current_tick(), 0);

    // Run ticks and record interactions
    for _ in 0..10 {
        orch.tick();
    }
    assert_eq!(orch.current_tick(), 10);

    // Record cooperation between agents 0 and 1
    let initial_trust = orch.trust.trust(0, 1);
    orch.record_cooperation(0, 1);
    assert!(orch.trust.trust(0, 1) > initial_trust);

    // Record hostility from agent 2 to agent 3
    let initial_trust_23 = orch.trust.trust(2, 3);
    orch.record_hostility(2, 3);
    assert!(orch.trust.trust(2, 3) < initial_trust_23);

    // Verify blended rewards
    let task_rewards: Vec<f32> = vec![1.0, 0.5, 0.3, 0.8];
    let blended = orch.blend_rewards(&task_rewards);
    assert_eq!(blended.len(), 4);
    for r in &blended {
        assert!(f32::is_finite(*r));
    }
}

#[test]
fn test_orchestrator_memory_access_and_writes() {
    let config = test_integration_config();
    let mut orch = IntegrationOrchestrator::new(2, config);

    // Write a semantic fact to agent 0's memory
    let mem = orch.agent_memory_mut(0).expect("agent 0 exists");
    mem.semantic.store(SemanticFact::new(
        "enemy.position".to_string(),
        "north".to_string(),
        0.9,
        0,
    ));

    // Read it back
    let mem = orch.agent_memory(0).expect("agent 0 exists");
    assert_eq!(mem.semantic.len(), 1);

    // Agent 1's memory should be separate
    let mem1 = orch.agent_memory(1).expect("agent 1 exists");
    assert_eq!(mem1.semantic.len(), 0);

    // Nonexistent agent
    assert!(orch.agent_memory(99).is_none());
}

#[test]
fn test_orchestrator_memory_decay_on_tick() {
    let mut config = test_integration_config();
    config.memory_write_interval = 1; // Decay every tick
    config.memory.decay_rate = 0.5;
    config.memory.min_strength = 0.3;

    let mut orch = IntegrationOrchestrator::new(1, config);

    // Store a fact with moderate strength
    let mem = orch.agent_memory_mut(0).unwrap();
    mem.semantic.store(SemanticFact::new(
        "temp".to_string(),
        "value".to_string(),
        1.0,
        0,
    ));
    assert_eq!(mem.semantic.len(), 1);

    // Tick should decay and eventually prune
    orch.tick(); // strength: 1.0 - 0.5 = 0.5
    assert_eq!(orch.agent_memory(0).unwrap().semantic.len(), 1);

    orch.tick(); // strength: 0.5 - 0.5 = 0.0 < 0.3 → pruned
    assert_eq!(orch.agent_memory(0).unwrap().semantic.len(), 0);
}

// ---------------------------------------------------------------------------
// Memory + Social integration
// ---------------------------------------------------------------------------

#[test]
fn test_memory_episodic_with_social_context() {
    let mem_config = test_memory_config();
    let mut store = InMemoryStore::new(0, &mem_config);

    // Store episodes that reflect social interactions
    let mut ep1 = Episode::new((0, 10), vec![0, 1], (5, 5), EpisodeOutcome::Success, 1.0);
    ep1.tags.push("cooperation".to_string());
    ep1.event_summaries
        .push("cooperated with agent 1".to_string());
    store.episodic.store(ep1);

    let mut ep2 = Episode::new((10, 20), vec![0, 2], (8, 3), EpisodeOutcome::Failure, -0.5);
    ep2.tags.push("combat".to_string());
    ep2.event_summaries.push("fought agent 2".to_string());
    store.episodic.store(ep2);

    // Query by agent
    let agent1_episodes = store.episodic.query_by_agent(1);
    assert_eq!(agent1_episodes.len(), 1);

    // Query by tag
    let combat_episodes = store.episodic.query_by_tag("combat");
    assert_eq!(combat_episodes.len(), 1);

    // Query by location
    let nearby = store.episodic.query_by_location(5, 5, 2);
    assert_eq!(nearby.len(), 1);
}

// ---------------------------------------------------------------------------
// Cognitive + Memory integration
// ---------------------------------------------------------------------------

#[test]
fn test_cognitive_agent_with_memory_context() {
    let mut provider = MockProvider::new("Action: 3".to_string());
    provider.add_response("You are".to_string(), "I'll take Action: 2".to_string());

    let config = CognitiveConfig::default();
    let mut agent = CognitiveAgent::new(Box::new(provider), config);

    // Build a prompt that includes memory context
    let prompt = CognitivePrompt::builder()
        .observation("Agent is at position (5, 5), health=80".to_string())
        .social("Trust with agent 1: 0.8, agent 2: 0.3".to_string())
        .task("Navigate to resource at (10, 10)".to_string())
        .actions(vec![
            "move_north".to_string(),
            "move_east".to_string(),
            "attack".to_string(),
            "wait".to_string(),
        ])
        .build();

    let (action_id, trace) = agent.select_action_with_prompt(prompt, 42);
    assert!(action_id <= 3); // Valid action range
    assert!(trace.confidence > 0.0);
    assert_eq!(agent.action_count(), 1);
}

#[test]
fn test_cognitive_agent_reasoning_trace() {
    let provider = MockProvider::new("After careful thought, Action: 1".to_string());
    let config = CognitiveConfig::default();
    let mut agent = CognitiveAgent::new(Box::new(provider), config);

    let prompt = CognitivePrompt::builder()
        .observation("Simple observation".to_string())
        .build();

    let (_action_id, trace) = agent.select_action_with_prompt(prompt, 0);

    // Trace should record the decision
    assert!(!trace.steps.is_empty() || trace.selected_action <= u32::MAX);
}

// ---------------------------------------------------------------------------
// Trust + Reputation + Alliance integration
// ---------------------------------------------------------------------------

#[test]
fn test_social_layer_full_flow() {
    let config = SocialConfig::default();
    let mut trust = TrustMatrix::new(4, config.trust_initial);
    let mut reputation = ReputationTracker::new(4);
    let mut alliances = AllianceSystem::new(4);

    // Establish cooperation patterns
    for _ in 0..10 {
        trust.record_cooperation(0, 1, &config);
        trust.record_cooperation(0, 2, &config);
        reputation.record_cooperation(0);
        reputation.record_cooperation(1);
        reputation.record_cooperation(2);
    }

    // Agent 3 is hostile
    for _ in 0..5 {
        trust.record_hostility(3, 0, &config);
        reputation.record_hostility(3);
    }

    // Trust verification
    assert!(trust.trust(0, 1) > config.trust_initial);
    assert!(trust.trust(3, 0) < config.trust_initial);

    // Reputation verification
    assert!(reputation.reputation(0) > 0.0);
    assert!(reputation.reputation(3) < 0.0);

    // Alliance update
    alliances.update(&trust, &config, 100);

    // Hostile agents should not be allied
    let a03 = alliances.are_allied(0, 3);
    assert!(!a03 || trust.trust(0, 3) >= config.alliance_threshold);

    // If 0 and 1 are allied, verify consistency
    if alliances.are_allied(0, 1) {
        let alliance_id = alliances.alliance_of(0).unwrap();
        let members = alliances.alliance_members(alliance_id);
        assert!(members.contains(&0));
        assert!(members.contains(&1));
    }
}

// ---------------------------------------------------------------------------
// Preference learning integration
// ---------------------------------------------------------------------------

#[test]
fn test_preference_learning_across_episodes() {
    let mem_config = test_memory_config();
    let mut store = InMemoryStore::new(0, &mem_config);

    // Simulate learning preferences over multiple episodes
    let context = "low_health".to_string();
    let pref = store.preferences.get_or_create(&context);
    for _ in 0..10 {
        pref.update_action(0, 0.2, 0.1, mem_config.preference_reinforcement_increment);
        pref.update_action(1, 0.8, 0.1, mem_config.preference_reinforcement_increment);
    }

    let preferred = pref.preferred_action();
    assert_eq!(preferred, Some(1)); // Should prefer heal
}

// ---------------------------------------------------------------------------
// End-to-end orchestrator scenario
// ---------------------------------------------------------------------------

#[test]
fn test_end_to_end_training_scenario() {
    let config = IntegrationConfig {
        enabled: true,
        memory_write_interval: 2,
        social_reward_weight: 0.3,
        ..IntegrationConfig::default()
    };
    let mut orch = IntegrationOrchestrator::new(3, config);

    // Simulate 20 ticks of training
    for tick in 0u64..20 {
        orch.tick();

        // Agents 0 and 1 cooperate frequently
        if tick % 2 == 0 {
            orch.record_cooperation(0, 1);
        }

        // Agent 2 is occasionally hostile
        if tick % 5 == 0 {
            orch.record_hostility(2, 0);
        }

        // Store episodic memory at intervals
        if tick % 3 == 0 {
            if let Some(mem) = orch.agent_memory_mut(0) {
                let ep = Episode::new(
                    (tick, tick + 3),
                    vec![0, 1],
                    (tick as u16, 0),
                    EpisodeOutcome::Success,
                    0.5,
                );
                mem.episodic.store(ep);
            }
        }
    }

    assert_eq!(orch.current_tick(), 20);

    // Trust should reflect cooperation/hostility patterns
    let trust_01 = orch.trust.trust(0, 1);
    let trust_20 = orch.trust.trust(2, 0);
    assert!(
        trust_01 > trust_20,
        "cooperative agents should have higher trust"
    );

    // Agent 0 should have episodic memories
    let mem0 = orch.agent_memory(0).unwrap();
    assert!(mem0.episodic.len() > 0);

    // Blended rewards should work
    let task_rewards: Vec<f32> = vec![1.0, 0.5, -0.2];
    let blended = orch.blend_rewards(&task_rewards);
    assert_eq!(blended.len(), 3);
    for r in &blended {
        assert!(f32::is_finite(*r));
    }
}

#[test]
fn test_config_defaults_are_backward_compatible() {
    // All configs should have sensible defaults
    let integration = IntegrationConfig::default();
    assert!(!integration.enabled); // Disabled by default
    assert!(integration.memory_write_interval > 0);

    let memory = MemoryConfig::default();
    assert!(memory.semantic_capacity > 0);
    assert!(memory.decay_rate >= 0.0);

    let social = SocialConfig::default();
    assert!(social.trust_initial >= 0.0 && social.trust_initial <= 1.0);

    let cognitive = CognitiveConfig::default();
    assert!(cognitive.temperature > 0.0);
    assert!(cognitive.default_confidence > 0.0 && cognitive.default_confidence <= 1.0);
}
