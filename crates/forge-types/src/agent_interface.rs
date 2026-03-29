//! Unified agent interface for the FORGE evaluation platform.
//!
//! This module defines the primary trait that all evaluated agents must
//! implement. It operates on [`Observation`] (partial information) rather
//! than full world state, making it suitable for LLM agents, RL policies,
//! human input, and any external agent.
//!
//! # Architecture
//!
//! The [`AgentInterface`] trait unifies three existing agent abstractions:
//! - `forge_agent::baselines::Agent` — takes `&WorldState` (privileged)
//! - `forge_mangomas::batch_runner::ActionPolicy` — takes `&Observation`
//! - `forge_cognitive::agent::CognitiveAgent` — uses LLM providers
//!
//! Adapters in `forge-agent` bridge each of these to `AgentInterface`,
//! preserving full backward compatibility with existing code.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::observation::Observation;

/// Default confidence value for agent responses (maximum confidence).
pub const DEFAULT_CONFIDENCE: f32 = 1.0;

/// Minimum valid confidence value.
pub const MIN_CONFIDENCE: f32 = 0.0;

/// Maximum valid confidence value.
pub const MAX_CONFIDENCE: f32 = 1.0;

/// Agent type string for heuristic/baseline agents.
pub const AGENT_TYPE_HEURISTIC: &str = "heuristic";

/// Agent type string for LLM-backed agents.
pub const AGENT_TYPE_LLM: &str = "llm";

/// Agent type string for reinforcement learning agents.
pub const AGENT_TYPE_RL: &str = "rl";

/// Agent type string for human players.
pub const AGENT_TYPE_HUMAN: &str = "human";

/// Response from an agent's action selection.
///
/// Wraps a discrete action ID with optional metadata useful for
/// evaluation scorecards, logging, and trajectory storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentResponse {
    /// The discrete action ID within the FORGE action space.
    pub action_id: u32,
    /// Optional reasoning trace (populated by LLM agents).
    pub reasoning: Option<String>,
    /// Agent's self-reported confidence in the action (0.0–1.0).
    pub confidence: f32,
    /// Wall-clock time taken to decide, in milliseconds.
    pub decision_time_ms: u64,
}

impl AgentResponse {
    /// Creates a minimal response with just an action ID.
    ///
    /// Confidence defaults to [`DEFAULT_CONFIDENCE`], decision_time_ms to 0.
    pub fn from_action(action_id: u32) -> Self {
        Self {
            action_id,
            reasoning: None,
            confidence: DEFAULT_CONFIDENCE,
            decision_time_ms: 0,
        }
    }

    /// Creates a response with timing measured from the given instant.
    pub fn with_timing(action_id: u32, started: Instant) -> Self {
        Self {
            action_id,
            reasoning: None,
            confidence: DEFAULT_CONFIDENCE,
            decision_time_ms: started.elapsed().as_millis() as u64,
        }
    }

    /// Clamps confidence to the valid range [`MIN_CONFIDENCE`]–[`MAX_CONFIDENCE`].
    ///
    /// NaN values are replaced with 0.0.
    pub fn clamp_confidence(mut self) -> Self {
        if self.confidence.is_nan() {
            self.confidence = MIN_CONFIDENCE;
        } else {
            self.confidence = self.confidence.clamp(MIN_CONFIDENCE, MAX_CONFIDENCE);
        }
        self
    }

    /// Returns true if confidence is within the valid range [0.0, 1.0] and not NaN.
    pub fn is_valid_confidence(&self) -> bool {
        !self.confidence.is_nan()
            && self.confidence >= MIN_CONFIDENCE
            && self.confidence <= MAX_CONFIDENCE
    }
}

impl Default for AgentResponse {
    fn default() -> Self {
        Self::from_action(0) // Noop
    }
}

impl fmt::Display for AgentResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Action({}, conf={:.2}, {}ms)",
            self.action_id, self.confidence, self.decision_time_ms
        )
    }
}

/// Agent metadata for evaluation scorecards and attribution.
///
/// Identifies the type, model, and configuration of an agent under
/// evaluation. Used by `forge-eval` to label scorecard results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentMetadata {
    /// Agent category: `"llm"`, `"rl"`, `"heuristic"`, `"human"`, or custom.
    pub agent_type: String,
    /// Model or checkpoint name (e.g., `"claude-3.5-sonnet"`, `"ppo-500k"`).
    pub model_name: String,
    /// Version string for the agent implementation.
    pub version: String,
    /// Arbitrary key-value parameters for the agent configuration.
    pub parameters: HashMap<String, String>,
}

impl AgentMetadata {
    /// Creates metadata for a heuristic/baseline agent.
    pub fn heuristic(name: &str) -> Self {
        Self {
            agent_type: AGENT_TYPE_HEURISTIC.to_string(),
            model_name: name.to_string(),
            ..Default::default()
        }
    }

    /// Creates metadata for an LLM-backed agent.
    pub fn llm(model_name: &str, version: &str) -> Self {
        Self {
            agent_type: AGENT_TYPE_LLM.to_string(),
            model_name: model_name.to_string(),
            version: version.to_string(),
            ..Default::default()
        }
    }

    /// Creates metadata for an RL policy agent.
    pub fn rl(checkpoint_name: &str) -> Self {
        Self {
            agent_type: AGENT_TYPE_RL.to_string(),
            model_name: checkpoint_name.to_string(),
            ..Default::default()
        }
    }
}

impl fmt::Display for AgentMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.version.is_empty() {
            write!(f, "{}({})", self.agent_type, self.model_name)
        } else {
            write!(
                f,
                "{}({} v{})",
                self.agent_type, self.model_name, self.version
            )
        }
    }
}

/// Unified agent interface for the FORGE evaluation platform.
///
/// This is the primary trait that all evaluated agents must implement.
/// It operates on [`Observation`] (partial information) rather than full
/// world state, making it suitable for LLMs, RL policies, human input,
/// and any external agent.
///
/// # Examples
///
/// ```rust,no_run
/// use forge_types::agent_interface::{AgentInterface, AgentResponse, AgentMetadata};
/// use forge_types::observation::Observation;
///
/// struct MyAgent;
///
/// impl AgentInterface for MyAgent {
///     fn select_action(&mut self, obs: &Observation, agent_idx: usize) -> AgentResponse {
///         AgentResponse::from_action(0) // always noop
///     }
///     fn name(&self) -> &str { "MyAgent" }
/// }
/// ```
pub trait AgentInterface: Send {
    /// Select a discrete action given an observation.
    ///
    /// Called once per tick per agent. Implementations may be synchronous
    /// (heuristics, RL forward pass) or internally block on async work
    /// (LLM API calls).
    fn select_action(&mut self, obs: &Observation, agent_idx: usize) -> AgentResponse;

    /// Returns the agent's name for logging and scorecard attribution.
    fn name(&self) -> &str;

    /// Returns agent metadata for evaluation scorecards.
    ///
    /// Override this to provide model name, version, and parameters.
    /// Default returns empty metadata.
    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::default()
    }

    /// Called at episode start. Agents can reset internal state.
    ///
    /// Default is a no-op. Override for agents that maintain episode-local
    /// state (e.g., memory buffers, conversation history).
    #[instrument(skip_all)]
    fn reset(&mut self) {}
}

/// Blanket implementation: a boxed `AgentInterface` is itself an `AgentInterface`.
impl AgentInterface for Box<dyn AgentInterface> {
    fn select_action(&mut self, obs: &Observation, agent_idx: usize) -> AgentResponse {
        (**self).select_action(obs, agent_idx)
    }

    fn name(&self) -> &str {
        (**self).name()
    }

    fn metadata(&self) -> AgentMetadata {
        (**self).metadata()
    }

    fn reset(&mut self) {
        (**self).reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use crate::observation::{InventoryObservation, TileObservation};

    fn make_test_observation() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation {
                slots: vec![(constants::OBS_EMPTY_SLOT_ITEM, 0)],
            },
            health: 1.0,
            stamina: 1.0,
            position: (5, 5),
            messages: vec![],
            day_phase: 1,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
        }
    }

    /// Minimal agent for testing.
    struct TestAgent {
        action: u32,
        call_count: u32,
    }

    impl TestAgent {
        fn new(action: u32) -> Self {
            Self {
                action,
                call_count: 0,
            }
        }
    }

    impl AgentInterface for TestAgent {
        fn select_action(&mut self, _obs: &Observation, _agent_idx: usize) -> AgentResponse {
            self.call_count += 1;
            AgentResponse::from_action(self.action)
        }

        fn name(&self) -> &str {
            "TestAgent"
        }

        fn metadata(&self) -> AgentMetadata {
            AgentMetadata::heuristic("TestAgent")
        }

        fn reset(&mut self) {
            self.call_count = 0;
        }
    }

    #[test]
    fn test_agent_response_from_action() {
        let resp = AgentResponse::from_action(5);
        assert_eq!(resp.action_id, 5);
        assert_eq!(resp.confidence, 1.0);
        assert_eq!(resp.decision_time_ms, 0);
        assert!(resp.reasoning.is_none());
    }

    #[test]
    fn test_agent_response_with_timing() {
        let started = Instant::now();
        let resp = AgentResponse::with_timing(3, started);
        assert_eq!(resp.action_id, 3);
        // Timing should be >= 0 (we just created it)
        assert!(resp.decision_time_ms < 1000);
    }

    #[test]
    fn test_agent_response_default() {
        let resp = AgentResponse::default();
        assert_eq!(resp.action_id, 0);
    }

    #[test]
    fn test_agent_metadata_heuristic() {
        let meta = AgentMetadata::heuristic("GreedyNav");
        assert_eq!(meta.agent_type, "heuristic");
        assert_eq!(meta.model_name, "GreedyNav");
    }

    #[test]
    fn test_agent_metadata_llm() {
        let meta = AgentMetadata::llm("claude-3.5-sonnet", "1.0");
        assert_eq!(meta.agent_type, "llm");
        assert_eq!(meta.model_name, "claude-3.5-sonnet");
        assert_eq!(meta.version, "1.0");
    }

    #[test]
    fn test_agent_metadata_rl() {
        let meta = AgentMetadata::rl("ppo-500k");
        assert_eq!(meta.agent_type, "rl");
        assert_eq!(meta.model_name, "ppo-500k");
    }

    #[test]
    fn test_agent_metadata_default() {
        let meta = AgentMetadata::default();
        assert!(meta.agent_type.is_empty());
        assert!(meta.model_name.is_empty());
        assert!(meta.parameters.is_empty());
    }

    #[test]
    fn test_agent_interface_select_action() {
        let mut agent = TestAgent::new(7);
        let obs = make_test_observation();
        let resp = agent.select_action(&obs, 0);
        assert_eq!(resp.action_id, 7);
        assert_eq!(agent.call_count, 1);
    }

    #[test]
    fn test_agent_interface_name() {
        let agent = TestAgent::new(0);
        assert_eq!(agent.name(), "TestAgent");
    }

    #[test]
    fn test_agent_interface_metadata() {
        let agent = TestAgent::new(0);
        let meta = agent.metadata();
        assert_eq!(meta.agent_type, "heuristic");
    }

    #[test]
    fn test_agent_interface_reset() {
        let mut agent = TestAgent::new(1);
        let obs = make_test_observation();
        agent.select_action(&obs, 0);
        agent.select_action(&obs, 0);
        assert_eq!(agent.call_count, 2);
        agent.reset();
        assert_eq!(agent.call_count, 0);
    }

    #[test]
    fn test_boxed_agent_interface() {
        let mut agent: Box<dyn AgentInterface> = Box::new(TestAgent::new(3));
        let obs = make_test_observation();
        let resp = agent.select_action(&obs, 0);
        assert_eq!(resp.action_id, 3);
        assert_eq!(agent.name(), "TestAgent");
    }

    #[test]
    fn test_agent_response_serde_roundtrip() {
        let resp = AgentResponse {
            action_id: 42,
            reasoning: Some("I should gather wood".to_string()),
            confidence: 0.85,
            decision_time_ms: 150,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: AgentResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.action_id, 42);
        assert_eq!(
            deserialized.reasoning.as_deref(),
            Some("I should gather wood")
        );
        assert_eq!(deserialized.confidence, 0.85);
        assert_eq!(deserialized.decision_time_ms, 150);
    }

    #[test]
    fn test_agent_metadata_serde_roundtrip() {
        let mut meta = AgentMetadata::llm("claude-3.5-sonnet", "2.0");
        meta.parameters
            .insert("temperature".to_string(), "0.7".to_string());
        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: AgentMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.agent_type, "llm");
        assert_eq!(deserialized.model_name, "claude-3.5-sonnet");
        assert_eq!(deserialized.version, "2.0");
        assert_eq!(deserialized.parameters.get("temperature").unwrap(), "0.7");
    }

    #[test]
    fn test_multiple_agents_independent_state() {
        let mut agent_a = TestAgent::new(1);
        let mut agent_b = TestAgent::new(2);
        let obs = make_test_observation();

        let resp_a = agent_a.select_action(&obs, 0);
        let resp_b = agent_b.select_action(&obs, 1);

        assert_eq!(resp_a.action_id, 1);
        assert_eq!(resp_b.action_id, 2);
        assert_eq!(agent_a.call_count, 1);
        assert_eq!(agent_b.call_count, 1);
    }

    #[test]
    fn test_clamp_confidence_normal() {
        let resp = AgentResponse::from_action(1);
        let clamped = resp.clamp_confidence();
        assert_eq!(clamped.confidence, 1.0);
    }

    #[test]
    fn test_clamp_confidence_out_of_range_high() {
        let mut resp = AgentResponse::from_action(1);
        resp.confidence = 2.5;
        let clamped = resp.clamp_confidence();
        assert_eq!(clamped.confidence, MAX_CONFIDENCE);
    }

    #[test]
    fn test_clamp_confidence_out_of_range_low() {
        let mut resp = AgentResponse::from_action(1);
        resp.confidence = -0.5;
        let clamped = resp.clamp_confidence();
        assert_eq!(clamped.confidence, MIN_CONFIDENCE);
    }

    #[test]
    fn test_clamp_confidence_nan() {
        let mut resp = AgentResponse::from_action(1);
        resp.confidence = f32::NAN;
        assert!(!resp.is_valid_confidence());
        let clamped = resp.clamp_confidence();
        assert_eq!(clamped.confidence, MIN_CONFIDENCE);
        assert!(clamped.is_valid_confidence());
    }

    #[test]
    fn test_is_valid_confidence() {
        let resp = AgentResponse::from_action(1);
        assert!(resp.is_valid_confidence());

        let mut bad = AgentResponse::from_action(1);
        bad.confidence = 1.5;
        assert!(!bad.is_valid_confidence());
    }

    #[test]
    fn test_agent_response_display() {
        let resp = AgentResponse::from_action(5);
        let display = format!("{resp}");
        assert!(display.contains("5"));
        assert!(display.contains("1.00"));
    }

    #[test]
    fn test_agent_metadata_display() {
        let meta = AgentMetadata::llm("claude-3.5", "2.0");
        let display = format!("{meta}");
        assert!(display.contains("llm"));
        assert!(display.contains("claude-3.5"));
        assert!(display.contains("v2.0"));

        let meta2 = AgentMetadata::heuristic("NoopAgent");
        let display2 = format!("{meta2}");
        assert!(display2.contains("heuristic"));
        assert!(!display2.contains(" v")); // no version
    }

    #[test]
    fn test_boxed_agent_metadata_delegation() {
        let mut agent: Box<dyn AgentInterface> = Box::new(TestAgent::new(3));
        let meta = agent.metadata();
        assert_eq!(meta.agent_type, AGENT_TYPE_HEURISTIC);
        assert_eq!(meta.model_name, "TestAgent");

        // Test reset delegation
        let obs = make_test_observation();
        agent.select_action(&obs, 0);
        agent.reset();
        // After reset, call_count should be 0 — verify via another select
        let resp = agent.select_action(&obs, 0);
        assert_eq!(resp.action_id, 3);
    }

    #[test]
    fn test_agent_type_constants() {
        assert_eq!(AGENT_TYPE_HEURISTIC, "heuristic");
        assert_eq!(AGENT_TYPE_LLM, "llm");
        assert_eq!(AGENT_TYPE_RL, "rl");
        assert_eq!(AGENT_TYPE_HUMAN, "human");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// AgentResponse serde roundtrip preserves all fields.
        #[test]
        fn agent_response_serde_roundtrip(
            action_id in 0_u32..1000,
            confidence in 0.0_f32..=1.0,
            decision_time_ms in 0_u64..100_000,
        ) {
            let resp = AgentResponse {
                action_id,
                reasoning: Some(format!("action {action_id}")),
                confidence,
                decision_time_ms,
            };
            let json = serde_json::to_string(&resp).unwrap();
            let deser: AgentResponse = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(deser.action_id, action_id);
            prop_assert_eq!(deser.confidence, confidence);
            prop_assert_eq!(deser.decision_time_ms, decision_time_ms);
        }

        /// AgentMetadata serde roundtrip preserves all fields.
        #[test]
        fn agent_metadata_serde_roundtrip(
            agent_type in "[a-z]{1,10}",
            model_name in "[a-z0-9-]{1,30}",
            version in "[0-9]{1,3}\\.[0-9]{1,3}",
        ) {
            let meta = AgentMetadata {
                agent_type: agent_type.clone(),
                model_name: model_name.clone(),
                version: version.clone(),
                parameters: HashMap::new(),
            };
            let json = serde_json::to_string(&meta).unwrap();
            let deser: AgentMetadata = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(&deser.agent_type, &agent_type);
            prop_assert_eq!(&deser.model_name, &model_name);
            prop_assert_eq!(&deser.version, &version);
        }

        /// from_action always produces confidence 1.0 and timing 0.
        #[test]
        fn from_action_invariants(action_id in 0_u32..10_000) {
            let resp = AgentResponse::from_action(action_id);
            prop_assert_eq!(resp.action_id, action_id);
            prop_assert_eq!(resp.confidence, 1.0);
            prop_assert_eq!(resp.decision_time_ms, 0);
            prop_assert!(resp.reasoning.is_none());
        }
    }
}
