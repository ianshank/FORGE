//! Cognitive agent: LLM-backed agent that reasons about the world.
//!
//! The [`CognitiveAgent`] uses a [`CognitiveProvider`](crate::provider::CognitiveProvider)
//! to generate actions through structured reasoning, integrating memory
//! and social context into each decision.

use tracing::{instrument, warn};

use crate::config::CognitiveConfig;
use crate::prompt::CognitivePrompt;
use crate::provider::{CognitiveProvider, CompletionConfig};
use crate::reasoning::{ReasoningStep, ReasoningTrace, ReasoningType};

/// An agent that uses an LLM to reason about actions.
pub struct CognitiveAgent {
    /// The cognitive provider (LLM backend).
    provider: Box<dyn CognitiveProvider>,
    /// Configuration.
    config: CognitiveConfig,
    /// Trace of the most recent reasoning process.
    last_trace: ReasoningTrace,
    /// Total number of actions taken.
    action_count: u64,
}

impl CognitiveAgent {
    /// Creates a new cognitive agent with the given provider and config.
    #[instrument(skip_all)]
    pub fn new(provider: Box<dyn CognitiveProvider>, config: CognitiveConfig) -> Self {
        Self {
            provider,
            config,
            last_trace: ReasoningTrace::new(),
            action_count: 0,
        }
    }

    /// Selects an action by constructing a prompt and querying the provider.
    ///
    /// Returns the selected action ID and the reasoning trace.
    #[instrument(skip_all)]
    pub fn select_action_with_prompt(
        &mut self,
        prompt: CognitivePrompt,
        tick: u64,
    ) -> (u32, ReasoningTrace) {
        let mut trace = ReasoningTrace::new();

        // Step 1: Observe
        trace.add_step(ReasoningStep::new(
            ReasoningType::Observe,
            prompt.observation.clone(),
            tick,
        ));

        // Step 2: Remember (memory context)
        if !prompt.memory_context.is_empty() {
            trace.add_step(ReasoningStep::new(
                ReasoningType::Remember,
                prompt.memory_context.join("; "),
                tick,
            ));
        }

        // Step 3: Think (query the provider)
        let rendered = prompt.render();
        let completion_config = CompletionConfig {
            model: self.config.model.clone(),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        let response = self.provider.complete(&rendered, &completion_config);

        let (action_id, reasoning_text) = match response {
            Ok(resp) => {
                let action_id = parse_action_id(&resp.text).unwrap_or(0);
                (action_id, resp.text)
            }
            Err(e) => {
                warn!(error = %e, "cognitive provider failed, defaulting to noop");
                (0, format!("Error: {e}"))
            }
        };

        trace.add_step(ReasoningStep::new(
            ReasoningType::Think,
            reasoning_text,
            tick,
        ));

        trace.selected_action = action_id;
        trace.confidence = self.config.default_confidence;
        self.last_trace = trace.clone();
        self.action_count += 1;

        (action_id, trace)
    }

    /// Returns the most recent reasoning trace.
    pub fn last_trace(&self) -> &ReasoningTrace {
        &self.last_trace
    }

    /// Returns the total number of actions taken.
    pub fn action_count(&self) -> u64 {
        self.action_count
    }

    /// Returns the provider name.
    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }
}

/// Parses an action ID from the provider's response text.
///
/// Looks for patterns like "action: 3" or "Action ID: 3" or just a number.
fn parse_action_id(text: &str) -> Option<u32> {
    // Try to find an "action" token (case-insensitive) followed by a number.
    //
    // We avoid using indices from a lowercased copy of the string to slice
    // the original, since `to_lowercase` can change string length for
    // non-ASCII text. Instead, we scan tokens directly.
    let mut tokens = text.split_whitespace().peekable();

    while let Some(token) = tokens.next() {
        // Normalize alphabetic part of the token for comparison.
        let lower = token.to_lowercase();
        let alpha_core = lower.trim_matches(|c: char| !c.is_ascii_alphabetic());

        if alpha_core == "action" {
            // Look ahead up to two tokens for a number (handles "Action: 3"
            // and "action id: 0").
            for _ in 0..2 {
                if let Some(next_tok) = tokens.next() {
                    let cleaned = next_tok.trim_matches(|c: char| !c.is_ascii_digit());
                    if let Ok(id) = cleaned.parse::<u32>() {
                        return Some(id);
                    }
                } else {
                    break;
                }
            }
        }
    }

    // Fallback: find the last number in the text.
    text.split_whitespace().rev().find_map(|w| {
        w.trim_matches(|c: char| !c.is_ascii_digit())
            .parse::<u32>()
            .ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    #[test]
    fn test_cognitive_agent_basic() {
        let provider = MockProvider::new("I think I should move up. Action: 1".into());
        let config = CognitiveConfig::default();
        let mut agent = CognitiveAgent::new(Box::new(provider), config);

        let prompt = CognitivePrompt::builder()
            .observation("I see a forest.".into())
            .actions(vec!["Noop".into(), "MoveUp".into()])
            .build();

        let (action, trace) = agent.select_action_with_prompt(prompt, 100);
        assert_eq!(action, 1);
        assert!(!trace.steps.is_empty());
        assert_eq!(agent.action_count(), 1);
    }

    #[test]
    fn test_parse_action_id() {
        assert_eq!(parse_action_id("Action: 3"), Some(3));
        assert_eq!(parse_action_id("I choose action 5"), Some(5));
        assert_eq!(parse_action_id("noop, action id: 0"), Some(0));
        assert_eq!(parse_action_id("just some text 7"), Some(7));
    }

    #[test]
    fn test_provider_failure_defaults_to_noop() {
        // MockProvider always succeeds, but we can verify the fallback path
        // by checking that action 0 is the default
        let provider = MockProvider::new("no action mentioned here".into());
        let config = CognitiveConfig::default();
        let mut agent = CognitiveAgent::new(Box::new(provider), config);

        let prompt = CognitivePrompt::builder()
            .observation("test".into())
            .build();

        let (action, _) = agent.select_action_with_prompt(prompt, 0);
        // "no action mentioned here" has no number → fallback
        // Actually parse_action_id returns None → unwrap_or(0)
        assert_eq!(action, 0);
    }

    #[test]
    fn test_agent_action_count_increments() {
        let provider = MockProvider::new("action: 2".into());
        let config = CognitiveConfig::default();
        let mut agent = CognitiveAgent::new(Box::new(provider), config);

        assert_eq!(agent.action_count(), 0);

        for i in 1u64..=3 {
            let prompt = CognitivePrompt::builder()
                .observation("test".into())
                .build();
            agent.select_action_with_prompt(prompt, i);
            assert_eq!(agent.action_count(), i);
        }
    }

    #[test]
    fn test_agent_last_trace_updated() {
        let provider = MockProvider::new("action: 5".into());
        let config = CognitiveConfig::default();
        let mut agent = CognitiveAgent::new(Box::new(provider), config);

        let prompt = CognitivePrompt::builder()
            .observation("I see enemies".into())
            .build();
        let (action, _) = agent.select_action_with_prompt(prompt, 42);

        assert_eq!(action, 5);
        let last = agent.last_trace();
        assert_eq!(last.selected_action, 5);
        assert!(!last.steps.is_empty());
    }

    #[test]
    fn test_agent_provider_name() {
        let provider = MockProvider::new("action: 0".into());
        let config = CognitiveConfig::default();
        let agent = CognitiveAgent::new(Box::new(provider), config);
        assert_eq!(agent.provider_name(), "mock");
    }

    #[test]
    fn test_agent_with_memory_context() {
        let provider = MockProvider::new("I remember the forest. action: 3".into());
        let config = CognitiveConfig::default();
        let mut agent = CognitiveAgent::new(Box::new(provider), config);

        let mut prompt = CognitivePrompt::builder()
            .observation("forest ahead".into())
            .build();
        prompt.memory_context = vec!["wood is useful".into(), "built shelter before".into()];

        let (action, trace) = agent.select_action_with_prompt(prompt, 10);
        assert_eq!(action, 3);
        // Should have Observe, Remember, and Think steps
        assert!(trace.steps.len() >= 3);
        assert_eq!(trace.steps[0].step_type, ReasoningType::Observe);
        assert_eq!(trace.steps[1].step_type, ReasoningType::Remember);
        assert_eq!(trace.steps[2].step_type, ReasoningType::Think);
    }

    #[test]
    fn test_parse_action_id_edge_cases() {
        // Empty string
        assert_eq!(parse_action_id(""), None);
        // Just whitespace
        assert_eq!(parse_action_id("   "), None);
        // Multiple numbers - action pattern takes priority
        assert_eq!(parse_action_id("I have 3 options, action: 7"), Some(7));
        // Case insensitive
        assert_eq!(parse_action_id("ACTION: 2"), Some(2));
        assert_eq!(parse_action_id("Action 4"), Some(4));
    }

    #[test]
    fn test_parse_action_id_bare_word_action() {
        // "action" alone with no following number — no digits at all → None
        assert_eq!(parse_action_id("action"), None);
    }

    #[test]
    fn test_parse_action_id_negative_number() {
        // "action -1" — the dash is stripped by trim_matches, leaving "1"
        assert_eq!(parse_action_id("action -1"), Some(1));
    }

    #[test]
    fn test_parse_action_id_large_number() {
        assert_eq!(parse_action_id("action 999999"), Some(999999));
    }

    /// A provider that always returns an error.
    struct ErrorProvider;

    impl CognitiveProvider for ErrorProvider {
        fn name(&self) -> &str {
            "error"
        }

        fn complete(
            &self,
            _prompt: &str,
            _config: &CompletionConfig,
        ) -> Result<crate::provider::CompletionResponse, String> {
            Err("simulated LLM failure".to_string())
        }
    }

    #[test]
    fn test_error_response_from_provider_defaults_to_noop() {
        let config = CognitiveConfig::default();
        let mut agent = CognitiveAgent::new(Box::new(ErrorProvider), config);

        let prompt = CognitivePrompt::builder()
            .observation("I see a wall.".into())
            .actions(vec!["Noop".into(), "MoveUp".into()])
            .build();

        let (action, trace) = agent.select_action_with_prompt(prompt, 5);
        // Provider error → default action 0
        assert_eq!(action, 0);
        assert_eq!(trace.selected_action, 0);
        // The Think step content should contain the error message
        let think_step = trace.steps.iter().find(|s| s.step_type == ReasoningType::Think).unwrap();
        assert!(think_step.content.contains("simulated LLM failure"));
    }

    #[test]
    fn test_empty_prompt_no_observation_no_memory_no_social() {
        let provider = MockProvider::new("action: 0".into());
        let config = CognitiveConfig::default();
        let mut agent = CognitiveAgent::new(Box::new(provider), config);

        let prompt = CognitivePrompt::builder().build();

        let (action, trace) = agent.select_action_with_prompt(prompt, 0);
        assert_eq!(action, 0);
        // Should have Observe + Think (no Remember since memory_context is empty)
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.steps[0].step_type, ReasoningType::Observe);
        assert_eq!(trace.steps[1].step_type, ReasoningType::Think);
        // Observe content is empty string
        assert!(trace.steps[0].content.is_empty());
    }

    #[test]
    fn test_multiple_sequential_action_selections() {
        let provider = MockProvider::new("action: 1".into());
        let config = CognitiveConfig::default();
        let mut agent = CognitiveAgent::new(Box::new(provider), config);

        for i in 1u64..=5 {
            let prompt = CognitivePrompt::builder()
                .observation(format!("tick {i}"))
                .build();
            let (action, _) = agent.select_action_with_prompt(prompt, i);
            assert_eq!(action, 1);
            assert_eq!(agent.action_count(), i);
        }
        // After 5 calls, last_trace should correspond to the final call
        assert_eq!(agent.last_trace().selected_action, 1);
        assert_eq!(agent.action_count(), 5);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::provider::MockProvider;
    use proptest::prelude::*;

    proptest! {
        /// Any text containing "action N" should parse to Some(N).
        #[test]
        fn parse_action_id_finds_explicit_action(id in 0_u32..1000) {
            let text = format!("I think the best move is action {id}.");
            let parsed = parse_action_id(&text);
            prop_assert_eq!(parsed, Some(id));
        }

        /// Fallback: any text ending with a bare number should parse to that number.
        #[test]
        fn parse_action_id_fallback_last_number(id in 0_u32..1000) {
            let text = format!("reasoning complete {id}");
            let parsed = parse_action_id(&text);
            prop_assert_eq!(parsed, Some(id));
        }

        /// Text with no digits returns None.
        #[test]
        fn parse_action_id_no_digits(text in "[a-zA-Z ]{0,100}") {
            let has_digit = text.chars().any(|c| c.is_ascii_digit());
            if !has_digit {
                prop_assert_eq!(parse_action_id(&text), None);
            }
        }

        /// Agent trace always has at least one step (Observe) and uses config confidence.
        #[test]
        fn agent_trace_invariants(
            confidence in 0.0_f32..=1.0,
            action_id in 0_u32..10,
            tick in 0_u64..10_000,
        ) {
            let response_text = format!("action: {action_id}");
            let provider = MockProvider::new(response_text);
            let config = CognitiveConfig {
                default_confidence: confidence,
                ..CognitiveConfig::default()
            };
            let mut agent = CognitiveAgent::new(Box::new(provider), config);

            let prompt = CognitivePrompt::builder()
                .observation("test observation".into())
                .build();

            let (parsed_action, trace) = agent.select_action_with_prompt(prompt, tick);

            // Trace always has at least Observe + Think steps
            prop_assert!(trace.steps.len() >= 2);
            // First step is always Observe
            prop_assert_eq!(trace.steps[0].step_type.clone(), ReasoningType::Observe);
            // Confidence matches config
            prop_assert_eq!(trace.confidence, confidence);
            // Action count incremented
            prop_assert_eq!(agent.action_count(), 1);
            // Parsed action matches
            prop_assert_eq!(parsed_action, action_id);
        }

        /// CognitiveConfig serde roundtrip preserves all fields.
        #[test]
        fn config_serde_roundtrip(
            temperature in 0.0_f32..=2.0,
            max_tokens in 1_u32..4096,
            confidence in 0.0_f32..=1.0,
        ) {
            let config = CognitiveConfig {
                temperature,
                max_tokens,
                default_confidence: confidence,
                ..CognitiveConfig::default()
            };
            let json = serde_json::to_string(&config).unwrap();
            let deser: CognitiveConfig = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(deser.temperature, config.temperature);
            prop_assert_eq!(deser.max_tokens, config.max_tokens);
            prop_assert_eq!(deser.default_confidence, config.default_confidence);
        }
    }
}
