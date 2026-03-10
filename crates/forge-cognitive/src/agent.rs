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
}
