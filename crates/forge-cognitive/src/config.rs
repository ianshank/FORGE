//! Configuration for the cognitive agent system.

use serde::{Deserialize, Serialize};

/// Default number of reasoning steps per action.
const DEFAULT_REASONING_STEPS: u32 = 5;
/// Default temperature for LLM sampling.
const DEFAULT_TEMPERATURE: f32 = 0.7;
/// Default maximum tokens per LLM completion.
const DEFAULT_MAX_TOKENS: u32 = 1024;
/// Default confidence assigned when the provider doesn't return one.
const DEFAULT_CONFIDENCE: f32 = 0.8;
/// Default system prompt for the cognitive agent.
pub(crate) const DEFAULT_SYSTEM_PROMPT: &str =
    "You are an intelligent agent in a grid-based simulation. \
     Reason step by step, then select an action.";

/// Configuration for the cognitive agent system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CognitiveConfig {
    /// Whether the cognitive system is enabled.
    pub enabled: bool,
    /// Provider name (e.g., "mock", "anthropic", "openai", "local").
    pub provider: String,
    /// Model identifier.
    pub model: String,
    /// LLM sampling temperature.
    pub temperature: f32,
    /// Maximum tokens per completion.
    pub max_tokens: u32,
    /// Number of reasoning steps per action selection.
    pub reasoning_steps: u32,
    /// Base URL for the API endpoint (empty = use provider default).
    pub api_base_url: String,
    /// Default confidence when the provider doesn't return one.
    pub default_confidence: f32,
    /// System prompt template for the cognitive agent.
    pub system_prompt: String,
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: "mock".to_string(),
            model: String::new(),
            temperature: DEFAULT_TEMPERATURE,
            max_tokens: DEFAULT_MAX_TOKENS,
            reasoning_steps: DEFAULT_REASONING_STEPS,
            api_base_url: String::new(),
            default_confidence: DEFAULT_CONFIDENCE,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CognitiveConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.provider, "mock");
        assert_eq!(config.reasoning_steps, DEFAULT_REASONING_STEPS);
        assert_eq!(config.default_confidence, DEFAULT_CONFIDENCE);
        assert!(!config.system_prompt.is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = CognitiveConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deser: CognitiveConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.provider, config.provider);
        assert_eq!(deser.temperature, config.temperature);
        assert_eq!(deser.default_confidence, config.default_confidence);
    }
}
