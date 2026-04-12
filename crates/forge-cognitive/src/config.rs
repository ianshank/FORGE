//! Configuration for the cognitive agent system.

use forge_types::constants;
use serde::{Deserialize, Serialize};

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
            temperature: constants::DEFAULT_COGNITIVE_TEMPERATURE,
            max_tokens: constants::DEFAULT_COGNITIVE_MAX_TOKENS,
            reasoning_steps: constants::DEFAULT_COGNITIVE_REASONING_STEPS,
            api_base_url: String::new(),
            default_confidence: constants::DEFAULT_COGNITIVE_CONFIDENCE,
            system_prompt: constants::DEFAULT_COGNITIVE_SYSTEM_PROMPT.to_string(),
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
        assert_eq!(
            config.reasoning_steps,
            constants::DEFAULT_COGNITIVE_REASONING_STEPS
        );
        assert_eq!(
            config.default_confidence,
            constants::DEFAULT_COGNITIVE_CONFIDENCE
        );
        assert!(!config.system_prompt.is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(CognitiveConfig);
    }

    #[test]
    fn test_defaults_valid() {
        forge_types::assert_config_defaults_valid!(CognitiveConfig);
    }
}
