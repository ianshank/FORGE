//! Provider-agnostic cognitive provider trait and implementations.
//!
//! The [`CognitiveProvider`] trait abstracts over different LLM backends,
//! enabling the cognitive agent to work with any completion API.
//! [`MockProvider`] provides deterministic responses for testing.

use std::collections::HashMap;

use tracing::instrument;

/// Configuration for a completion request.
#[derive(Debug, Clone)]
pub struct CompletionConfig {
    /// Model to use.
    pub model: String,
    /// Sampling temperature.
    pub temperature: f32,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
}

/// Response from a cognitive provider.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The generated text.
    pub text: String,
    /// Number of input tokens consumed.
    pub input_tokens: u32,
    /// Number of output tokens generated.
    pub output_tokens: u32,
}

/// Trait for LLM backends that can generate completions.
///
/// Implementations include mock (for testing), and real providers
/// (Anthropic, OpenAI, local) implemented in the Python layer.
pub trait CognitiveProvider: Send + Sync {
    /// Returns the provider name.
    fn name(&self) -> &str;

    /// Generates a completion for the given prompt.
    ///
    /// # Errors
    ///
    /// Returns a string error if the completion fails.
    fn complete(
        &self,
        prompt: &str,
        config: &CompletionConfig,
    ) -> Result<CompletionResponse, String>;
}

/// Mock provider that returns deterministic responses for testing.
///
/// Responses are looked up by prompt prefix, with a fallback default response.
#[derive(Debug)]
pub struct MockProvider {
    /// Map of prompt prefixes to responses.
    responses: HashMap<String, String>,
    /// Default response when no prefix matches.
    default_response: String,
}

impl MockProvider {
    /// Creates a new mock provider with a default response.
    #[instrument(skip_all)]
    pub fn new(default_response: String) -> Self {
        Self {
            responses: HashMap::new(),
            default_response,
        }
    }

    /// Adds a response for prompts starting with the given prefix.
    #[instrument(skip_all)]
    pub fn add_response(&mut self, prefix: String, response: String) {
        self.responses.insert(prefix, response);
    }
}

impl CognitiveProvider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    #[instrument(skip_all)]
    fn complete(
        &self,
        prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<CompletionResponse, String> {
        let response_text = self
            .responses
            .iter()
            .find(|(prefix, _)| prompt.starts_with(prefix.as_str()))
            .map(|(_, response)| response.clone())
            .unwrap_or_else(|| self.default_response.clone());

        Ok(CompletionResponse {
            text: response_text,
            input_tokens: prompt.len() as u32,
            output_tokens: 10,
        })
    }
}

/// Registry of cognitive providers, selected by name at runtime.
#[derive(Default)]
pub struct ProviderRegistry {
    /// Registered providers.
    providers: HashMap<String, Box<dyn CognitiveProvider>>,
}

impl ProviderRegistry {
    /// Creates a new empty registry.
    #[instrument(skip_all)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a provider under the given name.
    #[instrument(skip(self, provider))]
    pub fn register(&mut self, name: String, provider: Box<dyn CognitiveProvider>) {
        self.providers.insert(name, provider);
    }

    /// Gets a provider by name.
    #[instrument(skip_all)]
    pub fn get(&self, name: &str) -> Option<&dyn CognitiveProvider> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    /// Returns the names of all registered providers.
    #[instrument(skip_all)]
    pub fn available(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_provider_default_response() {
        let provider = MockProvider::new("default action: noop".into());
        let config = CompletionConfig {
            model: "test".into(),
            temperature: 0.0,
            max_tokens: 100,
        };
        let response = provider.complete("some prompt", &config).unwrap();
        assert_eq!(response.text, "default action: noop");
    }

    #[test]
    fn test_mock_provider_prefix_matching() {
        let mut provider = MockProvider::new("default".into());
        provider.add_response("observe:".into(), "I see resources nearby".into());
        let config = CompletionConfig {
            model: "test".into(),
            temperature: 0.0,
            max_tokens: 100,
        };

        let r1 = provider.complete("observe: grid view", &config).unwrap();
        assert_eq!(r1.text, "I see resources nearby");

        let r2 = provider.complete("act: move", &config).unwrap();
        assert_eq!(r2.text, "default");
    }

    #[test]
    fn test_provider_registry() {
        let mut registry = ProviderRegistry::new();
        registry.register("mock".into(), Box::new(MockProvider::new("ok".into())));
        assert!(registry.get("mock").is_some());
        assert!(registry.get("anthropic").is_none());
        assert_eq!(registry.available().len(), 1);
    }

    #[test]
    fn test_registry_duplicate_names_overwrites() {
        let mut registry = ProviderRegistry::new();
        registry.register("mock".into(), Box::new(MockProvider::new("first".into())));
        registry.register("mock".into(), Box::new(MockProvider::new("second".into())));

        // Should still have only one provider
        assert_eq!(registry.available().len(), 1);

        // The second registration should overwrite the first
        let config = CompletionConfig {
            model: "test".into(),
            temperature: 0.0,
            max_tokens: 100,
        };
        let provider = registry.get("mock").unwrap();
        let response = provider.complete("anything", &config).unwrap();
        assert_eq!(response.text, "second");
    }

    #[test]
    fn test_mock_provider_no_matching_prefix_falls_back() {
        let mut provider = MockProvider::new("fallback response".into());
        provider.add_response("alpha:".into(), "alpha hit".into());
        provider.add_response("beta:".into(), "beta hit".into());

        let config = CompletionConfig {
            model: "test".into(),
            temperature: 0.0,
            max_tokens: 100,
        };

        // Prompt does not start with any registered prefix
        let response = provider.complete("gamma: something", &config).unwrap();
        assert_eq!(response.text, "fallback response");
    }

    #[test]
    fn test_mock_provider_empty_default_response() {
        let provider = MockProvider::new(String::new());
        let config = CompletionConfig {
            model: "test".into(),
            temperature: 0.0,
            max_tokens: 100,
        };

        let response = provider.complete("any prompt", &config).unwrap();
        assert_eq!(response.text, "");
    }

    #[test]
    fn test_completion_response_token_counts() {
        let provider = MockProvider::new("action: 1".into());
        let config = CompletionConfig {
            model: "test".into(),
            temperature: 0.5,
            max_tokens: 256,
        };

        let prompt = "hello world"; // 11 bytes
        let response = provider.complete(prompt, &config).unwrap();
        assert_eq!(response.input_tokens, prompt.len() as u32);
        assert_eq!(response.output_tokens, 10);
        assert_eq!(response.text, "action: 1");
    }
}
