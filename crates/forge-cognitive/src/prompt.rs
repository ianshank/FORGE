//! Prompt construction from world state and memory context.
//!
//! Converts FORGE observations, memory retrieval results, and social state
//! into structured prompts for the cognitive provider.

use tracing::instrument;

use crate::config::DEFAULT_SYSTEM_PROMPT;
use forge_memory::store::MemoryEntry;

/// A structured prompt for the cognitive provider.
#[derive(Debug, Clone)]
pub struct CognitivePrompt {
    /// System instructions for the agent.
    pub system: String,
    /// Current observation summary.
    pub observation: String,
    /// Retrieved memory context.
    pub memory_context: Vec<String>,
    /// Social context (trust, reputation, alliances).
    pub social_context: String,
    /// Current task/goal description.
    pub task_description: String,
    /// Available actions.
    pub available_actions: Vec<String>,
}

impl CognitivePrompt {
    /// Creates a new prompt builder.
    pub fn builder() -> CognitivePromptBuilder {
        CognitivePromptBuilder::new()
    }

    /// Renders the prompt as a single string for the provider.
    #[instrument(skip_all)]
    pub fn render(&self) -> String {
        let mut parts = Vec::new();

        parts.push(format!("# System\n{}\n", self.system));
        parts.push(format!("# Observation\n{}\n", self.observation));

        if !self.memory_context.is_empty() {
            parts.push("# Relevant Memories\n".to_string());
            for mem in &self.memory_context {
                parts.push(format!("- {mem}\n"));
            }
        }

        if !self.social_context.is_empty() {
            parts.push(format!("# Social Context\n{}\n", self.social_context));
        }

        if !self.task_description.is_empty() {
            parts.push(format!("# Current Task\n{}\n", self.task_description));
        }

        if !self.available_actions.is_empty() {
            parts.push("# Available Actions\n".to_string());
            for (i, action) in self.available_actions.iter().enumerate() {
                parts.push(format!("{i}: {action}\n"));
            }
        }

        parts.push("\n# Respond with your reasoning and selected action ID.\n".to_string());

        parts.join("")
    }
}

/// Builder for constructing cognitive prompts.
pub struct CognitivePromptBuilder {
    system: String,
    observation: String,
    memory_context: Vec<String>,
    social_context: String,
    task_description: String,
    available_actions: Vec<String>,
}

impl CognitivePromptBuilder {
    /// Creates a new builder with defaults.
    fn new() -> Self {
        Self {
            system: DEFAULT_SYSTEM_PROMPT.to_string(),
            observation: String::new(),
            memory_context: Vec::new(),
            social_context: String::new(),
            task_description: String::new(),
            available_actions: Vec::new(),
        }
    }

    /// Overrides the system prompt from a config.
    pub fn system_prompt(mut self, prompt: String) -> Self {
        self.system = prompt;
        self
    }

    /// Sets the observation text.
    pub fn observation(mut self, obs: String) -> Self {
        self.observation = obs;
        self
    }

    /// Adds memory entries to the context.
    pub fn memories(mut self, entries: &[MemoryEntry]) -> Self {
        for entry in entries {
            let summary = match entry {
                MemoryEntry::Semantic(fact) => {
                    format!(
                        "[Fact] {}: {} (confidence: {:.2})",
                        fact.key, fact.value, fact.confidence
                    )
                }
                MemoryEntry::Episodic(ep) => {
                    format!(
                        "[Episode] ticks {}-{} at ({},{}) → {:?}, reward: {:.2}",
                        ep.tick_range.0,
                        ep.tick_range.1,
                        ep.location.0,
                        ep.location.1,
                        ep.outcome,
                        ep.reward
                    )
                }
                MemoryEntry::PreferredAction { context, action_id } => {
                    format!("[Preference] {context} → action {action_id}")
                }
            };
            self.memory_context.push(summary);
        }
        self
    }

    /// Sets the social context string.
    pub fn social(mut self, context: String) -> Self {
        self.social_context = context;
        self
    }

    /// Sets the task description.
    pub fn task(mut self, description: String) -> Self {
        self.task_description = description;
        self
    }

    /// Sets the available actions.
    pub fn actions(mut self, actions: Vec<String>) -> Self {
        self.available_actions = actions;
        self
    }

    /// Builds the prompt.
    pub fn build(self) -> CognitivePrompt {
        CognitivePrompt {
            system: self.system,
            observation: self.observation,
            memory_context: self.memory_context,
            social_context: self.social_context,
            task_description: self.task_description,
            available_actions: self.available_actions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_builder() {
        let prompt = CognitivePrompt::builder()
            .observation("I see a forest with wood.".into())
            .task("Gather 3 wood.".into())
            .actions(vec!["Noop".into(), "MoveUp".into(), "PickUp".into()])
            .build();

        let rendered = prompt.render();
        assert!(rendered.contains("forest"));
        assert!(rendered.contains("Gather 3 wood"));
        assert!(rendered.contains("MoveUp"));
    }

    #[test]
    fn test_prompt_with_memories() {
        let entries = vec![MemoryEntry::PreferredAction {
            context: "gathering".into(),
            action_id: 5,
        }];
        let prompt = CognitivePrompt::builder()
            .observation("test".into())
            .memories(&entries)
            .build();

        let rendered = prompt.render();
        assert!(rendered.contains("[Preference]"));
        assert!(rendered.contains("gathering"));
    }

    #[test]
    fn test_prompt_builder_all_fields_set() {
        let prompt = CognitivePrompt::builder()
            .system_prompt("Custom system prompt.".into())
            .observation("I see a river to the north.".into())
            .social("Ally: agent_42, Trust: 0.9".into())
            .task("Cross the river and gather fish.".into())
            .actions(vec!["Noop".into(), "MoveNorth".into(), "Fish".into()])
            .build();

        // Attach memory context directly (builder .memories() requires MemoryEntry)
        let mut prompt = prompt;
        prompt.memory_context = vec!["Caught fish before at tick 50".into()];

        let rendered = prompt.render();

        // All sections should be present and non-empty
        assert!(rendered.contains("# System\nCustom system prompt."));
        assert!(rendered.contains("# Observation\nI see a river to the north."));
        assert!(rendered.contains("# Relevant Memories"));
        assert!(rendered.contains("- Caught fish before at tick 50"));
        assert!(rendered.contains("# Social Context\nAlly: agent_42, Trust: 0.9"));
        assert!(rendered.contains("# Current Task\nCross the river and gather fish."));
        assert!(rendered.contains("# Available Actions"));
        assert!(rendered.contains("0: Noop"));
        assert!(rendered.contains("1: MoveNorth"));
        assert!(rendered.contains("2: Fish"));
        assert!(rendered.contains("# Respond with your reasoning"));
    }

    #[test]
    fn test_prompt_builder_only_required_fields() {
        let prompt = CognitivePrompt::builder()
            .observation("minimal observation".into())
            .build();

        let rendered = prompt.render();

        // System and observation are always present
        assert!(rendered.contains("# System"));
        assert!(rendered.contains("# Observation\nminimal observation"));
        // Optional sections should be absent
        assert!(!rendered.contains("# Relevant Memories"));
        assert!(!rendered.contains("# Social Context"));
        assert!(!rendered.contains("# Current Task"));
        assert!(!rendered.contains("# Available Actions"));
        // Footer is always present
        assert!(rendered.contains("# Respond with your reasoning"));
    }

    #[test]
    fn test_prompt_render_with_special_characters() {
        let prompt = CognitivePrompt::builder()
            .observation("Agent sees: <wood> & \"stone\" at (3,5)".into())
            .task("Collect items worth > $100 & < $500".into())
            .actions(vec![
                "Pick<Wood>".into(),
                "Mine \"Stone\"".into(),
                "Noop & Wait".into(),
            ])
            .build();

        let rendered = prompt.render();

        // Special characters should be preserved verbatim
        assert!(rendered.contains("<wood>"));
        assert!(rendered.contains("& \"stone\""));
        assert!(rendered.contains("(3,5)"));
        assert!(rendered.contains("> $100 & < $500"));
        assert!(rendered.contains("Pick<Wood>"));
        assert!(rendered.contains("Mine \"Stone\""));
        assert!(rendered.contains("Noop & Wait"));
    }
}
