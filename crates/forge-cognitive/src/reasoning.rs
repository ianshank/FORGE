//! Reasoning structures for chain-of-thought cognitive processing.
//!
//! Reasoning steps model the cognitive process from observation through
//! planning to action selection, enabling transparent decision-making
//! and post-hoc analysis of agent behavior.

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Type of reasoning step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReasoningType {
    /// Observing the current state of the world.
    Observe,
    /// Retrieving relevant memories.
    Remember,
    /// Thinking about the situation.
    Think,
    /// Planning a sequence of actions.
    Plan,
    /// Selecting a concrete action.
    Act,
    /// Reflecting on the outcome.
    Reflect,
}

/// A single step in the agent's reasoning process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Type of this reasoning step.
    pub step_type: ReasoningType,
    /// Content of the reasoning (natural language or structured data).
    pub content: String,
    /// Tick when this step occurred.
    pub tick: u64,
}

impl ReasoningStep {
    /// Creates a new reasoning step.
    #[instrument(skip_all)]
    pub fn new(step_type: ReasoningType, content: String, tick: u64) -> Self {
        Self {
            step_type,
            content,
            tick,
        }
    }
}

/// A complete reasoning trace for one action selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// Ordered steps in the reasoning process.
    pub steps: Vec<ReasoningStep>,
    /// Final selected action (discrete action ID).
    pub selected_action: u32,
    /// Confidence in the selected action.
    pub confidence: f32,
}

impl ReasoningTrace {
    /// Creates a new empty trace.
    #[instrument(skip_all)]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            selected_action: 0,
            confidence: 0.0,
        }
    }

    /// Adds a step to the trace.
    #[instrument(skip_all)]
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.steps.push(step);
    }
}

impl Default for ReasoningTrace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_trace() {
        let mut trace = ReasoningTrace::new();
        trace.add_step(ReasoningStep::new(
            ReasoningType::Observe,
            "I see a forest biome with wood resources".into(),
            100,
        ));
        trace.add_step(ReasoningStep::new(
            ReasoningType::Think,
            "I need wood to craft an axe".into(),
            100,
        ));
        trace.add_step(ReasoningStep::new(
            ReasoningType::Act,
            "Move toward the wood resource".into(),
            100,
        ));
        trace.selected_action = 1; // MoveUp
        trace.confidence = 0.85;

        assert_eq!(trace.steps.len(), 3);
        assert_eq!(trace.steps[0].step_type, ReasoningType::Observe);
        assert_eq!(trace.selected_action, 1);
    }

    #[test]
    fn test_serialization() {
        let mut trace = ReasoningTrace::new();
        trace.add_step(ReasoningStep::new(
            ReasoningType::Plan,
            "plan step".into(),
            50,
        ));
        let json = serde_json::to_string(&trace).unwrap();
        let deser: ReasoningTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.steps.len(), 1);
    }
}
