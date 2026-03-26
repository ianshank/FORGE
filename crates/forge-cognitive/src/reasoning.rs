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

    #[test]
    fn test_reasoning_type_all_variants() {
        let types = vec![
            ReasoningType::Observe,
            ReasoningType::Remember,
            ReasoningType::Think,
            ReasoningType::Plan,
            ReasoningType::Act,
            ReasoningType::Reflect,
        ];
        for t in &types {
            let step = ReasoningStep::new(t.clone(), "test".into(), 0);
            let json = serde_json::to_string(&step).unwrap();
            let deser: ReasoningStep = serde_json::from_str(&json).unwrap();
            assert_eq!(deser.step_type, *t);
        }
    }

    #[test]
    fn test_reasoning_step_preserves_tick() {
        let step = ReasoningStep::new(ReasoningType::Observe, "obs".into(), 999);
        assert_eq!(step.tick, 999);
        assert_eq!(step.content, "obs");
    }

    #[test]
    fn test_trace_default() {
        let trace = ReasoningTrace::default();
        assert!(trace.steps.is_empty());
        assert_eq!(trace.selected_action, 0);
        assert_eq!(trace.confidence, 0.0);
    }

    #[test]
    fn test_trace_clone() {
        let mut trace = ReasoningTrace::new();
        trace.add_step(ReasoningStep::new(
            ReasoningType::Think,
            "thinking...".into(),
            10,
        ));
        trace.selected_action = 5;
        trace.confidence = 0.9;

        let cloned = trace.clone();
        assert_eq!(cloned.steps.len(), 1);
        assert_eq!(cloned.selected_action, 5);
        assert_eq!(cloned.confidence, 0.9);
    }

    #[test]
    fn test_full_trace_serialization_roundtrip() {
        let mut trace = ReasoningTrace::new();
        trace.add_step(ReasoningStep::new(
            ReasoningType::Observe,
            "I see a forest".into(),
            1,
        ));
        trace.add_step(ReasoningStep::new(
            ReasoningType::Remember,
            "Wood is useful".into(),
            1,
        ));
        trace.add_step(ReasoningStep::new(
            ReasoningType::Think,
            "I should gather wood".into(),
            1,
        ));
        trace.add_step(ReasoningStep::new(
            ReasoningType::Act,
            "Moving toward wood".into(),
            1,
        ));
        trace.selected_action = 2;
        trace.confidence = 0.85;

        let json = serde_json::to_string(&trace).unwrap();
        let deser: ReasoningTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.steps.len(), 4);
        assert_eq!(deser.selected_action, 2);
        assert_eq!(deser.confidence, 0.85);
        assert_eq!(deser.steps[0].step_type, ReasoningType::Observe);
        assert_eq!(deser.steps[3].step_type, ReasoningType::Act);
    }

    #[test]
    fn test_trace_with_many_steps() {
        let mut trace = ReasoningTrace::new();
        let types = [
            ReasoningType::Observe,
            ReasoningType::Remember,
            ReasoningType::Think,
            ReasoningType::Plan,
            ReasoningType::Act,
            ReasoningType::Reflect,
        ];

        // Add 12 steps cycling through all types
        for i in 0..12 {
            let step_type = types[i % types.len()].clone();
            trace.add_step(ReasoningStep::new(
                step_type,
                format!("step {i}"),
                i as u64,
            ));
        }

        assert_eq!(trace.steps.len(), 12);
        // Verify ordering is preserved
        for (i, step) in trace.steps.iter().enumerate() {
            assert_eq!(step.content, format!("step {i}"));
            assert_eq!(step.tick, i as u64);
        }

        // Serialization roundtrip with many steps
        let json = serde_json::to_string(&trace).unwrap();
        let deser: ReasoningTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.steps.len(), 12);
    }

    #[test]
    fn test_reasoning_type_all_variants_accessible() {
        // Verify each variant can be constructed, cloned, debug-printed, and compared
        let variants = vec![
            ReasoningType::Observe,
            ReasoningType::Remember,
            ReasoningType::Think,
            ReasoningType::Plan,
            ReasoningType::Act,
            ReasoningType::Reflect,
        ];

        assert_eq!(variants.len(), 6);

        // All variants are distinct
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }

        // Debug formatting works for all
        for v in &variants {
            let debug = format!("{:?}", v);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_empty_trace_properties() {
        let trace = ReasoningTrace::new();

        assert!(trace.steps.is_empty());
        assert_eq!(trace.steps.len(), 0);
        assert_eq!(trace.selected_action, 0);
        assert_eq!(trace.confidence, 0.0);

        // Empty trace serializes and deserializes correctly
        let json = serde_json::to_string(&trace).unwrap();
        let deser: ReasoningTrace = serde_json::from_str(&json).unwrap();
        assert!(deser.steps.is_empty());
        assert_eq!(deser.selected_action, 0);
        assert_eq!(deser.confidence, 0.0);

        // Clone of empty trace is also empty
        let cloned = trace.clone();
        assert!(cloned.steps.is_empty());
    }
}
