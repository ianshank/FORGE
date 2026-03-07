//! Intent declarations for the FORGE neuro-symbolic bridge.
//!
//! This module provides types for declaring, tracking, and serializing
//! agent intents — the symbolic layer that bridges neural network outputs
//! (e.g., MCTS rollouts) with human-interpretable goals.

use serde::{Deserialize, Serialize};

/// Default maximum number of historical intents retained by [`AgentIntent`].
const DEFAULT_MAX_HISTORY: usize = 100;

/// High-level intent label describing an agent's current strategic goal.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum IntentLabel {
    /// Move through an area on a recurring patrol route.
    Patrol,
    /// Engage a hostile target.
    Attack,
    /// Hold a position or protect an asset.
    Defend,
    /// Collect resources from the environment.
    Gather,
    /// Explore unknown or partially observed terrain.
    Scout,
    /// Withdraw from a dangerous situation.
    Retreat,
    /// Assist a nearby allied agent.
    Support,
    /// No active goal; the agent is waiting for new orders.
    #[default]
    Idle,
}

/// A fully qualified intent declaration produced by the planning subsystem.
///
/// Each declaration captures the label, the planner metrics that led to it,
/// human-readable preconditions, and the simulation tick at which it was made.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntentDeclaration {
    /// The high-level intent label.
    pub intent_label: IntentLabel,
    /// Planner confidence in this intent, clamped to `[0.0, 1.0]`.
    pub confidence: f64,
    /// MCTS search depth that produced this declaration.
    pub search_depth: u32,
    /// UCB1 score at the point of selection.
    pub ucb1_score: f64,
    /// Human-readable preconditions that must hold for the intent.
    pub preconditions: Vec<String>,
    /// Description of the expected outcome if the intent succeeds.
    pub expected_outcome: String,
    /// Simulation tick when this intent was declared.
    pub tick: u64,
}

impl Default for IntentDeclaration {
    fn default() -> Self {
        Self {
            intent_label: IntentLabel::Idle,
            confidence: 0.0,
            search_depth: 0,
            ucb1_score: 0.0,
            preconditions: Vec::new(),
            expected_outcome: String::new(),
            tick: 0,
        }
    }
}

/// Per-agent intent state, holding the current declaration and a bounded history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentIntent {
    /// The currently active intent declaration.
    pub current: IntentDeclaration,
    /// Past intent declarations, ordered oldest-first.
    pub history: Vec<IntentDeclaration>,
    /// Maximum number of entries retained in [`history`](Self::history).
    pub max_history: usize,
}

impl Default for AgentIntent {
    fn default() -> Self {
        Self {
            current: IntentDeclaration::default(),
            history: Vec::new(),
            max_history: DEFAULT_MAX_HISTORY,
        }
    }
}

impl AgentIntent {
    /// Create a new [`AgentIntent`] with the given history capacity.
    pub fn new(max_history: usize) -> Self {
        Self {
            max_history,
            ..Self::default()
        }
    }

    /// Declare a new intent, archiving the previous one into history.
    ///
    /// If the history length would exceed [`max_history`](Self::max_history),
    /// the oldest entry is evicted first.
    pub fn declare(&mut self, intent: IntentDeclaration) {
        let old = std::mem::replace(&mut self.current, intent);
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(old);
    }

    /// Return a reference to the current intent's label.
    pub fn current_label(&self) -> &IntentLabel {
        &self.current.intent_label
    }

    /// Return the number of entries currently stored in the history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_label_default_is_idle() {
        assert_eq!(IntentLabel::default(), IntentLabel::Idle);
    }

    #[test]
    fn test_intent_label_variants() {
        let variants = [
            IntentLabel::Patrol,
            IntentLabel::Attack,
            IntentLabel::Defend,
            IntentLabel::Gather,
            IntentLabel::Scout,
            IntentLabel::Retreat,
            IntentLabel::Support,
            IntentLabel::Idle,
        ];
        // All variants are distinct.
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_intent_declaration_default() {
        let decl = IntentDeclaration::default();
        assert_eq!(decl.intent_label, IntentLabel::Idle);
        assert!((decl.confidence - 0.0).abs() < f64::EPSILON);
        assert_eq!(decl.search_depth, 0);
        assert!((decl.ucb1_score - 0.0).abs() < f64::EPSILON);
        assert!(decl.preconditions.is_empty());
        assert!(decl.expected_outcome.is_empty());
        assert_eq!(decl.tick, 0);
    }

    #[test]
    fn test_agent_intent_default() {
        let ai = AgentIntent::default();
        assert_eq!(ai.current.intent_label, IntentLabel::Idle);
        assert!(ai.history.is_empty());
        assert_eq!(ai.max_history, DEFAULT_MAX_HISTORY);
    }

    #[test]
    fn test_declare_and_history_tracking() {
        let mut ai = AgentIntent::new(10);

        let attack = IntentDeclaration {
            intent_label: IntentLabel::Attack,
            confidence: 0.85,
            search_depth: 5,
            ucb1_score: 1.42,
            preconditions: vec!["enemy_visible".to_string()],
            expected_outcome: "eliminate target".to_string(),
            tick: 42,
        };

        ai.declare(attack);

        assert_eq!(*ai.current_label(), IntentLabel::Attack);
        assert_eq!(ai.history_len(), 1);
        // The old default (Idle) should be in history.
        assert_eq!(ai.history[0].intent_label, IntentLabel::Idle);
    }

    #[test]
    fn test_history_eviction() {
        let max = 3;
        let mut ai = AgentIntent::new(max);

        // Declare max + 2 intents so eviction is clearly exercised.
        // Ticks: 1..=(max+2). The initial default (tick 0) is also pushed.
        for i in 1..=(max + 2) {
            let decl = IntentDeclaration {
                intent_label: IntentLabel::Patrol,
                tick: i as u64,
                ..IntentDeclaration::default()
            };
            ai.declare(decl);
        }

        // History should be capped at max.
        assert_eq!(ai.history_len(), max);
        // The oldest surviving entry should have tick == 2
        // (default tick-0 and tick-1 were evicted).
        assert_eq!(ai.history[0].tick, 2);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut ai = AgentIntent::new(50);
        ai.declare(IntentDeclaration {
            intent_label: IntentLabel::Gather,
            confidence: 0.73,
            search_depth: 8,
            ucb1_score: 2.1,
            preconditions: vec![
                "resource_nearby".to_string(),
                "inventory_not_full".to_string(),
            ],
            expected_outcome: "collect 5 wood".to_string(),
            tick: 100,
        });

        let json = serde_json::to_string(&ai).expect("serialize");
        let restored: AgentIntent = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(*restored.current_label(), IntentLabel::Gather);
        assert!((restored.current.confidence - 0.73).abs() < f64::EPSILON);
        assert_eq!(restored.current.search_depth, 8);
        assert_eq!(restored.current.preconditions.len(), 2);
        assert_eq!(restored.history_len(), 1);
        assert_eq!(restored.max_history, 50);
    }

    #[test]
    fn test_current_label_accessor() {
        let ai = AgentIntent::default();
        assert_eq!(*ai.current_label(), IntentLabel::Idle);

        let mut ai2 = AgentIntent::new(5);
        ai2.declare(IntentDeclaration {
            intent_label: IntentLabel::Scout,
            ..IntentDeclaration::default()
        });
        assert_eq!(*ai2.current_label(), IntentLabel::Scout);
    }
}
