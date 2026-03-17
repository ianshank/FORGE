//! Task predicate types and composition operators.
//!
//! Tasks are defined as reward functions composed from atomic predicates
//! using logical composition operators. This forms the Task DSL.

use serde::{Deserialize, Serialize};

use crate::entity::{AgentId, ObjectId};
use crate::grid::Position;
use crate::resource::ItemType;

/// Atomic predicates that evaluate against world state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Predicate {
    /// Agent is at a specific position.
    AgentAt(AgentId, Position),
    /// Agent has at least N of an item type.
    AgentHas(AgentId, ItemType, u16),
    /// Two agents are within a given Manhattan distance.
    AgentNear(AgentId, AgentId, u16),
    /// An object is at a specific position.
    ObjectAt(ObjectId, Position),
    /// An object is in a specific state.
    ObjectInState(ObjectId, String),
    /// A certain number of ticks have elapsed.
    TimeElapsed(u64),
    /// Agent's health is above a threshold (normalized 0.0-1.0).
    HealthAbove(AgentId, f32),
    /// Agent has at least N of a resource type.
    ResourceCount(AgentId, ItemType, u16),
    /// All agents of a team are alive.
    TeamAlive(u8),
    /// Agent is on a specific terrain type.
    AgentOnTerrain(AgentId, u8),
    /// Agent is at a specific altitude.
    AgentAtAltitude(AgentId, u8),
    /// Agent's normalized battery is above threshold (0.0-1.0).
    BatteryAbove(AgentId, f32),
    /// Agent is airborne (altitude > 0).
    AgentAirborne(AgentId),
    /// Agent has landed (altitude == 0).
    AgentLanded(AgentId),
}

/// Composition operators for building complex tasks from predicates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TaskComposition {
    /// A single atomic predicate.
    Atom(Predicate),
    /// All sub-tasks must be satisfied simultaneously.
    And(Vec<TaskComposition>),
    /// At least one sub-task must be satisfied.
    Or(Vec<TaskComposition>),
    /// Sub-tasks must be achieved in order (each locks in when achieved).
    Sequence(Vec<TaskComposition>),
    /// Sub-task must be achieved before the deadline tick.
    Before(Box<TaskComposition>, u64),
    /// First sub-task must remain true while achieving the second.
    While(Box<TaskComposition>, Box<TaskComposition>),
    /// Sub-task must be achieved without performing a specific action type.
    Without(Box<TaskComposition>, u32),
}

/// Task difficulty tiers (1-6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskTier(pub u8);

impl TaskTier {
    /// Creates a new task tier, clamped to valid range [1, 6].
    pub fn new(tier: u8) -> Self {
        Self(tier.clamp(1, 6))
    }

    /// Returns the tier number.
    pub fn value(&self) -> u8 {
        self.0
    }
}

/// A fully specified task with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    /// Unique task identifier.
    pub id: u64,
    /// Human-readable task description.
    pub description: String,
    /// The task composition tree defining success conditions.
    pub goal: TaskComposition,
    /// Estimated difficulty tier.
    pub tier: TaskTier,
    /// Estimated minimum solution length (oracle actions).
    pub estimated_steps: u32,
    /// Reward for completing the task.
    pub reward: f32,
    /// Optional partial rewards for intermediate progress.
    pub dense_reward_weights: Vec<f32>,
}

/// Runtime state of an active task being tracked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTask {
    /// The task definition.
    pub definition: TaskDefinition,
    /// Current progress for each predicate in the task tree (0.0-1.0).
    pub progress: Vec<f32>,
    /// Index of the current sequence step (for Sequence compositions).
    pub sequence_index: usize,
    /// Whether this task has been completed.
    pub completed: bool,
    /// Whether this task has failed (deadline passed, forbidden action taken).
    pub failed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_tier_clamping() {
        assert_eq!(TaskTier::new(0).value(), 1);
        assert_eq!(TaskTier::new(3).value(), 3);
        assert_eq!(TaskTier::new(10).value(), 6);
    }

    #[test]
    fn test_predicate_serialization() {
        let pred = Predicate::AgentAt(0, Position::new(5, 5));
        let json = serde_json::to_string(&pred).unwrap();
        let recovered: Predicate = serde_json::from_str(&json).unwrap();
        assert_eq!(pred, recovered);
    }

    #[test]
    fn test_composition_nesting() {
        let task = TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 3)),
            TaskComposition::Or(vec![
                TaskComposition::Atom(Predicate::AgentAt(0, Position::new(0, 0))),
                TaskComposition::Atom(Predicate::AgentAt(0, Position::new(10, 10))),
            ]),
        ]);
        // Verify serialization roundtrip
        let json = serde_json::to_string(&task).unwrap();
        let recovered: TaskComposition = serde_json::from_str(&json).unwrap();
        assert_eq!(task, recovered);
    }

    #[test]
    fn test_sequence_composition() {
        let task = TaskComposition::Sequence(vec![
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 2)),
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Axe, 1)),
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
        ]);
        let json = serde_json::to_string(&task).unwrap();
        let _: TaskComposition = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_task_definition() {
        let task = TaskDefinition {
            id: 1,
            description: "Collect 3 wood and deliver to base".to_string(),
            goal: TaskComposition::Sequence(vec![
                TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 3)),
                TaskComposition::Atom(Predicate::AgentAt(0, Position::new(0, 0))),
            ]),
            tier: TaskTier::new(2),
            estimated_steps: 50,
            reward: 1.0,
            dense_reward_weights: vec![0.5, 0.5],
        };
        assert_eq!(task.tier.value(), 2);
    }

    #[test]
    fn test_active_task_defaults() {
        let definition = TaskDefinition {
            id: 42,
            description: "Test task".to_string(),
            goal: TaskComposition::Atom(Predicate::AgentAt(0, Position::new(1, 1))),
            tier: TaskTier::new(1),
            estimated_steps: 10,
            reward: 1.0,
            dense_reward_weights: vec![1.0],
        };
        let active = ActiveTask {
            definition,
            progress: vec![0.0],
            sequence_index: 0,
            completed: false,
            failed: false,
        };
        assert_eq!(active.definition.id, 42);
        assert_eq!(active.sequence_index, 0);
        assert!(!active.completed);
        assert!(!active.failed);
        assert_eq!(active.progress.len(), 1);
        assert_eq!(active.progress[0], 0.0);
    }

    #[test]
    fn test_task_composition_while() {
        let condition = TaskComposition::Atom(Predicate::HealthAbove(0, 0.5));
        let goal = TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 3));
        let while_task = TaskComposition::While(Box::new(condition), Box::new(goal));

        // Verify structure via serialization roundtrip.
        let json = serde_json::to_string(&while_task).unwrap();
        let recovered: TaskComposition = serde_json::from_str(&json).unwrap();
        assert_eq!(while_task, recovered);

        // Verify it matches the While variant.
        match &while_task {
            TaskComposition::While(cond, g) => {
                match cond.as_ref() {
                    TaskComposition::Atom(Predicate::HealthAbove(id, t)) => {
                        assert_eq!(*id, 0);
                        assert_eq!(*t, 0.5);
                    }
                    _ => panic!("expected HealthAbove predicate"),
                }
                match g.as_ref() {
                    TaskComposition::Atom(Predicate::AgentHas(id, item, count)) => {
                        assert_eq!(*id, 0);
                        assert_eq!(*item, ItemType::Wood);
                        assert_eq!(*count, 3);
                    }
                    _ => panic!("expected AgentHas predicate"),
                }
            }
            _ => panic!("expected While variant"),
        }
    }

    #[test]
    fn test_task_composition_before() {
        let subtask = TaskComposition::Atom(Predicate::AgentAt(0, Position::new(10, 10)));
        let before_task = TaskComposition::Before(Box::new(subtask), 500);

        let json = serde_json::to_string(&before_task).unwrap();
        let recovered: TaskComposition = serde_json::from_str(&json).unwrap();
        assert_eq!(before_task, recovered);

        match &before_task {
            TaskComposition::Before(sub, deadline) => {
                assert_eq!(*deadline, 500);
                match sub.as_ref() {
                    TaskComposition::Atom(Predicate::AgentAt(id, pos)) => {
                        assert_eq!(*id, 0);
                        assert_eq!(*pos, Position::new(10, 10));
                    }
                    _ => panic!("expected AgentAt predicate"),
                }
            }
            _ => panic!("expected Before variant"),
        }
    }

    #[test]
    fn test_task_composition_without() {
        let subtask = TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Stone, 5));
        let without_task = TaskComposition::Without(Box::new(subtask), 0); // forbidden: Noop

        let json = serde_json::to_string(&without_task).unwrap();
        let recovered: TaskComposition = serde_json::from_str(&json).unwrap();
        assert_eq!(without_task, recovered);

        match &without_task {
            TaskComposition::Without(sub, forbidden_action) => {
                assert_eq!(*forbidden_action, 0);
                match sub.as_ref() {
                    TaskComposition::Atom(Predicate::AgentHas(id, item, count)) => {
                        assert_eq!(*id, 0);
                        assert_eq!(*item, ItemType::Stone);
                        assert_eq!(*count, 5);
                    }
                    _ => panic!("expected AgentHas predicate"),
                }
            }
            _ => panic!("expected Without variant"),
        }
    }

    #[test]
    fn test_drone_predicate_serde_roundtrip() {
        let preds = vec![
            Predicate::AgentAtAltitude(0, 5),
            Predicate::BatteryAbove(0, 0.5),
            Predicate::AgentAirborne(0),
            Predicate::AgentLanded(0),
        ];
        for pred in &preds {
            let json = serde_json::to_string(pred).unwrap();
            let deserialized: Predicate = serde_json::from_str(&json).unwrap();
            assert_eq!(format!("{:?}", pred), format!("{:?}", deserialized));
        }
    }
}
