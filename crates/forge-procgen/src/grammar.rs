//! Objective grammar DSL for composing complex mission objectives.
//!
//! Objectives are built from primitives (e.g., reach a location, eliminate a target)
//! and composed using logical combinators (And, Or, Sequence, Timed).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::instrument;

/// Atomic objective primitives that can be directly evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectivePrimitive {
    /// Navigate to a specific grid position.
    ReachLocation {
        /// Target x coordinate.
        x: u16,
        /// Target y coordinate.
        y: u16,
    },
    /// Hold a circular area for a duration.
    HoldArea {
        /// Center x coordinate.
        x: u16,
        /// Center y coordinate.
        y: u16,
        /// Radius of the area to hold.
        radius: u16,
        /// Number of ticks to hold the area.
        duration_ticks: u64,
    },
    /// Eliminate a specific target unit.
    EliminateTarget {
        /// ID of the target to eliminate.
        target_id: u32,
    },
    /// Collect a specified amount of a resource type.
    CollectResource {
        /// Type of resource to collect.
        resource_type: String,
        /// Number of units to collect.
        count: u32,
    },
    /// Escort a unit to a destination.
    EscortUnit {
        /// ID of the unit to escort.
        unit_id: u32,
        /// Destination x coordinate.
        dest_x: u16,
        /// Destination y coordinate.
        dest_y: u16,
    },
    /// Deny enemy access to an area for a duration.
    DenyArea {
        /// Center x coordinate.
        x: u16,
        /// Center y coordinate.
        y: u16,
        /// Radius of the denied area.
        radius: u16,
        /// Number of ticks to deny the area.
        duration_ticks: u64,
    },
    /// Survive for a specified duration.
    Survive {
        /// Number of ticks to survive.
        duration_ticks: u64,
    },
}

impl ObjectivePrimitive {
    /// Returns a unique string identifier for this primitive instance.
    ///
    /// Used for tracking completion status in a `HashSet<String>`.
    #[instrument(skip(self))]
    pub fn id(&self) -> String {
        match self {
            Self::ReachLocation { x, y } => format!("reach_{x}_{y}"),
            Self::HoldArea {
                x,
                y,
                radius,
                duration_ticks,
            } => {
                format!("hold_{x}_{y}_{radius}_{duration_ticks}")
            }
            Self::EliminateTarget { target_id } => format!("eliminate_{target_id}"),
            Self::CollectResource {
                resource_type,
                count,
            } => {
                format!("collect_{resource_type}_{count}")
            }
            Self::EscortUnit {
                unit_id,
                dest_x,
                dest_y,
            } => {
                format!("escort_{unit_id}_{dest_x}_{dest_y}")
            }
            Self::DenyArea {
                x,
                y,
                radius,
                duration_ticks,
            } => {
                format!("deny_{x}_{y}_{radius}_{duration_ticks}")
            }
            Self::Survive { duration_ticks } => format!("survive_{duration_ticks}"),
        }
    }
}

/// Composite objective built from primitives and logical combinators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Objective {
    /// A single primitive objective.
    Primitive(ObjectivePrimitive),
    /// All sub-objectives must be completed (in any order).
    And(Vec<Objective>),
    /// At least one sub-objective must be completed.
    Or(Vec<Objective>),
    /// Sub-objectives must be completed in order.
    Sequence(Vec<Objective>),
    /// An objective with a time limit (in ticks).
    Timed {
        /// The objective to complete within the time limit.
        objective: Box<Objective>,
        /// Maximum ticks allowed.
        time_limit: u64,
    },
}

/// Status of an objective evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveStatus {
    /// The objective is still being worked on.
    InProgress,
    /// The objective has been successfully completed.
    Completed,
    /// The objective has been failed.
    Failed,
}

impl Objective {
    /// Recursively evaluates the objective status given the current tick
    /// and the set of completed primitive IDs.
    #[instrument(skip(self, completed_primitives))]
    pub fn evaluate(&self, tick: u64, completed_primitives: &HashSet<String>) -> ObjectiveStatus {
        match self {
            Objective::Primitive(prim) => {
                if completed_primitives.contains(&prim.id()) {
                    ObjectiveStatus::Completed
                } else {
                    ObjectiveStatus::InProgress
                }
            }
            Objective::And(objectives) => {
                let mut all_completed = true;
                for obj in objectives {
                    match obj.evaluate(tick, completed_primitives) {
                        ObjectiveStatus::Failed => return ObjectiveStatus::Failed,
                        ObjectiveStatus::InProgress => all_completed = false,
                        ObjectiveStatus::Completed => {}
                    }
                }
                if all_completed {
                    ObjectiveStatus::Completed
                } else {
                    ObjectiveStatus::InProgress
                }
            }
            Objective::Or(objectives) => {
                let mut any_in_progress = false;
                for obj in objectives {
                    match obj.evaluate(tick, completed_primitives) {
                        ObjectiveStatus::Completed => return ObjectiveStatus::Completed,
                        ObjectiveStatus::InProgress => any_in_progress = true,
                        ObjectiveStatus::Failed => {}
                    }
                }
                if any_in_progress {
                    ObjectiveStatus::InProgress
                } else {
                    ObjectiveStatus::Failed
                }
            }
            Objective::Sequence(objectives) => {
                for obj in objectives {
                    match obj.evaluate(tick, completed_primitives) {
                        ObjectiveStatus::Failed => return ObjectiveStatus::Failed,
                        ObjectiveStatus::InProgress => return ObjectiveStatus::InProgress,
                        ObjectiveStatus::Completed => continue,
                    }
                }
                ObjectiveStatus::Completed
            }
            Objective::Timed {
                objective,
                time_limit,
            } => {
                let status = objective.evaluate(tick, completed_primitives);
                match status {
                    ObjectiveStatus::Completed => ObjectiveStatus::Completed,
                    ObjectiveStatus::Failed => ObjectiveStatus::Failed,
                    ObjectiveStatus::InProgress => {
                        if tick >= *time_limit {
                            ObjectiveStatus::Failed
                        } else {
                            ObjectiveStatus::InProgress
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_ids_unique() {
        let primitives = [
            ObjectivePrimitive::ReachLocation { x: 1, y: 2 },
            ObjectivePrimitive::ReachLocation { x: 3, y: 4 },
            ObjectivePrimitive::HoldArea {
                x: 1,
                y: 2,
                radius: 3,
                duration_ticks: 100,
            },
            ObjectivePrimitive::EliminateTarget { target_id: 1 },
            ObjectivePrimitive::CollectResource {
                resource_type: "gold".into(),
                count: 5,
            },
            ObjectivePrimitive::EscortUnit {
                unit_id: 1,
                dest_x: 10,
                dest_y: 10,
            },
            ObjectivePrimitive::DenyArea {
                x: 5,
                y: 5,
                radius: 2,
                duration_ticks: 50,
            },
            ObjectivePrimitive::Survive {
                duration_ticks: 200,
            },
        ];
        let ids: HashSet<String> = primitives.iter().map(|p| p.id()).collect();
        assert_eq!(
            ids.len(),
            primitives.len(),
            "All primitive IDs must be unique"
        );
    }

    #[test]
    fn test_evaluate_and_all_completed() {
        let obj = Objective::And(vec![
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 1, y: 1 }),
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 2, y: 2 }),
        ]);
        let mut completed = HashSet::new();
        completed.insert("reach_1_1".to_string());
        completed.insert("reach_2_2".to_string());
        assert_eq!(obj.evaluate(0, &completed), ObjectiveStatus::Completed);
    }

    #[test]
    fn test_evaluate_and_partial() {
        let obj = Objective::And(vec![
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 1, y: 1 }),
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 2, y: 2 }),
        ]);
        let mut completed = HashSet::new();
        completed.insert("reach_1_1".to_string());
        assert_eq!(obj.evaluate(0, &completed), ObjectiveStatus::InProgress);
    }

    #[test]
    fn test_evaluate_or_one_completed() {
        let obj = Objective::Or(vec![
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 1, y: 1 }),
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 2, y: 2 }),
        ]);
        let mut completed = HashSet::new();
        completed.insert("reach_1_1".to_string());
        assert_eq!(obj.evaluate(0, &completed), ObjectiveStatus::Completed);
    }

    #[test]
    fn test_evaluate_or_none_completed() {
        let obj = Objective::Or(vec![
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 1, y: 1 }),
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 2, y: 2 }),
        ]);
        let completed = HashSet::new();
        assert_eq!(obj.evaluate(0, &completed), ObjectiveStatus::InProgress);
    }

    #[test]
    fn test_evaluate_sequence() {
        let obj = Objective::Sequence(vec![
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 1, y: 1 }),
            Objective::Primitive(ObjectivePrimitive::ReachLocation { x: 2, y: 2 }),
        ]);

        // Nothing completed — blocked on first
        let completed = HashSet::new();
        assert_eq!(obj.evaluate(0, &completed), ObjectiveStatus::InProgress);

        // First completed — blocked on second
        let mut completed = HashSet::new();
        completed.insert("reach_1_1".to_string());
        assert_eq!(obj.evaluate(0, &completed), ObjectiveStatus::InProgress);

        // Both completed
        completed.insert("reach_2_2".to_string());
        assert_eq!(obj.evaluate(0, &completed), ObjectiveStatus::Completed);
    }

    #[test]
    fn test_evaluate_timed_within_limit() {
        let obj = Objective::Timed {
            objective: Box::new(Objective::Primitive(ObjectivePrimitive::ReachLocation {
                x: 1,
                y: 1,
            })),
            time_limit: 100,
        };
        let completed = HashSet::new();
        assert_eq!(obj.evaluate(50, &completed), ObjectiveStatus::InProgress);
    }

    #[test]
    fn test_evaluate_timed_exceeded() {
        let obj = Objective::Timed {
            objective: Box::new(Objective::Primitive(ObjectivePrimitive::ReachLocation {
                x: 1,
                y: 1,
            })),
            time_limit: 100,
        };
        let completed = HashSet::new();
        assert_eq!(obj.evaluate(100, &completed), ObjectiveStatus::Failed);
    }

    #[test]
    fn test_evaluate_timed_completed_in_time() {
        let obj = Objective::Timed {
            objective: Box::new(Objective::Primitive(ObjectivePrimitive::ReachLocation {
                x: 1,
                y: 1,
            })),
            time_limit: 100,
        };
        let mut completed = HashSet::new();
        completed.insert("reach_1_1".to_string());
        assert_eq!(obj.evaluate(50, &completed), ObjectiveStatus::Completed);
    }
}
