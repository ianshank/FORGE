//! Task DSL mapping: FORGE task compositions → MangoMAS mission IR.
//!
//! Converts FORGE's composable task operators (Sequence, While, Before, etc.)
//! into a JSON intermediate representation that MangoMAS can consume
//! for mission decomposition via its LLMGateway.

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::{MangoMasError, MangoMasResult};

/// MangoMAS mission intermediate representation.
///
/// A tree structure representing a mission plan that MangoMAS
/// can decompose into goal vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MissionIR {
    /// Single atomic objective.
    #[serde(rename = "atomic")]
    Atomic {
        /// Predicate name (e.g., "agent_at", "has_item").
        predicate: String,
        /// Parameters for the predicate.
        params: serde_json::Value,
    },
    /// All sub-missions must be completed simultaneously.
    #[serde(rename = "all")]
    All {
        /// Sub-missions to complete.
        missions: Vec<MissionIR>,
    },
    /// At least one sub-mission must be completed.
    #[serde(rename = "any")]
    Any {
        /// Sub-missions (any one suffices).
        missions: Vec<MissionIR>,
    },
    /// Sub-missions must be completed in order.
    #[serde(rename = "sequence")]
    Sequence {
        /// Ordered steps to complete.
        steps: Vec<MissionIR>,
    },
    /// Mission must complete before a deadline.
    #[serde(rename = "deadline")]
    Deadline {
        /// The mission to complete.
        mission: Box<MissionIR>,
        /// Deadline in ticks.
        deadline_ticks: u64,
    },
    /// Condition must remain true while pursuing the goal.
    #[serde(rename = "conditional")]
    Conditional {
        /// Condition that must hold.
        condition: Box<MissionIR>,
        /// Goal to achieve while condition holds.
        goal: Box<MissionIR>,
    },
    /// Goal must be achieved without performing forbidden action.
    #[serde(rename = "forbidden")]
    Forbidden {
        /// Goal to achieve.
        goal: Box<MissionIR>,
        /// Forbidden action ID.
        forbidden_action: u32,
    },
}

/// Maps FORGE task DSL concepts to MangoMAS mission IR.
pub struct TaskDslMapper;

impl TaskDslMapper {
    /// Creates a new mapper.
    pub fn new() -> Self {
        Self
    }

    /// Serializes a mission IR to JSON.
    #[instrument(skip_all)]
    pub fn to_json(&self, mission: &MissionIR) -> MangoMasResult<String> {
        serde_json::to_string_pretty(mission)
            .map_err(|e| MangoMasError::Curriculum(format!("mission serialization failed: {}", e)))
    }

    /// Parses a mission IR from JSON.
    #[instrument(skip_all)]
    pub fn from_json(&self, json: &str) -> MangoMasResult<MissionIR> {
        serde_json::from_str(json)
            .map_err(|e| MangoMasError::Curriculum(format!("mission parse failed: {}", e)))
    }

    /// Creates a navigation waypoint mission.
    pub fn navigate_to(&self, x: u16, y: u16) -> MissionIR {
        MissionIR::Atomic {
            predicate: "agent_at".to_string(),
            params: serde_json::json!({"x": x, "y": y}),
        }
    }

    /// Creates a resource gathering mission.
    pub fn gather_resource(&self, resource_type: &str, count: u16) -> MissionIR {
        MissionIR::Atomic {
            predicate: "has_resource".to_string(),
            params: serde_json::json!({"resource": resource_type, "count": count}),
        }
    }

    /// Creates a multi-waypoint patrol mission.
    pub fn patrol_route(&self, waypoints: &[(u16, u16)]) -> MissionIR {
        let steps = waypoints
            .iter()
            .map(|&(x, y)| self.navigate_to(x, y))
            .collect();
        MissionIR::Sequence { steps }
    }

    /// Creates a battery-aware mission (maintain battery above threshold).
    pub fn battery_aware_mission(&self, goal: MissionIR, min_battery: f32) -> MissionIR {
        MissionIR::Conditional {
            condition: Box::new(MissionIR::Atomic {
                predicate: "battery_above".to_string(),
                params: serde_json::json!({"threshold": min_battery}),
            }),
            goal: Box::new(goal),
        }
    }

    /// Creates a timed mission with deadline.
    pub fn timed_mission(&self, goal: MissionIR, deadline_ticks: u64) -> MissionIR {
        MissionIR::Deadline {
            mission: Box::new(goal),
            deadline_ticks,
        }
    }
}

impl Default for TaskDslMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigate_to_serialization() {
        let mapper = TaskDslMapper::new();
        let mission = mapper.navigate_to(10, 20);
        let json = mapper.to_json(&mission).unwrap();
        assert!(json.contains("agent_at"));
        assert!(json.contains("10"));
    }

    #[test]
    fn test_roundtrip_serialization() {
        let mapper = TaskDslMapper::new();
        let mission = mapper.patrol_route(&[(5, 5), (10, 10), (15, 15)]);
        let json = mapper.to_json(&mission).unwrap();
        let recovered = mapper.from_json(&json).unwrap();
        let json2 = mapper.to_json(&recovered).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn test_sequence_has_correct_steps() {
        let mapper = TaskDslMapper::new();
        let mission = mapper.patrol_route(&[(1, 2), (3, 4)]);
        match mission {
            MissionIR::Sequence { steps } => assert_eq!(steps.len(), 2),
            _ => panic!("expected Sequence"),
        }
    }

    #[test]
    fn test_battery_aware_mission() {
        let mapper = TaskDslMapper::new();
        let goal = mapper.navigate_to(10, 10);
        let mission = mapper.battery_aware_mission(goal, 0.2);
        match mission {
            MissionIR::Conditional { condition, goal: _ } => {
                let json = mapper.to_json(&condition).unwrap();
                assert!(json.contains("battery_above"));
            }
            _ => panic!("expected Conditional"),
        }
    }

    #[test]
    fn test_timed_mission() {
        let mapper = TaskDslMapper::new();
        let goal = mapper.navigate_to(10, 10);
        let mission = mapper.timed_mission(goal, 500);
        match mission {
            MissionIR::Deadline { deadline_ticks, .. } => assert_eq!(deadline_ticks, 500),
            _ => panic!("expected Deadline"),
        }
    }

    #[test]
    fn test_complex_nested_mission() {
        let mapper = TaskDslMapper::new();
        let patrol = mapper.patrol_route(&[(5, 5), (10, 10)]);
        let timed = mapper.timed_mission(patrol, 1000);
        let safe = mapper.battery_aware_mission(timed, 0.3);
        let json = mapper.to_json(&safe).unwrap();
        assert!(json.contains("conditional"));
        assert!(json.contains("deadline"));
        assert!(json.contains("sequence"));
    }
}
