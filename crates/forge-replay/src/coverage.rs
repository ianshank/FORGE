//! Behavioral coverage measured on a [`CompactReplay`].
//!
//! Exact replay-time measurement (tiles visited, action-type histogram,
//! predicate-arm activations, seeds, constraint-violation classes). Compare
//! to ECC ([arXiv:2605.16143](https://arxiv.org/abs/2605.16143)) rather than
//! claiming a new coverage invention. The publishable property is bit-faithful
//! CompactReplay reconstruction, not a novel coverage metric.

use std::collections::{BTreeMap, BTreeSet};

use forge_types::config::GridType;
use forge_types::task::{Predicate, TaskComposition};
use forge_types::Action;
use serde::{Deserialize, Serialize};

use crate::compact::{CompactReplay, ReplayError};

/// Coverage histogram collected by replaying a compact recording.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BehavioralCoverage {
    /// Unique grid cells occupied by any agent during replay.
    pub tiles_visited: usize,
    /// Counts of decoded [`Action`] variant names.
    pub action_type_histogram: BTreeMap<String, u64>,
    /// Predicate-arm hits from attached `scenario_tasks` during replay.
    pub predicate_arm_activations: BTreeMap<String, u64>,
    /// Seeds represented by this recording (usually one).
    pub seeds: Vec<u64>,
    /// Constraint-violation class counts (battery, altitude cap, unknown ids).
    pub constraint_violations: BTreeMap<String, u64>,
}

impl BehavioralCoverage {
    /// Replays `replay` and records coverage counters.
    ///
    /// # Errors
    ///
    /// Propagates [`ReplayError`] when the recording cannot be reconstructed.
    pub fn from_replay(replay: &CompactReplay) -> Result<Self, ReplayError> {
        let mut coverage = Self {
            seeds: vec![replay.seed],
            ..Self::default()
        };
        catalog_predicate_arms(replay, &mut coverage.predicate_arm_activations);
        accumulate_action_histogram(replay, &mut coverage)?;
        accumulate_world_coverage(replay, &mut coverage)?;
        Ok(coverage)
    }
}

fn catalog_predicate_arms(replay: &CompactReplay, arms: &mut BTreeMap<String, u64>) {
    for task in &replay.config.task.scenario_tasks {
        walk_atoms(&task.goal, &mut |pred| {
            arms.entry(pred_arm(pred).to_string()).or_insert(0);
        });
    }
}

fn accumulate_action_histogram(
    replay: &CompactReplay,
    coverage: &mut BehavioralCoverage,
) -> Result<(), ReplayError> {
    let drone = replay.config.drone.enabled;
    let agri = replay.config.agri.enabled && drone;
    let hex = matches!(replay.config.world.grid_type, GridType::Hex);
    let comm = replay.config.agents.comm_vocab_size;
    let space_size = Action::space_size_full(comm, drone, agri, hex);

    for (step, ids) in replay.actions.iter().enumerate() {
        for &id in ids {
            match Action::from_discrete_full(id, comm, drone, agri, hex) {
                Some(action) => {
                    *coverage
                        .action_type_histogram
                        .entry(action_arm(&action).to_string())
                        .or_insert(0) += 1;
                }
                None => {
                    *coverage
                        .constraint_violations
                        .entry("unknown_action_id".to_string())
                        .or_insert(0) += 1;
                    return Err(ReplayError::UnknownActionId {
                        step: step as u32,
                        id,
                        space_size,
                    });
                }
            }
        }
    }
    Ok(())
}

fn accumulate_world_coverage(
    replay: &CompactReplay,
    coverage: &mut BehavioralCoverage,
) -> Result<(), ReplayError> {
    let max_altitude = replay.config.drone.max_altitude;
    let mut tiles: BTreeSet<(u16, u16)> = BTreeSet::new();
    let mut iter = replay.replay()?;

    while let Some(step) = iter.next() {
        step?;
        let world = iter.world();
        for agent in &world.agents {
            tiles.insert((agent.position.x, agent.position.y));
            if agent.battery < 0 {
                bump(&mut coverage.constraint_violations, "battery_negative");
            }
            if agent.altitude > max_altitude {
                bump(&mut coverage.constraint_violations, "altitude_cap");
            }
        }
        for task in &world.tasks {
            record_task_arms(task, &mut coverage.predicate_arm_activations);
        }
    }

    coverage.tiles_visited = tiles.len();
    Ok(())
}

fn record_task_arms(task: &forge_types::task::ActiveTask, arms: &mut BTreeMap<String, u64>) {
    let progress = task.progress.first().copied().unwrap_or(0.0);
    walk_atoms(&task.definition.goal, &mut |pred| {
        let name = pred_arm(pred);
        if task.completed {
            bump(arms, &format!("{name}:completed"));
        } else if task.failed {
            bump(arms, &format!("{name}:failed"));
        } else if progress > 0.0 {
            bump(arms, name);
        }
    });
}

fn bump(map: &mut BTreeMap<String, u64>, key: &str) {
    *map.entry(key.to_string()).or_insert(0) += 1;
}

fn walk_atoms(comp: &TaskComposition, visit: &mut impl FnMut(&Predicate)) {
    match comp {
        TaskComposition::Atom(pred) => visit(pred),
        TaskComposition::And(xs) | TaskComposition::Or(xs) | TaskComposition::Sequence(xs) => {
            for child in xs {
                walk_atoms(child, visit);
            }
        }
        TaskComposition::Before(inner, _) | TaskComposition::Without(inner, _) => {
            walk_atoms(inner, visit);
        }
        TaskComposition::While(cond, goal) => {
            walk_atoms(cond, visit);
            walk_atoms(goal, visit);
        }
        _ => {}
    }
}

fn pred_arm(pred: &Predicate) -> &'static str {
    match pred {
        Predicate::AgentAt(_, _) => "AgentAt",
        Predicate::AgentHas(_, _, _) => "AgentHas",
        Predicate::AgentNear(_, _, _) => "AgentNear",
        Predicate::ObjectAt(_, _) => "ObjectAt",
        Predicate::ObjectInState(_, _) => "ObjectInState",
        Predicate::TimeElapsed(_) => "TimeElapsed",
        Predicate::HealthAbove(_, _) => "HealthAbove",
        Predicate::ResourceCount(_, _, _) => "ResourceCount",
        Predicate::TeamAlive(_) => "TeamAlive",
        Predicate::AgentOnTerrain(_, _) => "AgentOnTerrain",
        Predicate::AgentAtAltitude(_, _) => "AgentAtAltitude",
        Predicate::BatteryAbove(_, _) => "BatteryAbove",
        Predicate::AgentAirborne(_) => "AgentAirborne",
        Predicate::AgentLanded(_) => "AgentLanded",
        Predicate::CropHealthBelow(_, _) => "CropHealthBelow",
        Predicate::DiseaseDetected(_, _) => "DiseaseDetected",
        Predicate::FieldSurveyed(_) => "FieldSurveyed",
        Predicate::SoilDataCollected(_, _) => "SoilDataCollected",
        Predicate::FieldReportGenerated(_) => "FieldReportGenerated",
        Predicate::IrrigationMapped(_) => "IrrigationMapped",
        Predicate::AreaSprayed(_) => "AreaSprayed",
        _ => "unknown",
    }
}

fn action_arm(action: &Action) -> &'static str {
    match action {
        Action::Noop => "Noop",
        Action::Move(_) => "Move",
        Action::PickUp => "PickUp",
        Action::Drop(_) => "Drop",
        Action::Use(_) => "Use",
        Action::Craft(_) => "Craft",
        Action::Push(_) => "Push",
        Action::Communicate(_) => "Communicate",
        Action::Interact => "Interact",
        Action::Ascend => "Ascend",
        Action::Descend => "Descend",
        Action::Hover => "Hover",
        Action::TakeOff => "TakeOff",
        Action::Land => "Land",
        Action::Scan(_) => "Scan",
        Action::DropPayload(_) => "DropPayload",
        Action::Spray(_) => "Spray",
        Action::ScanMultispectral => "ScanMultispectral",
        Action::ScanThermal => "ScanThermal",
        Action::RelaySoilData => "RelaySoilData",
        Action::GenerateReport => "GenerateReport",
        Action::MoveHex(_) => "MoveHex",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::CompactReplay;
    use forge_types::config::ForgeConfig;

    fn tiny_replay() -> CompactReplay {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 32;
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0]);
        builder.record_tick(vec![4]); // Move Right
        builder.record_tick(vec![0]);
        builder.build()
    }

    #[test]
    fn coverage_counts_tiles_and_action_types() {
        let replay = tiny_replay();
        let coverage = BehavioralCoverage::from_replay(&replay).unwrap();
        assert_eq!(coverage.seeds, vec![42]);
        assert!(coverage.tiles_visited >= 1);
        assert_eq!(coverage.action_type_histogram.get("Noop"), Some(&2));
        assert_eq!(coverage.action_type_histogram.get("Move"), Some(&1));
        assert!(!coverage
            .constraint_violations
            .contains_key("unknown_action_id"));
    }

    #[test]
    fn coverage_catalogs_altitude_and_report_from_scenario_tasks() {
        use forge_types::task::{Predicate, TaskComposition, TaskDefinition, TaskTier};

        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 32;
        config.task.enabled = true;
        config.task.scenario_tasks = vec![
            TaskDefinition {
                id: 1,
                description: "altitude".to_string(),
                goal: TaskComposition::Atom(Predicate::AgentAtAltitude(0, 3)),
                tier: TaskTier::new(1),
                estimated_steps: 8,
                reward: 1.0,
                dense_reward_weights: vec![1.0],
            },
            TaskDefinition {
                id: 2,
                description: "report".to_string(),
                goal: TaskComposition::Atom(Predicate::FieldReportGenerated(0)),
                tier: TaskTier::new(1),
                estimated_steps: 8,
                reward: 1.0,
                dense_reward_weights: vec![1.0],
            },
        ];
        let mut builder = CompactReplay::builder(config, 42);
        builder.record_tick(vec![0]);
        let replay = builder.build();
        let coverage = BehavioralCoverage::from_replay(&replay).unwrap();
        assert!(coverage
            .predicate_arm_activations
            .contains_key("AgentAtAltitude"));
        assert!(coverage
            .predicate_arm_activations
            .contains_key("FieldReportGenerated"));
    }

    #[test]
    fn coverage_rejects_unknown_action_ids() {
        let mut replay = tiny_replay();
        replay.actions[0] = vec![u32::MAX];
        assert!(matches!(
            BehavioralCoverage::from_replay(&replay),
            Err(ReplayError::UnknownActionId { .. })
        ));
    }
}
