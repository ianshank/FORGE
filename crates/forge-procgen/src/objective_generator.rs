//! Random objective tree generation.
//!
//! Composes objective trees from the grammar DSL with configurable
//! depth, breadth, and difficulty scaling.

use crate::grammar::{Objective, ObjectivePrimitive};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Configuration for random objective generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectiveGenConfig {
    /// Maximum depth of the objective tree.
    pub max_depth: u32,
    /// Maximum branching factor at each combinator node.
    pub max_breadth: u32,
    /// Difficulty tier (1-6). Higher tiers produce deeper, more complex objectives.
    pub difficulty_tier: u8,
    /// Range of time limits for timed objectives `(min_ticks, max_ticks)`.
    pub time_limit_range: (u64, u64),
    /// Probability of emitting a primitive node instead of a combinator at non-root depth.
    pub primitive_probability: f64,
    /// Inclusive range for the hold-area radius `(min, max)`.
    pub hold_radius_range: (u16, u16),
    /// Range for hold-area duration in ticks `(min, max)`.
    pub hold_duration_range: (u64, u64),
    /// Range for survive duration in ticks `(min, max)`.
    pub survive_duration_range: (u64, u64),
    /// Range for resource collection count `(min, max)`.
    pub collect_count_range: (u32, u32),
    /// Available resource type names for `CollectResource` objectives.
    pub resource_types: Vec<String>,
}

impl Default for ObjectiveGenConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_breadth: 3,
            difficulty_tier: 1,
            time_limit_range: (1000, 5000),
            primitive_probability: 0.4,
            hold_radius_range: (1, 5),
            hold_duration_range: (50, 500),
            survive_duration_range: (100, 1000),
            collect_count_range: (1, 20),
            resource_types: vec![
                "gold".to_string(),
                "wood".to_string(),
                "stone".to_string(),
                "food".to_string(),
            ],
        }
    }
}

/// Generates a random objective tree from the grammar.
///
/// Uses seed-based RNG for deterministic generation. Higher difficulty tiers
/// produce deeper trees with more complex compositions.
#[instrument(skip(config))]
pub fn generate_objective(
    config: &ObjectiveGenConfig,
    grid_width: u16,
    grid_height: u16,
    seed: u64,
) -> Objective {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let effective_depth = config.max_depth.min(config.difficulty_tier as u32 + 1);
    build_objective(
        &mut rng,
        config,
        grid_width,
        grid_height,
        0,
        effective_depth,
    )
}

/// Recursively builds an objective tree.
fn build_objective(
    rng: &mut Pcg64Mcg,
    config: &ObjectiveGenConfig,
    grid_width: u16,
    grid_height: u16,
    depth: u32,
    max_depth: u32,
) -> Objective {
    // At max depth or with some probability, emit a primitive
    if depth >= max_depth || (depth > 0 && rng.gen_bool(config.primitive_probability)) {
        return Objective::Primitive(random_primitive(rng, config, grid_width, grid_height));
    }

    let breadth = rng.gen_range(2..=config.max_breadth.max(2));
    let children: Vec<Objective> = (0..breadth)
        .map(|_| build_objective(rng, config, grid_width, grid_height, depth + 1, max_depth))
        .collect();

    // Choose a combinator type
    let combinator = rng.gen_range(0u8..4);
    match combinator {
        0 => Objective::And(children),
        1 => Objective::Or(children),
        2 => Objective::Sequence(children),
        3 => {
            let time_limit = rng.gen_range(config.time_limit_range.0..=config.time_limit_range.1);
            let inner = Objective::And(children);
            Objective::Timed {
                objective: Box::new(inner),
                time_limit,
            }
        }
        _ => unreachable!(),
    }
}

/// Generates a random primitive objective within grid bounds.
fn random_primitive(
    rng: &mut Pcg64Mcg,
    config: &ObjectiveGenConfig,
    grid_width: u16,
    grid_height: u16,
) -> ObjectivePrimitive {
    let w = grid_width.max(1);
    let h = grid_height.max(1);
    match rng.gen_range(0u8..5) {
        0 => ObjectivePrimitive::ReachLocation {
            x: rng.gen_range(0..w),
            y: rng.gen_range(0..h),
        },
        1 => ObjectivePrimitive::HoldArea {
            x: rng.gen_range(0..w),
            y: rng.gen_range(0..h),
            radius: rng.gen_range(config.hold_radius_range.0..=config.hold_radius_range.1),
            duration_ticks: rng
                .gen_range(config.hold_duration_range.0..config.hold_duration_range.1),
        },
        2 => ObjectivePrimitive::EliminateTarget {
            target_id: rng.gen_range(0..10),
        },
        3 => {
            if config.resource_types.is_empty() {
                // Fall back to Survive when no resource types are configured.
                ObjectivePrimitive::Survive {
                    duration_ticks: rng.gen_range(
                        config.survive_duration_range.0..config.survive_duration_range.1,
                    ),
                }
            } else {
                ObjectivePrimitive::CollectResource {
                    resource_type: config.resource_types
                        [rng.gen_range(0..config.resource_types.len())]
                    .clone(),
                    count: rng
                        .gen_range(config.collect_count_range.0..config.collect_count_range.1),
                }
            }
        }
        4 => ObjectivePrimitive::Survive {
            duration_ticks: rng
                .gen_range(config.survive_duration_range.0..config.survive_duration_range.1),
        },
        _ => unreachable!(),
    }
}

/// Returns the depth of an objective tree.
#[cfg(test)]
fn objective_depth(obj: &Objective) -> u32 {
    match obj {
        Objective::Primitive(_) => 1,
        Objective::And(children) | Objective::Or(children) | Objective::Sequence(children) => {
            1 + children.iter().map(objective_depth).max().unwrap_or(0)
        }
        Objective::Timed { objective, .. } => 1 + objective_depth(objective),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_deterministic() {
        let config = ObjectiveGenConfig::default();
        let a = generate_objective(&config, 64, 64, 42);
        let b = generate_objective(&config, 64, 64, 42);
        let a_json = serde_json::to_string(&a).unwrap();
        let b_json = serde_json::to_string(&b).unwrap();
        assert_eq!(a_json, b_json);
    }

    #[test]
    fn test_generate_different_seeds() {
        let config = ObjectiveGenConfig::default();
        let a = generate_objective(&config, 64, 64, 1);
        let b = generate_objective(&config, 64, 64, 2);
        let a_json = serde_json::to_string(&a).unwrap();
        let b_json = serde_json::to_string(&b).unwrap();
        assert_ne!(a_json, b_json);
    }

    #[test]
    fn test_generate_reasonable_depth() {
        let config = ObjectiveGenConfig {
            max_depth: 3,
            difficulty_tier: 3,
            ..Default::default()
        };
        let obj = generate_objective(&config, 64, 64, 42);
        let depth = objective_depth(&obj);
        // Depth should not exceed max_depth + some margin for nesting
        assert!(depth <= 10, "Depth {depth} is unreasonably large");
    }

    #[test]
    fn test_generate_valid_objective() {
        let config = ObjectiveGenConfig::default();
        let obj = generate_objective(&config, 32, 32, 99);
        // Should serialize without error
        let json = serde_json::to_string(&obj).unwrap();
        assert!(!json.is_empty());
    }

    #[test]
    fn test_timed_with_zero_time_limit() {
        // With time_limit=0, evaluating at tick=0 should fail (tick >= time_limit).
        let obj = Objective::Timed {
            objective: Box::new(Objective::Primitive(ObjectivePrimitive::ReachLocation {
                x: 0,
                y: 0,
            })),
            time_limit: 0,
        };
        let completed = std::collections::HashSet::new();
        // At tick=0 with time_limit=0, the inner is InProgress but tick >= time_limit => Failed
        assert_eq!(
            obj.evaluate(0, &completed),
            crate::grammar::ObjectiveStatus::Failed
        );
    }

    #[test]
    fn test_deeply_nested_objective_depth_5() {
        let config = ObjectiveGenConfig {
            max_depth: 5,
            difficulty_tier: 5,
            primitive_probability: 0.0, // Maximize nesting
            max_breadth: 2,
            ..Default::default()
        };
        let obj = generate_objective(&config, 32, 32, 42);
        let depth = objective_depth(&obj);
        // With max_depth=5 and difficulty_tier=5, effective_depth = min(5, 6) = 5
        // Depth should be > 1 since primitive_probability is 0 and we have room.
        assert!(depth > 1, "Expected nested objective, got depth={depth}");
    }

    #[test]
    fn test_higher_difficulty_potentially_deeper() {
        let low = ObjectiveGenConfig {
            difficulty_tier: 1,
            ..Default::default()
        };
        let high = ObjectiveGenConfig {
            difficulty_tier: 5,
            max_depth: 6,
            ..Default::default()
        };
        // Generate many and check average depth tends to be higher for high difficulty
        let mut low_depths = 0u32;
        let mut high_depths = 0u32;
        for seed in 0..50 {
            low_depths += objective_depth(&generate_objective(&low, 64, 64, seed));
            high_depths += objective_depth(&generate_objective(&high, 64, 64, seed));
        }
        // High difficulty should generally produce deeper trees
        assert!(
            high_depths >= low_depths,
            "Higher difficulty should produce at least as deep trees: low={low_depths}, high={high_depths}"
        );
    }
}
