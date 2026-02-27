//! Procedural task generation across tiers 1-6.
//!
//! Generates randomized tasks of calibrated difficulty using the task DSL.
//! Each tier introduces more complex composition operators and predicates.

use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{instrument, warn};

use forge_types::grid::Position;
use forge_types::resource::ItemType;
use forge_types::task::{ActiveTask, Predicate, TaskComposition, TaskDefinition};

use crate::difficulty::{estimate_difficulty, estimate_min_steps};

/// Raw resource items suitable for task generation.
const RAW_ITEMS: [ItemType; 6] = [
    ItemType::Wood,
    ItemType::Stone,
    ItemType::Ore,
    ItemType::Fish,
    ItemType::Fiber,
    ItemType::Clay,
];

/// Configuration for procedural task generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGenConfig {
    /// Maximum tier to generate (1-6).
    pub max_tier: u8,
    /// World width in tiles.
    pub world_width: u16,
    /// World height in tiles.
    pub world_height: u16,
    /// Number of agents in the simulation.
    pub num_agents: u32,
    /// Maximum number of predicates per task.
    pub max_predicates: u16,
    /// Base reward for tier 1 tasks (scales with tier).
    pub base_reward: f32,
}

impl Default for TaskGenConfig {
    fn default() -> Self {
        Self {
            max_tier: 6,
            world_width: 64,
            world_height: 64,
            num_agents: 1,
            max_predicates: 32,
            base_reward: 1.0,
        }
    }
}

/// Generates a random position within the world bounds.
fn random_position<R: Rng>(rng: &mut R, config: &TaskGenConfig) -> Position {
    let x = rng.gen_range(0..config.world_width);
    let y = rng.gen_range(0..config.world_height);
    Position::new(x, y)
}

/// Selects a random raw resource item type.
fn random_raw_item<R: Rng>(rng: &mut R) -> ItemType {
    RAW_ITEMS[rng.gen_range(0..RAW_ITEMS.len())]
}

/// Selects a random agent ID within the configured agent count.
fn random_agent_id<R: Rng>(rng: &mut R, config: &TaskGenConfig) -> u32 {
    rng.gen_range(0..config.num_agents)
}

/// Generates a task definition of the given tier.
///
/// The tier determines the structural complexity of the generated task.
/// Reward scales linearly with the tier.
#[instrument(skip_all)]
pub fn generate_task<R: Rng>(
    rng: &mut R,
    tier: u8,
    config: &TaskGenConfig,
    task_id: u64,
) -> TaskDefinition {
    let tier = tier.clamp(1, config.max_tier.clamp(1, 6));
    let goal = match tier {
        1 => generate_tier1(rng, config),
        2 => generate_tier2(rng, config),
        3 => generate_tier3(rng, config),
        4 => generate_tier4(rng, config),
        5 => generate_tier5(rng, config),
        _ => generate_tier6(rng, config),
    };

    let estimated_tier = estimate_difficulty(&goal);
    let estimated_steps = estimate_min_steps(&goal);
    let reward = config.base_reward * tier as f32;
    let description = describe_task(&goal);

    // Generate dense reward weights proportional to subtask count.
    let weight_count = count_atoms(&goal).max(1);
    let dense_reward_weights = vec![1.0 / weight_count as f32; weight_count];

    TaskDefinition {
        id: task_id,
        description,
        goal,
        tier: estimated_tier,
        estimated_steps,
        reward,
        dense_reward_weights,
    }
}

/// Generates an active task ready for evaluation.
#[instrument(skip_all)]
pub fn generate_active_task<R: Rng>(
    rng: &mut R,
    tier: u8,
    config: &TaskGenConfig,
    task_id: u64,
) -> ActiveTask {
    let definition = generate_task(rng, tier, config, task_id);
    let progress_len = count_atoms(&definition.goal).max(1);
    ActiveTask {
        definition,
        progress: vec![0.0; progress_len],
        sequence_index: 0,
        completed: false,
        failed: false,
    }
}

// ---------------------------------------------------------------------------
// Tier generators
// ---------------------------------------------------------------------------

/// Tier 1: single atomic predicate.
fn generate_tier1<R: Rng>(rng: &mut R, config: &TaskGenConfig) -> TaskComposition {
    if rng.gen_bool(0.5) {
        // Navigate to a random position
        let pos = random_position(rng, config);
        TaskComposition::Atom(Predicate::AgentAt(0, pos))
    } else {
        // Collect a resource
        let item = random_raw_item(rng);
        TaskComposition::Atom(Predicate::AgentHas(0, item, 1))
    }
}

/// Tier 2: two predicates combined with AND.
fn generate_tier2<R: Rng>(rng: &mut R, config: &TaskGenConfig) -> TaskComposition {
    if rng.gen_bool(0.5) {
        // Navigate and collect
        let pos = random_position(rng, config);
        let item = random_raw_item(rng);
        let count = rng.gen_range(1..=3);
        TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, pos)),
            TaskComposition::Atom(Predicate::AgentHas(0, item, count)),
        ])
    } else {
        // Collect two different resources
        let item1 = random_raw_item(rng);
        let mut item2 = random_raw_item(rng);
        while item2 == item1 {
            item2 = random_raw_item(rng);
        }
        let count1 = rng.gen_range(1..=3);
        let count2 = rng.gen_range(1..=3);
        TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::AgentHas(0, item1, count1)),
            TaskComposition::Atom(Predicate::AgentHas(0, item2, count2)),
        ])
    }
}

/// Tier 3: sequences, OR, or deadline constraints.
fn generate_tier3<R: Rng>(rng: &mut R, config: &TaskGenConfig) -> TaskComposition {
    let variant = rng.gen_range(0..3);
    match variant {
        0 => {
            // Sequence: collect then deliver
            let item = random_raw_item(rng);
            let count = rng.gen_range(1..=3);
            let pos = random_position(rng, config);
            TaskComposition::Sequence(vec![
                TaskComposition::Atom(Predicate::AgentHas(0, item, count)),
                TaskComposition::Atom(Predicate::AgentAt(0, pos)),
            ])
        }
        1 => {
            // OR: reach a position or collect a resource
            let pos1 = random_position(rng, config);
            TaskComposition::Or(vec![
                TaskComposition::Atom(Predicate::AgentAt(0, pos1)),
                TaskComposition::Atom(Predicate::AgentHas(0, random_raw_item(rng), 1)),
            ])
        }
        _ => {
            // Before: complete task within deadline
            let inner = generate_tier1(rng, config);
            let deadline = rng.gen_range(50..=500);
            TaskComposition::Before(Box::new(inner), deadline)
        }
    }
}

/// Tier 4: complex compositions with conditions.
fn generate_tier4<R: Rng>(rng: &mut R, config: &TaskGenConfig) -> TaskComposition {
    let variant = rng.gen_range(0..3);
    match variant {
        0 => {
            // While: maintain health above threshold while completing task
            let agent_id = random_agent_id(rng, config);
            let threshold = rng.gen_range(3..=8) as f32 / 10.0;
            let inner_task = generate_tier2(rng, config);
            TaskComposition::While(
                Box::new(TaskComposition::Atom(Predicate::HealthAbove(
                    agent_id, threshold,
                ))),
                Box::new(inner_task),
            )
        }
        1 => {
            // Without: complete task without using Noop (action 0)
            let inner_task = generate_tier2(rng, config);
            TaskComposition::Without(Box::new(inner_task), 0)
        }
        _ => {
            // Multi-step with deadline
            let item = random_raw_item(rng);
            let count = rng.gen_range(2..=5);
            let pos = random_position(rng, config);
            let deadline = rng.gen_range(200..=1000);
            TaskComposition::Before(
                Box::new(TaskComposition::Sequence(vec![
                    TaskComposition::Atom(Predicate::AgentHas(0, item, count)),
                    TaskComposition::Atom(Predicate::AgentAt(0, pos)),
                ])),
                deadline,
            )
        }
    }
}

/// Tier 5: multi-agent tasks (requires num_agents >= 2).
fn generate_tier5<R: Rng>(rng: &mut R, config: &TaskGenConfig) -> TaskComposition {
    // Fall back to tier 4 if only one agent
    if config.num_agents < 2 {
        return generate_tier4(rng, config);
    }

    let variant = rng.gen_range(0..2);
    match variant {
        0 => {
            // Two agents must be near each other
            let dist = rng.gen_range(1..=5);
            let item = random_raw_item(rng);
            let count = rng.gen_range(1..=3);
            TaskComposition::And(vec![
                TaskComposition::Atom(Predicate::AgentNear(0, 1, dist)),
                TaskComposition::Atom(Predicate::AgentHas(0, item, count)),
            ])
        }
        _ => {
            // Both agents collect resources
            let item1 = random_raw_item(rng);
            let item2 = random_raw_item(rng);
            let count1 = rng.gen_range(1..=3);
            let count2 = rng.gen_range(1..=3);
            TaskComposition::And(vec![
                TaskComposition::Atom(Predicate::AgentHas(0, item1, count1)),
                TaskComposition::Atom(Predicate::AgentHas(1, item2, count2)),
            ])
        }
    }
}

/// Tier 6: maximum complexity with all operators combined.
fn generate_tier6<R: Rng>(rng: &mut R, config: &TaskGenConfig) -> TaskComposition {
    let agent_id: u32 = 0;
    let item1 = random_raw_item(rng);
    let item2 = random_raw_item(rng);
    let pos = random_position(rng, config);
    let deadline = rng.gen_range(500..=2000);

    // Build nested composition: While(health > threshold, Before(deadline,
    //   Sequence([collect item1, Without(collect item2, noop), go to pos])))
    let threshold = rng.gen_range(3..=7) as f32 / 10.0;
    let count1 = rng.gen_range(2..=5);
    let count2 = rng.gen_range(1..=3);

    let inner_sequence = TaskComposition::Sequence(vec![
        TaskComposition::Atom(Predicate::AgentHas(agent_id, item1, count1)),
        TaskComposition::Without(
            Box::new(TaskComposition::Atom(Predicate::AgentHas(
                agent_id, item2, count2,
            ))),
            0, // forbidden: Noop
        ),
        TaskComposition::Atom(Predicate::AgentAt(agent_id, pos)),
    ]);

    let with_deadline = TaskComposition::Before(Box::new(inner_sequence), deadline);

    // Add multi-agent requirement if available
    if config.num_agents >= 2 {
        let dist = rng.gen_range(1..=5);
        TaskComposition::While(
            Box::new(TaskComposition::Atom(Predicate::HealthAbove(
                agent_id, threshold,
            ))),
            Box::new(TaskComposition::And(vec![
                with_deadline,
                TaskComposition::Atom(Predicate::AgentNear(0, 1, dist)),
            ])),
        )
    } else {
        TaskComposition::While(
            Box::new(TaskComposition::Atom(Predicate::HealthAbove(
                agent_id, threshold,
            ))),
            Box::new(with_deadline),
        )
    }
}

// ---------------------------------------------------------------------------
// Description generation
// ---------------------------------------------------------------------------

/// Generates a human-readable description of a task composition.
#[instrument(skip_all)]
pub fn describe_task(goal: &TaskComposition) -> String {
    match goal {
        TaskComposition::Atom(pred) => describe_predicate(pred),

        TaskComposition::And(subtasks) => {
            let parts: Vec<String> = subtasks.iter().map(describe_task).collect();
            parts.join(" AND ")
        }

        TaskComposition::Or(subtasks) => {
            let parts: Vec<String> = subtasks.iter().map(describe_task).collect();
            parts.join(" OR ")
        }

        TaskComposition::Sequence(subtasks) => {
            let parts: Vec<String> = subtasks
                .iter()
                .enumerate()
                .map(|(i, t)| format!("{}. {}", i + 1, describe_task(t)))
                .collect();
            format!("In order: {}", parts.join(", then "))
        }

        TaskComposition::Before(subtask, deadline) => {
            format!("{} before tick {}", describe_task(subtask), deadline)
        }

        TaskComposition::While(condition, goal) => {
            format!(
                "While {}, {}",
                describe_task(condition),
                describe_task(goal)
            )
        }

        TaskComposition::Without(subtask, action_id) => {
            format!(
                "{} without using action {}",
                describe_task(subtask),
                action_id
            )
        }

        _ => {
            warn!("unknown TaskComposition variant in describe_task");
            "unknown task".to_string()
        }
    }
}

/// Generates a human-readable description of a predicate.
fn describe_predicate(pred: &Predicate) -> String {
    match pred {
        Predicate::AgentAt(id, pos) => {
            format!("agent {} reaches ({}, {})", id, pos.x, pos.y)
        }
        Predicate::AgentHas(id, item, count) => {
            format!("agent {} collects {} {:?}", id, count, item)
        }
        Predicate::AgentNear(a, b, dist) => {
            format!("agent {} is within {} of agent {}", a, dist, b)
        }
        Predicate::ObjectAt(id, pos) => {
            format!("object {} is at ({}, {})", id, pos.x, pos.y)
        }
        Predicate::ObjectInState(id, state) => {
            format!("object {} is in state {}", id, state)
        }
        Predicate::TimeElapsed(ticks) => {
            format!("{} ticks have elapsed", ticks)
        }
        Predicate::HealthAbove(id, threshold) => {
            format!("agent {} health above {:.0}%", id, threshold * 100.0)
        }
        Predicate::ResourceCount(id, item, count) => {
            format!("agent {} has {} {:?}", id, count, item)
        }
        Predicate::TeamAlive(team) => {
            format!("team {} is alive", team)
        }
        Predicate::AgentOnTerrain(id, terrain) => {
            format!("agent {} is on terrain {}", id, terrain)
        }

        _ => {
            warn!("unknown Predicate variant in describe_predicate");
            "unknown predicate".to_string()
        }
    }
}

/// Counts the number of atomic predicates in a composition tree.
fn count_atoms(composition: &TaskComposition) -> usize {
    match composition {
        TaskComposition::Atom(_) => 1,
        TaskComposition::And(subtasks)
        | TaskComposition::Or(subtasks)
        | TaskComposition::Sequence(subtasks) => subtasks.iter().map(count_atoms).sum(),
        TaskComposition::Before(subtask, _) => count_atoms(subtask),
        TaskComposition::While(cond, goal) => count_atoms(cond) + count_atoms(goal),
        TaskComposition::Without(subtask, _) => count_atoms(subtask),
        _ => {
            warn!("unknown TaskComposition variant in count_atoms");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn default_config() -> TaskGenConfig {
        TaskGenConfig {
            max_tier: 6,
            world_width: 64,
            world_height: 64,
            num_agents: 2,
            max_predicates: 32,
            base_reward: 1.0,
        }
    }

    fn make_rng(seed: u64) -> Pcg64 {
        Pcg64::seed_from_u64(seed)
    }

    #[test]
    fn test_generate_tier1() {
        let config = default_config();
        let mut rng = make_rng(42);
        for i in 0..10 {
            let task = generate_task(&mut rng, 1, &config, i);
            assert!(task.tier.value() <= 2, "tier 1 task should be easy");
            assert!(!task.description.is_empty());
        }
    }

    #[test]
    fn test_generate_tier2() {
        let config = default_config();
        let mut rng = make_rng(123);
        for i in 0..10 {
            let task = generate_task(&mut rng, 2, &config, i);
            assert!(task.tier.value() >= 1);
            assert!(!task.description.is_empty());
        }
    }

    #[test]
    fn test_generate_tier3() {
        let config = default_config();
        let mut rng = make_rng(456);
        for i in 0..10 {
            let task = generate_task(&mut rng, 3, &config, i);
            assert!(!task.description.is_empty());
        }
    }

    #[test]
    fn test_generate_tier4() {
        let config = default_config();
        let mut rng = make_rng(789);
        for i in 0..10 {
            let task = generate_task(&mut rng, 4, &config, i);
            assert!(!task.description.is_empty());
        }
    }

    #[test]
    fn test_generate_tier5_multi_agent() {
        let config = TaskGenConfig {
            num_agents: 2,
            ..default_config()
        };
        let mut rng = make_rng(101);
        for i in 0..10 {
            let task = generate_task(&mut rng, 5, &config, i);
            assert!(!task.description.is_empty());
        }
    }

    #[test]
    fn test_generate_tier5_single_agent_fallback() {
        let config = TaskGenConfig {
            num_agents: 1,
            ..default_config()
        };
        let mut rng = make_rng(202);
        let task = generate_task(&mut rng, 5, &config, 0);
        // Should fall back to tier 4 complexity
        assert!(!task.description.is_empty());
    }

    #[test]
    fn test_generate_tier6() {
        let config = default_config();
        let mut rng = make_rng(303);
        for i in 0..10 {
            let task = generate_task(&mut rng, 6, &config, i);
            assert!(!task.description.is_empty());
            assert!(task.estimated_steps >= 1);
        }
    }

    #[test]
    fn test_generate_active_task() {
        let config = default_config();
        let mut rng = make_rng(404);
        let active = generate_active_task(&mut rng, 3, &config, 1);
        assert!(!active.completed);
        assert!(!active.failed);
        assert_eq!(active.sequence_index, 0);
        assert!(!active.progress.is_empty());
        assert!(active.progress.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn test_tier_clamping() {
        let config = TaskGenConfig {
            max_tier: 3,
            ..default_config()
        };
        let mut rng = make_rng(505);
        // Requesting tier 6 but max_tier is 3; should clamp to 3
        let task = generate_task(&mut rng, 6, &config, 0);
        assert!(!task.description.is_empty());
    }

    #[test]
    fn test_generate_task_max_tier_zero() {
        let config = TaskGenConfig {
            max_tier: 0,
            ..default_config()
        };
        let mut rng = make_rng(42);
        // max_tier=0 means the tier clamp range is 1..min(0, 6) = 1..0,
        // so clamp(1, 0) yields 0 which then goes to the default arm.
        // The function should not panic; we just verify it produces a valid task.
        let task = generate_task(&mut rng, 1, &config, 0);
        assert!(!task.description.is_empty());
        assert!(task.estimated_steps >= 1);
    }

    #[test]
    fn test_reward_scales_with_tier() {
        let config = default_config();
        let mut rng = make_rng(606);
        let task1 = generate_task(&mut rng, 1, &config, 0);
        let task3 = generate_task(&mut rng, 3, &config, 1);
        let task6 = generate_task(&mut rng, 6, &config, 2);
        assert!(task3.reward > task1.reward);
        assert!(task6.reward > task3.reward);
    }

    #[test]
    fn test_describe_task_atom() {
        let task = TaskComposition::Atom(Predicate::AgentAt(0, Position::new(10, 20)));
        let desc = describe_task(&task);
        assert!(desc.contains("agent 0"));
        assert!(desc.contains("10"));
        assert!(desc.contains("20"));
    }

    #[test]
    fn test_describe_task_and() {
        let task = TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 3)),
        ]);
        let desc = describe_task(&task);
        assert!(desc.contains("AND"));
    }

    #[test]
    fn test_describe_task_sequence() {
        let task = TaskComposition::Sequence(vec![
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Stone, 2)),
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(0, 0))),
        ]);
        let desc = describe_task(&task);
        assert!(desc.contains("In order"));
    }

    #[test]
    fn test_describe_task_before() {
        let task = TaskComposition::Before(
            Box::new(TaskComposition::Atom(Predicate::AgentAt(
                0,
                Position::new(5, 5),
            ))),
            100,
        );
        let desc = describe_task(&task);
        assert!(desc.contains("before tick 100"));
    }

    #[test]
    fn test_random_position_in_bounds() {
        let config = TaskGenConfig {
            world_width: 16,
            world_height: 16,
            ..default_config()
        };
        let mut rng = make_rng(707);
        for _ in 0..100 {
            let pos = random_position(&mut rng, &config);
            assert!(pos.x < 16);
            assert!(pos.y < 16);
        }
    }

    #[test]
    fn test_deterministic_generation() {
        let config = default_config();
        let mut rng1 = make_rng(999);
        let mut rng2 = make_rng(999);
        let task1 = generate_task(&mut rng1, 3, &config, 0);
        let task2 = generate_task(&mut rng2, 3, &config, 0);
        assert_eq!(task1.description, task2.description);
        assert_eq!(task1.goal, task2.goal);
    }

    #[test]
    fn test_dense_reward_weights_sum() {
        let config = default_config();
        let mut rng = make_rng(808);
        for tier in 1..=6 {
            let task = generate_task(&mut rng, tier, &config, tier as u64);
            let sum: f32 = task.dense_reward_weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "dense reward weights should sum to ~1.0 for tier {}, got {}",
                tier,
                sum
            );
        }
    }

    #[test]
    fn test_count_atoms() {
        let task = TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(0, 0))),
            TaskComposition::Sequence(vec![
                TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 1)),
                TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Stone, 1)),
            ]),
        ]);
        assert_eq!(count_atoms(&task), 3);
    }
}
