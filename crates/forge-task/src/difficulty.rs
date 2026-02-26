//! Task difficulty estimation.
//!
//! Estimates how hard a task is based on its composition structure,
//! assigning it to a difficulty tier (1-6).

use forge_types::task::{TaskComposition, TaskTier};

/// Estimates the difficulty tier of a task composition.
pub fn estimate_difficulty(composition: &TaskComposition) -> TaskTier {
    let score = difficulty_score(composition);
    let tier = match score {
        0..=1 => 1,
        2..=3 => 2,
        4..=6 => 3,
        7..=10 => 4,
        11..=15 => 5,
        _ => 6,
    };
    TaskTier::new(tier)
}

/// Computes a raw difficulty score for a composition tree.
fn difficulty_score(composition: &TaskComposition) -> u32 {
    match composition {
        TaskComposition::Atom(_) => 1,

        TaskComposition::And(subtasks) => {
            let sum: u32 = subtasks.iter().map(difficulty_score).sum();
            sum + subtasks.len().saturating_sub(1) as u32
        }

        TaskComposition::Or(subtasks) => {
            // Or is easier than And — take the minimum subtask difficulty
            subtasks.iter().map(difficulty_score).min().unwrap_or(0)
        }

        TaskComposition::Sequence(subtasks) => {
            let sum: u32 = subtasks.iter().map(difficulty_score).sum();
            // Sequences are harder because ordering matters
            sum + subtasks.len() as u32
        }

        TaskComposition::Before(subtask, _deadline) => {
            // Deadline adds pressure
            difficulty_score(subtask) + 2
        }

        TaskComposition::While(condition, goal) => {
            // Maintaining a condition while achieving a goal is harder
            difficulty_score(condition) + difficulty_score(goal) + 3
        }

        TaskComposition::Without(subtask, _forbidden) => {
            // Constraints add difficulty
            difficulty_score(subtask) + 1
        }
    }
}

/// Estimates the minimum number of actions an oracle agent would need.
pub fn estimate_min_steps(composition: &TaskComposition) -> u32 {
    match composition {
        TaskComposition::Atom(_) => 1,
        TaskComposition::And(subtasks) => subtasks.iter().map(estimate_min_steps).sum(),
        TaskComposition::Or(subtasks) => subtasks.iter().map(estimate_min_steps).min().unwrap_or(0),
        TaskComposition::Sequence(subtasks) => subtasks.iter().map(estimate_min_steps).sum(),
        TaskComposition::Before(subtask, _) => estimate_min_steps(subtask),
        TaskComposition::While(_, goal) => estimate_min_steps(goal),
        TaskComposition::Without(subtask, _) => estimate_min_steps(subtask),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::grid::Position;
    use forge_types::resource::ItemType;
    use forge_types::task::Predicate;

    #[test]
    fn test_single_predicate_tier_1() {
        let task = TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5)));
        let tier = estimate_difficulty(&task);
        assert_eq!(tier.value(), 1);
    }

    #[test]
    fn test_and_increases_difficulty() {
        let task = TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 3)),
        ]);
        let tier = estimate_difficulty(&task);
        assert!(tier.value() >= 2);
    }

    #[test]
    fn test_sequence_harder_than_and() {
        let subtasks = vec![
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 3)),
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Axe, 1)),
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
        ];
        let and_task = TaskComposition::And(subtasks.clone());
        let seq_task = TaskComposition::Sequence(subtasks);

        let and_tier = estimate_difficulty(&and_task);
        let seq_tier = estimate_difficulty(&seq_task);
        assert!(seq_tier.value() >= and_tier.value());
    }

    #[test]
    fn test_or_reduces_difficulty() {
        let task = TaskComposition::Or(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(0, 0))),
        ]);
        let tier = estimate_difficulty(&task);
        assert_eq!(tier.value(), 1); // easiest subtask
    }

    #[test]
    fn test_deadline_increases_difficulty() {
        let base = TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5)));
        let with_deadline = TaskComposition::Before(Box::new(base.clone()), 100);
        let base_tier = estimate_difficulty(&base);
        let deadline_tier = estimate_difficulty(&with_deadline);
        assert!(deadline_tier.value() >= base_tier.value());
    }

    #[test]
    fn test_estimate_min_steps() {
        let task = TaskComposition::Sequence(vec![
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 3)),
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
        ]);
        let steps = estimate_min_steps(&task);
        assert_eq!(steps, 2); // at least 2 actions
    }
}
