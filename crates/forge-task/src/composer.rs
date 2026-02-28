//! Task composition evaluation.
//!
//! Evaluates composite tasks built from atomic predicates using
//! logical operators (AND, OR, SEQUENCE, BEFORE, WHILE, WITHOUT).

use forge_types::task::TaskComposition;
use tracing::{instrument, trace, warn};

use crate::predicate::{evaluate_predicate, EvalContext, PredicateResult};

/// Evaluates a composite task against the current world state.
///
/// Returns the overall satisfaction and progress of the task tree.
#[instrument(skip_all)]
pub fn evaluate_composition(
    composition: &TaskComposition,
    ctx: &EvalContext,
    sequence_index: &mut usize,
    forbidden_actions: &[u32],
) -> PredicateResult {
    match composition {
        TaskComposition::Atom(predicate) => evaluate_predicate(predicate, ctx),

        TaskComposition::And(subtasks) => {
            if subtasks.is_empty() {
                return PredicateResult {
                    satisfied: true,
                    progress: 1.0,
                };
            }
            let results: Vec<PredicateResult> = subtasks
                .iter()
                .map(|t| evaluate_composition(t, ctx, sequence_index, forbidden_actions))
                .collect();
            let all_satisfied = results.iter().all(|r| r.satisfied);
            let avg_progress =
                results.iter().map(|r| r.progress).sum::<f32>() / results.len() as f32;
            PredicateResult {
                satisfied: all_satisfied,
                progress: avg_progress,
            }
        }

        TaskComposition::Or(subtasks) => {
            if subtasks.is_empty() {
                return PredicateResult {
                    satisfied: false,
                    progress: 0.0,
                };
            }
            let results: Vec<PredicateResult> = subtasks
                .iter()
                .map(|t| evaluate_composition(t, ctx, sequence_index, forbidden_actions))
                .collect();
            let any_satisfied = results.iter().any(|r| r.satisfied);
            let max_progress = results.iter().map(|r| r.progress).fold(0.0_f32, f32::max);
            PredicateResult {
                satisfied: any_satisfied,
                progress: max_progress,
            }
        }

        TaskComposition::Sequence(subtasks) => {
            if subtasks.is_empty() {
                return PredicateResult {
                    satisfied: true,
                    progress: 1.0,
                };
            }

            let current_idx = *sequence_index;
            if current_idx >= subtasks.len() {
                return PredicateResult {
                    satisfied: true,
                    progress: 1.0,
                };
            }

            let current_result = evaluate_composition(
                &subtasks[current_idx],
                ctx,
                sequence_index,
                forbidden_actions,
            );

            if current_result.satisfied {
                trace!(step = current_idx, "sequence step completed");
                *sequence_index = current_idx + 1;
                if *sequence_index >= subtasks.len() {
                    return PredicateResult {
                        satisfied: true,
                        progress: 1.0,
                    };
                }
            }

            let completed_steps = *sequence_index;
            let total_steps = subtasks.len();
            let step_progress = current_result.progress;
            let overall_progress = (completed_steps as f32 + step_progress) / total_steps as f32;

            PredicateResult {
                satisfied: false,
                progress: overall_progress,
            }
        }

        TaskComposition::Before(subtask, deadline) => {
            if ctx.tick > *deadline {
                // Deadline passed — task failed
                PredicateResult {
                    satisfied: false,
                    progress: 0.0,
                }
            } else {
                evaluate_composition(subtask, ctx, sequence_index, forbidden_actions)
            }
        }

        TaskComposition::While(condition, goal) => {
            let cond_result =
                evaluate_composition(condition, ctx, sequence_index, forbidden_actions);
            if !cond_result.satisfied {
                // Condition violated — task failed
                PredicateResult {
                    satisfied: false,
                    progress: 0.0,
                }
            } else {
                evaluate_composition(goal, ctx, sequence_index, forbidden_actions)
            }
        }

        TaskComposition::Without(subtask, forbidden_action_id) => {
            if forbidden_actions.contains(forbidden_action_id) {
                // Forbidden action was taken — task failed
                PredicateResult {
                    satisfied: false,
                    progress: 0.0,
                }
            } else {
                evaluate_composition(subtask, ctx, sequence_index, forbidden_actions)
            }
        }

        _ => {
            warn!("unknown TaskComposition variant in evaluate_composition");
            PredicateResult {
                satisfied: false,
                progress: 0.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::AgentConfig;
    use forge_types::entity::Agent;
    use forge_types::grid::Position;
    use forge_types::resource::ItemType;
    use forge_types::task::Predicate;

    fn make_agent(id: u32, x: u16, y: u16) -> Agent {
        Agent::new(id, Position::new(x, y), &AgentConfig::default())
    }

    fn make_ctx(agents: &[Agent], tick: u64) -> EvalContext<'_> {
        EvalContext {
            agents,
            tick,
            grid: None,
            objects: None,
        }
    }

    #[test]
    fn test_and_all_satisfied() {
        let agents = vec![make_agent(0, 5, 5)];
        let ctx = make_ctx(&agents, 100);
        let task = TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            TaskComposition::Atom(Predicate::TimeElapsed(50)),
        ]);
        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        assert!(result.satisfied);
    }

    #[test]
    fn test_and_partial() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 100);
        let task = TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))), // NOT satisfied
            TaskComposition::Atom(Predicate::TimeElapsed(50)),                 // satisfied
        ]);
        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        assert!(!result.satisfied);
        assert!(result.progress > 0.0);
    }

    #[test]
    fn test_or_one_satisfied() {
        let agents = vec![make_agent(0, 5, 5)];
        let ctx = make_ctx(&agents, 0);
        let task = TaskComposition::Or(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(0, 0))),
        ]);
        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        assert!(result.satisfied);
    }

    #[test]
    fn test_sequence_progression() {
        let mut agents = vec![make_agent(0, 0, 0)];
        agents[0].inventory.add_item(ItemType::Wood, 3);

        let ctx = make_ctx(&agents, 0);
        let task = TaskComposition::Sequence(vec![
            TaskComposition::Atom(Predicate::AgentHas(0, ItemType::Wood, 3)), // satisfied
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))), // not satisfied
        ]);

        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        // First step should be completed, second not yet
        assert!(!result.satisfied);
        assert_eq!(seq_idx, 1); // advanced to step 1
        assert!(result.progress > 0.4); // at least half done
    }

    #[test]
    fn test_before_deadline_passed() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 200);
        let task = TaskComposition::Before(
            Box::new(TaskComposition::Atom(Predicate::AgentAt(
                0,
                Position::new(5, 5),
            ))),
            100, // deadline at tick 100
        );
        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0); // failed
    }

    #[test]
    fn test_without_forbidden_action() {
        let agents = vec![make_agent(0, 5, 5)];
        let ctx = make_ctx(&agents, 0);
        let task = TaskComposition::Without(
            Box::new(TaskComposition::Atom(Predicate::AgentAt(
                0,
                Position::new(5, 5),
            ))),
            42, // forbidden action ID
        );

        // Without the forbidden action, should succeed
        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        assert!(result.satisfied);

        // With the forbidden action, should fail
        let result2 = evaluate_composition(&task, &ctx, &mut seq_idx, &[42]);
        assert!(!result2.satisfied);
    }

    #[test]
    fn test_while_condition_maintained() {
        let agents = vec![make_agent(0, 5, 5)]; // full health
        let ctx = make_ctx(&agents, 100);
        let task = TaskComposition::While(
            Box::new(TaskComposition::Atom(Predicate::HealthAbove(0, 0.5))),
            Box::new(TaskComposition::Atom(Predicate::TimeElapsed(50))),
        );
        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        assert!(result.satisfied);
    }

    #[test]
    fn test_empty_and() {
        let agents = vec![];
        let ctx = make_ctx(&agents, 0);
        let task = TaskComposition::And(vec![]);
        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        assert!(result.satisfied); // vacuous truth
    }

    #[test]
    fn test_empty_or() {
        let agents = vec![];
        let ctx = make_ctx(&agents, 0);
        let task = TaskComposition::Or(vec![]);
        let mut seq_idx = 0;
        let result = evaluate_composition(&task, &ctx, &mut seq_idx, &[]);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_composer_deeply_nested() {
        // Build deeply nested AND/OR/SEQUENCE composition:
        // AND(
        //   OR(
        //     AgentAt(0, (5,5)),
        //     SEQUENCE(
        //       AgentAt(0, (0,0)),
        //       AND(
        //         TimeElapsed(10),
        //         OR(
        //           AgentAt(0, (5,5)),
        //           TimeElapsed(5)
        //         )
        //       )
        //     )
        //   ),
        //   TimeElapsed(1)
        // )
        let agents = vec![make_agent(0, 5, 5)];
        let ctx = make_ctx(&agents, 100);

        let deep_or = TaskComposition::Or(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            TaskComposition::Atom(Predicate::TimeElapsed(5)),
        ]);
        let deep_and = TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::TimeElapsed(10)),
            deep_or,
        ]);
        let sequence = TaskComposition::Sequence(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(0, 0))),
            deep_and,
        ]);
        let top_or = TaskComposition::Or(vec![
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            sequence,
        ]);
        let top_and = TaskComposition::And(vec![
            top_or,
            TaskComposition::Atom(Predicate::TimeElapsed(1)),
        ]);

        let mut seq_idx = 0;
        let result = evaluate_composition(&top_and, &ctx, &mut seq_idx, &[]);
        // The top AND requires both: OR (satisfied via AgentAt(0,(5,5))) AND TimeElapsed(1) (satisfied at tick 100)
        assert!(
            result.satisfied,
            "deeply nested composition should be satisfied"
        );
        assert!((result.progress - 1.0).abs() < 0.01);
    }
}
