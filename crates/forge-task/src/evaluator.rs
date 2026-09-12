//! Task completion evaluation and reward computation.
//!
//! Evaluates active tasks against the current world state,
//! computes rewards, and detects completion/failure.

use forge_types::task::ActiveTask;
use tracing::{debug, instrument, trace};

use crate::composer::evaluate_composition;
use crate::predicate::EvalContext;

/// Result of evaluating all active tasks for a single tick.
#[derive(Debug, Clone)]
pub struct TaskEvalResult {
    /// Per-agent reward for this tick.
    pub rewards: Vec<f32>,
    /// Indices of tasks that were completed this tick.
    pub completed_tasks: Vec<usize>,
    /// Indices of tasks that failed this tick.
    pub failed_tasks: Vec<usize>,
    /// Whether the episode should terminate (all tasks completed or failed).
    pub should_terminate: bool,
}

/// Evaluates all active tasks and computes rewards.
///
/// `ctx` supplies agents, tick, and optional world slices (`grid`, `objects`,
/// `crop_states`, `soil_nodes`). `forbidden_actions` is this tick's discrete
/// action ids — used by [`TaskComposition::Without`].
#[instrument(skip_all)]
pub fn evaluate_tasks(
    tasks: &mut [ActiveTask],
    ctx: &EvalContext<'_>,
    reward_scale: f32,
    forbidden_actions: &[u32],
) -> TaskEvalResult {
    let agents = ctx.agents;
    let mut rewards = vec![0.0_f32; agents.len()];
    let mut completed_tasks = Vec::new();
    let mut failed_tasks = Vec::new();

    for (task_idx, task) in tasks.iter_mut().enumerate() {
        if task.completed || task.failed {
            continue;
        }

        let result = evaluate_composition(
            &task.definition.goal,
            ctx,
            &mut task.sequence_index,
            forbidden_actions,
        );

        // Update progress
        let old_progress: f32 =
            task.progress.iter().sum::<f32>() / task.progress.len().max(1) as f32;
        if !task.progress.is_empty() {
            task.progress[0] = result.progress;
        }

        // Dense reward: reward for progress improvement
        if !task.definition.dense_reward_weights.is_empty() {
            let new_progress = result.progress;
            let delta = new_progress - old_progress;
            if delta > 0.0 {
                let dense_reward = delta * reward_scale;
                // Distribute reward to all alive agents
                for (i, agent) in agents.iter().enumerate() {
                    if agent.alive {
                        rewards[i] += dense_reward;
                    }
                }
                trace!(task_idx, delta, dense_reward, "dense reward for progress");
            }
        }

        if result.failed {
            task.failed = true;
            failed_tasks.push(task_idx);
            debug!(
                task_idx,
                task_id = task.definition.id,
                "task failed (constraint or deadline)"
            );
        } else if result.satisfied {
            task.completed = true;
            completed_tasks.push(task_idx);
            // Completion reward
            let completion_reward = task.definition.reward * reward_scale;
            for (i, agent) in agents.iter().enumerate() {
                if agent.alive {
                    rewards[i] += completion_reward;
                }
            }
            debug!(
                task_idx,
                task_id = task.definition.id,
                completion_reward,
                "task completed"
            );
        }
    }

    // Check episode termination: all tasks completed or all failed
    let all_done = tasks.iter().all(|t| t.completed || t.failed);
    let should_terminate = !tasks.is_empty() && all_done;

    TaskEvalResult {
        rewards,
        completed_tasks,
        failed_tasks,
        should_terminate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::agriculture::CropState;
    use forge_types::config::AgentConfig;
    use forge_types::entity::Agent;
    use forge_types::grid::{Grid, Position, TerrainType};
    use forge_types::task::{Predicate, TaskComposition, TaskDefinition, TaskTier};

    fn make_agent(id: u32, x: u16, y: u16) -> Agent {
        Agent::new(id, Position::new(x, y), &AgentConfig::default())
    }

    fn make_simple_task(id: u64, goal: TaskComposition, reward: f32) -> ActiveTask {
        ActiveTask::from_definition(TaskDefinition {
            id,
            description: format!("Test task {}", id),
            goal,
            tier: TaskTier::new(1),
            estimated_steps: 10,
            reward,
            dense_reward_weights: vec![1.0],
        })
    }

    fn eval(
        tasks: &mut [ActiveTask],
        agents: &[Agent],
        scale: f32,
        forbidden: &[u32],
    ) -> TaskEvalResult {
        let ctx = EvalContext::new(agents, 0);
        evaluate_tasks(tasks, &ctx, scale, forbidden)
    }

    #[test]
    fn test_task_completion_gives_reward() {
        let agents = vec![make_agent(0, 5, 5)];
        let mut tasks = vec![make_simple_task(
            1,
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            10.0,
        )];

        let result = eval(&mut tasks, &agents, 1.0, &[]);
        assert!(result.rewards[0] > 0.0);
        assert_eq!(result.completed_tasks, vec![0]);
        assert!(tasks[0].completed);
    }

    #[test]
    fn test_incomplete_task_no_completion_reward() {
        let agents = vec![make_agent(0, 0, 0)];
        let mut tasks = vec![make_simple_task(
            1,
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            10.0,
        )];

        let result = eval(&mut tasks, &agents, 1.0, &[]);
        assert!(!tasks[0].completed);
        assert!(result.completed_tasks.is_empty());
    }

    #[test]
    fn test_episode_termination_all_complete() {
        let agents = vec![make_agent(0, 5, 5)];
        let mut tasks = vec![make_simple_task(
            1,
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            10.0,
        )];

        let result = eval(&mut tasks, &agents, 1.0, &[]);
        assert!(result.should_terminate);
    }

    #[test]
    fn test_multi_agent_reward_distribution() {
        let agents = vec![make_agent(0, 5, 5), make_agent(1, 0, 0)];
        let mut tasks = vec![make_simple_task(
            1,
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            10.0,
        )];

        let result = eval(&mut tasks, &agents, 1.0, &[]);
        // Both alive agents should get reward
        assert!(result.rewards[0] > 0.0);
        assert!(result.rewards[1] > 0.0);
    }

    #[test]
    fn test_reward_scale() {
        let agents = vec![make_agent(0, 5, 5)];
        let mut tasks1 = vec![make_simple_task(
            1,
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            10.0,
        )];
        let mut tasks2 = vec![make_simple_task(
            1,
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            10.0,
        )];

        let r1 = eval(&mut tasks1, &agents, 1.0, &[]);
        let r2 = eval(&mut tasks2, &agents, 2.0, &[]);
        assert!((r2.rewards[0] / r1.rewards[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_evaluate_empty_tasks() {
        let agents = vec![make_agent(0, 5, 5)];
        let mut tasks: Vec<ActiveTask> = vec![];
        let result = eval(&mut tasks, &agents, 1.0, &[]);
        assert_eq!(result.rewards.len(), 1);
        assert_eq!(result.rewards[0], 0.0);
        assert!(result.completed_tasks.is_empty());
        assert!(result.failed_tasks.is_empty());
        // Empty task list should not terminate (there's nothing to complete)
        assert!(!result.should_terminate);
    }

    #[test]
    fn test_already_completed_task_skipped() {
        let agents = vec![make_agent(0, 5, 5)];
        let mut tasks = vec![make_simple_task(
            1,
            TaskComposition::Atom(Predicate::AgentAt(0, Position::new(5, 5))),
            10.0,
        )];
        tasks[0].completed = true;

        let result = eval(&mut tasks, &agents, 1.0, &[]);
        assert!(result.completed_tasks.is_empty()); // not re-completed
        assert!(result.should_terminate); // still terminates
    }

    #[test]
    fn field_surveyed_satisfied_with_surveyed_crop_vector() {
        let agents = vec![make_agent(0, 0, 0)];
        let mut grid = Grid::new(4, 4);
        for tile in grid.tiles.iter_mut() {
            tile.terrain = TerrainType::Cropland;
        }
        let mut crops = vec![CropState::default(); 16];
        for crop in &mut crops {
            crop.surveyed_tick = 1;
        }
        let mut tasks = vec![make_simple_task(
            1,
            TaskComposition::Atom(Predicate::FieldSurveyed(0.8)),
            1.0,
        )];
        let ctx = EvalContext {
            grid: Some(&grid),
            crop_states: Some(&crops),
            ..EvalContext::new(&agents, 0)
        };
        let result = evaluate_tasks(&mut tasks, &ctx, 1.0, &[]);
        assert!(tasks[0].completed);
        assert!(result.should_terminate);
        assert!(result.rewards[0] > 0.0);
    }

    #[test]
    fn without_fails_when_forbidden_id_taken_this_tick() {
        let agents = vec![make_agent(0, 5, 5)];
        let mut tasks = vec![make_simple_task(
            1,
            TaskComposition::Without(
                Box::new(TaskComposition::Atom(Predicate::AgentAt(
                    0,
                    Position::new(5, 5),
                ))),
                42,
            ),
            1.0,
        )];
        let result = eval(&mut tasks, &agents, 1.0, &[42]);
        assert!(tasks[0].failed);
        assert!(!tasks[0].completed);
        assert_eq!(result.failed_tasks, vec![0]);
        assert!(result.should_terminate);
    }

    #[test]
    fn without_completes_when_forbidden_id_not_taken() {
        let agents = vec![make_agent(0, 5, 5)];
        let mut tasks = vec![make_simple_task(
            1,
            TaskComposition::Without(
                Box::new(TaskComposition::Atom(Predicate::AgentAt(
                    0,
                    Position::new(5, 5),
                ))),
                42,
            ),
            1.0,
        )];
        let result = eval(&mut tasks, &agents, 1.0, &[0]);
        assert!(tasks[0].completed);
        assert!(!tasks[0].failed);
        assert_eq!(result.completed_tasks, vec![0]);
    }
}
