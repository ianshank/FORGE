//! Predicate evaluation against world state.
//!
//! Atomic predicates are the building blocks of the task DSL.
//! Each predicate evaluates to a boolean (satisfied or not) and
//! optionally a progress value in [0.0, 1.0].

use forge_types::entity::{Agent, AgentId};
use forge_types::grid::Position;
use forge_types::resource::ItemType;
use forge_types::task::Predicate;
use tracing::trace;

/// Context for evaluating predicates against the current world state.
pub struct EvalContext<'a> {
    /// All agents in the simulation.
    pub agents: &'a [Agent],
    /// Current tick.
    pub tick: u64,
}

/// Result of evaluating a predicate.
#[derive(Debug, Clone, Copy)]
pub struct PredicateResult {
    /// Whether the predicate is satisfied.
    pub satisfied: bool,
    /// Progress toward satisfaction (0.0 = no progress, 1.0 = complete).
    pub progress: f32,
}

impl PredicateResult {
    fn satisfied() -> Self {
        Self {
            satisfied: true,
            progress: 1.0,
        }
    }

    fn unsatisfied(progress: f32) -> Self {
        Self {
            satisfied: false,
            progress: progress.clamp(0.0, 1.0),
        }
    }
}

/// Evaluates an atomic predicate against the current world state.
pub fn evaluate_predicate(predicate: &Predicate, ctx: &EvalContext) -> PredicateResult {
    match predicate {
        Predicate::AgentAt(agent_id, target_pos) => eval_agent_at(ctx, *agent_id, target_pos),
        Predicate::AgentHas(agent_id, item_type, count) => {
            eval_agent_has(ctx, *agent_id, *item_type, *count)
        }
        Predicate::AgentNear(a_id, b_id, distance) => eval_agent_near(ctx, *a_id, *b_id, *distance),
        Predicate::TimeElapsed(target_tick) => eval_time_elapsed(ctx, *target_tick),
        Predicate::HealthAbove(agent_id, threshold) => {
            eval_health_above(ctx, *agent_id, *threshold)
        }
        Predicate::ResourceCount(agent_id, item_type, count) => {
            eval_agent_has(ctx, *agent_id, *item_type, *count)
        }
        Predicate::TeamAlive(team_id) => eval_team_alive(ctx, *team_id),
        Predicate::AgentOnTerrain(_agent_id, _terrain) => {
            // Requires grid access — handled at a higher level
            PredicateResult::unsatisfied(0.0)
        }
        Predicate::ObjectAt(_obj_id, _pos) => {
            // Requires object list — handled at a higher level
            PredicateResult::unsatisfied(0.0)
        }
        Predicate::ObjectInState(_obj_id, _state) => {
            // Requires object list — handled at a higher level
            PredicateResult::unsatisfied(0.0)
        }
    }
}

fn find_agent(agents: &[Agent], id: AgentId) -> Option<&Agent> {
    agents.iter().find(|a| a.id == id)
}

fn eval_agent_at(ctx: &EvalContext, agent_id: AgentId, target: &Position) -> PredicateResult {
    if let Some(agent) = find_agent(ctx.agents, agent_id) {
        if agent.position == *target {
            trace!(agent_id, "predicate AgentAt satisfied");
            PredicateResult::satisfied()
        } else {
            let distance = agent.position.manhattan_distance(target) as f32;
            // Progress based on proximity (inverse of distance, capped)
            let max_dist = 100.0_f32;
            let progress = 1.0 - (distance / max_dist).min(1.0);
            PredicateResult::unsatisfied(progress)
        }
    } else {
        PredicateResult::unsatisfied(0.0)
    }
}

fn eval_agent_has(
    ctx: &EvalContext,
    agent_id: AgentId,
    item_type: ItemType,
    count: u16,
) -> PredicateResult {
    if let Some(agent) = find_agent(ctx.agents, agent_id) {
        let current = agent.inventory.count_item(item_type);
        if current >= count {
            trace!(agent_id, ?item_type, count, "predicate AgentHas satisfied");
            PredicateResult::satisfied()
        } else {
            let progress = if count > 0 {
                current as f32 / count as f32
            } else {
                1.0
            };
            PredicateResult::unsatisfied(progress)
        }
    } else {
        PredicateResult::unsatisfied(0.0)
    }
}

fn eval_agent_near(
    ctx: &EvalContext,
    a_id: AgentId,
    b_id: AgentId,
    max_distance: u16,
) -> PredicateResult {
    let a = find_agent(ctx.agents, a_id);
    let b = find_agent(ctx.agents, b_id);

    if let (Some(a), Some(b)) = (a, b) {
        let dist = a.position.manhattan_distance(&b.position);
        if dist <= max_distance as u32 {
            PredicateResult::satisfied()
        } else {
            let progress = 1.0 - ((dist as f32 - max_distance as f32) / 50.0).min(1.0);
            PredicateResult::unsatisfied(progress.max(0.0))
        }
    } else {
        PredicateResult::unsatisfied(0.0)
    }
}

fn eval_time_elapsed(ctx: &EvalContext, target_tick: u64) -> PredicateResult {
    if ctx.tick >= target_tick {
        PredicateResult::satisfied()
    } else if target_tick > 0 {
        let progress = ctx.tick as f32 / target_tick as f32;
        PredicateResult::unsatisfied(progress)
    } else {
        PredicateResult::satisfied()
    }
}

fn eval_health_above(ctx: &EvalContext, agent_id: AgentId, threshold: f32) -> PredicateResult {
    if let Some(agent) = find_agent(ctx.agents, agent_id) {
        // Health is stored as fixed-point i32. Threshold is normalized 0.0-1.0.
        // max_health from constants = 655360 (10.0 in fixed-point)
        let max_health = forge_types::constants::DEFAULT_MAX_HEALTH as f32;
        let normalized = if max_health > 0.0 {
            agent.health as f32 / max_health
        } else {
            0.0
        };
        if normalized >= threshold {
            PredicateResult::satisfied()
        } else {
            let progress = if threshold > 0.0 {
                normalized / threshold
            } else {
                1.0
            };
            PredicateResult::unsatisfied(progress)
        }
    } else {
        PredicateResult::unsatisfied(0.0)
    }
}

fn eval_team_alive(ctx: &EvalContext, team_id: u8) -> PredicateResult {
    let team_agents: Vec<&Agent> = ctx.agents.iter().filter(|a| a.team == team_id).collect();
    if team_agents.is_empty() {
        return PredicateResult::unsatisfied(0.0);
    }
    let alive_count = team_agents.iter().filter(|a| a.alive).count();
    if alive_count == team_agents.len() {
        PredicateResult::satisfied()
    } else {
        let progress = alive_count as f32 / team_agents.len() as f32;
        PredicateResult::unsatisfied(progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::AgentConfig;

    fn make_agent(id: u32, x: u16, y: u16) -> Agent {
        Agent::new(id, Position::new(x, y), &AgentConfig::default())
    }

    fn make_ctx(agents: &[Agent], tick: u64) -> EvalContext<'_> {
        EvalContext { agents, tick }
    }

    #[test]
    fn test_agent_at_satisfied() {
        let agents = vec![make_agent(0, 5, 5)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentAt(0, Position::new(5, 5)), &ctx);
        assert!(result.satisfied);
        assert_eq!(result.progress, 1.0);
    }

    #[test]
    fn test_agent_at_unsatisfied() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentAt(0, Position::new(5, 5)), &ctx);
        assert!(!result.satisfied);
        assert!(result.progress > 0.0); // some progress toward target
    }

    #[test]
    fn test_agent_has_satisfied() {
        let mut agents = vec![make_agent(0, 0, 0)];
        agents[0].inventory.add_item(ItemType::Wood, 5);
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentHas(0, ItemType::Wood, 3), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_agent_has_partial_progress() {
        let mut agents = vec![make_agent(0, 0, 0)];
        agents[0].inventory.add_item(ItemType::Wood, 2);
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentHas(0, ItemType::Wood, 4), &ctx);
        assert!(!result.satisfied);
        assert!((result.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_agent_near_satisfied() {
        let agents = vec![make_agent(0, 5, 5), make_agent(1, 6, 5)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 3), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_agent_near_unsatisfied() {
        let agents = vec![make_agent(0, 0, 0), make_agent(1, 50, 50)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 3), &ctx);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_time_elapsed() {
        let agents = vec![];
        let ctx = make_ctx(&agents, 100);
        let result = evaluate_predicate(&Predicate::TimeElapsed(50), &ctx);
        assert!(result.satisfied);

        let ctx2 = make_ctx(&agents, 25);
        let result2 = evaluate_predicate(&Predicate::TimeElapsed(50), &ctx2);
        assert!(!result2.satisfied);
        assert!((result2.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_health_above() {
        let agents = vec![make_agent(0, 0, 0)]; // full health
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::HealthAbove(0, 0.5), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_team_alive() {
        let mut agents = vec![make_agent(0, 0, 0), make_agent(1, 5, 5)];
        agents[0].team = 1;
        agents[1].team = 1;
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::TeamAlive(1), &ctx);
        assert!(result.satisfied);

        // Kill one agent
        let mut agents2 = agents.clone();
        agents2[0].alive = false;
        let ctx2 = make_ctx(&agents2, 0);
        let result2 = evaluate_predicate(&Predicate::TeamAlive(1), &ctx2);
        assert!(!result2.satisfied);
        assert!((result2.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_unknown_agent() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentAt(99, Position::new(0, 0)), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }
}
