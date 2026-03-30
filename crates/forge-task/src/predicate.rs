//! Predicate evaluation against world state.
//!
//! Atomic predicates are the building blocks of the task DSL.
//! Each predicate evaluates to a boolean (satisfied or not) and
//! optionally a progress value in [0.0, 1.0].

use forge_types::agriculture::CropState;
use forge_types::entity::{Agent, AgentId, ObjectState};
use forge_types::grid::{Grid, Position, TerrainType};
use forge_types::resource::ItemType;
use forge_types::task::Predicate;
use forge_types::Object;
use tracing::{instrument, trace, warn};

/// Context for evaluating predicates against the current world state.
///
/// The `grid` and `objects` fields are optional for backwards compatibility:
/// predicates that don't require spatial lookups (e.g. `AgentHas`, `TimeElapsed`)
/// work without them. Predicates like `AgentOnTerrain`, `ObjectAt`, and
/// `ObjectInState` return unsatisfied(0.0) when the required data is absent.
pub struct EvalContext<'a> {
    /// All agents in the simulation.
    pub agents: &'a [Agent],
    /// Current tick.
    pub tick: u64,
    /// The world grid (optional — required for terrain predicates).
    pub grid: Option<&'a Grid>,
    /// All objects in the simulation (optional — required for object predicates).
    pub objects: Option<&'a [Object]>,
    /// Per-tile crop states (optional — required for agricultural predicates).
    pub crop_states: Option<&'a [CropState]>,
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
#[instrument(skip_all)]
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
        Predicate::AgentOnTerrain(agent_id, terrain_id) => {
            eval_agent_on_terrain(ctx, *agent_id, *terrain_id)
        }
        Predicate::ObjectAt(obj_id, pos) => eval_object_at(ctx, *obj_id, pos),
        Predicate::ObjectInState(obj_id, state_name) => {
            eval_object_in_state(ctx, *obj_id, state_name)
        }

        // Agricultural predicates
        Predicate::CropHealthBelow(pos, threshold) => eval_crop_health_below(ctx, pos, *threshold),
        Predicate::DiseaseDetected(_agent_id, count) => {
            // Counts diseased tiles across all crop states
            eval_disease_detected(ctx, *count)
        }
        Predicate::FieldSurveyed(threshold) => eval_field_surveyed(ctx, *threshold),
        Predicate::SoilDataCollected(_agent_id, _count) => {
            // Not evaluatable without soil node state — falls through to wildcard
            warn!("SoilDataCollected predicate requires runtime tracking");
            PredicateResult::unsatisfied(0.0)
        }
        Predicate::FieldReportGenerated(_agent_id) => {
            // Requires runtime report tracking — falls through
            warn!("FieldReportGenerated predicate requires runtime tracking");
            PredicateResult::unsatisfied(0.0)
        }
        Predicate::IrrigationMapped(threshold) => eval_irrigation_mapped(ctx, *threshold),
        Predicate::AreaSprayed(threshold) => eval_area_sprayed(ctx, *threshold),
        _ => {
            warn!("unknown predicate variant encountered — treating as unsatisfied");
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

/// Checks if agent is standing on a specific terrain type.
fn eval_agent_on_terrain(ctx: &EvalContext, agent_id: AgentId, terrain_id: u8) -> PredicateResult {
    let grid = match ctx.grid {
        Some(g) => g,
        None => return PredicateResult::unsatisfied(0.0),
    };
    let agent = match find_agent(ctx.agents, agent_id) {
        Some(a) => a,
        None => return PredicateResult::unsatisfied(0.0),
    };
    let expected_terrain = match TerrainType::from_u8(terrain_id) {
        Some(t) => t,
        None => return PredicateResult::unsatisfied(0.0),
    };
    if let Some(tile) = grid.get(agent.position.x, agent.position.y) {
        if tile.terrain == expected_terrain {
            trace!(
                agent_id,
                ?expected_terrain,
                "predicate AgentOnTerrain satisfied"
            );
            PredicateResult::satisfied()
        } else {
            PredicateResult::unsatisfied(0.0)
        }
    } else {
        PredicateResult::unsatisfied(0.0)
    }
}

/// Checks if an object is at a specific position.
fn eval_object_at(ctx: &EvalContext, obj_id: u32, target: &Position) -> PredicateResult {
    let objects = match ctx.objects {
        Some(o) => o,
        None => return PredicateResult::unsatisfied(0.0),
    };
    if let Some(obj) = objects.iter().find(|o| o.id == obj_id) {
        if obj.position == *target {
            trace!(obj_id, "predicate ObjectAt satisfied");
            PredicateResult::satisfied()
        } else {
            let distance = obj.position.manhattan_distance(target) as f32;
            let max_dist = 100.0_f32;
            let progress = 1.0 - (distance / max_dist).min(1.0);
            PredicateResult::unsatisfied(progress)
        }
    } else {
        PredicateResult::unsatisfied(0.0)
    }
}

/// Checks if an object is in a specific state (by state name).
fn eval_object_in_state(ctx: &EvalContext, obj_id: u32, state_name: &str) -> PredicateResult {
    let objects = match ctx.objects {
        Some(o) => o,
        None => return PredicateResult::unsatisfied(0.0),
    };
    let expected_state = match state_name {
        "Active" | "active" => ObjectState::Active,
        "Inactive" | "inactive" => ObjectState::Inactive,
        "Open" | "open" => ObjectState::Open,
        "Closed" | "closed" => ObjectState::Closed,
        "Destroyed" | "destroyed" => ObjectState::Destroyed,
        _ => return PredicateResult::unsatisfied(0.0),
    };
    if let Some(obj) = objects.iter().find(|o| o.id == obj_id) {
        if obj.state == expected_state {
            trace!(obj_id, ?expected_state, "predicate ObjectInState satisfied");
            PredicateResult::satisfied()
        } else {
            PredicateResult::unsatisfied(0.0)
        }
    } else {
        PredicateResult::unsatisfied(0.0)
    }
}

// ---- Agricultural predicate evaluators ----

/// Checks if crop at the given position has health below threshold.
fn eval_crop_health_below(ctx: &EvalContext, pos: &Position, threshold: f32) -> PredicateResult {
    let grid = match ctx.grid {
        Some(g) => g,
        None => return PredicateResult::unsatisfied(0.0),
    };
    let crop_states = match ctx.crop_states {
        Some(cs) => cs,
        None => return PredicateResult::unsatisfied(0.0),
    };

    let idx = pos.y as usize * grid.width as usize + pos.x as usize;
    if idx >= crop_states.len() {
        return PredicateResult::unsatisfied(0.0);
    }

    let health_normalized =
        crop_states[idx].health as f32 / forge_types::constants::FIXED_POINT_ONE as f32;
    if health_normalized < threshold {
        PredicateResult::satisfied()
    } else {
        let progress = if threshold > 0.0 {
            (1.0 - health_normalized / threshold).max(0.0)
        } else {
            0.0
        };
        PredicateResult::unsatisfied(progress)
    }
}

/// Counts diseased tiles and checks if total meets threshold.
fn eval_disease_detected(ctx: &EvalContext, count: u32) -> PredicateResult {
    let crop_states = match ctx.crop_states {
        Some(cs) => cs,
        None => return PredicateResult::unsatisfied(0.0),
    };

    let diseased = crop_states.iter().filter(|c| c.is_diseased()).count() as u32;
    if diseased >= count {
        PredicateResult::satisfied()
    } else if count > 0 {
        PredicateResult::unsatisfied(diseased as f32 / count as f32)
    } else {
        PredicateResult::satisfied()
    }
}

/// Checks if the fraction of cropland tiles surveyed meets threshold.
fn eval_field_surveyed(ctx: &EvalContext, threshold: f32) -> PredicateResult {
    let grid = match ctx.grid {
        Some(g) => g,
        None => return PredicateResult::unsatisfied(0.0),
    };
    let crop_states = match ctx.crop_states {
        Some(cs) => cs,
        None => return PredicateResult::unsatisfied(0.0),
    };

    let mut total_crop = 0u32;
    let mut surveyed = 0u32;

    for (i, tile) in grid.tiles.iter().enumerate() {
        if tile.terrain == TerrainType::Cropland || tile.terrain == TerrainType::Orchard {
            total_crop += 1;
            if i < crop_states.len() && crop_states[i].is_surveyed() {
                surveyed += 1;
            }
        }
    }

    if total_crop == 0 {
        return PredicateResult::unsatisfied(0.0);
    }

    let fraction = surveyed as f32 / total_crop as f32;
    if fraction >= threshold {
        PredicateResult::satisfied()
    } else {
        PredicateResult::unsatisfied(fraction / threshold.max(0.001))
    }
}

/// Checks if the fraction of field thermally mapped meets threshold.
/// Uses surveyed_tick as proxy (thermal scan also marks surveyed).
fn eval_irrigation_mapped(ctx: &EvalContext, threshold: f32) -> PredicateResult {
    // Reuses same logic as field_surveyed
    eval_field_surveyed(ctx, threshold)
}

/// Checks if the fraction of diseased tiles that have been sprayed meets threshold.
fn eval_area_sprayed(ctx: &EvalContext, threshold: f32) -> PredicateResult {
    let crop_states = match ctx.crop_states {
        Some(cs) => cs,
        None => return PredicateResult::unsatisfied(0.0),
    };

    let mut diseased = 0u32;
    let mut sprayed_diseased = 0u32;

    for crop in crop_states {
        if crop.is_diseased() || crop.sprayed {
            diseased += 1;
            if crop.sprayed {
                sprayed_diseased += 1;
            }
        }
    }

    if diseased == 0 {
        // No disease to spray — consider satisfied
        return PredicateResult::satisfied();
    }

    let fraction = sprayed_diseased as f32 / diseased as f32;
    if fraction >= threshold {
        PredicateResult::satisfied()
    } else {
        PredicateResult::unsatisfied(fraction / threshold.max(0.001))
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
        EvalContext {
            agents,
            tick,
            grid: None,
            objects: None,
            crop_states: None,
        }
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
    fn test_agent_near_exact_max_distance_satisfied() {
        let agents = vec![make_agent(0, 0, 0), make_agent(1, 2, 1)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 3), &ctx);
        assert!(result.satisfied);
        assert_eq!(result.progress, 1.0);
    }

    #[test]
    fn test_agent_near_far_distance_progress_clamps_to_zero() {
        let agents = vec![make_agent(0, 0, 0), make_agent(1, 200, 200)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 1), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
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

    #[test]
    fn test_resource_count_predicate() {
        let mut agents = vec![make_agent(0, 0, 0)];
        agents[0].inventory.add_item(ItemType::Stone, 5);
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::ResourceCount(0, ItemType::Stone, 3), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_agent_has_zero_count() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 0);
        // Requesting 0 items should always be satisfied
        let result = evaluate_predicate(&Predicate::AgentHas(0, ItemType::Wood, 0), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_agent_has_nonexistent_agent() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentHas(99, ItemType::Wood, 1), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_agent_near_both_missing() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 0);
        // Agent 5 and 6 don't exist
        let result = evaluate_predicate(&Predicate::AgentNear(5, 6, 3), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_time_elapsed_zero_deadline() {
        let agents = vec![];
        let ctx = make_ctx(&agents, 0);
        // target_tick == 0 should be satisfied immediately
        let result = evaluate_predicate(&Predicate::TimeElapsed(0), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_health_above_low_health() {
        let mut agents = vec![make_agent(0, 0, 0)];
        // Set health to very low value
        agents[0].health = 1;
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::HealthAbove(0, 0.5), &ctx);
        assert!(!result.satisfied);
        assert!(result.progress >= 0.0);
    }

    #[test]
    fn test_health_above_zero_threshold() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 0);
        // threshold == 0.0 should always be satisfied
        let result = evaluate_predicate(&Predicate::HealthAbove(0, 0.0), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_health_above_nonexistent_agent() {
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::HealthAbove(99, 0.5), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_team_alive_empty_team() {
        let agents = vec![make_agent(0, 0, 0)];
        // Agent 0 defaults to team 0, so team 99 is empty
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::TeamAlive(99), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_agent_on_terrain_invalid_terrain_id() {
        let agents = vec![make_agent(0, 3, 3)];
        let grid = Grid::new(16, 16);
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: Some(&grid),
            objects: None,
            crop_states: None,
        };
        // terrain_id 255 is invalid
        let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 255), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_agent_on_terrain_next_after_last_valid_id_is_unsatisfied() {
        let agents = vec![make_agent(0, 3, 3)];
        let grid = Grid::new(16, 16);
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: Some(&grid),
            objects: None,
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 8), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_agent_on_terrain_nonexistent_agent() {
        let agents = vec![make_agent(0, 3, 3)];
        let grid = Grid::new(16, 16);
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: Some(&grid),
            objects: None,
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::AgentOnTerrain(99, 0), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_agent_on_terrain_all_terrain_ids() {
        let agents = vec![make_agent(0, 3, 3)];
        let grid = Grid::new(16, 16);
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: Some(&grid),
            objects: None,
            crop_states: None,
        };
        // Test all valid terrain IDs (0-7) don't panic
        for terrain_id in 0..=7 {
            let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, terrain_id), &ctx);
            // Only terrain_id 0 (Ground) should be satisfied on default grid
            if terrain_id == 0 {
                assert!(result.satisfied);
            } else {
                assert!(!result.satisfied);
            }
        }
    }

    #[test]
    fn test_object_at_nonexistent_object() {
        use forge_types::entity::ObjectType;
        let objects = vec![Object {
            id: 0,
            position: Position::new(5, 5),
            object_type: ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Active,
        }];
        let ctx = EvalContext {
            agents: &[],
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        // Object 99 doesn't exist
        let result = evaluate_predicate(&Predicate::ObjectAt(99, Position::new(5, 5)), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_object_in_state_no_objects() {
        let ctx = EvalContext {
            agents: &[],
            tick: 0,
            grid: None,
            objects: None,
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectInState(0, "Active".to_string()), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_object_in_state_invalid_state_name() {
        use forge_types::entity::ObjectType;
        let objects = vec![Object {
            id: 0,
            position: Position::new(0, 0),
            object_type: ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Active,
        }];
        let ctx = EvalContext {
            agents: &[],
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        let result = evaluate_predicate(
            &Predicate::ObjectInState(0, "InvalidState".to_string()),
            &ctx,
        );
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_object_in_state_nonexistent_object() {
        use forge_types::entity::ObjectType;
        let objects = vec![Object {
            id: 0,
            position: Position::new(0, 0),
            object_type: ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Active,
        }];
        let ctx = EvalContext {
            agents: &[],
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectInState(99, "Active".to_string()), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_object_in_state_all_valid_states() {
        use forge_types::entity::ObjectType;
        let objects = vec![Object {
            id: 0,
            position: Position::new(0, 0),
            object_type: ObjectType::Door,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Closed,
        }];
        let ctx = EvalContext {
            agents: &[],
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        // Test all valid state names
        for state_name in &[
            "Active",
            "active",
            "Inactive",
            "inactive",
            "Open",
            "open",
            "Closed",
            "closed",
            "Destroyed",
            "destroyed",
        ] {
            let result =
                evaluate_predicate(&Predicate::ObjectInState(0, state_name.to_string()), &ctx);
            // Only "Closed" and "closed" should match
            if *state_name == "Closed" || *state_name == "closed" {
                assert!(
                    result.satisfied,
                    "Expected satisfied for state '{}'",
                    state_name
                );
            } else {
                assert!(
                    !result.satisfied,
                    "Expected unsatisfied for state '{}'",
                    state_name
                );
            }
        }
    }

    // ---- AgentOnTerrain tests ----

    #[test]
    fn test_agent_on_terrain_satisfied() {
        let agents = vec![make_agent(0, 3, 3)];
        let mut grid = Grid::new(16, 16);
        grid.get_mut(3, 3).unwrap().terrain = TerrainType::Forest;
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: Some(&grid),
            objects: None,
            crop_states: None,
        };
        // Forest = terrain_id 6
        let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 6), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_agent_on_terrain_unsatisfied() {
        let agents = vec![make_agent(0, 3, 3)];
        let grid = Grid::new(16, 16); // all Ground = 0
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: Some(&grid),
            objects: None,
            crop_states: None,
        };
        // Forest = terrain_id 6, but agent is on Ground
        let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 6), &ctx);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_agent_on_terrain_no_grid() {
        let agents = vec![make_agent(0, 3, 3)];
        let ctx = make_ctx(&agents, 0); // grid is None
        let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 0), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    // ---- ObjectAt tests ----

    #[test]
    fn test_object_at_satisfied() {
        use forge_types::entity::ObjectType;
        let agents = vec![];
        let objects = vec![Object {
            id: 0,
            position: Position::new(5, 5),
            object_type: ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Active,
        }];
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectAt(0, Position::new(5, 5)), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_object_at_unsatisfied() {
        use forge_types::entity::ObjectType;
        let agents = vec![];
        let objects = vec![Object {
            id: 0,
            position: Position::new(1, 1),
            object_type: ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Active,
        }];
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectAt(0, Position::new(5, 5)), &ctx);
        assert!(!result.satisfied);
        assert!(result.progress > 0.0); // partial progress from proximity
    }

    #[test]
    fn test_object_at_no_objects() {
        let agents = vec![];
        let ctx = make_ctx(&agents, 0); // objects is None
        let result = evaluate_predicate(&Predicate::ObjectAt(0, Position::new(5, 5)), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    // ---- ObjectInState tests ----

    #[test]
    fn test_object_in_state_satisfied() {
        use forge_types::entity::ObjectType;
        let agents = vec![];
        let objects = vec![Object {
            id: 0,
            position: Position::new(0, 0),
            object_type: ObjectType::Door,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Open,
        }];
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectInState(0, "Open".to_string()), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_object_in_state_unsatisfied() {
        use forge_types::entity::ObjectType;
        let agents = vec![];
        let objects = vec![Object {
            id: 0,
            position: Position::new(0, 0),
            object_type: ObjectType::Door,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Closed,
        }];
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectInState(0, "Open".to_string()), &ctx);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_object_in_state_case_insensitive() {
        use forge_types::entity::ObjectType;
        let agents = vec![];
        let objects = vec![Object {
            id: 0,
            position: Position::new(0, 0),
            object_type: ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Active,
        }];
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectInState(0, "active".to_string()), &ctx);
        assert!(result.satisfied);
    }

    #[test]
    fn test_object_in_state_uppercase_name_is_rejected() {
        use forge_types::entity::ObjectType;
        let agents = vec![];
        let objects = vec![Object {
            id: 0,
            position: Position::new(0, 0),
            object_type: ObjectType::Boulder,
            mass: 65536,
            durability: 655360,
            state: ObjectState::Active,
        }];
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: None,
            objects: Some(&objects),
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectInState(0, "ACTIVE".to_string()), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    // ---- Coverage gap tests: drone predicate wildcard branch ----

    #[test]
    fn test_unknown_predicate_agent_at_altitude() {
        let agents = vec![make_agent(0, 3, 3)];
        let ctx = make_ctx(&agents, 0);
        // AgentAtAltitude hits the wildcard `_` branch
        let result = evaluate_predicate(&Predicate::AgentAtAltitude(0, 5), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_unknown_predicate_battery_above() {
        let agents = vec![make_agent(0, 3, 3)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::BatteryAbove(0, 0.5), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_unknown_predicate_agent_airborne() {
        let agents = vec![make_agent(0, 3, 3)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentAirborne(0), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_unknown_predicate_agent_landed() {
        let agents = vec![make_agent(0, 3, 3)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentLanded(0), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_agent_on_terrain_position_outside_grid() {
        // Agent at position (200, 200) on a 16x16 grid — grid.get returns None
        let agents = vec![make_agent(0, 200, 200)];
        let grid = Grid::new(16, 16);
        let ctx = EvalContext {
            agents: &agents,
            tick: 0,
            grid: Some(&grid),
            objects: None,
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::AgentOnTerrain(0, 0), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_object_at_with_none_objects() {
        let ctx = EvalContext {
            agents: &[],
            tick: 0,
            grid: None,
            objects: None,
            crop_states: None,
        };
        let result = evaluate_predicate(&Predicate::ObjectAt(0, Position::new(5, 5)), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_predicate_result_unsatisfied_clamps_progress() {
        let result = PredicateResult::unsatisfied(-0.5);
        assert_eq!(result.progress, 0.0);

        let result2 = PredicateResult::unsatisfied(1.5);
        assert_eq!(result2.progress, 1.0);
    }

    #[test]
    fn test_agent_near_one_missing() {
        // Only agent 0 exists, agent 1 missing
        let agents = vec![make_agent(0, 5, 5)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentNear(0, 1, 3), &ctx);
        assert!(!result.satisfied);
        assert_eq!(result.progress, 0.0);
    }

    #[test]
    fn test_agent_at_far_away() {
        // Agent very far from target — progress should be near 0
        let agents = vec![make_agent(0, 0, 0)];
        let ctx = make_ctx(&agents, 0);
        let result = evaluate_predicate(&Predicate::AgentAt(0, Position::new(200, 200)), &ctx);
        assert!(!result.satisfied);
        // Manhattan distance 400 / max_dist 100 -> clamped to 1.0 -> progress 0.0
        assert_eq!(result.progress, 0.0);
    }
}
