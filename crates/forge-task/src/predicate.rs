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
#[path = "predicate/tests.rs"]
mod tests;
