//! High-level scenario TOML compiler.
//!
//! `configs/scenarios/*.toml` uses a `[scenario]` schema (map, drone, agri,
//! objectives) that is **not** [`ForgeConfig`]. This module compiles that
//! document into a [`ForgeConfig`] plus [`TaskDefinition`] values so Python
//! Gymnasium episodes and `forge-eval` consume the same artifact.
//!
//! Extra keys in existing scenario files (waypoints, fog of war, etc.) are
//! ignored. Unmapped objective types (patrol, escort, SAR) yield an empty
//! `scenario_tasks` list rather than a fake [`Predicate::TimeElapsed`].

use std::path::Path;

use serde::Deserialize;
use tracing::{debug, instrument, warn};

use crate::config::{AgriConfig, DroneConfig, ForgeConfig};
use crate::constants;
use crate::error::ConfigError;
use crate::grid::Position;
use crate::task::{Predicate, TaskComposition, TaskDefinition, TaskTier};

/// Result of compiling a high-level `[scenario]` TOML document.
#[derive(Debug, Clone)]
pub struct CompiledScenario {
    /// Stable identifier derived from `scenario.name` (or the file stem).
    pub id: String,
    /// Human-readable scenario name.
    pub name: String,
    /// Optional description copied from the manifest.
    pub description: Option<String>,
    /// Difficulty tier in `1..=6`.
    pub tier: u8,
    /// Simulation configuration, including compiled `task.scenario_tasks`.
    pub forge_config: ForgeConfig,
    /// Episode step cap from `objectives.time_limit`, when present.
    pub max_steps: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HighLevelScenarioFile {
    scenario: HighLevelScenario,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct HighLevelScenario {
    name: Option<String>,
    description: Option<String>,
    min_agents: Option<u32>,
    max_agents: Option<u32>,
    map: HighLevelMap,
    drone: HighLevelDrone,
    agri: HighLevelAgri,
    objectives: HighLevelObjectives,
    difficulty: HighLevelDifficulty,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct HighLevelMap {
    grid_size: Option<u16>,
    cropland_density: Option<f32>,
    pasture_density: Option<f32>,
    geofence_enabled: Option<bool>,
    geofence_margin: Option<u16>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
struct HighLevelPosition {
    x: u16,
    y: u16,
}

impl From<HighLevelPosition> for Position {
    fn from(value: HighLevelPosition) -> Self {
        Position::new(value.x, value.y)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct HighLevelDrone {
    enabled: Option<bool>,
    num_aerial: Option<u32>,
    num_ground_vehicles: Option<u32>,
    max_altitude: Option<u8>,
    starting_battery: Option<i32>,
    max_battery: Option<i32>,
    aerial_drain_rate: Option<i32>,
    recharge_rate: Option<i32>,
    scan_range: Option<u8>,
    scan_cost: Option<i32>,
    hover_cost: Option<i32>,
    ascend_cost: Option<i32>,
    descend_cost: Option<i32>,
    altitude_vision_bonus: Option<u8>,
    vehicle_turn_radius: Option<u8>,
    fall_damage_per_level: Option<i32>,
    restrict_recharge_to_chargers: Option<bool>,
    charger_tiles: Vec<HighLevelPosition>,
    spawn_home: Option<HighLevelPosition>,
    battery_action_floor: Option<i32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct HighLevelAgri {
    enabled: Option<bool>,
    ndvi_scan_radius: Option<u8>,
    thermal_scan_radius: Option<u8>,
    scan_battery_cost: Option<i32>,
    spray_radius: Option<u8>,
    spray_efficacy: Option<i32>,
    spray_battery_cost: Option<i32>,
    disease_spread_rate: Option<i32>,
    num_soil_nodes: Option<u16>,
    soil_relay_range: Option<u8>,
    soil_reading_interval: Option<u32>,
    report_generation_cost: Option<i32>,
    report_scan_radius: Option<u8>,
    moisture_drain_rate: Option<i32>,
    cropland_density: Option<f32>,
    pasture_density: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct HighLevelObjectives {
    #[serde(rename = "type")]
    objective_type: Option<String>,
    steps: Vec<String>,
    survey_threshold: Option<f32>,
    spray_threshold: Option<f32>,
    collect_threshold: Option<u16>,
    battery_threshold: Option<f32>,
    home: Option<HighLevelPosition>,
    time_limit: Option<u64>,
    completion_bonus: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct HighLevelDifficulty {
    base_tier: Option<u8>,
    disease_spread_rate: Option<i32>,
}

/// Compiles a high-level scenario TOML string.
///
/// `source_stem` is used when `scenario.name` is missing (typically the
/// file stem).
#[instrument(skip_all, fields(stem = source_stem))]
pub fn compile_high_level_scenario(
    toml_str: &str,
    source_stem: &str,
) -> Result<CompiledScenario, ConfigError> {
    let file: HighLevelScenarioFile =
        toml::from_str(toml_str).map_err(|e| ConfigError::ParseError(e.to_string()))?;
    Ok(compile_parsed(&file.scenario, source_stem))
}

/// Compiles a high-level scenario TOML file from disk.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn compile_high_level_path(path: impl AsRef<Path>) -> Result<CompiledScenario, ConfigError> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .map_err(|e| ConfigError::ParseError(format!("read {}: {e}", path.display())))?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scenario");
    compile_high_level_scenario(&raw, stem)
}

fn compile_parsed(scenario: &HighLevelScenario, source_stem: &str) -> CompiledScenario {
    let name = scenario
        .name
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| source_stem.to_string());
    let id = {
        let normalized = normalize_identifier(&name);
        if normalized.is_empty() {
            normalize_identifier(source_stem)
        } else {
            normalized
        }
    };

    let mut config = ForgeConfig::default();

    if let Some(grid_size) = scenario.map.grid_size {
        config.world.width = grid_size;
        config.world.height = grid_size;
    }

    if let Some(n) = scenario.min_agents {
        config.agents.num_agents = n.max(1);
    }

    if let Some(time_limit) = scenario.objectives.time_limit {
        config.task.max_episode_length = time_limit;
    }

    let tier = scenario.difficulty.base_tier.unwrap_or(1).clamp(1, 6);
    config.task.max_tier = tier;

    apply_drone(&mut config.drone, &scenario.drone);
    apply_agri(&mut config.agri, &scenario.agri, &scenario.map);
    apply_geofence(&mut config.world, &scenario.map);
    if let Some(rate) = scenario.difficulty.disease_spread_rate {
        config.agri.disease_spread_rate = rate;
    }
    if config.drone.restrict_recharge_to_chargers && config.drone.charger_tiles.is_empty() {
        if let Some(home) = config.drone.spawn_home {
            config.drone.charger_tiles.push(home);
        }
    }

    let home = resolve_coverage_home(&scenario.objectives, &config);
    let tasks = compile_objectives(&scenario.objectives, tier, config.agri.num_soil_nodes, home);
    let needs_agri = tasks.iter().any(task_needs_agri);
    if needs_agri {
        config.agri.enabled = true;
        config.drone.enabled = true;
    }
    if config.drone.enabled && config.drone.num_aerial == 0 && config.drone.num_ground_vehicles == 0
    {
        config.drone.num_aerial = constants::DEFAULT_SCENARIO_NUM_AERIAL;
    }

    config.task.scenario_tasks = tasks;

    debug!(
        id = %id,
        drone = config.drone.enabled,
        agri = config.agri.enabled,
        tasks = config.task.scenario_tasks.len(),
        "compiled high-level scenario"
    );

    CompiledScenario {
        id,
        name,
        description: scenario.description.clone(),
        tier,
        max_steps: scenario.objectives.time_limit,
        forge_config: config,
    }
}

fn apply_drone(dst: &mut DroneConfig, src: &HighLevelDrone) {
    if let Some(v) = src.enabled {
        dst.enabled = v;
    }
    if let Some(v) = src.num_aerial {
        dst.num_aerial = v;
    }
    if let Some(v) = src.num_ground_vehicles {
        dst.num_ground_vehicles = v;
    }
    if let Some(v) = src.max_altitude {
        dst.max_altitude = v;
    }
    if let Some(v) = src.starting_battery {
        dst.starting_battery = v;
    }
    if let Some(v) = src.max_battery {
        dst.max_battery = v;
    }
    if let Some(v) = src.aerial_drain_rate {
        dst.aerial_drain_rate = v;
    }
    if let Some(v) = src.recharge_rate {
        dst.recharge_rate = v;
    }
    if let Some(v) = src.scan_range {
        dst.scan_range = v;
    }
    if let Some(v) = src.scan_cost {
        dst.scan_cost = v;
    }
    if let Some(v) = src.hover_cost {
        dst.hover_cost = v;
    }
    if let Some(v) = src.ascend_cost {
        dst.ascend_cost = v;
    }
    if let Some(v) = src.descend_cost {
        dst.descend_cost = v;
    }
    if let Some(v) = src.altitude_vision_bonus {
        dst.altitude_vision_bonus = v;
    }
    if let Some(v) = src.vehicle_turn_radius {
        dst.vehicle_turn_radius = v;
    }
    if let Some(v) = src.fall_damage_per_level {
        dst.fall_damage_per_level = v;
    }
    if let Some(v) = src.restrict_recharge_to_chargers {
        dst.restrict_recharge_to_chargers = v;
    }
    if !src.charger_tiles.is_empty() {
        dst.charger_tiles = src.charger_tiles.iter().copied().map(Into::into).collect();
    }
    if let Some(home) = src.spawn_home {
        dst.spawn_home = Some(home.into());
    }
    if let Some(v) = src.battery_action_floor {
        dst.battery_action_floor = v;
    }
}

fn apply_agri(dst: &mut AgriConfig, src: &HighLevelAgri, map: &HighLevelMap) {
    if let Some(v) = src.enabled {
        dst.enabled = v;
    }
    if let Some(v) = src.ndvi_scan_radius {
        dst.ndvi_scan_radius = v;
    }
    if let Some(v) = src.thermal_scan_radius {
        dst.thermal_scan_radius = v;
    }
    if let Some(v) = src.scan_battery_cost {
        dst.scan_battery_cost = v;
    }
    if let Some(v) = src.spray_radius {
        dst.spray_radius = v;
    }
    if let Some(v) = src.spray_efficacy {
        dst.spray_efficacy = v;
    }
    if let Some(v) = src.spray_battery_cost {
        dst.spray_battery_cost = v;
    }
    if let Some(v) = src.disease_spread_rate {
        dst.disease_spread_rate = v;
    }
    if let Some(v) = src.num_soil_nodes {
        dst.num_soil_nodes = v;
    }
    if let Some(v) = src.soil_relay_range {
        dst.soil_relay_range = v;
    }
    if let Some(v) = src.soil_reading_interval {
        dst.soil_reading_interval = v;
    }
    if let Some(v) = src.report_generation_cost {
        dst.report_generation_cost = v;
    }
    if let Some(v) = src.report_scan_radius {
        dst.report_scan_radius = v;
    }
    if let Some(v) = src.moisture_drain_rate {
        dst.moisture_drain_rate = v;
    }
    if let Some(v) = src.cropland_density.or(map.cropland_density) {
        dst.cropland_density = v;
    }
    if let Some(v) = src.pasture_density.or(map.pasture_density) {
        dst.pasture_density = v;
    }
}

fn apply_geofence(world: &mut crate::config::WorldConfig, map: &HighLevelMap) {
    if let Some(v) = map.geofence_enabled {
        world.geofence_enabled = v;
    }
    if let Some(v) = map.geofence_margin {
        world.geofence_margin = v;
    }
}

fn resolve_coverage_home(obj: &HighLevelObjectives, config: &ForgeConfig) -> Position {
    if let Some(home) = obj.home {
        return home.into();
    }
    if let Some(home) = config.drone.spawn_home {
        return home;
    }
    if let Some(tile) = config.drone.charger_tiles.first() {
        return *tile;
    }
    Position::new(
        constants::DEFAULT_SPAWN_HOME_X,
        constants::DEFAULT_SPAWN_HOME_Y,
    )
}

fn compile_objectives(
    obj: &HighLevelObjectives,
    tier: u8,
    soil_nodes: u16,
    home: Position,
) -> Vec<TaskDefinition> {
    let reward = obj
        .completion_bonus
        .unwrap_or(constants::DEFAULT_REWARD_SCALE);
    let estimated_steps = obj
        .time_limit
        .unwrap_or(constants::DEFAULT_MAX_EPISODE_LENGTH) as u32;
    let kind = obj.objective_type.as_deref().unwrap_or("");

    let goal = match kind {
        "survey" => Some(TaskComposition::Atom(Predicate::FieldSurveyed(
            obj.survey_threshold
                .unwrap_or(constants::DEFAULT_SURVEY_THRESHOLD),
        ))),
        "collect" => Some(TaskComposition::Atom(Predicate::SoilDataCollected(
            0,
            obj.collect_threshold
                .unwrap_or(soil_nodes.max(constants::DEFAULT_SOIL_COLLECT_COUNT)),
        ))),
        "coverage" | "orchard" => Some(TaskComposition::And(vec![
            TaskComposition::Atom(Predicate::FieldSurveyed(
                obj.survey_threshold
                    .unwrap_or(constants::DEFAULT_SURVEY_THRESHOLD),
            )),
            TaskComposition::Atom(Predicate::BatteryAbove(
                0,
                obj.battery_threshold
                    .unwrap_or(constants::DEFAULT_COVERAGE_BATTERY_THRESHOLD),
            )),
            TaskComposition::Atom(Predicate::AgentAt(0, home)),
        ])),
        "sequence" => {
            let steps: Vec<TaskComposition> = obj
                .steps
                .iter()
                .filter_map(|step| map_sequence_step(step, obj, soil_nodes))
                .collect();
            if steps.is_empty() {
                None
            } else {
                Some(TaskComposition::Sequence(steps))
            }
        }
        "" | "patrol" | "escort" | "search_and_rescue" | "sar" => None,
        other => {
            warn!(objective_type = other, "unmapped high-level objective type");
            None
        }
    };

    match goal {
        Some(goal) => vec![make_task(
            1,
            format!("{kind} objective"),
            goal,
            tier,
            estimated_steps,
            reward,
        )],
        None => Vec::new(),
    }
}

fn map_sequence_step(
    step: &str,
    obj: &HighLevelObjectives,
    soil_nodes: u16,
) -> Option<TaskComposition> {
    let atom = match step {
        "survey" => Predicate::FieldSurveyed(
            obj.survey_threshold
                .unwrap_or(constants::DEFAULT_SURVEY_THRESHOLD),
        ),
        "spray" => Predicate::AreaSprayed(
            obj.spray_threshold
                .unwrap_or(constants::DEFAULT_SPRAY_THRESHOLD),
        ),
        "relay_soil" | "collect" => Predicate::SoilDataCollected(
            0,
            obj.collect_threshold
                .unwrap_or(soil_nodes.max(constants::DEFAULT_SOIL_COLLECT_COUNT)),
        ),
        "generate_report" | "report" => Predicate::FieldReportGenerated(0),
        other => {
            warn!(step = other, "unmapped sequence step; skipping");
            return None;
        }
    };
    Some(TaskComposition::Atom(atom))
}

fn make_task(
    id: u64,
    description: String,
    goal: TaskComposition,
    tier: u8,
    estimated_steps: u32,
    reward: f32,
) -> TaskDefinition {
    TaskDefinition {
        id,
        description,
        goal,
        tier: TaskTier::new(tier),
        estimated_steps,
        reward,
        dense_reward_weights: vec![1.0],
    }
}

fn task_needs_agri(task: &TaskDefinition) -> bool {
    composition_needs_agri(&task.goal)
}

fn composition_needs_agri(comp: &TaskComposition) -> bool {
    match comp {
        TaskComposition::Atom(pred) => matches!(
            pred,
            Predicate::FieldSurveyed(_)
                | Predicate::AreaSprayed(_)
                | Predicate::SoilDataCollected(_, _)
                | Predicate::FieldReportGenerated(_)
                | Predicate::CropHealthBelow(_, _)
                | Predicate::DiseaseDetected(_, _)
                | Predicate::IrrigationMapped(_)
        ),
        TaskComposition::And(xs) | TaskComposition::Or(xs) | TaskComposition::Sequence(xs) => {
            xs.iter().any(composition_needs_agri)
        }
        TaskComposition::Before(inner, _) | TaskComposition::Without(inner, _) => {
            composition_needs_agri(inner)
        }
        TaskComposition::While(cond, goal) => {
            composition_needs_agri(cond) || composition_needs_agri(goal)
        }
    }
}

fn normalize_identifier(value: &str) -> String {
    let mut out = String::new();
    let mut prev_sep = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        } else if !out.is_empty() && !prev_sep {
            out.push('_');
            prev_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CROP_SCOUT: &str = include_str!("../../../configs/scenarios/crop_scout.toml");
    const PATROL: &str = include_str!("../../../configs/scenarios/patrol.toml");
    const SPRAY: &str = include_str!("../../../configs/scenarios/spray_mission.toml");
    const SOIL: &str = include_str!("../../../configs/scenarios/soil_relay.toml");
    const ORCHARD: &str = include_str!("../../../configs/scenarios/orchard_coverage.toml");

    #[test]
    fn crop_scout_enables_drone_and_agri_and_survey_task() {
        let compiled = compile_high_level_scenario(CROP_SCOUT, "crop_scout").unwrap();
        assert_eq!(compiled.id, "crop_scout");
        assert!(compiled.forge_config.drone.enabled);
        assert!(compiled.forge_config.agri.enabled);
        assert_eq!(compiled.forge_config.drone.num_aerial, 1);
        assert_eq!(compiled.forge_config.world.width, 48);
        assert_eq!(compiled.forge_config.task.max_episode_length, 1500);
        assert_eq!(compiled.forge_config.task.scenario_tasks.len(), 1);
        match &compiled.forge_config.task.scenario_tasks[0].goal {
            TaskComposition::Atom(Predicate::FieldSurveyed(threshold)) => {
                assert!((threshold - 0.8).abs() < f32::EPSILON);
            }
            other => panic!("expected FieldSurveyed, got {other:?}"),
        }
    }

    #[test]
    fn patrol_does_not_invent_tasks() {
        let compiled = compile_high_level_scenario(PATROL, "patrol").unwrap();
        assert!(compiled.forge_config.task.scenario_tasks.is_empty());
        assert!(!compiled.forge_config.agri.enabled);
        assert_eq!(compiled.forge_config.world.width, 64);
        assert_eq!(compiled.forge_config.task.max_episode_length, 2000);
    }

    #[test]
    fn spray_mission_sequence_survey_then_spray() {
        let compiled = compile_high_level_scenario(SPRAY, "spray_mission").unwrap();
        match &compiled.forge_config.task.scenario_tasks[0].goal {
            TaskComposition::Sequence(steps) => {
                assert_eq!(steps.len(), 2);
                assert!(matches!(
                    steps[0],
                    TaskComposition::Atom(Predicate::FieldSurveyed(_))
                ));
                assert!(matches!(
                    steps[1],
                    TaskComposition::Atom(Predicate::AreaSprayed(_))
                ));
            }
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    #[test]
    fn soil_relay_collects_nodes() {
        let compiled = compile_high_level_scenario(SOIL, "soil_relay").unwrap();
        match &compiled.forge_config.task.scenario_tasks[0].goal {
            TaskComposition::Atom(Predicate::SoilDataCollected(agent, count)) => {
                assert_eq!(*agent, 0);
                assert_eq!(*count, 10);
            }
            other => panic!("expected SoilDataCollected, got {other:?}"),
        }
    }

    #[test]
    fn orchard_coverage_emits_survey_battery_home_and() {
        let compiled = compile_high_level_scenario(ORCHARD, "orchard_coverage").unwrap();
        assert!(compiled.forge_config.drone.enabled);
        assert!(compiled.forge_config.agri.enabled);
        assert!(compiled.forge_config.drone.restrict_recharge_to_chargers);
        assert_eq!(
            compiled.forge_config.drone.spawn_home,
            Some(Position::new(0, 0))
        );
        assert_eq!(
            compiled.forge_config.drone.charger_tiles,
            vec![Position::new(0, 0)]
        );
        assert!(compiled.forge_config.world.geofence_enabled);
        match &compiled.forge_config.task.scenario_tasks[0].goal {
            TaskComposition::And(parts) => {
                assert_eq!(parts.len(), 3);
                assert!(matches!(
                    parts[0],
                    TaskComposition::Atom(Predicate::FieldSurveyed(_))
                ));
                assert!(matches!(
                    parts[1],
                    TaskComposition::Atom(Predicate::BatteryAbove(0, _))
                ));
                assert!(matches!(
                    parts[2],
                    TaskComposition::Atom(Predicate::AgentAt(0, Position { x: 0, y: 0 }))
                ));
            }
            other => panic!("expected And coverage goal, got {other:?}"),
        }
    }

    #[test]
    fn extra_keys_are_ignored() {
        let toml = r#"
[scenario]
name = "crop_scout"
min_agents = 1
unknown_top = true

[scenario.map]
grid_size = 16
terrain_type = "agricultural"
num_waypoints = 9

[scenario.drone]
enabled = true
num_aerial = 1

[scenario.agri]
enabled = true
initial_disease_tiles = 10

[scenario.objectives]
type = "survey"
survey_threshold = 0.5
time_limit = 100
"#;
        let compiled = compile_high_level_scenario(toml, "x").unwrap();
        assert!(compiled.forge_config.drone.enabled);
        assert!(compiled.forge_config.agri.enabled);
    }

    #[test]
    fn normalize_identifier_matches_python() {
        assert_eq!(normalize_identifier("Patrol"), "patrol");
        assert_eq!(normalize_identifier("Crop Scout"), "crop_scout");
        assert_eq!(normalize_identifier("soil_relay"), "soil_relay");
    }

    #[test]
    fn xlang_orchard_coverage_pinned_to_known_good() {
        let compiled = compile_high_level_scenario(ORCHARD, "orchard_coverage").unwrap();
        assert_eq!(compiled.id, "orchard_coverage");
        assert_eq!(compiled.forge_config.world.width, 16);
        assert_eq!(compiled.forge_config.world.height, 16);
        assert!(compiled.forge_config.world.geofence_enabled);
        assert_eq!(compiled.forge_config.world.geofence_margin, 1);
        assert!(compiled.forge_config.drone.enabled);
        assert_eq!(compiled.forge_config.drone.num_aerial, 1);
        assert!(compiled.forge_config.drone.restrict_recharge_to_chargers);
        assert_eq!(
            compiled.forge_config.drone.spawn_home,
            Some(Position::new(0, 0))
        );
        assert_eq!(
            compiled.forge_config.drone.charger_tiles,
            vec![Position::new(0, 0)]
        );
        assert!(compiled.forge_config.agri.enabled);
        assert_eq!(compiled.forge_config.task.max_episode_length, 800);
        match &compiled.forge_config.task.scenario_tasks[0].goal {
            TaskComposition::And(parts) => {
                assert_eq!(parts.len(), 3);
                match &parts[0] {
                    TaskComposition::Atom(Predicate::FieldSurveyed(threshold)) => {
                        assert!((*threshold - 0.8).abs() < f32::EPSILON);
                    }
                    other => panic!("expected FieldSurveyed, got {other:?}"),
                }
                match &parts[1] {
                    TaskComposition::Atom(Predicate::BatteryAbove(0, threshold)) => {
                        assert!((*threshold - 0.2).abs() < f32::EPSILON);
                    }
                    other => panic!("expected BatteryAbove, got {other:?}"),
                }
                match &parts[2] {
                    TaskComposition::Atom(Predicate::AgentAt(0, pos)) => {
                        assert_eq!(*pos, Position::new(0, 0));
                    }
                    other => panic!("expected AgentAt home, got {other:?}"),
                }
            }
            other => panic!("expected And coverage goal, got {other:?}"),
        }
    }

    #[test]
    fn xlang_crop_scout_pinned_to_known_good() {
        let compiled = compile_high_level_scenario(CROP_SCOUT, "crop_scout").unwrap();
        assert_eq!(compiled.id, "crop_scout");
        assert_eq!(compiled.forge_config.world.width, 48);
        assert_eq!(compiled.forge_config.world.height, 48);
        assert!(!compiled.forge_config.world.geofence_enabled);
        assert!(compiled.forge_config.drone.enabled);
        assert_eq!(compiled.forge_config.drone.num_aerial, 1);
        assert!(compiled.forge_config.agri.enabled);
        assert_eq!(compiled.forge_config.task.max_episode_length, 1500);
        match &compiled.forge_config.task.scenario_tasks[0].goal {
            TaskComposition::Atom(Predicate::FieldSurveyed(threshold)) => {
                assert!((*threshold - 0.8).abs() < f32::EPSILON);
            }
            other => panic!("expected FieldSurveyed, got {other:?}"),
        }
    }
}
