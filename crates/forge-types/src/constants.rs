//! Default constants for the FORGE simulation platform.
//!
//! These provide default values for all configuration structs.
//! Every constant is overridable via the corresponding config field.
//! Fixed-point values use i32 with 16 fractional bits (FixedI32<U16>):
//! value 65536 = 1.0, value 32768 = 0.5, etc.

/// Default world width in tiles.
pub const DEFAULT_WORLD_WIDTH: u16 = 64;
/// Default world height in tiles.
pub const DEFAULT_WORLD_HEIGHT: u16 = 64;
/// Default RNG seed.
pub const DEFAULT_SEED: u64 = 0;
/// Default biome Perlin noise scale.
pub const DEFAULT_BIOME_SCALE: f32 = 0.1;
/// Default resource placement density (0.0–1.0).
pub const DEFAULT_RESOURCE_DENSITY: f32 = 0.3;
/// Default day/night cycle length in ticks.
pub const DEFAULT_DAY_NIGHT_CYCLE_LENGTH: u32 = 1000;
/// Default maximum number of entities in the world.
pub const DEFAULT_MAX_ENTITIES: u16 = 256;
/// Minimum allowed world dimension (width or height).
pub const MIN_WORLD_DIMENSION: u16 = 8;
/// Maximum allowed world dimension (width or height).
pub const MAX_WORLD_DIMENSION: u16 = 256;

/// Default stamina cost per movement action (fixed-point ~0.1).
pub const DEFAULT_STAMINA_COST_MOVE: i32 = 6554; // ~0.1
/// Default stamina regeneration rate per tick (fixed-point ~0.05).
pub const DEFAULT_STAMINA_REGEN_RATE: i32 = 3277; // ~0.05
/// Default maximum velocity in tiles per tick (fixed-point 1.0).
pub const DEFAULT_MAX_VELOCITY: i32 = 65536; // 1.0 tile/tick
/// Default friction coefficient (fixed-point ~0.8).
pub const DEFAULT_FRICTION: i32 = 52429; // ~0.8

/// Default number of agents.
pub const DEFAULT_NUM_AGENTS: u32 = 1;
/// Default agent vision radius in tiles.
pub const DEFAULT_VISION_RADIUS: u8 = 5;
/// Default agent inventory carry capacity.
pub const DEFAULT_CARRY_CAPACITY: u8 = 10;
/// Default starting health (fixed-point 10.0).
pub const DEFAULT_STARTING_HEALTH: i32 = 655360; // 10.0
/// Default maximum health (fixed-point 10.0).
pub const DEFAULT_MAX_HEALTH: i32 = 655360; // 10.0
/// Default starting stamina (fixed-point 10.0).
pub const DEFAULT_STARTING_STAMINA: i32 = 655360; // 10.0
/// Default maximum stamina (fixed-point 10.0).
pub const DEFAULT_MAX_STAMINA: i32 = 655360; // 10.0
/// Default communication vocabulary size.
pub const DEFAULT_COMM_VOCAB_SIZE: u16 = 16;
/// Default communication broadcast radius.
pub const DEFAULT_COMM_RADIUS: u16 = 10;
/// Default communication buffer size per agent.
pub const DEFAULT_COMM_BUFFER_SIZE: u8 = 8;

/// Default maximum number of ingredients per crafting recipe.
pub const DEFAULT_MAX_INGREDIENTS: u8 = 4;

/// Default maximum task difficulty tier.
pub const DEFAULT_MAX_TASK_TIER: u8 = 6;
/// Default maximum number of predicates per task.
pub const DEFAULT_MAX_PREDICATES: u16 = 32;
/// Default maximum episode length in ticks.
pub const DEFAULT_MAX_EPISODE_LENGTH: u64 = 10000;
/// Default reward scaling factor.
pub const DEFAULT_REWARD_SCALE: f32 = 1.0;

/// Default target success rate for curriculum advancement.
pub const DEFAULT_TARGET_SUCCESS_RATE: f32 = 0.5;
/// Default sliding window size for curriculum statistics.
pub const DEFAULT_CURRICULUM_WINDOW_SIZE: u32 = 100;
/// Default number of warmup episodes before curriculum adapts.
pub const DEFAULT_CURRICULUM_WARMUP: u32 = 50;
/// Default rate of curriculum difficulty adjustment.
pub const DEFAULT_CURRICULUM_ADJUSTMENT_RATE: f32 = 0.1;

/// Default render output width in pixels.
pub const DEFAULT_PIXEL_WIDTH: u32 = 64;
/// Default render output height in pixels.
pub const DEFAULT_PIXEL_HEIGHT: u32 = 64;

/// Default inventory size (number of slots).
pub const DEFAULT_INVENTORY_SIZE: usize = 10;
/// Maximum items per inventory stack.
pub const MAX_STACK_SIZE: u16 = 64;

/// Default ticks between resource respawn increments.
pub const DEFAULT_RESOURCE_RESPAWN_TICKS: u32 = 100;
/// Default maximum quantity per resource node.
pub const DEFAULT_RESOURCE_MAX_QUANTITY: u16 = 5;
/// Default object placement density scale.
pub const DEFAULT_OBJECT_DENSITY_SCALE: f32 = 1.0;

/// Number of terrain types (for observation encoding).
pub const NUM_TERRAIN_TYPES: usize = 8;

/// Default damage dealt by a sword attack per hit.
pub const DEFAULT_SWORD_DAMAGE: i32 = 196608; // 3.0
/// Default damage per tick from standing on lava.
pub const DEFAULT_LAVA_DAMAGE: i32 = 65536; // 1.0

/// Number of cardinal directions.
pub const NUM_DIRECTIONS: usize = 4;

/// Number of inventory drop action slots in the action encoding.
pub const ACTION_DROP_SLOTS: usize = 10;
/// Number of inventory use action slots in the action encoding.
pub const ACTION_USE_SLOTS: usize = 10;
/// Number of craft recipe action slots in the action encoding.
pub const ACTION_CRAFT_SLOTS: usize = 9;

/// Number of fractional bits in fixed-point representation (16 bits).
pub const FIXED_POINT_SHIFT: u32 = 16;
/// 1.0 in fixed-point representation.
pub const FIXED_POINT_ONE: i32 = 1 << FIXED_POINT_SHIFT; // 65536

/// Base water level threshold for biome classification (before scale adjustment).
pub const BIOME_WATER_LEVEL_BASE: f64 = 0.35;
/// Water level adjustment rate per biome scale unit.
pub const BIOME_WATER_LEVEL_SCALE_MULTIPLIER: f64 = 0.3;
/// Minimum water level threshold.
pub const BIOME_WATER_LEVEL_MIN: f64 = 0.10;
/// Maximum water level threshold.
pub const BIOME_WATER_LEVEL_MAX: f64 = 0.50;
/// Base mountain level threshold for biome classification.
pub const BIOME_MOUNTAIN_LEVEL_BASE: f64 = 0.72;
/// Mountain level adjustment rate per biome scale unit.
pub const BIOME_MOUNTAIN_LEVEL_SCALE_MULTIPLIER: f64 = 0.3;
/// Minimum mountain level threshold.
pub const BIOME_MOUNTAIN_LEVEL_MIN: f64 = 0.60;
/// Maximum mountain level threshold.
pub const BIOME_MOUNTAIN_LEVEL_MAX: f64 = 0.90;
/// Sand level offset from water level.
pub const BIOME_SAND_LEVEL_OFFSET: f64 = 0.05;
/// Moisture threshold for forest biome.
pub const BIOME_FOREST_MOISTURE_THRESHOLD: f64 = 0.45;
/// Moisture threshold for desert biome.
pub const BIOME_DESERT_MOISTURE_THRESHOLD: f64 = 0.25;
/// Multiplier used to compute terrain noise octaves from biome_scale.
/// Higher values produce more octaves (more detail) in terrain generation.
pub const TERRAIN_NOISE_OCTAVES_MULTIPLIER: f64 = 40.0;
/// Minimum octaves for terrain noise.
pub const TERRAIN_NOISE_OCTAVES_MIN: u32 = 2;
/// Maximum octaves for terrain noise.
pub const TERRAIN_NOISE_OCTAVES_MAX: u32 = 8;
/// Perlin noise persistence value (controls fractal amplitude decay).
pub const TERRAIN_NOISE_PERSISTENCE: f64 = 0.5;

/// Sentinel value for "no object" in observation encoding.
pub const OBS_NO_OBJECT: u8 = 255;
/// Sentinel value for "no resource" in observation encoding.
pub const OBS_NO_RESOURCE: u8 = 255;
/// Sentinel value for empty inventory slot item type.
pub const OBS_EMPTY_SLOT_ITEM: u8 = 255;

/// Dawn phase index.
pub const DAY_PHASE_DAWN: u8 = 0;
/// Day phase index.
pub const DAY_PHASE_DAY: u8 = 1;
/// Dusk phase index.
pub const DAY_PHASE_DUSK: u8 = 2;
/// Night phase index.
pub const DAY_PHASE_NIGHT: u8 = 3;
/// Number of day/night phases.
pub const NUM_DAY_PHASES: u8 = 4;

/// Vision multiplier during day.
pub const VISION_MODIFIER_DAY: f32 = 1.0;
/// Vision multiplier during dawn/dusk.
pub const VISION_MODIFIER_TWILIGHT: f32 = 0.75;
/// Vision multiplier during night.
pub const VISION_MODIFIER_NIGHT: f32 = 0.5;

/// Number of features per tile in grid observation encoding.
pub const OBS_FEATURES_PER_TILE: usize = 7;

/// Number of additional observation fields when drone mechanics are enabled
/// (altitude, battery, morphology, heading).
pub const OBS_DRONE_FIELDS_COUNT: usize = 4;

// ---------- Terrain movement costs (fixed-point) ----------

/// Movement cost for Ground terrain (1.0x — normal speed).
pub const TERRAIN_COST_GROUND: i32 = FIXED_POINT_ONE;
/// Movement cost for Ice terrain (0.5x — slippery, reduced stamina).
pub const TERRAIN_COST_ICE: i32 = FIXED_POINT_ONE / 2;
/// Movement cost for Sand terrain (1.5x — slower movement).
pub const TERRAIN_COST_SAND: i32 = FIXED_POINT_ONE + FIXED_POINT_ONE / 2;
/// Movement cost for Forest terrain (2.0x — dense vegetation).
pub const TERRAIN_COST_FOREST: i32 = FIXED_POINT_ONE * 2;

// ---------- Item type classification boundaries ----------

/// Items with discriminant below this are raw (harvestable) resources.
pub const ITEM_TYPE_RAW_MAX: u8 = 10;
/// Items with discriminant in [ITEM_TYPE_CRAFTED_MIN, ITEM_TYPE_CRAFTED_MAX) are crafted.
pub const ITEM_TYPE_CRAFTED_MIN: u8 = 10;
/// Upper exclusive bound for crafted item discriminants.
pub const ITEM_TYPE_CRAFTED_MAX: u8 = 30;

// ---------- Drone defaults ----------

/// Whether drone mechanics are enabled by default.
pub const DEFAULT_DRONE_ENABLED: bool = false;
/// Default maximum altitude for aerial agents.
pub const DEFAULT_MAX_ALTITUDE: u8 = 10;
/// Default battery drain per tick while airborne (fixed-point ~0.15).
pub const DEFAULT_AERIAL_DRAIN_RATE: i32 = 9830;
/// Default battery cost to ascend one level (fixed-point ~0.2).
pub const DEFAULT_ASCEND_COST: i32 = 13107;
/// Default battery cost to descend one level (fixed-point ~0.05).
pub const DEFAULT_DESCEND_COST: i32 = 3277;
/// Default hover cost per tick (fixed-point ~0.1).
pub const DEFAULT_HOVER_COST: i32 = 6554;
/// Default scan action battery cost (fixed-point ~0.08).
pub const DEFAULT_SCAN_COST: i32 = 5243;
/// Default scan range in tiles.
pub const DEFAULT_SCAN_RANGE: u8 = 12;
/// Default starting battery for aerial agents (fixed-point 10.0).
pub const DEFAULT_STARTING_BATTERY: i32 = 655360;
/// Default maximum battery (fixed-point 10.0).
pub const DEFAULT_MAX_BATTERY: i32 = 655360;
/// Default battery recharge rate per tick when landed (fixed-point ~0.03).
pub const DEFAULT_RECHARGE_RATE: i32 = 1966;
/// Default vision bonus per altitude level for aerial agents.
pub const DEFAULT_ALTITUDE_VISION_BONUS: u8 = 2;
/// Default ground vehicle turn radius.
pub const DEFAULT_VEHICLE_TURN_RADIUS: u8 = 1;
/// Number of discrete drone action slots in the action space.
pub const DRONE_ACTION_COUNT: u32 = 19;
/// Default ground vehicle terrain costs [Ground, Water, Wall, Lava, Ice, Sand, Forest, Mountain].
/// Fixed-point values. i32::MAX = impassable.
pub const DEFAULT_VEHICLE_TERRAIN_COSTS: [i32; 8] = [
    32768,    // Ground: 0.5x (faster)
    i32::MAX, // Water: impassable
    i32::MAX, // Wall: impassable
    i32::MAX, // Lava: impassable
    49152,    // Ice: 0.75x
    45875,    // Sand: 0.7x
    i32::MAX, // Forest: impassable
    i32::MAX, // Mountain: impassable
];
/// Number of terrain types used for vehicle terrain cost array sizing.
pub const NUM_VEHICLE_TERRAIN_TYPES: usize = 8;
/// Default fall damage per altitude level during emergency landing (fixed-point 1.0).
pub const DEFAULT_FALL_DAMAGE_PER_LEVEL: i32 = FIXED_POINT_ONE;

// ---------- Health monitoring defaults ----------

/// Whether health monitoring is enabled by default.
pub const DEFAULT_HEALTH_MONITORING_ENABLED: bool = false;
/// Number of component types tracked for degradation (Motor, Sensor, Structure).
pub const NUM_COMPONENT_TYPES: usize = 3;
/// Number of discrete degradation levels per component for the health state library.
pub const DEFAULT_NUM_DEGRADATION_LEVELS: u8 = 5;
/// Per-tick degradation rate for components under active use (fixed-point ~0.0005).
pub const DEFAULT_DEGRADATION_RATE: i32 = 33;
/// Minimum motor efficiency below which movement fails (fixed-point 0.25).
pub const DEFAULT_MOTOR_EFFICIENCY_FLOOR: i32 = FIXED_POINT_ONE / 4;
/// Per-tick sensor drift rate that increases observation noise (fixed-point ~0.00025).
pub const DEFAULT_SENSOR_DRIFT_RATE: i32 = 16;
/// Minimum observation noise sigma (fixed-point 0.0).
pub const DEFAULT_SENSOR_NOISE_FLOOR: i32 = 0;
/// Maximum observation noise sigma (fixed-point 0.5).
pub const DEFAULT_SENSOR_NOISE_CEILING: i32 = FIXED_POINT_ONE / 2;
/// Damage scale factor for structural degradation (fixed-point 1.0).
pub const DEFAULT_STRUCTURAL_DAMAGE_SCALE: i32 = FIXED_POINT_ONE;
/// Base noise applied to health/stamina/battery observation readings (fixed-point ~0.1).
pub const DEFAULT_OBSERVATION_NOISE_SCALE: i32 = 6554;
/// Number of additional observation fields when health monitoring is enabled.
/// [motor_integrity, sensor_integrity, structure_integrity, noisy_health,
///  noisy_stamina, noisy_battery, integrity_estimate].
pub const OBS_HEALTH_MONITORING_FIELDS_COUNT: usize = 7;
/// Whether component degradation is included in observations by default.
pub const DEFAULT_OBSERVABLE_DEGRADATION: bool = true;

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_one_equals_shift() {
        assert_eq!(FIXED_POINT_ONE, 1 << FIXED_POINT_SHIFT);
        assert_eq!(FIXED_POINT_ONE, 65536);
    }

    #[test]
    fn test_starting_health_equals_max_health() {
        assert_eq!(DEFAULT_STARTING_HEALTH, DEFAULT_MAX_HEALTH);
    }

    #[test]
    fn test_starting_stamina_equals_max_stamina() {
        assert_eq!(DEFAULT_STARTING_STAMINA, DEFAULT_MAX_STAMINA);
    }

    #[test]
    fn test_num_day_phases() {
        assert_eq!(NUM_DAY_PHASES, 4);
    }

    #[test]
    fn test_min_less_than_max_world_dimension() {
        const { assert!(MIN_WORLD_DIMENSION < MAX_WORLD_DIMENSION) };
    }

    #[test]
    fn test_default_world_dimensions_within_range() {
        const { assert!(DEFAULT_WORLD_WIDTH >= MIN_WORLD_DIMENSION) };
        const { assert!(DEFAULT_WORLD_WIDTH <= MAX_WORLD_DIMENSION) };
        const { assert!(DEFAULT_WORLD_HEIGHT >= MIN_WORLD_DIMENSION) };
        const { assert!(DEFAULT_WORLD_HEIGHT <= MAX_WORLD_DIMENSION) };
    }

    #[test]
    fn test_obs_features_per_tile() {
        assert_eq!(OBS_FEATURES_PER_TILE, 7);
    }

    #[test]
    fn test_drone_constants_starting_battery_equals_max() {
        assert_eq!(DEFAULT_STARTING_BATTERY, DEFAULT_MAX_BATTERY);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_drone_ascend_cost_within_battery() {
        assert!(DEFAULT_ASCEND_COST < DEFAULT_STARTING_BATTERY);
    }

    #[test]
    fn test_vehicle_terrain_costs_length() {
        assert_eq!(
            DEFAULT_VEHICLE_TERRAIN_COSTS.len(),
            NUM_VEHICLE_TERRAIN_TYPES
        );
    }

    #[test]
    fn test_drone_action_count() {
        // 5 basic (Ascend, Descend, Hover, TakeOff, Land) + 4 Scan + 10 DropPayload
        assert_eq!(DRONE_ACTION_COUNT, 19);
    }

    #[test]
    fn test_fall_damage_default_matches_fixed_point_one() {
        assert_eq!(DEFAULT_FALL_DAMAGE_PER_LEVEL, FIXED_POINT_ONE);
    }

    #[test]
    fn test_drone_costs_are_positive() {
        assert!(DEFAULT_AERIAL_DRAIN_RATE > 0);
        assert!(DEFAULT_ASCEND_COST > 0);
        assert!(DEFAULT_DESCEND_COST > 0);
        assert!(DEFAULT_HOVER_COST > 0);
        assert!(DEFAULT_SCAN_COST > 0);
        assert!(DEFAULT_RECHARGE_RATE > 0);
    }

    #[test]
    fn test_vehicle_terrain_costs_has_impassable() {
        // Water, Wall, Lava, Forest, Mountain should be impassable
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[1], i32::MAX); // Water
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[2], i32::MAX); // Wall
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[3], i32::MAX); // Lava
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[6], i32::MAX); // Forest
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[7], i32::MAX); // Mountain
    }

    #[test]
    fn test_vehicle_terrain_costs_ground_is_faster() {
        // Ground cost < FIXED_POINT_ONE means faster than walking
        assert!(DEFAULT_VEHICLE_TERRAIN_COSTS[0] < FIXED_POINT_ONE);
        assert!(DEFAULT_VEHICLE_TERRAIN_COSTS[0] > 0);
    }

    #[test]
    fn test_terrain_costs_derived_from_fixed_point() {
        assert_eq!(TERRAIN_COST_GROUND, FIXED_POINT_ONE);
        assert_eq!(TERRAIN_COST_ICE, FIXED_POINT_ONE / 2);
        assert_eq!(TERRAIN_COST_SAND, FIXED_POINT_ONE + FIXED_POINT_ONE / 2);
        assert_eq!(TERRAIN_COST_FOREST, FIXED_POINT_ONE * 2);
    }

    #[test]
    fn test_terrain_cost_ordering() {
        // Ice < Ground < Sand < Forest
        assert!(TERRAIN_COST_ICE < TERRAIN_COST_GROUND);
        assert!(TERRAIN_COST_GROUND < TERRAIN_COST_SAND);
        assert!(TERRAIN_COST_SAND < TERRAIN_COST_FOREST);
    }

    #[test]
    fn test_item_type_boundaries_consistent() {
        assert_eq!(ITEM_TYPE_RAW_MAX, ITEM_TYPE_CRAFTED_MIN);
        assert!(ITEM_TYPE_CRAFTED_MIN < ITEM_TYPE_CRAFTED_MAX);
    }

    #[test]
    fn test_health_monitoring_defaults() {
        assert!(!DEFAULT_HEALTH_MONITORING_ENABLED);
        assert_eq!(NUM_COMPONENT_TYPES, 3);
        assert!(DEFAULT_DEGRADATION_RATE > 0);
        assert!(DEFAULT_MOTOR_EFFICIENCY_FLOOR > 0);
        assert!(DEFAULT_MOTOR_EFFICIENCY_FLOOR < FIXED_POINT_ONE);
        assert!(DEFAULT_SENSOR_DRIFT_RATE > 0);
        assert!(DEFAULT_SENSOR_NOISE_FLOOR <= DEFAULT_SENSOR_NOISE_CEILING);
        assert!(DEFAULT_SENSOR_NOISE_CEILING <= FIXED_POINT_ONE);
        assert_eq!(DEFAULT_STRUCTURAL_DAMAGE_SCALE, FIXED_POINT_ONE);
        assert!(DEFAULT_OBSERVATION_NOISE_SCALE > 0);
        assert_eq!(OBS_HEALTH_MONITORING_FIELDS_COUNT, 7);
    }

    #[test]
    fn test_num_degradation_levels_nonzero() {
        assert!(DEFAULT_NUM_DEGRADATION_LEVELS >= 2);
    }
}
