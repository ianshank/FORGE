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

#[cfg(test)]
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
}
