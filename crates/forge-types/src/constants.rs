//! Default constants for the FORGE simulation platform.
//!
//! These provide default values for all configuration structs.
//! Every constant is overridable via the corresponding config field.
//! Fixed-point values use i32 with 16 fractional bits (FixedI32<U16>):
//! value 65536 = 1.0, value 32768 = 0.5, etc.

// World defaults
pub const DEFAULT_WORLD_WIDTH: u16 = 64;
pub const DEFAULT_WORLD_HEIGHT: u16 = 64;
pub const DEFAULT_SEED: u64 = 0;
pub const DEFAULT_BIOME_SCALE: f32 = 0.1;
pub const DEFAULT_RESOURCE_DENSITY: f32 = 0.3;
pub const DEFAULT_DAY_NIGHT_CYCLE_LENGTH: u32 = 1000;
pub const DEFAULT_MAX_ENTITIES: u16 = 256;
pub const MIN_WORLD_DIMENSION: u16 = 8;
pub const MAX_WORLD_DIMENSION: u16 = 256;

// Physics defaults (fixed-point i32 with 16 fractional bits: 65536 = 1.0)
pub const DEFAULT_STAMINA_COST_MOVE: i32 = 6554; // ~0.1
pub const DEFAULT_STAMINA_REGEN_RATE: i32 = 3277; // ~0.05
pub const DEFAULT_MAX_VELOCITY: i32 = 65536; // 1.0 tile/tick
pub const DEFAULT_FRICTION: i32 = 52429; // ~0.8

// Agent defaults
pub const DEFAULT_NUM_AGENTS: u32 = 1;
pub const DEFAULT_VISION_RADIUS: u8 = 5;
pub const DEFAULT_CARRY_CAPACITY: u8 = 10;
pub const DEFAULT_STARTING_HEALTH: i32 = 655360; // 10.0
pub const DEFAULT_MAX_HEALTH: i32 = 655360; // 10.0
pub const DEFAULT_STARTING_STAMINA: i32 = 655360; // 10.0
pub const DEFAULT_MAX_STAMINA: i32 = 655360; // 10.0
pub const DEFAULT_COMM_VOCAB_SIZE: u16 = 16;
pub const DEFAULT_COMM_RADIUS: u16 = 10;
pub const DEFAULT_COMM_BUFFER_SIZE: u8 = 8;

// Crafting defaults
pub const DEFAULT_MAX_INGREDIENTS: u8 = 4;

// Task defaults
pub const DEFAULT_MAX_TASK_TIER: u8 = 6;
pub const DEFAULT_MAX_PREDICATES: u16 = 32;
pub const DEFAULT_MAX_EPISODE_LENGTH: u64 = 10000;
pub const DEFAULT_REWARD_SCALE: f32 = 1.0;

// Curriculum defaults
pub const DEFAULT_TARGET_SUCCESS_RATE: f32 = 0.5;
pub const DEFAULT_CURRICULUM_WINDOW_SIZE: u32 = 100;
pub const DEFAULT_CURRICULUM_WARMUP: u32 = 50;
pub const DEFAULT_CURRICULUM_ADJUSTMENT_RATE: f32 = 0.1;

// Rendering defaults
pub const DEFAULT_PIXEL_WIDTH: u32 = 64;
pub const DEFAULT_PIXEL_HEIGHT: u32 = 64;

// Inventory
pub const DEFAULT_INVENTORY_SIZE: usize = 10;
pub const MAX_STACK_SIZE: u16 = 64;

// Resource respawn
pub const DEFAULT_RESOURCE_RESPAWN_TICKS: u32 = 100;

// Terrain type count (for observation encoding)
pub const NUM_TERRAIN_TYPES: usize = 8;

// Combat defaults (fixed-point i32 with 16 fractional bits: 65536 = 1.0)
/// Default damage dealt by a sword attack per hit.
pub const DEFAULT_SWORD_DAMAGE: i32 = 196608; // 3.0
/// Default damage per tick from standing on lava.
pub const DEFAULT_LAVA_DAMAGE: i32 = 65536; // 1.0

// Direction count
pub const NUM_DIRECTIONS: usize = 4;

// Fixed-point arithmetic
/// Number of fractional bits in fixed-point representation (16 bits).
pub const FIXED_POINT_SHIFT: u32 = 16;
/// 1.0 in fixed-point representation.
pub const FIXED_POINT_ONE: i32 = 1 << FIXED_POINT_SHIFT; // 65536

// Observation encoding sentinels
/// Sentinel value for "no object" in observation encoding.
pub const OBS_NO_OBJECT: u8 = 255;
/// Sentinel value for "no resource" in observation encoding.
pub const OBS_NO_RESOURCE: u8 = 255;
/// Sentinel value for empty inventory slot item type.
pub const OBS_EMPTY_SLOT_ITEM: u8 = 255;

// Day/night phases
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

// Vision modifiers per day phase
/// Vision multiplier during day.
pub const VISION_MODIFIER_DAY: f32 = 1.0;
/// Vision multiplier during dawn/dusk.
pub const VISION_MODIFIER_TWILIGHT: f32 = 0.75;
/// Vision multiplier during night.
pub const VISION_MODIFIER_NIGHT: f32 = 0.5;

// Observation feature count per tile
/// Number of features per tile in grid observation encoding.
pub const OBS_FEATURES_PER_TILE: usize = 7;

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
}
