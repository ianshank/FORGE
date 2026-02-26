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
