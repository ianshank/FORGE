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
pub const NUM_TERRAIN_TYPES: usize = 11;

// ---------- Service port defaults ----------

/// Default port for the forge-server HTTP/WebSocket API.
pub const DEFAULT_SERVER_PORT: u16 = 8080;
/// Default port for the React dashboard frontend.
pub const DEFAULT_DASHBOARD_PORT: u16 = 3000;
/// Default port for the FastAPI demo UI.
pub const DEFAULT_DEMO_UI_PORT: u16 = 8765;
/// Default port for the frontend dev server (Vite).
pub const DEFAULT_FRONTEND_DEV_PORT: u16 = 5173;
/// Default bind host used by services unless overridden.
pub const DEFAULT_BIND_HOST: &str = "127.0.0.1";
/// Default startup wait in seconds before health-checking services.
pub const DEFAULT_STARTUP_WAIT_SECS: u64 = 2;

// ---------- Cognitive defaults ----------

/// Default model for cognitive completion requests.
pub const DEFAULT_COGNITIVE_MODEL: &str = "claude-haiku-4-5-20251001";
/// Default sampling temperature for cognitive completion.
pub const DEFAULT_COGNITIVE_TEMPERATURE: f32 = 0.7;
/// Default maximum tokens for cognitive completion.
pub const DEFAULT_COGNITIVE_MAX_TOKENS: u32 = 1024;
/// Default number of reasoning steps per cognitive action selection.
pub const DEFAULT_COGNITIVE_REASONING_STEPS: u32 = 5;
/// Default confidence assigned when the provider doesn't return one.
pub const DEFAULT_COGNITIVE_CONFIDENCE: f32 = 0.8;
/// Default system prompt for the cognitive agent.
pub const DEFAULT_COGNITIVE_SYSTEM_PROMPT: &str =
    "You are an intelligent agent in a grid-based simulation. \
     Reason step by step, then select an action.";
/// Default healthcheck interval in seconds.
pub const DEFAULT_HEALTHCHECK_INTERVAL_S: u32 = 15;
/// Default healthcheck timeout in seconds.
pub const DEFAULT_HEALTHCHECK_TIMEOUT_S: u32 = 3;
/// Default healthcheck start period in seconds.
pub const DEFAULT_HEALTHCHECK_START_PERIOD_S: u32 = 10;
/// Default healthcheck retry count.
pub const DEFAULT_HEALTHCHECK_RETRIES: u32 = 3;

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
/// Default ground vehicle terrain costs indexed by TerrainType discriminant.
/// Fixed-point values. i32::MAX = impassable.
pub const DEFAULT_VEHICLE_TERRAIN_COSTS: [i32; NUM_TERRAIN_TYPES] = [
    32768,    // Ground: 0.5x (faster)
    i32::MAX, // Water: impassable
    i32::MAX, // Wall: impassable
    i32::MAX, // Lava: impassable
    49152,    // Ice: 0.75x
    45875,    // Sand: 0.7x
    i32::MAX, // Forest: impassable
    i32::MAX, // Mountain: impassable
    32768,    // Cropland: 0.5x (flat fields, fast)
    32768,    // Pasture: 0.5x (open grazing land)
    i32::MAX, // Orchard: impassable (tree rows)
];
/// Number of terrain types used for vehicle terrain cost array sizing.
pub const NUM_VEHICLE_TERRAIN_TYPES: usize = NUM_TERRAIN_TYPES;

// ---------- MCTS defaults ----------

/// Inline capacity for SmallVec in physics hot path.
/// Avoids heap allocation when the number of agents is at or below this threshold.
pub const PHYSICS_SMALLVEC_CAPACITY: usize = 16;

/// Default PUCT exploration constant (c_puct) for MCTS.
pub const DEFAULT_MCTS_C_PUCT: f32 = 1.41;
/// Default number of MCTS simulations per search.
pub const DEFAULT_MCTS_NUM_SIMULATIONS: u32 = 100;
/// Default maximum MCTS tree depth.
pub const DEFAULT_MCTS_MAX_DEPTH: u32 = 50;
/// Default MCTS discount factor for future rewards.
pub const DEFAULT_MCTS_DISCOUNT: f32 = 0.99;
/// Default MCTS temperature for action selection (1.0 = proportional to visits).
pub const DEFAULT_MCTS_TEMPERATURE: f32 = 1.0;
/// Default MCTS action space size.
pub const DEFAULT_MCTS_ACTION_SPACE: u32 = 32;
/// Default fall damage per altitude level during emergency landing (fixed-point 1.0).
pub const DEFAULT_FALL_DAMAGE_PER_LEVEL: i32 = FIXED_POINT_ONE;

// ---------- Agricultural defaults ----------

/// Whether agricultural systems are enabled by default.
pub const DEFAULT_AGRI_ENABLED: bool = false;
/// Default crop growth rate in ticks per growth stage advance (fixed-point ~0.01).
pub const DEFAULT_AGRI_CROP_GROWTH_RATE: i32 = 655; // ~0.01 per tick
/// Default disease spread probability per tick (fixed-point ~0.005).
pub const DEFAULT_AGRI_DISEASE_SPREAD_RATE: i32 = 328;
/// Default natural disease decay rate (fixed-point ~0.002).
pub const DEFAULT_AGRI_DISEASE_DECAY_RATE: i32 = 131;
/// Default maximum crop growth stages.
pub const DEFAULT_AGRI_MAX_GROWTH_STAGES: u8 = 5;
/// Default initial crop health (fixed-point 1.0).
pub const DEFAULT_AGRI_INITIAL_CROP_HEALTH: i32 = FIXED_POINT_ONE;
/// Default per-tick moisture drain (fixed-point ~0.003).
pub const DEFAULT_AGRI_MOISTURE_DRAIN_RATE: i32 = 197;
/// Default per-tick nutrient drain (fixed-point ~0.002).
pub const DEFAULT_AGRI_NUTRIENT_DRAIN_RATE: i32 = 131;
/// Default spray radius in tiles.
pub const DEFAULT_AGRI_SPRAY_RADIUS: u8 = 3;
/// Default spray disease reduction efficacy (fixed-point ~0.5).
pub const DEFAULT_AGRI_SPRAY_EFFICACY: i32 = FIXED_POINT_ONE / 2;
/// Default battery cost to spray (fixed-point ~0.15).
pub const DEFAULT_AGRI_SPRAY_BATTERY_COST: i32 = 9830;
/// Default NDVI multispectral scan radius in tiles.
pub const DEFAULT_AGRI_NDVI_SCAN_RADIUS: u8 = 8;
/// Default thermal scan radius in tiles.
pub const DEFAULT_AGRI_THERMAL_SCAN_RADIUS: u8 = 6;
/// Default soil sensor relay range in tiles.
pub const DEFAULT_AGRI_SOIL_RELAY_RANGE: u8 = 10;
/// Default battery cost per agricultural scan (fixed-point ~0.1).
pub const DEFAULT_AGRI_SCAN_BATTERY_COST: i32 = 6554;
/// Default number of ground-deployed soil sensor nodes.
pub const DEFAULT_AGRI_NUM_SOIL_NODES: u16 = 0;
/// Default ticks between soil sensor readings.
pub const DEFAULT_AGRI_SOIL_READING_INTERVAL: u32 = 50;
/// Default battery cost for VLM report generation (fixed-point ~0.2).
pub const DEFAULT_AGRI_REPORT_GENERATION_COST: i32 = 13107;
/// Default observation radius for VLM report generation.
pub const DEFAULT_AGRI_REPORT_SCAN_RADIUS: u8 = 10;
/// Default fraction of Ground tiles converted to Cropland.
pub const DEFAULT_AGRI_CROPLAND_DENSITY: f32 = 0.3;
/// Default fraction of Ground tiles converted to Pasture.
pub const DEFAULT_AGRI_PASTURE_DENSITY: f32 = 0.1;
/// Number of discrete agricultural action slots in the action space.
/// Spray(10 slots) + ScanMultispectral + ScanThermal + RelaySoilData + GenerateReport = 14.
pub const AGRI_ACTION_COUNT: u32 = 14;
/// Number of discrete hex movement action slots (6 directions: NE, E, SE, SW, W, NW).
pub const HEX_ACTION_COUNT: u32 = 6;
/// Number of additional observation fields when agriculture is enabled.
/// (disease_detections, area_surveyed_frac, soil_nodes_collected, report_ready).
pub const OBS_AGRI_FIELDS_COUNT: usize = 4;
/// Movement cost for Cropland terrain (1.2x — slightly slower).
pub const TERRAIN_COST_CROPLAND: i32 = FIXED_POINT_ONE + FIXED_POINT_ONE / 5;
/// Movement cost for Pasture terrain (1.0x — normal speed).
pub const TERRAIN_COST_PASTURE: i32 = FIXED_POINT_ONE;
/// Movement cost for Orchard terrain (1.5x — tree rows).
pub const TERRAIN_COST_ORCHARD: i32 = FIXED_POINT_ONE + FIXED_POINT_ONE / 2;
/// Upper exclusive bound for agricultural item discriminants.
pub const ITEM_TYPE_AGRI_MIN: u8 = 40;
/// Upper exclusive bound for agricultural item discriminants.
pub const ITEM_TYPE_AGRI_MAX: u8 = 50;

// ======================== Cloud training defaults ========================

/// Default number of rollout workers for cloud training.
pub const DEFAULT_CLOUD_NUM_WORKERS: u32 = 4;
/// Default replay batch size (replays collected before syncing to coordinator).
pub const DEFAULT_CLOUD_REPLAY_BATCH_SIZE: u32 = 64;
/// Default worker heartbeat interval in seconds.
pub const DEFAULT_CLOUD_HEARTBEAT_INTERVAL_S: u32 = 10;
/// Default worker heartbeat timeout in seconds before marking as dead.
pub const DEFAULT_CLOUD_HEARTBEAT_TIMEOUT_S: u32 = 30;
/// Default maximum replay payload size in bytes (10 MB).
pub const DEFAULT_CLOUD_MAX_REPLAY_SIZE_BYTES: u64 = 10_485_760;
/// Default replay transport compression setting (0 = disabled, non-zero = enabled).
pub const DEFAULT_CLOUD_COMPRESSION_LEVEL: u8 = 3;
/// Default port for worker coordination service.
pub const DEFAULT_CLOUD_COORDINATOR_PORT: u16 = 9090;
/// Default replay archive path for local storage backend.
pub const DEFAULT_CLOUD_REPLAY_ARCHIVE_PATH: &str = "replays";
/// Default model registry path for local storage backend.
pub const DEFAULT_CLOUD_MODEL_REGISTRY_PATH: &str = "models";
/// Default checkpoint path for training state persistence.
pub const DEFAULT_CLOUD_CHECKPOINT_PATH: &str = "checkpoints";
/// Default maximum model versions retained in the registry.
pub const DEFAULT_CLOUD_MODEL_VERSION_RETENTION: u32 = 10;
/// Default checkpoint interval in training steps.
pub const DEFAULT_CLOUD_CHECKPOINT_INTERVAL_STEPS: u64 = 1000;
/// Default storage backend identifier (`"local"` or `"gcs"`).
pub const DEFAULT_CLOUD_STORAGE_BACKEND: &str = "local";
/// Default GCS bucket name (empty = unconfigured).
pub const DEFAULT_CLOUD_GCS_BUCKET: &str = "";
/// Default key prefix within the GCS bucket.
pub const DEFAULT_CLOUD_GCS_PREFIX: &str = "forge/";
/// Default GCP project ID (empty = use ADC default).
pub const DEFAULT_CLOUD_GCP_PROJECT: &str = "";
/// Default GCP region.
pub const DEFAULT_CLOUD_GCP_REGION: &str = "us-central1";
/// Default GCP service account email (empty = use ADC).
pub const DEFAULT_CLOUD_GCP_SERVICE_ACCOUNT: &str = "";

// ======================== Edge runtime defaults ========================

/// Default MCTS latency budget in milliseconds for edge planning.
pub const DEFAULT_EDGE_MCTS_LATENCY_BUDGET_MS: u32 = 50;
/// Default minimum MCTS simulations on edge (floor even under time pressure).
pub const DEFAULT_EDGE_MCTS_MIN_SIMULATIONS: u32 = 8;
/// Default maximum MCTS simulations on edge (cap for battery saving).
pub const DEFAULT_EDGE_MCTS_MAX_SIMULATIONS: u32 = 200;
/// Default telemetry upload interval in seconds (store-and-forward).
pub const DEFAULT_EDGE_TELEMETRY_INTERVAL_S: u32 = 300;
/// Default maximum telemetry buffer size in bytes (1 MB).
pub const DEFAULT_EDGE_TELEMETRY_BUFFER_BYTES: u64 = 1_048_576;
/// Whether edge telemetry compression is enabled by default.
/// Defaults to false until transport-level compression is implemented.
pub const DEFAULT_EDGE_COMPRESS_TELEMETRY: bool = false;
/// Default ONNX inference batch size on edge.
pub const DEFAULT_EDGE_ONNX_BATCH_SIZE: u32 = 1;
/// Default ONNX thread count on edge.
pub const DEFAULT_EDGE_ONNX_NUM_THREADS: u32 = 1;
/// Default model update check interval in seconds.
pub const DEFAULT_EDGE_MODEL_UPDATE_INTERVAL_S: u32 = 3600;
/// Default exponential moving average alpha for edge latency estimation.
pub const DEFAULT_EDGE_LATENCY_EMA_ALPHA: f32 = 0.3;
/// Default number of upload retries for edge telemetry.
pub const DEFAULT_EDGE_UPLOAD_RETRY_COUNT: u32 = 4;
/// Default base delay in milliseconds for exponential backoff retries.
pub const DEFAULT_EDGE_UPLOAD_RETRY_BASE_MS: u64 = 2000;
/// Default GCS bucket for edge model pulls (empty = unconfigured).
pub const DEFAULT_EDGE_GCS_MODEL_BUCKET: &str = "";
/// Default GCS prefix for edge model artifacts.
pub const DEFAULT_EDGE_GCS_MODEL_PREFIX: &str = "forge/models/";

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
        // Water, Wall, Lava, Forest, Mountain, Orchard should be impassable
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[1], i32::MAX); // Water
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[2], i32::MAX); // Wall
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[3], i32::MAX); // Lava
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[6], i32::MAX); // Forest
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[7], i32::MAX); // Mountain
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS[10], i32::MAX); // Orchard
    }

    #[test]
    fn test_vehicle_terrain_costs_ground_is_faster() {
        // Ground cost < FIXED_POINT_ONE means faster than walking
        assert!(DEFAULT_VEHICLE_TERRAIN_COSTS[0] < FIXED_POINT_ONE);
        assert!(DEFAULT_VEHICLE_TERRAIN_COSTS[0] > 0);
    }

    #[test]
    fn test_service_port_constants_non_zero() {
        assert!(DEFAULT_SERVER_PORT > 0);
        assert!(DEFAULT_DASHBOARD_PORT > 0);
        assert!(DEFAULT_DEMO_UI_PORT > 0);
        assert!(DEFAULT_FRONTEND_DEV_PORT > 0);
    }

    #[test]
    fn test_service_port_constants_distinct() {
        // All four service ports should be different from each other
        let ports = [
            DEFAULT_SERVER_PORT,
            DEFAULT_DASHBOARD_PORT,
            DEFAULT_DEMO_UI_PORT,
            DEFAULT_FRONTEND_DEV_PORT,
        ];
        for i in 0..ports.len() {
            for j in (i + 1)..ports.len() {
                assert_ne!(
                    ports[i], ports[j],
                    "ports[{i}]={} and ports[{j}]={} should be different",
                    ports[i], ports[j]
                );
            }
        }
    }

    #[test]
    fn test_bind_host_is_loopback() {
        assert_eq!(DEFAULT_BIND_HOST, "127.0.0.1");
    }

    #[test]
    fn test_healthcheck_interval_greater_than_timeout() {
        assert!(DEFAULT_HEALTHCHECK_INTERVAL_S > DEFAULT_HEALTHCHECK_TIMEOUT_S);
    }

    #[test]
    fn test_healthcheck_retries_nonzero() {
        assert!(DEFAULT_HEALTHCHECK_RETRIES > 0);
    }

    #[test]
    fn test_service_ports_non_privileged() {
        // All default ports should be non-privileged (>= 1024)
        for port in [
            DEFAULT_SERVER_PORT,
            DEFAULT_DASHBOARD_PORT,
            DEFAULT_DEMO_UI_PORT,
            DEFAULT_FRONTEND_DEV_PORT,
        ] {
            assert!(
                port >= 1024,
                "Port {port} should be non-privileged (>= 1024)"
            );
        }
    }

    #[test]
    fn test_cognitive_defaults_valid() {
        assert!(!DEFAULT_COGNITIVE_MODEL.is_empty());
        assert!(DEFAULT_COGNITIVE_TEMPERATURE >= 0.0 && DEFAULT_COGNITIVE_TEMPERATURE <= 2.0);
        assert!(DEFAULT_COGNITIVE_MAX_TOKENS > 0);
        assert!(DEFAULT_COGNITIVE_REASONING_STEPS > 0);
        assert!(DEFAULT_COGNITIVE_CONFIDENCE >= 0.0 && DEFAULT_COGNITIVE_CONFIDENCE <= 1.0);
        assert!(!DEFAULT_COGNITIVE_SYSTEM_PROMPT.is_empty());
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
        assert!(ITEM_TYPE_CRAFTED_MAX <= ITEM_TYPE_AGRI_MIN);
        assert!(ITEM_TYPE_AGRI_MIN < ITEM_TYPE_AGRI_MAX);
    }

    #[test]
    fn test_agri_constants_are_positive() {
        assert!(DEFAULT_AGRI_CROP_GROWTH_RATE > 0);
        assert!(DEFAULT_AGRI_DISEASE_SPREAD_RATE > 0);
        assert!(DEFAULT_AGRI_DISEASE_DECAY_RATE > 0);
        assert!(DEFAULT_AGRI_MOISTURE_DRAIN_RATE > 0);
        assert!(DEFAULT_AGRI_NUTRIENT_DRAIN_RATE > 0);
        assert!(DEFAULT_AGRI_SPRAY_EFFICACY > 0);
        assert!(DEFAULT_AGRI_SPRAY_BATTERY_COST > 0);
        assert!(DEFAULT_AGRI_SCAN_BATTERY_COST > 0);
        assert!(DEFAULT_AGRI_REPORT_GENERATION_COST > 0);
    }

    #[test]
    fn test_agri_terrain_costs_derived_from_fixed_point() {
        assert_eq!(TERRAIN_COST_CROPLAND, FIXED_POINT_ONE + FIXED_POINT_ONE / 5);
        assert_eq!(TERRAIN_COST_PASTURE, FIXED_POINT_ONE);
        assert_eq!(TERRAIN_COST_ORCHARD, FIXED_POINT_ONE + FIXED_POINT_ONE / 2);
    }

    #[test]
    fn test_num_terrain_types_matches_vehicle_costs() {
        assert_eq!(NUM_TERRAIN_TYPES, NUM_VEHICLE_TERRAIN_TYPES);
        assert_eq!(DEFAULT_VEHICLE_TERRAIN_COSTS.len(), NUM_TERRAIN_TYPES);
    }

    #[test]
    fn test_agri_action_count() {
        // 10 Spray slots + ScanMultispectral + ScanThermal + RelaySoilData + GenerateReport
        assert_eq!(AGRI_ACTION_COUNT, 14);
    }

    // ---- Cloud constants validation ----

    #[test]
    fn test_cloud_constants_are_positive() {
        assert!(DEFAULT_CLOUD_NUM_WORKERS > 0);
        assert!(DEFAULT_CLOUD_REPLAY_BATCH_SIZE > 0);
        assert!(DEFAULT_CLOUD_HEARTBEAT_INTERVAL_S > 0);
        assert!(DEFAULT_CLOUD_HEARTBEAT_TIMEOUT_S > 0);
        assert!(DEFAULT_CLOUD_MAX_REPLAY_SIZE_BYTES > 0);
        assert!(DEFAULT_CLOUD_MODEL_VERSION_RETENTION > 0);
        assert!(DEFAULT_CLOUD_CHECKPOINT_INTERVAL_STEPS > 0);
    }

    #[test]
    fn test_cloud_compression_level_in_range() {
        assert!(DEFAULT_CLOUD_COMPRESSION_LEVEL <= 9);
    }

    #[test]
    fn test_cloud_heartbeat_timeout_exceeds_interval() {
        assert!(DEFAULT_CLOUD_HEARTBEAT_TIMEOUT_S > DEFAULT_CLOUD_HEARTBEAT_INTERVAL_S);
    }

    #[test]
    fn test_cloud_coordinator_port_non_privileged() {
        assert!(DEFAULT_CLOUD_COORDINATOR_PORT >= 1024);
    }

    #[test]
    fn test_cloud_ports_distinct_from_existing() {
        assert_ne!(DEFAULT_CLOUD_COORDINATOR_PORT, DEFAULT_SERVER_PORT);
        assert_ne!(DEFAULT_CLOUD_COORDINATOR_PORT, DEFAULT_DASHBOARD_PORT);
        assert_ne!(DEFAULT_CLOUD_COORDINATOR_PORT, DEFAULT_DEMO_UI_PORT);
    }

    #[test]
    fn test_cloud_paths_not_empty() {
        assert!(!DEFAULT_CLOUD_REPLAY_ARCHIVE_PATH.is_empty());
        assert!(!DEFAULT_CLOUD_MODEL_REGISTRY_PATH.is_empty());
        assert!(!DEFAULT_CLOUD_CHECKPOINT_PATH.is_empty());
    }

    // ---- Edge constants validation ----

    #[test]
    fn test_edge_mcts_min_le_max() {
        assert!(DEFAULT_EDGE_MCTS_MIN_SIMULATIONS <= DEFAULT_EDGE_MCTS_MAX_SIMULATIONS);
    }

    #[test]
    fn test_edge_latency_budget_positive() {
        assert!(DEFAULT_EDGE_MCTS_LATENCY_BUDGET_MS > 0);
    }

    #[test]
    fn test_edge_constants_are_positive() {
        assert!(DEFAULT_EDGE_MCTS_MIN_SIMULATIONS > 0);
        assert!(DEFAULT_EDGE_MCTS_MAX_SIMULATIONS > 0);
        assert!(DEFAULT_EDGE_TELEMETRY_INTERVAL_S > 0);
        assert!(DEFAULT_EDGE_TELEMETRY_BUFFER_BYTES > 0);
        assert!(DEFAULT_EDGE_ONNX_BATCH_SIZE > 0);
        assert!(DEFAULT_EDGE_ONNX_NUM_THREADS > 0);
        assert!(DEFAULT_EDGE_MODEL_UPDATE_INTERVAL_S > 0);
        assert!(DEFAULT_EDGE_UPLOAD_RETRY_COUNT > 0);
        assert!(DEFAULT_EDGE_UPLOAD_RETRY_BASE_MS > 0);
    }

    #[test]
    fn test_edge_latency_ema_alpha_in_range() {
        assert!(DEFAULT_EDGE_LATENCY_EMA_ALPHA > 0.0);
        assert!(DEFAULT_EDGE_LATENCY_EMA_ALPHA <= 1.0);
    }
}
