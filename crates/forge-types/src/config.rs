//! Configuration types for the FORGE simulation platform.
//!
//! All simulation parameters are configurable through these structs.
//! No hard-coded values — defaults are provided via `Default` trait
//! and can be overridden at construction time.
//!
//! # Loading from TOML
//!
//! ```no_run
//! use forge_types::config::ForgeConfig;
//! let config = ForgeConfig::from_toml("forge.toml").unwrap();
//! ```
//!
//! Environment variables with the `FORGE_` prefix override file values:
//! `FORGE_WORLD_WIDTH=128` overrides `world.width`.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::constants;
use crate::error::{ConfigError, ForgeError};
use crate::grid::Position;
use crate::skill::SkillsConfig;

/// Top-level configuration for a FORGE simulation instance.
///
/// # Schema Difference from Python
///
/// This Rust struct uses field names like `world`, `physics`, `agents`, etc.
/// The **Python** `ForgeConfig` (in `python/forge/config.py`) uses different
/// section names: `hardware`, `simulation`, `training`. The root `forge.toml`
/// file uses the **Python** schema. Use [`ForgeConfig::default()`] in Rust
/// code, or create a separate TOML file with the Rust schema if needed.
///
/// The Python config provides [`to_rust_config()`](python/forge/config.py)
/// to bridge between the two schemas.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForgeConfig {
    /// World generation and layout parameters.
    pub world: WorldConfig,
    /// Physics simulation parameters.
    pub physics: PhysicsConfig,
    /// Crafting system parameters.
    pub crafting: CraftingConfig,
    /// Agent configuration.
    pub agents: AgentConfig,
    /// Task system parameters.
    pub task: TaskConfig,
    /// Curriculum controller parameters.
    pub curriculum: CurriculumConfig,
    /// Rendering and visualization parameters.
    pub rendering: RenderConfig,
    /// Drone-specific mechanics parameters.
    pub drone: DroneConfig,
    /// Agricultural simulation parameters.
    pub agri: AgriConfig,
    /// Cloud training pipeline parameters.
    pub cloud: CloudConfig,
    /// Edge deployment runtime parameters.
    pub edge: EdgeConfig,
    /// Hierarchical skill catalog over primitive actions (opt-in).
    pub skills: SkillsConfig,
    /// External orchestration and controller parameters.
    pub orchestration: OrchestrationConfig,
    /// Honcho memory mirror parameters.
    pub honcho: HonchoConfig,
}

/// Grid topology type for the simulation world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GridType {
    /// Standard 4-neighbor square grid (cardinal directions).
    #[default]
    #[serde(alias = "Square")]
    Square,
    /// 6-neighbor hexagonal grid (odd-r offset layout).
    #[serde(alias = "Hex")]
    Hex,
}

/// World generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorldConfig {
    /// Grid topology type (square or hex).
    pub grid_type: GridType,
    /// Grid width in tiles.
    pub width: u16,
    /// Grid height in tiles.
    pub height: u16,
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Scale factor for Perlin noise terrain generation.
    pub biome_scale: f32,
    /// Resource spawn density (0.0 = none, 1.0 = maximum).
    pub resource_density: f32,
    /// Number of ticks in a full day/night cycle. 0 disables the cycle.
    pub day_night_cycle_length: u32,
    /// Maximum number of entities (agents + NPCs + objects) allowed.
    pub max_entities: u16,
    /// Minimum world dimension (width or height).
    pub min_dimension: u16,
    /// Maximum world dimension (width or height).
    pub max_dimension: u16,
    /// Resource respawn rate (ticks between respawn increments).
    pub resource_respawn_rate: u32,
    /// Maximum quantity per resource node.
    pub resource_max_quantity: u16,
    /// Object placement density scale (multiplied with base probability).
    pub object_density_scale: f32,
    /// When true, `Move` / `MoveHex` that would leave the interior
    /// `[margin, width-margin) × [margin, height-margin)` become `Noop`.
    pub geofence_enabled: bool,
    /// Interior margin in tiles. Ignored unless `geofence_enabled`.
    pub geofence_margin: u16,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            grid_type: GridType::default(),
            width: constants::DEFAULT_WORLD_WIDTH,
            height: constants::DEFAULT_WORLD_HEIGHT,
            seed: constants::DEFAULT_SEED,
            biome_scale: constants::DEFAULT_BIOME_SCALE,
            resource_density: constants::DEFAULT_RESOURCE_DENSITY,
            day_night_cycle_length: constants::DEFAULT_DAY_NIGHT_CYCLE_LENGTH,
            max_entities: constants::DEFAULT_MAX_ENTITIES,
            min_dimension: constants::MIN_WORLD_DIMENSION,
            max_dimension: constants::MAX_WORLD_DIMENSION,
            resource_respawn_rate: constants::DEFAULT_RESOURCE_RESPAWN_TICKS,
            resource_max_quantity: constants::DEFAULT_RESOURCE_MAX_QUANTITY,
            object_density_scale: constants::DEFAULT_OBJECT_DENSITY_SCALE,
            geofence_enabled: constants::DEFAULT_GEOFENCE_ENABLED,
            geofence_margin: constants::DEFAULT_GEOFENCE_MARGIN,
        }
    }
}

impl WorldConfig {
    /// Whether `pos` is a legal tile under the configured geofence.
    ///
    /// Out-of-world coordinates are never allowed. When geofencing is
    /// disabled, every in-world tile is allowed.
    pub fn allows_position(&self, pos: Position) -> bool {
        if pos.x >= self.width || pos.y >= self.height {
            return false;
        }
        if !self.geofence_enabled {
            return true;
        }
        let margin = self.geofence_margin;
        pos.x >= margin
            && pos.y >= margin
            && pos.x + margin < self.width
            && pos.y + margin < self.height
    }
}

/// Physics system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhysicsConfig {
    /// Whether collision detection is enabled.
    pub collision_enabled: bool,
    /// Base movement cost in stamina units (fixed-point, stored as i32 with 16 fractional bits).
    pub stamina_cost_move: i32,
    /// Stamina regeneration per tick (fixed-point).
    pub stamina_regen_rate: i32,
    /// Maximum agent velocity in tiles per tick (fixed-point).
    pub max_velocity: i32,
    /// Friction coefficient (fixed-point). Affects sliding on ice, etc.
    pub friction: i32,
    /// Whether projectile physics is enabled.
    pub projectiles_enabled: bool,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            collision_enabled: true,
            stamina_cost_move: constants::DEFAULT_STAMINA_COST_MOVE,
            stamina_regen_rate: constants::DEFAULT_STAMINA_REGEN_RATE,
            max_velocity: constants::DEFAULT_MAX_VELOCITY,
            friction: constants::DEFAULT_FRICTION,
            projectiles_enabled: false,
        }
    }
}

/// Crafting system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CraftingConfig {
    /// Whether the crafting system is enabled.
    pub enabled: bool,
    /// Whether recipes vary per world seed (procedural recipes).
    pub procedural_recipes: bool,
    /// Maximum number of ingredients per recipe.
    pub max_ingredients: u8,
    /// Whether agents must discover recipes before crafting.
    pub require_discovery: bool,
}

impl Default for CraftingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            procedural_recipes: false,
            max_ingredients: constants::DEFAULT_MAX_INGREDIENTS,
            require_discovery: false,
        }
    }
}

/// Agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentConfig {
    /// Number of agents in the simulation.
    pub num_agents: u32,
    /// Default vision radius in tiles.
    pub default_vision_radius: u8,
    /// Default carry capacity (inventory slots).
    pub default_carry_capacity: u8,
    /// Starting health (fixed-point).
    pub starting_health: i32,
    /// Maximum health (fixed-point).
    pub max_health: i32,
    /// Starting stamina (fixed-point).
    pub starting_stamina: i32,
    /// Maximum stamina (fixed-point).
    pub max_stamina: i32,
    /// Whether agents can have heterogeneous capabilities.
    pub heterogeneous: bool,
    /// Communication vocabulary size (0 = communication disabled).
    pub comm_vocab_size: u16,
    /// Communication broadcast radius in tiles. 0 = global broadcast.
    pub comm_radius: u16,
    /// Maximum messages stored in agent's communication buffer.
    pub comm_buffer_size: u8,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            num_agents: constants::DEFAULT_NUM_AGENTS,
            default_vision_radius: constants::DEFAULT_VISION_RADIUS,
            default_carry_capacity: constants::DEFAULT_CARRY_CAPACITY,
            starting_health: constants::DEFAULT_STARTING_HEALTH,
            max_health: constants::DEFAULT_MAX_HEALTH,
            starting_stamina: constants::DEFAULT_STARTING_STAMINA,
            max_stamina: constants::DEFAULT_MAX_STAMINA,
            heterogeneous: false,
            comm_vocab_size: constants::DEFAULT_COMM_VOCAB_SIZE,
            comm_radius: constants::DEFAULT_COMM_RADIUS,
            comm_buffer_size: constants::DEFAULT_COMM_BUFFER_SIZE,
        }
    }
}

/// Task system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TaskConfig {
    /// Whether the task system is enabled.
    pub enabled: bool,
    /// Maximum task tier to generate (1-6).
    pub max_tier: u8,
    /// Maximum number of predicates in a single task.
    pub max_predicates: u16,
    /// Maximum episode length in ticks. 0 = unlimited.
    pub max_episode_length: u64,
    /// Reward scale factor.
    pub reward_scale: f32,
    /// Whether to provide dense reward shaping.
    pub dense_rewards: bool,
    /// Tasks attached when a world is created or reset, if [`Self::enabled`].
    ///
    /// Populated by the high-level scenario compiler. Empty by default so
    /// Gymnasium episodes without a scenario stay task-free (and remain on
    /// the zero-alloc hot path). Procedural curriculum tasks from
    /// `forge-data` are attached separately.
    #[serde(default)]
    pub scenario_tasks: Vec<crate::task::TaskDefinition>,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_tier: constants::DEFAULT_MAX_TASK_TIER,
            max_predicates: constants::DEFAULT_MAX_PREDICATES,
            max_episode_length: constants::DEFAULT_MAX_EPISODE_LENGTH,
            reward_scale: constants::DEFAULT_REWARD_SCALE,
            dense_rewards: true,
            scenario_tasks: Vec::new(),
        }
    }
}

/// Curriculum controller configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurriculumConfig {
    /// Whether the curriculum controller is enabled.
    pub enabled: bool,
    /// Target success rate to maintain (0.0-1.0).
    pub target_success_rate: f32,
    /// Window size for computing rolling success rate.
    pub window_size: u32,
    /// Minimum episodes before adjusting difficulty.
    pub warmup_episodes: u32,
    /// How aggressively to adjust difficulty (learning rate).
    pub adjustment_rate: f32,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_success_rate: constants::DEFAULT_TARGET_SUCCESS_RATE,
            window_size: constants::DEFAULT_CURRICULUM_WINDOW_SIZE,
            warmup_episodes: constants::DEFAULT_CURRICULUM_WARMUP,
            adjustment_rate: constants::DEFAULT_CURRICULUM_ADJUSTMENT_RATE,
        }
    }
}

/// Rendering and visualization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderConfig {
    /// Whether to generate pixel observations (slower).
    pub pixel_observations: bool,
    /// Pixel observation width (if enabled).
    pub pixel_width: u32,
    /// Pixel observation height (if enabled).
    pub pixel_height: u32,
    /// Whether to record replays.
    pub record_replays: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            pixel_observations: false,
            pixel_width: constants::DEFAULT_PIXEL_WIDTH,
            pixel_height: constants::DEFAULT_PIXEL_HEIGHT,
            record_replays: false,
        }
    }
}

/// Configuration for drone-specific mechanics.
///
/// When `enabled` is false (default), all drone systems are skipped
/// and the simulation behaves identically to pre-drone versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DroneConfig {
    /// Whether drone mechanics are enabled.
    pub enabled: bool,
    /// Maximum altitude for aerial agents.
    pub max_altitude: u8,
    /// Battery drain per tick while airborne (fixed-point).
    pub aerial_drain_rate: i32,
    /// Battery cost to ascend one level (fixed-point).
    pub ascend_cost: i32,
    /// Battery cost to descend one level (fixed-point).
    pub descend_cost: i32,
    /// Hover cost per tick (fixed-point).
    pub hover_cost: i32,
    /// Battery cost for a scan action (fixed-point).
    pub scan_cost: i32,
    /// Scan range in tiles (extends beyond normal vision).
    pub scan_range: u8,
    /// Starting battery for aerial agents (fixed-point).
    pub starting_battery: i32,
    /// Maximum battery (fixed-point).
    pub max_battery: i32,
    /// Battery recharge rate when landed (fixed-point per tick).
    pub recharge_rate: i32,
    /// Ground vehicle terrain speed multipliers indexed by TerrainType.
    /// Fixed-point values. i32::MAX = impassable.
    pub vehicle_terrain_costs: [i32; constants::NUM_TERRAIN_TYPES],
    /// Vision radius bonus per altitude level for aerial agents.
    pub altitude_vision_bonus: u8,
    /// Turn radius for ground vehicles (0 = free, 1+ = restricted).
    pub vehicle_turn_radius: u8,
    /// Fall damage per altitude level during emergency landing (fixed-point).
    pub fall_damage_per_level: i32,
    /// Number of aerial agents to spawn.
    pub num_aerial: u32,
    /// Number of ground vehicle agents to spawn.
    pub num_ground_vehicles: u32,
    /// When true, landed aerial agents recharge only on `charger_tiles`.
    pub restrict_recharge_to_chargers: bool,
    /// Tiles that act as charging stations when recharge is restricted.
    pub charger_tiles: Vec<Position>,
    /// Optional spawn override for aerial agents (home / depot).
    pub spawn_home: Option<Position>,
    /// Energy-costing actions become `Noop` while `battery` is below this
    /// floor. `Land` and `Descend` remain legal so the agent can return.
    pub battery_action_floor: i32,
}

impl Default for DroneConfig {
    fn default() -> Self {
        Self {
            enabled: constants::DEFAULT_DRONE_ENABLED,
            max_altitude: constants::DEFAULT_MAX_ALTITUDE,
            aerial_drain_rate: constants::DEFAULT_AERIAL_DRAIN_RATE,
            ascend_cost: constants::DEFAULT_ASCEND_COST,
            descend_cost: constants::DEFAULT_DESCEND_COST,
            hover_cost: constants::DEFAULT_HOVER_COST,
            scan_cost: constants::DEFAULT_SCAN_COST,
            scan_range: constants::DEFAULT_SCAN_RANGE,
            starting_battery: constants::DEFAULT_STARTING_BATTERY,
            max_battery: constants::DEFAULT_MAX_BATTERY,
            recharge_rate: constants::DEFAULT_RECHARGE_RATE,
            vehicle_terrain_costs: constants::DEFAULT_VEHICLE_TERRAIN_COSTS,
            altitude_vision_bonus: constants::DEFAULT_ALTITUDE_VISION_BONUS,
            vehicle_turn_radius: constants::DEFAULT_VEHICLE_TURN_RADIUS,
            fall_damage_per_level: constants::DEFAULT_FALL_DAMAGE_PER_LEVEL,
            num_aerial: 0,
            num_ground_vehicles: 0,
            restrict_recharge_to_chargers: constants::DEFAULT_RESTRICT_RECHARGE_TO_CHARGERS,
            charger_tiles: Vec::new(),
            spawn_home: None,
            battery_action_floor: constants::DEFAULT_BATTERY_ACTION_FLOOR,
        }
    }
}

impl DroneConfig {
    /// Whether a landed aerial agent at `pos` may recharge this tick.
    pub fn allows_recharge_at(&self, pos: Position) -> bool {
        if !self.restrict_recharge_to_chargers {
            return true;
        }
        self.charger_tiles.contains(&pos)
    }
}

/// Configuration for agricultural drone simulation.
///
/// When `enabled` is false (default), all agricultural systems are skipped
/// and the simulation behaves identically to pre-agriculture versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgriConfig {
    /// Whether agricultural systems are enabled.
    pub enabled: bool,
    /// Crop growth increment per tick (fixed-point). Accumulated until a stage threshold.
    pub crop_growth_rate: i32,
    /// Disease spread probability per tick per diseased neighbor (fixed-point).
    pub disease_spread_rate: i32,
    /// Natural disease decay rate per tick (fixed-point).
    pub disease_decay_rate: i32,
    /// Maximum number of crop growth stages.
    pub max_growth_stages: u8,
    /// Initial crop health at planting (fixed-point).
    pub initial_crop_health: i32,
    /// Per-tick soil moisture drain (fixed-point).
    pub moisture_drain_rate: i32,
    /// Per-tick nutrient drain (fixed-point).
    pub nutrient_drain_rate: i32,
    /// Radius in tiles affected by a single spray action.
    pub spray_radius: u8,
    /// Disease reduction per spray application (fixed-point).
    pub spray_efficacy: i32,
    /// Battery cost per spray action (fixed-point).
    pub spray_battery_cost: i32,
    /// Multispectral NDVI scan radius in tiles.
    pub ndvi_scan_radius: u8,
    /// Thermal scan radius in tiles.
    pub thermal_scan_radius: u8,
    /// Range in tiles for soil sensor data relay.
    pub soil_relay_range: u8,
    /// Battery cost per agricultural scan action (fixed-point).
    pub scan_battery_cost: i32,
    /// Number of ground-deployed soil sensor nodes.
    pub num_soil_nodes: u16,
    /// Ticks between soil sensor readings being refreshed.
    pub soil_reading_interval: u32,
    /// Battery cost for VLM report generation (fixed-point).
    pub report_generation_cost: i32,
    /// Observation radius for report generation.
    pub report_scan_radius: u8,
    /// Fraction of Ground tiles converted to Cropland during worldgen.
    pub cropland_density: f32,
    /// Fraction of Ground tiles converted to Pasture during worldgen.
    pub pasture_density: f32,
}

impl Default for AgriConfig {
    fn default() -> Self {
        Self {
            enabled: constants::DEFAULT_AGRI_ENABLED,
            crop_growth_rate: constants::DEFAULT_AGRI_CROP_GROWTH_RATE,
            disease_spread_rate: constants::DEFAULT_AGRI_DISEASE_SPREAD_RATE,
            disease_decay_rate: constants::DEFAULT_AGRI_DISEASE_DECAY_RATE,
            max_growth_stages: constants::DEFAULT_AGRI_MAX_GROWTH_STAGES,
            initial_crop_health: constants::DEFAULT_AGRI_INITIAL_CROP_HEALTH,
            moisture_drain_rate: constants::DEFAULT_AGRI_MOISTURE_DRAIN_RATE,
            nutrient_drain_rate: constants::DEFAULT_AGRI_NUTRIENT_DRAIN_RATE,
            spray_radius: constants::DEFAULT_AGRI_SPRAY_RADIUS,
            spray_efficacy: constants::DEFAULT_AGRI_SPRAY_EFFICACY,
            spray_battery_cost: constants::DEFAULT_AGRI_SPRAY_BATTERY_COST,
            ndvi_scan_radius: constants::DEFAULT_AGRI_NDVI_SCAN_RADIUS,
            thermal_scan_radius: constants::DEFAULT_AGRI_THERMAL_SCAN_RADIUS,
            soil_relay_range: constants::DEFAULT_AGRI_SOIL_RELAY_RANGE,
            scan_battery_cost: constants::DEFAULT_AGRI_SCAN_BATTERY_COST,
            num_soil_nodes: constants::DEFAULT_AGRI_NUM_SOIL_NODES,
            soil_reading_interval: constants::DEFAULT_AGRI_SOIL_READING_INTERVAL,
            report_generation_cost: constants::DEFAULT_AGRI_REPORT_GENERATION_COST,
            report_scan_radius: constants::DEFAULT_AGRI_REPORT_SCAN_RADIUS,
            cropland_density: constants::DEFAULT_AGRI_CROPLAND_DENSITY,
            pasture_density: constants::DEFAULT_AGRI_PASTURE_DENSITY,
        }
    }
}

/// Cloud training pipeline configuration.
///
/// When `enabled` is false (default), all cloud training features are inactive
/// and the simulation behaves identically to pre-cloud versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CloudConfig {
    /// Whether cloud training pipeline is enabled.
    pub enabled: bool,
    /// Number of rollout workers.
    pub num_workers: u32,
    /// Replay batch size (replays collected before syncing to coordinator).
    pub replay_batch_size: u32,
    /// Worker heartbeat interval in seconds.
    pub heartbeat_interval_s: u32,
    /// Worker heartbeat timeout in seconds before marking as dead.
    pub heartbeat_timeout_s: u32,
    /// Maximum replay payload size in bytes.
    pub max_replay_size_bytes: u64,
    /// Compression setting for replay transport. 0 disables compression;
    /// non-zero values enable the built-in compression scheme.
    pub compression_level: u8,
    /// Port for worker coordination service.
    pub coordinator_port: u16,
    /// Path for replay archive (local storage backend).
    pub replay_archive_path: String,
    /// Path for model registry (local storage backend).
    pub model_registry_path: String,
    /// Path for training checkpoint persistence.
    pub checkpoint_path: String,
    /// Maximum model versions retained in the registry.
    pub model_version_retention: u32,
    /// Checkpoint interval in training steps.
    pub checkpoint_interval_steps: u64,
    /// Storage backend: `"local"` or `"gcs"`.
    pub storage_backend: String,
    /// GCS bucket name (required when `storage_backend` is `"gcs"`).
    pub gcs_bucket: String,
    /// Key prefix within the GCS bucket.
    pub gcs_prefix: String,
    /// GCP project ID (empty = use Application Default Credentials project).
    pub gcp_project: String,
    /// GCP region for storage and compute.
    pub gcp_region: String,
    /// GCP service account credentials for the GCS backend.
    /// Expected value: service account key JSON contents or a key-file path
    /// (passed to `GoogleCloudStorageBuilder::with_service_account_key`).
    /// Empty = use Application Default Credentials.
    pub gcp_service_account: String,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_workers: constants::DEFAULT_CLOUD_NUM_WORKERS,
            replay_batch_size: constants::DEFAULT_CLOUD_REPLAY_BATCH_SIZE,
            heartbeat_interval_s: constants::DEFAULT_CLOUD_HEARTBEAT_INTERVAL_S,
            heartbeat_timeout_s: constants::DEFAULT_CLOUD_HEARTBEAT_TIMEOUT_S,
            max_replay_size_bytes: constants::DEFAULT_CLOUD_MAX_REPLAY_SIZE_BYTES,
            compression_level: constants::DEFAULT_CLOUD_COMPRESSION_LEVEL,
            coordinator_port: constants::DEFAULT_CLOUD_COORDINATOR_PORT,
            replay_archive_path: constants::DEFAULT_CLOUD_REPLAY_ARCHIVE_PATH.to_string(),
            model_registry_path: constants::DEFAULT_CLOUD_MODEL_REGISTRY_PATH.to_string(),
            checkpoint_path: constants::DEFAULT_CLOUD_CHECKPOINT_PATH.to_string(),
            model_version_retention: constants::DEFAULT_CLOUD_MODEL_VERSION_RETENTION,
            checkpoint_interval_steps: constants::DEFAULT_CLOUD_CHECKPOINT_INTERVAL_STEPS,
            storage_backend: constants::DEFAULT_CLOUD_STORAGE_BACKEND.to_string(),
            gcs_bucket: constants::DEFAULT_CLOUD_GCS_BUCKET.to_string(),
            gcs_prefix: constants::DEFAULT_CLOUD_GCS_PREFIX.to_string(),
            gcp_project: constants::DEFAULT_CLOUD_GCP_PROJECT.to_string(),
            gcp_region: constants::DEFAULT_CLOUD_GCP_REGION.to_string(),
            gcp_service_account: constants::DEFAULT_CLOUD_GCP_SERVICE_ACCOUNT.to_string(),
        }
    }
}

/// Edge deployment runtime configuration.
///
/// When `enabled` is false (default), all edge-specific features are inactive
/// and agents use standard MCTS without latency budgeting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EdgeConfig {
    /// Whether edge runtime features are enabled.
    pub enabled: bool,
    /// MCTS latency budget in milliseconds for edge planning.
    pub mcts_latency_budget_ms: u32,
    /// Minimum MCTS simulations (floor even under time pressure).
    pub mcts_min_simulations: u32,
    /// Maximum MCTS simulations (cap for battery saving).
    pub mcts_max_simulations: u32,
    /// Telemetry upload interval in seconds (store-and-forward).
    pub telemetry_interval_s: u32,
    /// Maximum telemetry buffer size in bytes.
    pub telemetry_buffer_bytes: u64,
    /// Reserved flag for future telemetry compression support.
    ///
    /// The built-in telemetry collector currently uploads raw compact replay
    /// bytes regardless of this setting.
    pub compress_telemetry: bool,
    /// ONNX inference batch size on edge.
    pub onnx_batch_size: u32,
    /// ONNX thread count on edge.
    pub onnx_num_threads: u32,
    /// Model update check interval in seconds.
    pub model_update_interval_s: u32,
    /// Exponential moving average alpha for latency estimation.
    pub latency_ema_alpha: f32,
    /// Number of upload retries for edge telemetry.
    pub upload_retry_count: u32,
    /// Base delay in milliseconds for exponential backoff retries.
    pub upload_retry_base_ms: u64,
    /// GCS bucket for pulling model updates on edge (empty = unconfigured).
    pub gcs_model_bucket: String,
    /// GCS prefix for model artifacts on edge.
    pub gcs_model_prefix: String,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mcts_latency_budget_ms: constants::DEFAULT_EDGE_MCTS_LATENCY_BUDGET_MS,
            mcts_min_simulations: constants::DEFAULT_EDGE_MCTS_MIN_SIMULATIONS,
            mcts_max_simulations: constants::DEFAULT_EDGE_MCTS_MAX_SIMULATIONS,
            telemetry_interval_s: constants::DEFAULT_EDGE_TELEMETRY_INTERVAL_S,
            telemetry_buffer_bytes: constants::DEFAULT_EDGE_TELEMETRY_BUFFER_BYTES,
            compress_telemetry: constants::DEFAULT_EDGE_COMPRESS_TELEMETRY,
            onnx_batch_size: constants::DEFAULT_EDGE_ONNX_BATCH_SIZE,
            onnx_num_threads: constants::DEFAULT_EDGE_ONNX_NUM_THREADS,
            model_update_interval_s: constants::DEFAULT_EDGE_MODEL_UPDATE_INTERVAL_S,
            latency_ema_alpha: constants::DEFAULT_EDGE_LATENCY_EMA_ALPHA,
            upload_retry_count: constants::DEFAULT_EDGE_UPLOAD_RETRY_COUNT,
            upload_retry_base_ms: constants::DEFAULT_EDGE_UPLOAD_RETRY_BASE_MS,
            gcs_model_bucket: constants::DEFAULT_EDGE_GCS_MODEL_BUCKET.to_string(),
            gcs_model_prefix: constants::DEFAULT_EDGE_GCS_MODEL_PREFIX.to_string(),
        }
    }
}

/// External orchestration configuration (e.g., ADK, DeerFlow).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OrchestrationConfig {
    /// Whether external orchestration is enabled.
    pub enabled: bool,
    /// Controller type ("adk", "deerflow", or "none").
    pub controller_type: String,
    /// Whether DeerFlow sandbox is enforced.
    pub deerflow_sandboxed: bool,
    /// Artifact directory for DeerFlow execution.
    pub artifact_dir: String,
    /// Endpoint for external ADK requests.
    pub endpoint: String,
    /// Timeout for external controller actions in ms.
    pub timeout_ms: u32,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            controller_type: "none".to_string(),
            deerflow_sandboxed: false,
            artifact_dir: "/tmp/forge-artifacts".to_string(),
            endpoint: "http://localhost:8080".to_string(),
            timeout_ms: 5000,
        }
    }
}

/// Honcho memory mirror configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HonchoConfig {
    /// Whether Honcho mirroring is enabled.
    pub enabled: bool,
    /// Endpoint for Honcho ingest API.
    pub endpoint: String,
    /// Whether to strip latent states (must be true for safe execution).
    pub strip_latent_state: bool,
    /// Sync interval in ticks.
    pub sync_interval_ticks: u32,
}

impl Default for HonchoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:9090".to_string(),
            strip_latent_state: true,
            sync_interval_ticks: 100,
        }
    }
}

/// Environment variable prefix for config overrides.
const ENV_PREFIX: &str = "FORGE_";

impl ForgeConfig {
    /// Loads configuration from a TOML file, falling back to defaults for
    /// missing fields. Environment variable overrides are applied after loading.
    ///
    /// # Errors
    ///
    /// Returns [`ForgeError::Config`] if the file cannot be read or parsed.
    pub fn from_toml<P: AsRef<Path>>(path: P) -> Result<Self, ForgeError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(|e| {
            warn!(?path, error = %e, "failed to read config file");
            ConfigError::ParseError(format!("cannot read {}: {e}", path.display()))
        })?;
        let mut config = Self::from_toml_str(&contents)?;
        config.apply_env_overrides();
        debug!(?path, "loaded config from TOML");
        Ok(config)
    }

    /// Parses configuration from a TOML string, falling back to defaults for
    /// missing fields.
    ///
    /// # Errors
    ///
    /// Returns [`ForgeError::Config`] if the string is not valid TOML.
    pub fn from_toml_str(toml_str: &str) -> Result<Self, ForgeError> {
        toml::from_str(toml_str)
            .map_err(|e| {
                warn!(error = %e, "failed to parse TOML config");
                ConfigError::ParseError(format!("invalid TOML: {e}"))
            })
            .map_err(ForgeError::from)
    }

    /// Applies environment variable overrides with the `FORGE_` prefix.
    ///
    /// Mapping: `FORGE_<SECTION>_<FIELD>` (uppercase, underscores).
    /// For example, `FORGE_WORLD_WIDTH=128` sets `world.width = 128`.
    pub fn apply_env_overrides(&mut self) {
        macro_rules! env_override {
            ($section:ident . $field:ident, $ty:ty) => {
                let key = format!(
                    "{}{}_{}", ENV_PREFIX,
                    stringify!($section).to_uppercase(),
                    stringify!($field).to_uppercase()
                );
                if let Ok(val) = std::env::var(&key) {
                    match val.parse::<$ty>() {
                        Ok(parsed) => {
                            debug!(key = %key, value = %val, "applying env override");
                            self.$section.$field = parsed;
                        }
                        Err(e) => {
                            warn!(key = %key, value = %val, error = %e, "invalid env override");
                        }
                    }
                }
            };
        }

        // World overrides
        env_override!(world.width, u16);
        env_override!(world.height, u16);
        env_override!(world.seed, u64);
        env_override!(world.biome_scale, f32);
        env_override!(world.resource_density, f32);
        env_override!(world.day_night_cycle_length, u32);
        env_override!(world.max_entities, u16);

        // Physics overrides
        env_override!(physics.collision_enabled, bool);
        env_override!(physics.stamina_cost_move, i32);
        env_override!(physics.stamina_regen_rate, i32);
        env_override!(physics.max_velocity, i32);
        env_override!(physics.friction, i32);
        env_override!(physics.projectiles_enabled, bool);

        // Agent overrides
        env_override!(agents.num_agents, u32);
        env_override!(agents.default_vision_radius, u8);
        env_override!(agents.default_carry_capacity, u8);
        env_override!(agents.comm_vocab_size, u16);
        env_override!(agents.comm_radius, u16);

        // Task overrides
        env_override!(task.max_tier, u8);
        env_override!(task.max_episode_length, u64);
        env_override!(task.reward_scale, f32);
        env_override!(task.dense_rewards, bool);

        // Curriculum overrides
        env_override!(curriculum.enabled, bool);
        env_override!(curriculum.target_success_rate, f32);
        env_override!(curriculum.window_size, u32);

        // Rendering overrides
        env_override!(rendering.pixel_observations, bool);
        env_override!(rendering.pixel_width, u32);
        env_override!(rendering.pixel_height, u32);
        env_override!(rendering.record_replays, bool);

        // Agricultural overrides
        env_override!(agri.enabled, bool);
        env_override!(agri.spray_radius, u8);
        env_override!(agri.ndvi_scan_radius, u8);
        env_override!(agri.num_soil_nodes, u16);
        env_override!(agri.cropland_density, f32);
        env_override!(agri.pasture_density, f32);

        // Cloud overrides
        env_override!(cloud.enabled, bool);
        env_override!(cloud.num_workers, u32);
        env_override!(cloud.replay_batch_size, u32);
        env_override!(cloud.heartbeat_interval_s, u32);
        env_override!(cloud.heartbeat_timeout_s, u32);
        env_override!(cloud.max_replay_size_bytes, u64);
        env_override!(cloud.compression_level, u8);
        env_override!(cloud.coordinator_port, u16);
        env_override!(cloud.model_version_retention, u32);
        env_override!(cloud.checkpoint_interval_steps, u64);

        // Cloud string path overrides
        if let Ok(val) = std::env::var("FORGE_CLOUD_REPLAY_ARCHIVE_PATH") {
            debug!(key = "FORGE_CLOUD_REPLAY_ARCHIVE_PATH", value = %val, "applying env override");
            self.cloud.replay_archive_path = val;
        }
        if let Ok(val) = std::env::var("FORGE_CLOUD_MODEL_REGISTRY_PATH") {
            debug!(key = "FORGE_CLOUD_MODEL_REGISTRY_PATH", value = %val, "applying env override");
            self.cloud.model_registry_path = val;
        }
        if let Ok(val) = std::env::var("FORGE_CLOUD_CHECKPOINT_PATH") {
            debug!(key = "FORGE_CLOUD_CHECKPOINT_PATH", value = %val, "applying env override");
            self.cloud.checkpoint_path = val;
        }

        // Cloud GCP string overrides
        if let Ok(val) = std::env::var("FORGE_CLOUD_STORAGE_BACKEND") {
            debug!(key = "FORGE_CLOUD_STORAGE_BACKEND", value = %val, "applying env override");
            self.cloud.storage_backend = val;
        }
        if let Ok(val) = std::env::var("FORGE_CLOUD_GCS_BUCKET") {
            debug!(key = "FORGE_CLOUD_GCS_BUCKET", value = %val, "applying env override");
            self.cloud.gcs_bucket = val;
        }
        if let Ok(val) = std::env::var("FORGE_CLOUD_GCS_PREFIX") {
            debug!(key = "FORGE_CLOUD_GCS_PREFIX", value = %val, "applying env override");
            self.cloud.gcs_prefix = val;
        }
        if let Ok(val) = std::env::var("FORGE_CLOUD_GCP_PROJECT") {
            debug!(key = "FORGE_CLOUD_GCP_PROJECT", value = %val, "applying env override");
            self.cloud.gcp_project = val;
        }
        if let Ok(val) = std::env::var("FORGE_CLOUD_GCP_REGION") {
            debug!(key = "FORGE_CLOUD_GCP_REGION", value = %val, "applying env override");
            self.cloud.gcp_region = val;
        }
        if let Ok(val) = std::env::var("FORGE_CLOUD_GCP_SERVICE_ACCOUNT") {
            debug!(key = "FORGE_CLOUD_GCP_SERVICE_ACCOUNT", value = %val, "applying env override");
            self.cloud.gcp_service_account = val;
        }

        // Edge overrides
        env_override!(edge.enabled, bool);
        env_override!(edge.mcts_latency_budget_ms, u32);
        env_override!(edge.mcts_min_simulations, u32);
        env_override!(edge.mcts_max_simulations, u32);
        env_override!(edge.telemetry_interval_s, u32);
        env_override!(edge.telemetry_buffer_bytes, u64);
        env_override!(edge.compress_telemetry, bool);
        env_override!(edge.onnx_batch_size, u32);
        env_override!(edge.onnx_num_threads, u32);
        env_override!(edge.model_update_interval_s, u32);
        env_override!(edge.latency_ema_alpha, f32);
        env_override!(edge.upload_retry_count, u32);
        env_override!(edge.upload_retry_base_ms, u64);

        // Hierarchical skill catalog overrides
        env_override!(skills.enabled, bool);
        env_override!(skills.default_horizon, u32);
        if let Ok(val) = std::env::var("FORGE_SKILLS_DEFAULT_SKILL") {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                debug!(key = "FORGE_SKILLS_DEFAULT_SKILL", value = %trimmed, "applying env override");
                self.skills.default_skill = trimmed.to_string();
            }
        }

        // Orchestration overrides
        env_override!(orchestration.enabled, bool);
        env_override!(orchestration.deerflow_sandboxed, bool);
        env_override!(orchestration.timeout_ms, u32);
        if let Ok(val) = std::env::var("FORGE_ORCHESTRATION_CONTROLLER_TYPE") {
            debug!(key = "FORGE_ORCHESTRATION_CONTROLLER_TYPE", value = %val, "applying env override");
            self.orchestration.controller_type = val;
        }
        if let Ok(val) = std::env::var("FORGE_ORCHESTRATION_ARTIFACT_DIR") {
            debug!(key = "FORGE_ORCHESTRATION_ARTIFACT_DIR", value = %val, "applying env override");
            self.orchestration.artifact_dir = val;
        }
        if let Ok(val) = std::env::var("FORGE_ORCHESTRATION_ENDPOINT") {
            debug!(key = "FORGE_ORCHESTRATION_ENDPOINT", value = %val, "applying env override");
            self.orchestration.endpoint = val;
        }

        // Honcho overrides
        env_override!(honcho.enabled, bool);
        env_override!(honcho.strip_latent_state, bool);
        env_override!(honcho.sync_interval_ticks, u32);
        if let Ok(val) = std::env::var("FORGE_HONCHO_ENDPOINT") {
            debug!(key = "FORGE_HONCHO_ENDPOINT", value = %val, "applying env override");
            self.honcho.endpoint = val;
        }

        // Edge GCS string overrides
        if let Ok(val) = std::env::var("FORGE_EDGE_GCS_MODEL_BUCKET") {
            debug!(key = "FORGE_EDGE_GCS_MODEL_BUCKET", value = %val, "applying env override");
            self.edge.gcs_model_bucket = val;
        }
        if let Ok(val) = std::env::var("FORGE_EDGE_GCS_MODEL_PREFIX") {
            debug!(key = "FORGE_EDGE_GCS_MODEL_PREFIX", value = %val, "applying env override");
            self.edge.gcs_model_prefix = val;
        }
    }
}

/// Team configuration for multi-agent scenarios.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum TeamStructure {
    /// Every agent is independent.
    #[default]
    FreeForAll,
    /// Fixed teams specified by team assignments.
    FixedTeams(Vec<Vec<u32>>),
    /// Dynamic alliances formed through communication.
    DynamicAlliances,
}

#[cfg(test)]
#[path = "config/tests.rs"]
mod tests;
