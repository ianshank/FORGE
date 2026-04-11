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

/// Top-level configuration for a FORGE simulation instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
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
}

/// World generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorldConfig {
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
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
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
        }
    }
}

/// Physics system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
        }
    }
}

/// Curriculum controller configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
        }
    }
}

/// Configuration for agricultural drone simulation.
///
/// When `enabled` is false (default), all agricultural systems are skipped
/// and the simulation behaves identically to pre-agriculture versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    /// Mutex protecting environment variable access during tests.
    ///
    /// **IMPORTANT**: Tests that modify `std::env` (via `set_var`/`remove_var`) MUST
    /// acquire this lock to prevent race conditions when tests run in parallel.
    /// Environment variables are process-global state, so concurrent modification
    /// by multiple tests can cause non-deterministic failures.
    ///
    /// **Usage**: Call `let _lock = ENV_TEST_LOCK.lock().unwrap();` at the start of
    /// any test that modifies environment variables.
    ///
    /// **Monitoring Note**: If new tests are added that interact with `std::env`,
    /// ensure they also acquire this lock. Consider refactoring env-dependent code
    /// to accept a config parameter instead of reading from `std::env` directly
    /// to improve testability and eliminate this class of race conditions.
    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard for restoring environment variables on drop.
    ///
    /// Ensures that environment variables are restored even if the test panics,
    /// preventing variable leakage into subsequent tests.
    struct EnvironmentGuard {
        var_name: &'static str,
        original_value: Option<String>,
    }

    impl Drop for EnvironmentGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.original_value {
                std::env::set_var(self.var_name, value);
            } else {
                std::env::remove_var(self.var_name);
            }
        }
    }

    #[test]
    fn test_from_toml_str_partial() {
        let toml_str = r#"
[world]
width = 128
height = 128

[agents]
num_agents = 4
"#;
        let config = ForgeConfig::from_toml_str(toml_str).unwrap();
        assert_eq!(config.world.width, 128);
        assert_eq!(config.world.height, 128);
        assert_eq!(config.agents.num_agents, 4);
        // Defaults should fill in
        assert_eq!(config.world.seed, constants::DEFAULT_SEED);
        assert!(config.physics.collision_enabled);
    }

    #[test]
    fn test_from_toml_str_empty() {
        let config = ForgeConfig::from_toml_str("").unwrap();
        let defaults = ForgeConfig::default();
        assert_eq!(config.world.width, defaults.world.width);
        assert_eq!(config.agents.num_agents, defaults.agents.num_agents);
    }

    #[test]
    fn test_from_toml_str_invalid() {
        let result = ForgeConfig::from_toml_str("invalid [[[ toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_toml_file() {
        // Lock needed because from_toml() calls apply_env_overrides()
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join("forge_test_config");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "[world]\nwidth = 32\nheight = 32").unwrap();
        drop(f);

        let config = ForgeConfig::from_toml(&path).unwrap();
        assert_eq!(config.world.width, 32);
        assert_eq!(config.world.height, 32);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_toml_missing_file() {
        let result = ForgeConfig::from_toml("/nonexistent/path/config.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_env_overrides() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        // Use EnvironmentGuard to ensure vars are restored even if test panics
        let _width_guard = EnvironmentGuard {
            var_name: "FORGE_WORLD_WIDTH",
            original_value: std::env::var("FORGE_WORLD_WIDTH").ok(),
        };
        let _agents_guard = EnvironmentGuard {
            var_name: "FORGE_AGENTS_NUM_AGENTS",
            original_value: std::env::var("FORGE_AGENTS_NUM_AGENTS").ok(),
        };

        std::env::set_var("FORGE_WORLD_WIDTH", "200");
        std::env::set_var("FORGE_AGENTS_NUM_AGENTS", "8");
        let mut config = ForgeConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.world.width, 200);
        assert_eq!(config.agents.num_agents, 8);
        // Guards automatically restore on drop
    }

    #[test]
    fn test_env_overrides_invalid_value() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        // Use EnvironmentGuard to ensure var is restored even if test panics
        let _width_guard = EnvironmentGuard {
            var_name: "FORGE_WORLD_WIDTH",
            original_value: std::env::var("FORGE_WORLD_WIDTH").ok(),
        };

        std::env::set_var("FORGE_WORLD_WIDTH", "not_a_number");
        let mut config = ForgeConfig::default();
        let original_width = config.world.width;
        config.apply_env_overrides();
        // Invalid value should be ignored
        assert_eq!(config.world.width, original_width);
        // Guard automatically restores on drop
    }

    #[test]
    fn test_default_config_is_valid() {
        let config = ForgeConfig::default();
        assert!(config.world.width >= config.world.min_dimension);
        assert!(config.world.width <= config.world.max_dimension);
        assert!(config.world.height >= config.world.min_dimension);
        assert!(config.world.height <= config.world.max_dimension);
        assert!((0.0..=1.0).contains(&config.world.resource_density));
        assert!(config.agents.num_agents >= 1);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = ForgeConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ForgeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.world.width, config.world.width);
        assert_eq!(deserialized.world.height, config.world.height);
        assert_eq!(deserialized.world.seed, config.world.seed);
        assert_eq!(deserialized.agents.num_agents, config.agents.num_agents);
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let config = ForgeConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: ForgeConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(deserialized.world.width, config.world.width);
        assert_eq!(
            deserialized.physics.collision_enabled,
            config.physics.collision_enabled
        );
    }

    #[test]
    fn test_team_structure_variants() {
        let ffa = TeamStructure::FreeForAll;
        let fixed = TeamStructure::FixedTeams(vec![vec![0, 1], vec![2, 3]]);
        let dynamic = TeamStructure::DynamicAlliances;

        // Verify serialization works for all variants
        for team in &[ffa, fixed, dynamic] {
            let json = serde_json::to_string(team).unwrap();
            let _: TeamStructure = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_config_partial_json_deserialization() {
        // Only set a few fields; everything else should get defaults
        let json = r#"{"world": {"width": 32, "height": 128}, "agents": {"num_agents": 4}}"#;
        let config: ForgeConfig = serde_json::from_str(json).unwrap();

        // Explicitly set fields
        assert_eq!(config.world.width, 32);
        assert_eq!(config.world.height, 128);
        assert_eq!(config.agents.num_agents, 4);

        // Default-filled fields
        let defaults = ForgeConfig::default();
        assert_eq!(config.world.seed, defaults.world.seed);
        assert_eq!(config.world.biome_scale, defaults.world.biome_scale);
        assert_eq!(
            config.world.resource_density,
            defaults.world.resource_density
        );
        assert_eq!(
            config.physics.collision_enabled,
            defaults.physics.collision_enabled
        );
        assert_eq!(config.physics.friction, defaults.physics.friction);
        assert_eq!(config.crafting.enabled, defaults.crafting.enabled);
        assert_eq!(
            config.crafting.max_ingredients,
            defaults.crafting.max_ingredients
        );
        assert_eq!(config.task.max_tier, defaults.task.max_tier);
        assert_eq!(
            config.task.max_episode_length,
            defaults.task.max_episode_length
        );
        assert_eq!(config.curriculum.enabled, defaults.curriculum.enabled);
        assert_eq!(
            config.rendering.pixel_observations,
            defaults.rendering.pixel_observations
        );
    }

    #[cfg(test)]
    mod proptests {
        use super::*;
        use crate::constants::{MAX_WORLD_DIMENSION, MIN_WORLD_DIMENSION};
        use crate::validation::validate_config;
        use proptest::prelude::*;

        /// Strategy that generates world dimensions within the valid range
        /// [MIN_WORLD_DIMENSION, MAX_WORLD_DIMENSION].
        fn valid_dimension() -> impl Strategy<Value = u16> {
            MIN_WORLD_DIMENSION..=MAX_WORLD_DIMENSION
        }

        proptest! {
            /// Any config built from valid dimensions and a positive reward scale
            /// must pass `validate_config`.
            #[test]
            fn prop_valid_config_passes_validation(
                width  in valid_dimension(),
                height in valid_dimension(),
                seed   in 0u64..=u64::MAX,
            ) {
                let mut config = ForgeConfig::default();
                config.world.width  = width;
                config.world.height = height;
                config.world.seed   = seed;
                // Default vision radius is 5; ensure it fits both dimensions.
                let max_radius = ((width.min(height) - 1) / 2) as u8;
                config.agents.default_vision_radius =
                    config.agents.default_vision_radius.min(max_radius);
                prop_assert!(
                    validate_config(&config).is_ok(),
                    "config should be valid: {:?}",
                    validate_config(&config)
                );
            }

            /// A `ForgeConfig` serialized to TOML and deserialized must produce
            /// identical `world.width`, `world.height`, and `world.seed` values.
            ///
            /// Seeds are restricted to `[0, i64::MAX]` because the `toml` v0.8 crate
            /// serializes `u64` values via `i64` and rejects values that would overflow.
            #[test]
            fn prop_config_toml_roundtrip(
                width  in valid_dimension(),
                height in valid_dimension(),
                seed   in 0u64..=(i64::MAX as u64),
            ) {
                let mut config = ForgeConfig::default();
                config.world.width  = width;
                config.world.height = height;
                config.world.seed   = seed;

                let toml_str = toml::to_string(&config)
                    .expect("serialization must not fail");
                let restored: ForgeConfig = toml::from_str(&toml_str)
                    .expect("deserialization must not fail");

                prop_assert_eq!(restored.world.width,  config.world.width);
                prop_assert_eq!(restored.world.height, config.world.height);
                prop_assert_eq!(restored.world.seed,   config.world.seed);
                prop_assert_eq!(
                    restored.physics.collision_enabled,
                    config.physics.collision_enabled
                );
                prop_assert_eq!(
                    restored.agents.num_agents,
                    config.agents.num_agents
                );
            }
        }
    }

    #[test]
    fn test_all_configs_implement_default() {
        let _world = WorldConfig::default();
        let _physics = PhysicsConfig::default();
        let _crafting = CraftingConfig::default();
        let _agent = AgentConfig::default();
        let _task = TaskConfig::default();
        let _curriculum = CurriculumConfig::default();
        let _render = RenderConfig::default();
        let _agri = AgriConfig::default();
        let _forge = ForgeConfig::default();
        let _team = TeamStructure::default();

        // Verify sub-config defaults match constant values
        let world = WorldConfig::default();
        assert_eq!(world.width, constants::DEFAULT_WORLD_WIDTH);
        assert_eq!(world.height, constants::DEFAULT_WORLD_HEIGHT);

        let physics = PhysicsConfig::default();
        assert_eq!(
            physics.stamina_cost_move,
            constants::DEFAULT_STAMINA_COST_MOVE
        );
        assert_eq!(physics.max_velocity, constants::DEFAULT_MAX_VELOCITY);

        let crafting = CraftingConfig::default();
        assert_eq!(crafting.max_ingredients, constants::DEFAULT_MAX_INGREDIENTS);

        let agent = AgentConfig::default();
        assert_eq!(agent.starting_health, constants::DEFAULT_STARTING_HEALTH);
        assert_eq!(agent.max_stamina, constants::DEFAULT_MAX_STAMINA);

        let task = TaskConfig::default();
        assert_eq!(task.max_tier, constants::DEFAULT_MAX_TASK_TIER);
        assert_eq!(task.reward_scale, constants::DEFAULT_REWARD_SCALE);

        let curriculum = CurriculumConfig::default();
        assert_eq!(
            curriculum.target_success_rate,
            constants::DEFAULT_TARGET_SUCCESS_RATE
        );

        let render = RenderConfig::default();
        assert_eq!(render.pixel_width, constants::DEFAULT_PIXEL_WIDTH);
        assert_eq!(render.pixel_height, constants::DEFAULT_PIXEL_HEIGHT);
    }

    #[test]
    fn test_drone_config_default_disabled() {
        let config = DroneConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.num_aerial, 0);
        assert_eq!(config.num_ground_vehicles, 0);
    }

    #[test]
    fn test_drone_config_default_values_match_constants() {
        let config = DroneConfig::default();
        assert_eq!(config.max_altitude, constants::DEFAULT_MAX_ALTITUDE);
        assert_eq!(
            config.aerial_drain_rate,
            constants::DEFAULT_AERIAL_DRAIN_RATE
        );
        assert_eq!(config.starting_battery, constants::DEFAULT_STARTING_BATTERY);
        assert_eq!(config.max_battery, constants::DEFAULT_MAX_BATTERY);
    }

    #[test]
    fn test_forge_config_default_has_drone() {
        let config = ForgeConfig::default();
        assert!(!config.drone.enabled);
    }

    #[test]
    fn test_drone_config_serde_roundtrip() {
        let config = DroneConfig {
            enabled: true,
            num_aerial: 3,
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DroneConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.enabled);
        assert_eq!(deserialized.num_aerial, 3);
    }

    #[test]
    fn test_agri_config_default_disabled() {
        let config = AgriConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.num_soil_nodes, 0);
    }

    #[test]
    fn test_agri_config_default_values_match_constants() {
        let config = AgriConfig::default();
        assert_eq!(
            config.crop_growth_rate,
            constants::DEFAULT_AGRI_CROP_GROWTH_RATE
        );
        assert_eq!(config.spray_radius, constants::DEFAULT_AGRI_SPRAY_RADIUS);
        assert_eq!(
            config.spray_efficacy,
            constants::DEFAULT_AGRI_SPRAY_EFFICACY
        );
        assert_eq!(
            config.ndvi_scan_radius,
            constants::DEFAULT_AGRI_NDVI_SCAN_RADIUS
        );
        assert_eq!(
            config.initial_crop_health,
            constants::DEFAULT_AGRI_INITIAL_CROP_HEALTH
        );
    }

    #[test]
    fn test_agri_config_serde_roundtrip() {
        let config = AgriConfig {
            enabled: true,
            num_soil_nodes: 10,
            cropland_density: 0.5,
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AgriConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.enabled);
        assert_eq!(deserialized.num_soil_nodes, 10);
        assert_eq!(deserialized.cropland_density, 0.5);
    }

    #[test]
    fn test_forge_config_default_has_agri() {
        let config = ForgeConfig::default();
        assert!(!config.agri.enabled);
    }

    #[test]
    fn test_config_clone_equality() {
        let original = ForgeConfig::default();
        let cloned = original.clone();

        // WorldConfig fields
        assert_eq!(cloned.world.width, original.world.width);
        assert_eq!(cloned.world.height, original.world.height);
        assert_eq!(cloned.world.seed, original.world.seed);
        assert_eq!(cloned.world.biome_scale, original.world.biome_scale);
        assert_eq!(
            cloned.world.resource_density,
            original.world.resource_density
        );
        assert_eq!(
            cloned.world.day_night_cycle_length,
            original.world.day_night_cycle_length
        );
        assert_eq!(cloned.world.max_entities, original.world.max_entities);
        assert_eq!(cloned.world.min_dimension, original.world.min_dimension);
        assert_eq!(cloned.world.max_dimension, original.world.max_dimension);

        // PhysicsConfig fields
        assert_eq!(
            cloned.physics.collision_enabled,
            original.physics.collision_enabled
        );
        assert_eq!(
            cloned.physics.stamina_cost_move,
            original.physics.stamina_cost_move
        );
        assert_eq!(
            cloned.physics.stamina_regen_rate,
            original.physics.stamina_regen_rate
        );
        assert_eq!(cloned.physics.max_velocity, original.physics.max_velocity);
        assert_eq!(cloned.physics.friction, original.physics.friction);
        assert_eq!(
            cloned.physics.projectiles_enabled,
            original.physics.projectiles_enabled
        );

        // CraftingConfig fields
        assert_eq!(cloned.crafting.enabled, original.crafting.enabled);
        assert_eq!(
            cloned.crafting.procedural_recipes,
            original.crafting.procedural_recipes
        );
        assert_eq!(
            cloned.crafting.max_ingredients,
            original.crafting.max_ingredients
        );
        assert_eq!(
            cloned.crafting.require_discovery,
            original.crafting.require_discovery
        );

        // AgentConfig fields
        assert_eq!(cloned.agents.num_agents, original.agents.num_agents);
        assert_eq!(
            cloned.agents.default_vision_radius,
            original.agents.default_vision_radius
        );
        assert_eq!(
            cloned.agents.default_carry_capacity,
            original.agents.default_carry_capacity
        );
        assert_eq!(
            cloned.agents.starting_health,
            original.agents.starting_health
        );
        assert_eq!(cloned.agents.max_health, original.agents.max_health);
        assert_eq!(
            cloned.agents.starting_stamina,
            original.agents.starting_stamina
        );
        assert_eq!(cloned.agents.max_stamina, original.agents.max_stamina);
        assert_eq!(cloned.agents.heterogeneous, original.agents.heterogeneous);
        assert_eq!(
            cloned.agents.comm_vocab_size,
            original.agents.comm_vocab_size
        );
        assert_eq!(cloned.agents.comm_radius, original.agents.comm_radius);
        assert_eq!(
            cloned.agents.comm_buffer_size,
            original.agents.comm_buffer_size
        );

        // TaskConfig fields
        assert_eq!(cloned.task.enabled, original.task.enabled);
        assert_eq!(cloned.task.max_tier, original.task.max_tier);
        assert_eq!(cloned.task.max_predicates, original.task.max_predicates);
        assert_eq!(
            cloned.task.max_episode_length,
            original.task.max_episode_length
        );
        assert_eq!(cloned.task.reward_scale, original.task.reward_scale);
        assert_eq!(cloned.task.dense_rewards, original.task.dense_rewards);

        // CurriculumConfig fields
        assert_eq!(cloned.curriculum.enabled, original.curriculum.enabled);
        assert_eq!(
            cloned.curriculum.target_success_rate,
            original.curriculum.target_success_rate
        );
        assert_eq!(
            cloned.curriculum.window_size,
            original.curriculum.window_size
        );
        assert_eq!(
            cloned.curriculum.warmup_episodes,
            original.curriculum.warmup_episodes
        );
        assert_eq!(
            cloned.curriculum.adjustment_rate,
            original.curriculum.adjustment_rate
        );

        // RenderConfig fields
        assert_eq!(
            cloned.rendering.pixel_observations,
            original.rendering.pixel_observations
        );
        assert_eq!(cloned.rendering.pixel_width, original.rendering.pixel_width);
        assert_eq!(
            cloned.rendering.pixel_height,
            original.rendering.pixel_height
        );
        assert_eq!(
            cloned.rendering.record_replays,
            original.rendering.record_replays
        );
    }

    // ---- Proptest: config invariants ----

    mod proptests_roundtrip {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// ForgeConfig survives JSON roundtrip with arbitrary valid seeds.
            #[test]
            fn config_json_roundtrip(seed in 0u64..u64::MAX) {
                let mut config = ForgeConfig::default();
                config.world.seed = seed;
                let json = serde_json::to_string(&config).unwrap();
                let deser: ForgeConfig = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(deser.world.seed, seed);
                prop_assert_eq!(deser.world.width, config.world.width);
                prop_assert_eq!(deser.agents.num_agents, config.agents.num_agents);
            }

            /// Config with varying dimensions roundtrips correctly.
            #[test]
            fn config_dimensions_roundtrip(
                w in 8u16..512,
                h in 8u16..512,
            ) {
                let mut config = ForgeConfig::default();
                config.world.width = w;
                config.world.height = h;
                let json = serde_json::to_string(&config).unwrap();
                let deser: ForgeConfig = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(deser.world.width, w);
                prop_assert_eq!(deser.world.height, h);
            }
        }
    }
}
