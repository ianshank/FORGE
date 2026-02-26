//! Configuration types for the FORGE simulation platform.
//!
//! All simulation parameters are configurable through these structs.
//! No hard-coded values — defaults are provided via `Default` trait
//! and can be overridden at construction time.

use serde::{Deserialize, Serialize};

use crate::constants;

/// Top-level configuration for a FORGE simulation instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
}

/// World generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        }
    }
}

/// Physics system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Team configuration for multi-agent scenarios.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
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
}
