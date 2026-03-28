//! Configuration types for the MangoMAS integration layer.
//!
//! All constants flow through config structs with `Default` implementations.
//! No hard-coded values — every parameter is overridable.

use serde::{Deserialize, Serialize};

/// Target platform for MangoMAS training.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Platform {
    /// Autonomous car (2D continuous action space).
    Car,
    /// Autonomous drone (4D continuous action space).
    #[default]
    Drone,
}

// ── Action Adapter ──────────────────────────────────────────────────────

/// Default number of bins per continuous axis for discretization.
const DEFAULT_BINS_PER_AXIS: u32 = 7;
/// Default continuous action range minimum.
const DEFAULT_ACTION_RANGE_MIN: f32 = -1.0;
/// Default continuous action range maximum.
const DEFAULT_ACTION_RANGE_MAX: f32 = 1.0;

/// Configuration for the action space adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ActionAdapterConfig {
    /// Number of discrete bins per continuous axis.
    pub bins_per_axis: u32,
    /// Minimum value for continuous action range.
    pub action_range_min: f32,
    /// Maximum value for continuous action range.
    pub action_range_max: f32,
    /// Target platform determines continuous dimensionality (2 for car, 4 for drone).
    pub platform: Platform,
}

impl Default for ActionAdapterConfig {
    fn default() -> Self {
        Self {
            bins_per_axis: DEFAULT_BINS_PER_AXIS,
            action_range_min: DEFAULT_ACTION_RANGE_MIN,
            action_range_max: DEFAULT_ACTION_RANGE_MAX,
            platform: Platform::default(),
        }
    }
}

// ── Observation Adapter ─────��───────────────────────────────────────────

/// Default maximum number of grid summary features.
const DEFAULT_GRID_SUMMARY_DIM: usize = 64;
/// Whether to include raw grid tiles in the flat output.
const DEFAULT_INCLUDE_RAW_GRID: bool = false;
/// Default maximum world dimension for position normalization.
pub const DEFAULT_MAX_WORLD_DIM: f32 = 256.0;
/// Default maximum day phases (0-3).
pub const DEFAULT_MAX_DAY_PHASE: f32 = 3.0;
/// Default maximum altitude for normalization.
pub const DEFAULT_MAX_ALTITUDE: f32 = 10.0;
/// Default maximum morphology value.
pub const DEFAULT_MAX_MORPHOLOGY: f32 = 2.0;
/// Default maximum heading value.
pub const DEFAULT_MAX_HEADING: f32 = 3.0;
/// Default number of terrain types for normalization.
pub const DEFAULT_MAX_TERRAIN: f32 = 7.0;
/// Default maximum elevation for normalization.
pub const DEFAULT_MAX_ELEVATION: f32 = 10.0;
/// Default maximum object/resource type ID.
pub const DEFAULT_MAX_TYPE_ID: f32 = 255.0;
/// Default normalization factor for inventory item count.
pub const DEFAULT_INVENTORY_NORM: f32 = 100.0;

/// Number of grid summary features (3 counts + 8 terrain proportions).
pub const GRID_SUMMARY_FEATURES: usize = 11;
/// Number of scalar features (health, stamina, x, y, day_phase).
pub const SCALAR_FEATURES: usize = 5;
/// Number of inventory features (total_items, occupied_ratio).
pub const INVENTORY_FEATURES: usize = 2;
/// Number of drone-specific features (altitude, battery, morphology, heading).
pub const DRONE_FEATURES: usize = 4;

/// Configuration for the observation space adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObservationAdapterConfig {
    /// Dimensionality of the grid summary encoding.
    pub grid_summary_dim: usize,
    /// Whether to include raw grid tile features in the output.
    pub include_raw_grid: bool,
    /// Target platform determines which drone fields to include.
    pub platform: Platform,
    /// Maximum world dimension for position normalization.
    pub max_world_dim: f32,
    /// Maximum day phase value for normalization.
    pub max_day_phase: f32,
    /// Maximum altitude value for normalization.
    pub max_altitude: f32,
    /// Maximum morphology value for normalization.
    pub max_morphology: f32,
    /// Maximum heading value for normalization.
    pub max_heading: f32,
    /// Maximum terrain type for normalization.
    pub max_terrain: f32,
    /// Maximum elevation for normalization.
    pub max_elevation: f32,
    /// Maximum object/resource type ID for normalization.
    pub max_type_id: f32,
    /// Normalization factor for inventory item count.
    pub inventory_norm: f32,
}

impl Default for ObservationAdapterConfig {
    fn default() -> Self {
        Self {
            grid_summary_dim: DEFAULT_GRID_SUMMARY_DIM,
            include_raw_grid: DEFAULT_INCLUDE_RAW_GRID,
            platform: Platform::default(),
            max_world_dim: DEFAULT_MAX_WORLD_DIM,
            max_day_phase: DEFAULT_MAX_DAY_PHASE,
            max_altitude: DEFAULT_MAX_ALTITUDE,
            max_morphology: DEFAULT_MAX_MORPHOLOGY,
            max_heading: DEFAULT_MAX_HEADING,
            max_terrain: DEFAULT_MAX_TERRAIN,
            max_elevation: DEFAULT_MAX_ELEVATION,
            max_type_id: DEFAULT_MAX_TYPE_ID,
            inventory_norm: DEFAULT_INVENTORY_NORM,
        }
    }
}

// ── Sweep ───────────────────────────────────────────────────────────────

/// Default number of episodes per sweep configuration evaluation.
const DEFAULT_EPISODES_PER_CONFIG: u32 = 50;
/// Default number of parallel workers for sweep.
const DEFAULT_SWEEP_WORKERS: usize = 4;
/// Default minimum PUCT exploration constant for sweep range.
const DEFAULT_C_PUCT_MIN: f32 = 0.5;
/// Default maximum PUCT exploration constant for sweep range.
const DEFAULT_C_PUCT_MAX: f32 = 3.0;
/// Default number of steps for PUCT sweep.
const DEFAULT_C_PUCT_STEPS: u32 = 6;
/// Default minimum simulation budget.
const DEFAULT_SIM_BUDGET_MIN: u32 = 10;
/// Default maximum simulation budget.
const DEFAULT_SIM_BUDGET_MAX: u32 = 500;
/// Default number of steps for simulation budget sweep.
const DEFAULT_SIM_BUDGET_STEPS: u32 = 5;
/// Default minimum rollout depth.
const DEFAULT_DEPTH_MIN: u32 = 10;
/// Default maximum rollout depth.
const DEFAULT_DEPTH_MAX: u32 = 100;
/// Default number of steps for depth sweep.
const DEFAULT_DEPTH_STEPS: u32 = 4;
/// Default minimum discount factor.
const DEFAULT_DISCOUNT_MIN: f32 = 0.9;
/// Default maximum discount factor.
const DEFAULT_DISCOUNT_MAX: f32 = 0.999;
/// Default number of steps for discount sweep.
const DEFAULT_DISCOUNT_STEPS: u32 = 4;

/// Configuration for MCTS hyperparameter sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SweepConfig {
    /// Number of episodes to evaluate each parameter configuration.
    pub episodes_per_config: u32,
    /// Number of parallel workers for sweep execution.
    pub num_workers: usize,
    /// Sweep range for PUCT exploration constant.
    pub c_puct_range: (f32, f32),
    /// Number of steps in the PUCT sweep grid.
    pub c_puct_steps: u32,
    /// Sweep range for simulation budget.
    pub sim_budget_range: (u32, u32),
    /// Number of steps in the simulation budget sweep grid.
    pub sim_budget_steps: u32,
    /// Sweep range for rollout depth.
    pub depth_range: (u32, u32),
    /// Number of steps in the depth sweep grid.
    pub depth_steps: u32,
    /// Sweep range for discount factor.
    pub discount_range: (f32, f32),
    /// Number of steps in the discount sweep grid.
    pub discount_steps: u32,
    /// Random seed for deterministic sweep execution.
    pub seed: u64,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            episodes_per_config: DEFAULT_EPISODES_PER_CONFIG,
            num_workers: DEFAULT_SWEEP_WORKERS,
            c_puct_range: (DEFAULT_C_PUCT_MIN, DEFAULT_C_PUCT_MAX),
            c_puct_steps: DEFAULT_C_PUCT_STEPS,
            sim_budget_range: (DEFAULT_SIM_BUDGET_MIN, DEFAULT_SIM_BUDGET_MAX),
            sim_budget_steps: DEFAULT_SIM_BUDGET_STEPS,
            depth_range: (DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX),
            depth_steps: DEFAULT_DEPTH_STEPS,
            discount_range: (DEFAULT_DISCOUNT_MIN, DEFAULT_DISCOUNT_MAX),
            discount_steps: DEFAULT_DISCOUNT_STEPS,
            seed: 0,
        }
    }
}

// ── Surprise Validator ─────���────────────────────────────────────────────

/// Default cached simulation budget for low surprise.
const DEFAULT_LOW_SURPRISE_BUDGET: u32 = 25;
/// Default base simulation budget for medium surprise.
const DEFAULT_BASE_SURPRISE_BUDGET: u32 = 100;
/// Default full simulation budget for high surprise.
const DEFAULT_FULL_SURPRISE_BUDGET: u32 = 300;
/// Default low surprise threshold.
const DEFAULT_LOW_SURPRISE_THRESHOLD: f32 = 0.1;
/// Default high surprise threshold.
const DEFAULT_HIGH_SURPRISE_THRESHOLD: f32 = 0.5;
/// Default number of episodes per surprise level.
const DEFAULT_EPISODES_PER_SURPRISE_LEVEL: u32 = 100;

/// Configuration for surprise-adaptive budget validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SurpriseValidatorConfig {
    /// Simulation budget for low-surprise (cached) scenarios.
    pub low_surprise_budget: u32,
    /// Simulation budget for medium-surprise (base) scenarios.
    pub base_surprise_budget: u32,
    /// Simulation budget for high-surprise (full) scenarios.
    pub full_surprise_budget: u32,
    /// KL divergence threshold below which surprise is considered low.
    pub low_surprise_threshold: f32,
    /// KL divergence threshold above which surprise is considered high.
    pub high_surprise_threshold: f32,
    /// Number of episodes to run per surprise level.
    pub episodes_per_level: u32,
}

impl Default for SurpriseValidatorConfig {
    fn default() -> Self {
        Self {
            low_surprise_budget: DEFAULT_LOW_SURPRISE_BUDGET,
            base_surprise_budget: DEFAULT_BASE_SURPRISE_BUDGET,
            full_surprise_budget: DEFAULT_FULL_SURPRISE_BUDGET,
            low_surprise_threshold: DEFAULT_LOW_SURPRISE_THRESHOLD,
            high_surprise_threshold: DEFAULT_HIGH_SURPRISE_THRESHOLD,
            episodes_per_level: DEFAULT_EPISODES_PER_SURPRISE_LEVEL,
        }
    }
}

// ── Transfer ───────────────────────────────────────��────────────────────

/// Default number of BDI intention classes matching MangoMAS.
const DEFAULT_NUM_BDI_INTENTIONS: u8 = 8;
/// Default battery floor threshold for constitutional constraint.
pub const DEFAULT_BATTERY_THRESHOLD: f32 = 0.2;
/// Default altitude ceiling threshold (normalized).
pub const DEFAULT_ALTITUDE_THRESHOLD: f32 = 0.9;
/// Default speed ceiling threshold (normalized stamina usage).
pub const DEFAULT_SPEED_THRESHOLD: f32 = 0.8;
/// Default geofence proximity threshold (fraction of world edge).
pub const DEFAULT_GEOFENCE_THRESHOLD: f32 = 0.1;
/// Default threat exclusion proximity threshold.
pub const DEFAULT_THREAT_THRESHOLD: f32 = 0.3;
/// Default unknown constraint threshold.
pub const DEFAULT_UNKNOWN_CONSTRAINT_THRESHOLD: f32 = 0.5;
/// Default nearby-agent count above which threat is detected.
pub const DEFAULT_THREAT_AGENT_COUNT: usize = 1;
/// Default threat proximity value when threat is close.
pub const DEFAULT_THREAT_CLOSE_VALUE: f32 = 0.1;
/// Default threat proximity value when no threat is detected.
pub const DEFAULT_THREAT_SAFE_VALUE: f32 = 0.8;

/// A constraint threshold definition for constitutional mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintThreshold {
    /// Constraint name (must match a constitutional constraint name).
    pub name: String,
    /// Violation threshold value.
    pub threshold: f32,
    /// Whether this is a lower bound (true) or upper bound (false).
    pub is_lower_bound: bool,
}

/// Configuration for weight transfer and pre-training.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TransferConfig {
    /// Number of BDI intention classes (must match MangoMAS BDI model).
    pub num_bdi_intentions: u8,
    /// BDI action-to-intention mapping (FORGE discrete action ID → intention class index).
    /// If empty, a default semantic mapping is used.
    pub bdi_mapping_overrides: Vec<(u32, u8)>,
    /// Number of episodes to collect for BDI pre-training.
    pub bdi_collection_episodes: u32,
    /// RSSM sequence length for transition data.
    pub rssm_sequence_length: u32,
    /// Number of episodes to collect for RSSM pre-training.
    pub rssm_collection_episodes: u32,
    /// Names of the 5 constitutional safety constraints.
    pub constitutional_constraints: Vec<String>,
    /// Initial curiosity channel weights: [social, epistemic, perceptual, metacognitive].
    pub curiosity_weights: [f32; 4],
    /// Constraint thresholds for constitutional mapping. If empty, defaults are used.
    pub constraint_thresholds: Vec<ConstraintThreshold>,
    /// Maximum world dimension for geofence normalization.
    pub max_world_dim: f32,
    /// Minimum nearby agent count to trigger threat detection.
    pub threat_agent_count: usize,
    /// Proximity value returned when threat is close.
    pub threat_close_value: f32,
    /// Proximity value returned when no threat is detected.
    pub threat_safe_value: f32,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            num_bdi_intentions: DEFAULT_NUM_BDI_INTENTIONS,
            bdi_mapping_overrides: Vec::new(),
            bdi_collection_episodes: 10_000,
            rssm_sequence_length: 50,
            rssm_collection_episodes: 50_000,
            constitutional_constraints: vec![
                "battery_minimum".to_string(),
                "altitude_ceiling".to_string(),
                "speed_ceiling".to_string(),
                "geofence".to_string(),
                "threat_exclusion".to_string(),
            ],
            curiosity_weights: [0.4, 0.3, 0.2, 0.1],
            constraint_thresholds: vec![
                ConstraintThreshold {
                    name: "battery_minimum".to_string(),
                    threshold: DEFAULT_BATTERY_THRESHOLD,
                    is_lower_bound: true,
                },
                ConstraintThreshold {
                    name: "altitude_ceiling".to_string(),
                    threshold: DEFAULT_ALTITUDE_THRESHOLD,
                    is_lower_bound: false,
                },
                ConstraintThreshold {
                    name: "speed_ceiling".to_string(),
                    threshold: DEFAULT_SPEED_THRESHOLD,
                    is_lower_bound: false,
                },
                ConstraintThreshold {
                    name: "geofence".to_string(),
                    threshold: DEFAULT_GEOFENCE_THRESHOLD,
                    is_lower_bound: true,
                },
                ConstraintThreshold {
                    name: "threat_exclusion".to_string(),
                    threshold: DEFAULT_THREAT_THRESHOLD,
                    is_lower_bound: true,
                },
            ],
            max_world_dim: DEFAULT_MAX_WORLD_DIM,
            threat_agent_count: DEFAULT_THREAT_AGENT_COUNT,
            threat_close_value: DEFAULT_THREAT_CLOSE_VALUE,
            threat_safe_value: DEFAULT_THREAT_SAFE_VALUE,
        }
    }
}

// ── Curriculum ──────────��───────────────────────────────────────────────

/// Default number of curriculum tiers per platform.
const DEFAULT_NUM_TIERS: u8 = 5;
/// Default target success rate for curriculum progression.
const DEFAULT_CURRICULUM_TARGET: f32 = 0.6;
/// Default curriculum window size.
const DEFAULT_CURRICULUM_WINDOW: u32 = 100;
/// Default curriculum warmup episodes.
const DEFAULT_CURRICULUM_WARMUP: u32 = 20;
/// Default curriculum adjustment rate.
const DEFAULT_CURRICULUM_ADJUSTMENT: f32 = 0.1;

/// Configuration for platform-specific curriculum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PlatformCurriculumConfig {
    /// Target platform.
    pub platform: Platform,
    /// Number of difficulty tiers.
    pub num_tiers: u8,
    /// Target success rate for tier progression.
    pub target_success_rate: f32,
    /// Rolling window size for success rate computation.
    pub window_size: u32,
    /// Warmup episodes before adaptation begins.
    pub warmup_episodes: u32,
    /// Adjustment rate for tier weight shifting.
    pub adjustment_rate: f32,
}

impl Default for PlatformCurriculumConfig {
    fn default() -> Self {
        Self {
            platform: Platform::default(),
            num_tiers: DEFAULT_NUM_TIERS,
            target_success_rate: DEFAULT_CURRICULUM_TARGET,
            window_size: DEFAULT_CURRICULUM_WINDOW,
            warmup_episodes: DEFAULT_CURRICULUM_WARMUP,
            adjustment_rate: DEFAULT_CURRICULUM_ADJUSTMENT,
        }
    }
}

// ── Batch Runner ────────────────────────────────────────────────────────

/// Default maximum steps per episode.
const DEFAULT_MAX_EPISODE_STEPS: u64 = 1000;
/// Default number of parallel environments.
const DEFAULT_NUM_ENVS: usize = 8;

/// Configuration for the headless batch runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BatchRunnerConfig {
    /// Maximum steps per episode before truncation.
    pub max_episode_steps: u64,
    /// Number of parallel environments.
    pub num_envs: usize,
    /// Random seed base (each env gets seed + env_index).
    pub seed: u64,
    /// FORGE simulation configuration to use.
    pub forge_config: forge_types::config::ForgeConfig,
}

impl Default for BatchRunnerConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: DEFAULT_MAX_EPISODE_STEPS,
            num_envs: DEFAULT_NUM_ENVS,
            seed: 0,
            forge_config: forge_types::config::ForgeConfig::default(),
        }
    }
}

// ��─ Top-Level ─────────────���─────────────────────────────────────────────

/// Top-level configuration for MangoMAS integration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MangoMasConfig {
    /// Whether the MangoMAS integration layer is enabled.
    pub enabled: bool,
    /// Target platform for training.
    pub platform: Platform,
    /// Action space adapter configuration.
    pub action_adapter: ActionAdapterConfig,
    /// Observation space adapter configuration.
    pub observation_adapter: ObservationAdapterConfig,
    /// MCTS hyperparameter sweep configuration.
    pub sweep: SweepConfig,
    /// Weight transfer and pre-training configuration.
    pub transfer: TransferConfig,
    /// Platform-specific curriculum configuration.
    pub curriculum: PlatformCurriculumConfig,
    /// Batch runner configuration.
    pub batch_runner: BatchRunnerConfig,
    /// Surprise-adaptive budget validator configuration.
    pub surprise_validator: SurpriseValidatorConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_creation() {
        let config = MangoMasConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.platform, Platform::Drone);
    }

    #[test]
    fn test_platform_variants() {
        assert_eq!(Platform::default(), Platform::Drone);
        let car = Platform::Car;
        let drone = Platform::Drone;
        assert_ne!(car, drone);
    }

    #[test]
    fn test_action_adapter_config_defaults() {
        let config = ActionAdapterConfig::default();
        assert_eq!(config.bins_per_axis, DEFAULT_BINS_PER_AXIS);
        assert_eq!(config.action_range_min, DEFAULT_ACTION_RANGE_MIN);
        assert_eq!(config.action_range_max, DEFAULT_ACTION_RANGE_MAX);
    }

    #[test]
    fn test_sweep_config_defaults() {
        let config = SweepConfig::default();
        assert_eq!(
            config.c_puct_range,
            (DEFAULT_C_PUCT_MIN, DEFAULT_C_PUCT_MAX)
        );
        assert_eq!(config.episodes_per_config, DEFAULT_EPISODES_PER_CONFIG);
        assert!(config.c_puct_steps > 0);
    }

    #[test]
    fn test_transfer_config_defaults() {
        let config = TransferConfig::default();
        assert_eq!(config.num_bdi_intentions, DEFAULT_NUM_BDI_INTENTIONS);
        assert_eq!(config.constitutional_constraints.len(), 5);
        let weight_sum: f32 = config.curiosity_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_curriculum_config_defaults() {
        let config = PlatformCurriculumConfig::default();
        assert_eq!(config.num_tiers, DEFAULT_NUM_TIERS);
        assert!(config.target_success_rate > 0.0);
        assert!(config.target_success_rate < 1.0);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = MangoMasConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deser: MangoMasConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.enabled, config.enabled);
        assert_eq!(deser.platform, config.platform);
    }

    #[test]
    fn test_surprise_validator_config_defaults() {
        let config = SurpriseValidatorConfig::default();
        assert!(config.low_surprise_threshold < config.high_surprise_threshold);
        assert!(config.low_surprise_budget < config.base_surprise_budget);
        assert!(config.base_surprise_budget < config.full_surprise_budget);
    }

    #[test]
    fn test_batch_runner_config_defaults() {
        let config = BatchRunnerConfig::default();
        assert!(config.max_episode_steps > 0);
        assert!(config.num_envs > 0);
    }
}
