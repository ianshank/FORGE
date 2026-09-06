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
fn test_grid_type_parses_lower_and_pascal_case() {
    let lower = ForgeConfig::from_toml_str(
        r#"
[world]
grid_type = "hex"
"#,
    )
    .unwrap();
    assert_eq!(lower.world.grid_type, GridType::Hex);

    let pascal = ForgeConfig::from_toml_str(
        r#"
[world]
grid_type = "Hex"
"#,
    )
    .unwrap();
    assert_eq!(pascal.world.grid_type, GridType::Hex);
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
fn test_env_overrides_cloud_gcp_fields() {
    let _lock = ENV_TEST_LOCK.lock().unwrap();

    let env_vars = [
        "FORGE_CLOUD_STORAGE_BACKEND",
        "FORGE_CLOUD_GCS_BUCKET",
        "FORGE_CLOUD_GCS_PREFIX",
        "FORGE_CLOUD_GCP_PROJECT",
        "FORGE_CLOUD_GCP_REGION",
        "FORGE_CLOUD_GCP_SERVICE_ACCOUNT",
    ];
    let guards: Vec<EnvironmentGuard> = env_vars
        .iter()
        .map(|&var| EnvironmentGuard {
            var_name: var,
            original_value: std::env::var(var).ok(),
        })
        .collect();

    std::env::set_var("FORGE_CLOUD_STORAGE_BACKEND", "gcs");
    std::env::set_var("FORGE_CLOUD_GCS_BUCKET", "test-bucket");
    std::env::set_var("FORGE_CLOUD_GCS_PREFIX", "test-prefix/");
    std::env::set_var("FORGE_CLOUD_GCP_PROJECT", "my-project");
    std::env::set_var("FORGE_CLOUD_GCP_REGION", "europe-west1");
    std::env::set_var("FORGE_CLOUD_GCP_SERVICE_ACCOUNT", "sa@proj.iam");

    let mut config = ForgeConfig::default();
    config.apply_env_overrides();

    assert_eq!(config.cloud.storage_backend, "gcs");
    assert_eq!(config.cloud.gcs_bucket, "test-bucket");
    assert_eq!(config.cloud.gcs_prefix, "test-prefix/");
    assert_eq!(config.cloud.gcp_project, "my-project");
    assert_eq!(config.cloud.gcp_region, "europe-west1");
    assert_eq!(config.cloud.gcp_service_account, "sa@proj.iam");

    drop(guards);
}

#[test]
fn test_env_overrides_edge_gcs_fields() {
    let _lock = ENV_TEST_LOCK.lock().unwrap();

    let env_vars = ["FORGE_EDGE_GCS_MODEL_BUCKET", "FORGE_EDGE_GCS_MODEL_PREFIX"];
    let guards: Vec<EnvironmentGuard> = env_vars
        .iter()
        .map(|&var| EnvironmentGuard {
            var_name: var,
            original_value: std::env::var(var).ok(),
        })
        .collect();

    std::env::set_var("FORGE_EDGE_GCS_MODEL_BUCKET", "edge-bucket");
    std::env::set_var("FORGE_EDGE_GCS_MODEL_PREFIX", "edge/models/");

    let mut config = ForgeConfig::default();
    config.apply_env_overrides();

    assert_eq!(config.edge.gcs_model_bucket, "edge-bucket");
    assert_eq!(config.edge.gcs_model_prefix, "edge/models/");

    drop(guards);
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
    let _cloud = CloudConfig::default();
    let _edge = EdgeConfig::default();
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

// ---- Cloud config tests ----

#[test]
fn test_cloud_config_default_disabled() {
    let config = CloudConfig::default();
    assert!(!config.enabled);
}

#[test]
fn test_cloud_config_default_values_match_constants() {
    let config = CloudConfig::default();
    assert_eq!(config.num_workers, constants::DEFAULT_CLOUD_NUM_WORKERS);
    assert_eq!(
        config.replay_batch_size,
        constants::DEFAULT_CLOUD_REPLAY_BATCH_SIZE
    );
    assert_eq!(
        config.compression_level,
        constants::DEFAULT_CLOUD_COMPRESSION_LEVEL
    );
    assert_eq!(
        config.coordinator_port,
        constants::DEFAULT_CLOUD_COORDINATOR_PORT
    );
    assert_eq!(
        config.model_version_retention,
        constants::DEFAULT_CLOUD_MODEL_VERSION_RETENTION
    );
}

#[test]
fn test_cloud_config_serde_roundtrip() {
    let config = CloudConfig {
        enabled: true,
        num_workers: 16,
        compression_level: 6,
        ..Default::default()
    };
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: CloudConfig = serde_json::from_str(&json).unwrap();
    assert!(deserialized.enabled);
    assert_eq!(deserialized.num_workers, 16);
    assert_eq!(deserialized.compression_level, 6);
}

#[test]
fn test_forge_config_default_has_cloud() {
    let config = ForgeConfig::default();
    assert!(!config.cloud.enabled);
}

// ---- Edge config tests ----

#[test]
fn test_edge_config_default_disabled() {
    let config = EdgeConfig::default();
    assert!(!config.enabled);
}

#[test]
fn test_edge_config_default_values_match_constants() {
    let config = EdgeConfig::default();
    assert_eq!(
        config.mcts_latency_budget_ms,
        constants::DEFAULT_EDGE_MCTS_LATENCY_BUDGET_MS
    );
    assert_eq!(
        config.mcts_min_simulations,
        constants::DEFAULT_EDGE_MCTS_MIN_SIMULATIONS
    );
    assert_eq!(
        config.mcts_max_simulations,
        constants::DEFAULT_EDGE_MCTS_MAX_SIMULATIONS
    );
    assert_eq!(
        config.onnx_batch_size,
        constants::DEFAULT_EDGE_ONNX_BATCH_SIZE
    );
    assert_eq!(
        config.latency_ema_alpha,
        constants::DEFAULT_EDGE_LATENCY_EMA_ALPHA
    );
}

#[test]
fn test_edge_config_serde_roundtrip() {
    let config = EdgeConfig {
        enabled: true,
        mcts_latency_budget_ms: 100,
        mcts_max_simulations: 500,
        ..Default::default()
    };
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: EdgeConfig = serde_json::from_str(&json).unwrap();
    assert!(deserialized.enabled);
    assert_eq!(deserialized.mcts_latency_budget_ms, 100);
    assert_eq!(deserialized.mcts_max_simulations, 500);
}

#[test]
fn test_forge_config_default_has_edge() {
    let config = ForgeConfig::default();
    assert!(!config.edge.enabled);
}

#[test]
fn test_cloud_edge_backward_compatible_deserialization() {
    // Existing TOML without cloud/edge sections should still parse
    let json = r#"{"world": {"width": 32}, "agents": {"num_agents": 2}}"#;
    let config: ForgeConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.world.width, 32);
    assert_eq!(config.agents.num_agents, 2);
    // Cloud and edge should be disabled by default
    assert!(!config.cloud.enabled);
    assert!(!config.edge.enabled);
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

// ---- Proptest: config invariants (serde roundtrips) ----

mod proptests_serde {
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
