#![deny(missing_docs)]
#![deny(clippy::all)]

//! # forge-wasm
//!
//! WebAssembly visualization module for the FORGE simulation platform.
//!
//! This crate provides a JavaScript-friendly API via `wasm-bindgen` that
//! wraps the core simulation engine. All inputs and outputs are JSON strings,
//! making it easy to integrate with browser-based visualizations.

use forge_core::WorldState;
use forge_types::config::ForgeConfig;
use forge_types::observation::{ActionSpace, ObservationSpace};
use forge_types::Action;
use serde::Serialize;
use tracing::instrument;
use wasm_bindgen::prelude::*;

/// Response payload for a single simulation step, serialized to JSON.
#[derive(Debug, Serialize)]
struct StepResponse {
    /// Per-agent observations.
    observations: Vec<forge_types::observation::Observation>,
    /// Per-agent scalar rewards.
    rewards: Vec<f32>,
    /// Whether the episode terminated naturally (goal reached or agent died).
    terminated: bool,
    /// Whether the episode was truncated (max steps reached).
    truncated: bool,
    /// Diagnostic info about the current tick.
    info: forge_types::observation::StepInfo,
}

/// A FORGE simulation environment exposed to WebAssembly.
///
/// All public methods accept and return JSON strings so they can be called
/// directly from JavaScript without additional marshalling.
#[wasm_bindgen]
pub struct ForgeWasmEnv {
    /// The underlying simulation state.
    world: WorldState,
    /// Cached configuration for space queries.
    config: ForgeConfig,
}

#[wasm_bindgen]
impl ForgeWasmEnv {
    /// Creates a new FORGE environment from a JSON configuration string.
    ///
    /// # Arguments
    ///
    /// * `config_json` - A JSON string that deserializes into a `ForgeConfig`.
    ///   Pass an empty string or `"null"` to use all default values.
    ///
    /// # Panics
    ///
    /// Panics if `config_json` contains invalid JSON or fields that cannot be
    /// parsed into a `ForgeConfig`.
    #[instrument(skip_all)]
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Self {
        let trimmed = config_json.trim();
        let config: ForgeConfig = if trimmed.is_empty() || trimmed == "null" {
            ForgeConfig::default()
        } else {
            serde_json::from_str(config_json).expect("failed to parse ForgeConfig JSON")
        };
        let world = WorldState::new(config.clone()).expect("invalid config");
        Self { world, config }
    }

    /// Resets the simulation and returns the initial observation as a JSON string.
    ///
    /// # Arguments
    ///
    /// * `seed` - Optional RNG seed for the new episode. When `None`, a seed is
    ///   derived from the internal RNG.
    ///
    /// # Returns
    ///
    /// A JSON-encoded [`StepResponse`] containing the initial observations,
    /// zero rewards, and `terminated = false`.
    #[instrument(skip_all)]
    pub fn reset(&mut self, seed: Option<u64>) -> String {
        let result = self.world.reset(seed);
        let response = StepResponse {
            observations: result.observations,
            rewards: result.rewards,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        };
        serde_json::to_string(&response).expect("failed to serialize reset result")
    }

    /// Advances the simulation by one tick with the given discrete action.
    ///
    /// The action is decoded from a flat integer using the standard FORGE
    /// action encoding (see [`Action::from_discrete`]).
    ///
    /// # Arguments
    ///
    /// * `action` - A discrete action index for the first agent. Out-of-range
    ///   values are mapped to `Noop`.
    ///
    /// # Returns
    ///
    /// A JSON-encoded [`StepResponse`] with observations, rewards,
    /// termination flags, and diagnostic info.
    #[instrument(skip_all)]
    pub fn step(&mut self, action: u32) -> String {
        let comm_vocab = self.config.agents.comm_vocab_size;
        let decoded = Action::from_discrete(action, comm_vocab, self.config.drone.enabled)
            .unwrap_or(Action::Noop);
        let result = self.world.step(&[decoded]);
        let response = StepResponse {
            observations: result.observations,
            rewards: result.rewards,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        };
        serde_json::to_string(&response).expect("failed to serialize step result")
    }

    /// Returns an ASCII-art rendering of the current world grid.
    ///
    /// Useful for quick text-based visualization and debugging.
    /// Characters: `A` = agent, `O` = object, `R` = resource, `.` = ground,
    /// `~` = water, `#` = wall, `L` = lava, `I` = ice, `S` = sand,
    /// `T` = forest, `M` = mountain.
    #[instrument(skip_all)]
    pub fn render_ascii(&self) -> String {
        self.world.to_debug_grid()
    }

    /// Serializes the full simulation state to a JSON string.
    ///
    /// This includes tick count, grid dimensions, agent positions and stats,
    /// object and resource data, day phase, and termination flags. Intended
    /// for save/load, replay recording, or detailed inspection.
    #[instrument(skip_all)]
    pub fn get_state_json(&self) -> String {
        let state = SerializableState {
            tick: self.world.tick,
            grid_width: self.world.grid.width,
            grid_height: self.world.grid.height,
            num_agents: self.world.agents.len(),
            agents_alive: self.world.agents.iter().map(|a| a.alive).collect(),
            agent_positions: self
                .world
                .agents
                .iter()
                .map(|a| (a.position.x, a.position.y))
                .collect(),
            day_phase: self.world.day_phase,
            terminated: self.world.terminated,
            truncated: self.world.truncated,
            num_objects: self.world.objects.len(),
            num_resources: self.world.resources.len(),
        };
        serde_json::to_string(&state).expect("failed to serialize world state")
    }

    /// Returns a JSON description of the observation space.
    ///
    /// The response includes the flattened shape, value bounds, grid view
    /// dimensions, inventory size, and communication buffer size.
    #[instrument(skip_all)]
    pub fn observation_space_json(&self) -> String {
        let vr = self.config.agents.default_vision_radius;
        let view_side = 2 * vr as usize + 1;
        let features_per_tile = forge_types::constants::OBS_FEATURES_PER_TILE;

        let flat_size = forge_types::observation::Observation::flat_size(
            vr,
            self.config.agents.default_carry_capacity,
            self.config.agents.comm_buffer_size,
            self.config.task.max_predicates,
            self.config.drone.enabled,
        );

        let space = ObservationSpace {
            flat_shape: vec![flat_size],
            low: 0.0,
            high: u8::MAX as f32,
            grid_shape: (view_side, view_side, features_per_tile),
            inventory_size: self.config.agents.default_carry_capacity as usize,
            comm_buffer_size: self.config.agents.comm_buffer_size as usize,
        };
        serde_json::to_string(&space).expect("failed to serialize observation space")
    }

    /// Returns a JSON description of the action space.
    ///
    /// The response includes the total number of discrete actions and a list
    /// of human-readable action names indexed by action ID.
    #[instrument(skip_all)]
    pub fn action_space_json(&self) -> String {
        let space = ActionSpace::new(
            self.config.agents.comm_vocab_size,
            self.config.drone.enabled,
        );
        serde_json::to_string(&space).expect("failed to serialize action space")
    }
}

/// A simplified view of the world state for JSON serialization.
#[derive(Debug, Serialize)]
struct SerializableState {
    /// Current simulation tick.
    tick: u64,
    /// Grid width in tiles.
    grid_width: u16,
    /// Grid height in tiles.
    grid_height: u16,
    /// Total number of agents.
    num_agents: usize,
    /// Per-agent alive status.
    agents_alive: Vec<bool>,
    /// Per-agent (x, y) positions.
    agent_positions: Vec<(u16, u16)>,
    /// Current day/night phase.
    day_phase: u8,
    /// Whether the episode has terminated.
    terminated: bool,
    /// Whether the episode was truncated.
    truncated: bool,
    /// Number of objects in the world.
    num_objects: usize,
    /// Number of resource nodes in the world.
    num_resources: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a JSON string for a default `ForgeConfig`.
    fn default_config_json() -> String {
        serde_json::to_string(&ForgeConfig::default()).unwrap()
    }

    #[test]
    fn test_new_default_config() {
        let env = ForgeWasmEnv::new(&default_config_json());
        assert!(!env.world.terminated);
    }

    #[test]
    fn test_new_with_empty_string_uses_defaults() {
        let env = ForgeWasmEnv::new("");
        assert!(!env.world.terminated);
    }

    #[test]
    fn test_new_with_null_uses_defaults() {
        let env = ForgeWasmEnv::new("null");
        assert!(!env.world.terminated);
    }

    #[test]
    fn test_reset_returns_valid_json() {
        let mut env = ForgeWasmEnv::new(&default_config_json());
        let json = env.reset(Some(42));
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("observations").is_some());
        assert!(value.get("rewards").is_some());
        assert!(value.get("terminated").is_some());
        assert!(value.get("truncated").is_some());
    }

    #[test]
    fn test_step_returns_valid_json() {
        let mut env = ForgeWasmEnv::new(&default_config_json());
        let json = env.step(0); // Noop
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("observations").is_some());
        assert!(value.get("rewards").is_some());
    }

    #[test]
    fn test_render_ascii_nonempty() {
        let env = ForgeWasmEnv::new(&default_config_json());
        let grid = env.render_ascii();
        assert!(!grid.is_empty());
        assert!(grid.contains('A')); // At least one agent
    }

    #[test]
    fn test_get_state_json() {
        let env = ForgeWasmEnv::new(&default_config_json());
        let json = env.get_state_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("tick").is_some());
        assert!(value.get("grid_width").is_some());
        assert!(value.get("num_agents").is_some());
    }

    #[test]
    fn test_observation_space_json() {
        let env = ForgeWasmEnv::new(&default_config_json());
        let json = env.observation_space_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("flat_shape").is_some());
        assert!(value.get("grid_shape").is_some());
    }

    #[test]
    fn test_action_space_json() {
        let env = ForgeWasmEnv::new(&default_config_json());
        let json = env.action_space_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("n").is_some());
        assert!(value.get("action_names").is_some());
    }

    #[test]
    fn test_partial_config_json() {
        // Provide JSON with only some fields; the rest should use defaults
        let partial_json = r#"{"world": {"width": 16, "height": 16, "seed": 42}}"#;
        let env = ForgeWasmEnv::new(partial_json);
        assert_eq!(env.world.grid.width, 16);
        assert_eq!(env.world.grid.height, 16);
        assert!(!env.world.terminated);
        // Agents should still be created using defaults
        assert!(!env.world.agents.is_empty());
    }

    #[test]
    fn test_multi_step_episode() {
        let mut env = ForgeWasmEnv::new(&default_config_json());
        let reset_json = env.reset(Some(123));
        let reset_val: serde_json::Value = serde_json::from_str(&reset_json).unwrap();
        assert_eq!(reset_val["terminated"].as_bool(), Some(false));

        // Run 100 steps
        for i in 0..100 {
            let action = i % 5; // cycle through Noop, Up, Down, Left, Right
            let step_json = env.step(action);
            let step_val: serde_json::Value = serde_json::from_str(&step_json).unwrap();
            assert!(step_val.get("observations").is_some());
            assert!(step_val.get("rewards").is_some());
            // If terminated or truncated, stop early
            if step_val["terminated"].as_bool() == Some(true)
                || step_val["truncated"].as_bool() == Some(true)
            {
                break;
            }
        }

        // Verify the state advanced from tick 0
        let state_json = env.get_state_json();
        let state_val: serde_json::Value = serde_json::from_str(&state_json).unwrap();
        let tick = state_val["tick"].as_u64().unwrap();
        assert!(tick > 0, "tick should have advanced after stepping");
    }

    #[test]
    fn test_determinism_same_seed_same_actions() {
        let config_json =
            r#"{"world":{"width":16,"height":16,"seed":42},"agents":{"num_agents":1}}"#;
        let mut env_a = ForgeWasmEnv::new(config_json);
        let mut env_b = ForgeWasmEnv::new(config_json);

        env_a.reset(Some(42));
        env_b.reset(Some(42));

        for action in [0, 1, 2, 3, 4, 0, 1, 2] {
            let json_a = env_a.step(action);
            let json_b = env_b.step(action);
            assert_eq!(json_a, json_b, "Determinism broken at action {action}");
        }
    }

    #[test]
    fn test_reset_with_none_seed() {
        let mut env = ForgeWasmEnv::new(&default_config_json());
        let json = env.reset(None);
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("observations").is_some());
        assert_eq!(value["terminated"].as_bool(), Some(false));
    }

    #[test]
    fn test_get_state_json_tick_advances() {
        let mut env = ForgeWasmEnv::new(&default_config_json());
        env.reset(Some(1));

        let state_before = env.get_state_json();
        let before: serde_json::Value = serde_json::from_str(&state_before).unwrap();
        let tick_before = before["tick"].as_u64().unwrap();

        env.step(0); // Noop
        let state_after = env.get_state_json();
        let after: serde_json::Value = serde_json::from_str(&state_after).unwrap();
        let tick_after = after["tick"].as_u64().unwrap();

        assert!(tick_after > tick_before, "Tick should advance after step");
    }

    #[test]
    fn test_tiny_world() {
        let config_json = r#"{"world":{"width":8,"height":8,"seed":7},"agents":{"num_agents":1,"default_vision_radius":2}}"#;
        let mut env = ForgeWasmEnv::new(config_json);
        let json = env.reset(Some(7));
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("observations").is_some());

        // Should be able to step without crash
        for _ in 0..10 {
            let step_json = env.step(0);
            let _: serde_json::Value = serde_json::from_str(&step_json).unwrap();
        }
    }

    #[test]
    fn test_observation_space_matches_config() {
        let config_json =
            r#"{"world":{"width":32,"height":32},"agents":{"default_vision_radius":3}}"#;
        let env = ForgeWasmEnv::new(config_json);
        let json = env.observation_space_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        let grid_shape = value["grid_shape"].as_array().unwrap();
        // VR=3 -> view_side=7
        assert_eq!(grid_shape[0].as_u64().unwrap(), 7);
        assert_eq!(grid_shape[1].as_u64().unwrap(), 7);
        assert_eq!(grid_shape[2].as_u64().unwrap(), 7); // OBS_FEATURES_PER_TILE
    }

    #[test]
    fn test_action_space_n_matches() {
        let env = ForgeWasmEnv::new(&default_config_json());
        let json = env.action_space_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let n = value["n"].as_u64().unwrap();
        let expected = forge_types::Action::space_size(
            env.config.agents.comm_vocab_size,
            env.config.drone.enabled,
        ) as u64;
        assert_eq!(n, expected);
    }

    #[test]
    fn test_multiple_resets_produce_valid_output() {
        let mut env = ForgeWasmEnv::new(&default_config_json());
        for seed in [1, 2, 3, 42, 999] {
            let json = env.reset(Some(seed));
            let value: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(value.get("observations").is_some());
            assert_eq!(value["terminated"].as_bool(), Some(false));
        }
    }

    #[test]
    fn test_different_seeds_produce_different_states() {
        let mut env_a = ForgeWasmEnv::new(&default_config_json());
        let mut env_b = ForgeWasmEnv::new(&default_config_json());

        env_a.reset(Some(1));
        env_b.reset(Some(9999));

        // Step a few times
        for _ in 0..5 {
            env_a.step(1);
            env_b.step(1);
        }

        let state_a = env_a.get_state_json();
        let state_b = env_b.get_state_json();
        // Different seeds should produce different world states (positions, etc.)
        // This is probabilistic but extremely likely
        assert_ne!(state_a, state_b);
    }

    #[test]
    fn test_step_with_invalid_action_falls_back_to_noop() {
        let mut env = ForgeWasmEnv::new(&default_config_json());
        // Very large action index should fall back to Noop
        let json = env.step(99999);
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("observations").is_some());
    }
}
