//! Simulation reset logic.

use forge_types::observation::StepResult;
use tracing::{info, instrument};

use super::WorldState;

impl WorldState {
    /// Resets the simulation to initial state with a new seed.
    #[instrument(skip_all)]
    pub fn reset(&mut self, seed: Option<u64>) -> StepResult {
        let new_seed = seed.unwrap_or_else(|| self.rng.next_u64());

        info!(seed = new_seed, "resetting world");

        let mut new_config = (*self.config).clone();
        new_config.world.seed = new_seed;

        let new_state = WorldState::new(new_config).expect("validated config must be valid");

        self.tick = new_state.tick;
        self.grid = new_state.grid;
        self.agents = new_state.agents;
        self.objects = new_state.objects;
        self.resources = new_state.resources;
        self.tasks = new_state.tasks;
        self.recipe_book = new_state.recipe_book;
        self.day_phase = new_state.day_phase;
        self.rng = new_state.rng;
        self.config = new_state.config;
        self.terminated = false;
        self.truncated = false;
        self.last_task_rewards = None;
        self.topology = new_state.topology;
        self.crop_states = new_state.crop_states;
        self.soil_nodes = new_state.soil_nodes;
        self.agri_scratch.clear();
        self.step_actions.clear();
        self.validated_actions.clear();
        self.near_station.clear();
        self.crafting_object_map.clear();
        self.comm_messages.clear();
        self.push_scratch.clear();

        let mut result = StepResult::default();
        self.fill_step_result(&mut result);
        result
    }
}
