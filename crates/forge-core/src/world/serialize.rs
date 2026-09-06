//! Serialization and deserialization for WorldState snapshots.

use std::collections::HashMap;
use std::sync::Arc;

use forge_civ::grid_topology::GridTopologyKind;
use forge_civ::{HexTopology, SquareTopology};
use forge_types::agriculture::AgriScratch;
use forge_types::config::{ForgeConfig, GridType};
use forge_types::entity::{Agent, Object};
use forge_types::grid::Grid;
use forge_types::resource::{RecipeBook, ResourceNode};
use serde::Serialize;
use tracing::{error, instrument};

use super::WorldState;
use crate::physics::PhysicsScratch;
use crate::rng::ForgeRng;

/// Serializable subset of `WorldState` (excludes `Arc<ForgeConfig>`).
///
/// **Not** a full state snapshot. The following `WorldState` fields are
/// deliberately omitted and must be compared separately by anything that
/// needs whole-state coverage: `tasks`, `recipe_book`, `config`,
/// `last_task_rewards`, `physics_scratch`, `topology`, `crop_states`,
/// `soil_nodes`, `agri_scratch`, `step_actions`, `validated_actions`,
/// `near_station`, `crafting_object_map`, `comm_messages`, `push_scratch`.
#[derive(Serialize)]
struct SerializableWorldState<'a> {
    tick: u64,
    grid: &'a Grid,
    agents: &'a [Agent],
    objects: &'a [Object],
    resources: &'a [ResourceNode],
    day_phase: u8,
    rng_state: crate::rng::RngState,
    terminated: bool,
    truncated: bool,
}

/// Owned version for deserialization from bytes/JSON.
#[derive(serde::Deserialize)]
struct DeserializableWorldState {
    tick: u64,
    grid: Grid,
    agents: Vec<Agent>,
    objects: Vec<Object>,
    resources: Vec<ResourceNode>,
    day_phase: u8,
    rng_state: crate::rng::RngState,
    terminated: bool,
    truncated: bool,
}

impl WorldState {
    /// Serializes the world state to bytes for MCTS state snapshots,
    /// surfacing any serialization failure to the caller.
    ///
    /// This is the fallible twin of [`Self::to_bytes`]. Prefer it wherever an
    /// empty snapshot would be indistinguishable from a successful one — in
    /// particular when comparing two worlds byte-for-byte to verify the
    /// determinism invariant, where `to_bytes()`'s empty-on-error fallback
    /// would make the comparison pass vacuously.
    ///
    /// The serialized payload is the `SerializableWorldState` subset — tick,
    /// grid, agents, objects, resources, day phase, RNG state, terminated and
    /// truncated. It deliberately excludes `tasks`, `recipe_book`, `config`,
    /// `last_task_rewards`, `topology`, `crop_states`, `soil_nodes` and the
    /// per-tick scratch buffers, so callers that need whole-`WorldState`
    /// coverage must compare those fields separately.
    ///
    /// # Errors
    ///
    /// Returns [`forge_types::ForgeError::Serialization`] if bincode fails to
    /// encode the state.
    #[instrument(skip_all)]
    pub fn try_to_bytes(&self) -> forge_types::ForgeResult<Vec<u8>> {
        // Serialize the essential state (excluding config which is shared)
        let serializable = SerializableWorldState {
            tick: self.tick,
            grid: &self.grid,
            agents: &self.agents,
            objects: &self.objects,
            resources: &self.resources,
            day_phase: self.day_phase,
            rng_state: self.rng.save_state(),
            terminated: self.terminated,
            truncated: self.truncated,
        };
        bincode::serialize(&serializable)
            .map_err(|e| forge_types::ForgeError::Serialization(format!("bincode: {e}")))
    }

    /// Serializes the world state to bytes for MCTS state snapshots.
    ///
    /// Infallible wrapper around [`Self::try_to_bytes`]. On a serialization
    /// failure it logs at `error` level and returns an empty `Vec`, which is
    /// indistinguishable from a successful encoding of an empty payload — use
    /// [`Self::try_to_bytes`] when that ambiguity matters.
    #[instrument(skip_all)]
    pub fn to_bytes(&self) -> Vec<u8> {
        match self.try_to_bytes() {
            Ok(bytes) => bytes,
            Err(e) => {
                error!(
                    error = %e,
                    tick = self.tick,
                    agents = self.agents.len(),
                    "world state serialization failed; returning an empty snapshot"
                );
                Vec::new()
            }
        }
    }

    /// Deserializes a world state from bytes (bincode format).
    ///
    /// The config must be the same one used when the state was serialized.
    /// The RNG is reconstructed from the saved state to preserve determinism.
    #[instrument(skip_all)]
    pub fn from_bytes(bytes: &[u8], config: Arc<ForgeConfig>) -> forge_types::ForgeResult<Self> {
        let deserialized: DeserializableWorldState = bincode::deserialize(bytes)
            .map_err(|e| forge_types::ForgeError::Serialization(format!("bincode: {e}")))?;

        let mut physics_scratch = PhysicsScratch::default();
        physics_scratch.ensure_capacity(deserialized.agents.len());
        let topology = match config.world.grid_type {
            GridType::Square => GridTopologyKind::Square(SquareTopology),
            GridType::Hex => GridTopologyKind::Hex(HexTopology),
        };
        Ok(WorldState {
            tick: deserialized.tick,
            grid: deserialized.grid,
            agents: deserialized.agents,
            objects: deserialized.objects,
            resources: deserialized.resources,
            tasks: Vec::new(),
            recipe_book: RecipeBook::default(),
            day_phase: deserialized.day_phase,
            rng: ForgeRng::from_state(&deserialized.rng_state),
            config,
            terminated: deserialized.terminated,
            truncated: deserialized.truncated,
            last_task_rewards: None,
            physics_scratch,
            topology,
            crop_states: Vec::new(),
            soil_nodes: Vec::new(),
            agri_scratch: AgriScratch::default(),
            step_actions: Vec::new(),
            validated_actions: Vec::new(),
            near_station: Vec::new(),
            crafting_object_map: HashMap::new(),
            comm_messages: Vec::new(),
            push_scratch: Vec::new(),
        })
    }

    /// Serializes the world state to JSON for human-readable saves.
    #[instrument(skip_all)]
    pub fn to_json(&self) -> forge_types::ForgeResult<String> {
        let serializable = SerializableWorldState {
            tick: self.tick,
            grid: &self.grid,
            agents: &self.agents,
            objects: &self.objects,
            resources: &self.resources,
            day_phase: self.day_phase,
            rng_state: self.rng.save_state(),
            terminated: self.terminated,
            truncated: self.truncated,
        };
        serde_json::to_string(&serializable)
            .map_err(|e| forge_types::ForgeError::Serialization(format!("json: {e}")))
    }

    /// Deserializes a world state from JSON.
    ///
    /// The config must match what was used when the state was serialized.
    #[instrument(skip_all)]
    pub fn from_json(json: &str, config: Arc<ForgeConfig>) -> forge_types::ForgeResult<Self> {
        let deserialized: DeserializableWorldState = serde_json::from_str(json)
            .map_err(|e| forge_types::ForgeError::Serialization(format!("json: {e}")))?;

        let mut physics_scratch = PhysicsScratch::default();
        physics_scratch.ensure_capacity(deserialized.agents.len());
        let topology = match config.world.grid_type {
            GridType::Square => GridTopologyKind::Square(SquareTopology),
            GridType::Hex => GridTopologyKind::Hex(HexTopology),
        };
        Ok(WorldState {
            tick: deserialized.tick,
            grid: deserialized.grid,
            agents: deserialized.agents,
            objects: deserialized.objects,
            resources: deserialized.resources,
            tasks: Vec::new(),
            recipe_book: RecipeBook::default(),
            day_phase: deserialized.day_phase,
            rng: ForgeRng::from_state(&deserialized.rng_state),
            config,
            terminated: deserialized.terminated,
            truncated: deserialized.truncated,
            last_task_rewards: None,
            physics_scratch,
            topology,
            crop_states: Vec::new(),
            soil_nodes: Vec::new(),
            agri_scratch: AgriScratch::default(),
            step_actions: Vec::new(),
            validated_actions: Vec::new(),
            near_station: Vec::new(),
            crafting_object_map: HashMap::new(),
            comm_messages: Vec::new(),
            push_scratch: Vec::new(),
        })
    }
}
