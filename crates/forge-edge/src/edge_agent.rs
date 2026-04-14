//! Composite edge agent combining adaptive MCTS, telemetry, and fallback.
//!
//! Implements [`AgentInterface`] from forge-types, making it directly
//! compatible with `EvalHarness`, `BatchRunner`, and all FORGE evaluation
//! infrastructure.

use std::time::Instant;

use forge_agent::latent_mcts::model::LatentForwardModel;
use forge_agent::latent_mcts::search::LatentMctsConfig;
use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::config::EdgeConfig;
use forge_types::observation::Observation;
use tracing::{instrument, warn};

use crate::adaptive_mcts::AdaptiveMctsSearch;
use crate::telemetry::TelemetryCollector;

/// Flattens an [`Observation`] into a contiguous `f32` vector.
///
/// The flattened layout is defined by this Rust implementation in the
/// following order: grid tiles (7 features each), inventory slots (2 values
/// each), scalar fields (health, stamina, position x/y, day_phase), and
/// drone fields (altitude, battery, morphology, heading).
///
/// This ordering is an internal contract for the edge agent model input and
/// is not guaranteed to match Python `ForgeEnv` flattening unless the
/// trainer/exporter uses the same layout explicitly.
fn flatten_observation(obs: &Observation) -> Vec<f32> {
    let mut flat = Vec::new();

    // Grid tiles
    for tile in &obs.grid_view {
        flat.push(tile.terrain as f32);
        flat.push(if tile.has_agent { 1.0 } else { 0.0 });
        flat.push(if tile.has_object { 1.0 } else { 0.0 });
        flat.push(if tile.has_resource { 1.0 } else { 0.0 });
        flat.push(tile.elevation as f32);
        flat.push(tile.object_type as f32);
        flat.push(tile.resource_type as f32);
    }

    // Inventory
    for &(item_type, count) in &obs.inventory.slots {
        flat.push(item_type as f32);
        flat.push(count as f32);
    }

    // Scalars
    flat.push(obs.health);
    flat.push(obs.stamina);
    flat.push(obs.position.0 as f32);
    flat.push(obs.position.1 as f32);

    // Messages (fixed size: DEFAULT_COMM_BUFFER_SIZE)
    let max_messages = forge_types::constants::DEFAULT_COMM_BUFFER_SIZE as usize;
    for i in 0..max_messages {
        flat.push(obs.messages.get(i).copied().unwrap_or(0) as f32);
    }

    // Day phase
    flat.push(obs.day_phase as f32);

    // Task progress (fixed size: DEFAULT_MAX_PREDICATES)
    let max_predicates = forge_types::constants::DEFAULT_MAX_PREDICATES as usize;
    for i in 0..max_predicates {
        flat.push(obs.task_progress.get(i).copied().unwrap_or(0.0));
    }

    // Drone fields
    flat.push(obs.altitude as f32);
    flat.push(obs.battery);
    flat.push(obs.morphology as f32);
    flat.push(obs.heading as f32);

    flat
}

/// Composite edge agent combining adaptive MCTS, telemetry, and fallback.
///
/// Implements [`AgentInterface`] from forge-types, making it directly
/// compatible with `EvalHarness`, `BatchRunner`, and all FORGE evaluation
/// infrastructure.
///
/// On MCTS failure, the agent falls back to a configurable fallback action
/// (default: Noop = 0) and logs a warning.
pub struct EdgeAgent<M: LatentForwardModel + Clone> {
    /// Adaptive MCTS search engine.
    mcts: AdaptiveMctsSearch<M>,
    /// Telemetry collector for store-and-forward replay upload.
    telemetry: TelemetryCollector,
    /// Model version string for metadata.
    model_version: String,
    /// Agent display name.
    name: String,
    /// Fallback action when MCTS fails (0 = Noop).
    fallback_action: u32,
}

impl<M: LatentForwardModel + Clone> EdgeAgent<M> {
    /// Creates a new edge agent.
    ///
    /// # Arguments
    ///
    /// * `model` - The latent forward model for MCTS inference.
    /// * `edge_config` - Edge runtime configuration.
    /// * `mcts_config` - Base MCTS search configuration.
    /// * `model_version` - Version string for metadata attribution.
    pub fn new(
        model: M,
        edge_config: &EdgeConfig,
        mcts_config: LatentMctsConfig,
        model_version: String,
    ) -> Self {
        let mcts = AdaptiveMctsSearch::new(model, edge_config, mcts_config);
        let telemetry = TelemetryCollector::new(edge_config);
        Self {
            mcts,
            telemetry,
            model_version,
            name: "EdgeAgent".to_string(),
            fallback_action: 0,
        }
    }

    /// Returns a reference to the telemetry collector.
    pub fn telemetry(&self) -> &TelemetryCollector {
        &self.telemetry
    }

    /// Returns a mutable reference to the telemetry collector.
    pub fn telemetry_mut(&mut self) -> &mut TelemetryCollector {
        &mut self.telemetry
    }

    /// Returns the current model version string.
    pub fn model_version(&self) -> &str {
        &self.model_version
    }
}

impl<M: LatentForwardModel + Clone> AgentInterface for EdgeAgent<M> {
    #[instrument(skip_all, fields(agent_idx))]
    fn select_action(&mut self, obs: &Observation, _agent_idx: usize) -> AgentResponse {
        let started = Instant::now();
        let flat_obs = flatten_observation(obs);
        match self.mcts.search(&flat_obs) {
            Ok((action, _metrics)) => AgentResponse::with_timing(action, started),
            Err(e) => {
                warn!(
                    error = %e,
                    fallback = self.fallback_action,
                    "MCTS search failed, using fallback action"
                );
                AgentResponse::with_timing(self.fallback_action, started)
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::rl(&self.model_version)
    }

    fn reset(&mut self) {
        self.mcts.reset_estimator();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_agent::latent_mcts::model::{
        LatentForwardModel, LatentInferenceOutput, StubLatentModel,
    };
    use forge_agent::latent_mcts::state::LatentState;
    use forge_agent::mcts::tree::MctsConfig;
    use forge_types::constants;
    use forge_types::observation::{InventoryObservation, TileObservation};

    fn make_test_observation() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation {
                slots: vec![(constants::OBS_EMPTY_SLOT_ITEM, 0)],
            },
            health: 1.0,
            stamina: 1.0,
            position: (5, 5),
            messages: vec![],
            day_phase: 1,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    fn make_edge_config() -> EdgeConfig {
        EdgeConfig {
            mcts_latency_budget_ms: 100,
            mcts_min_simulations: 4,
            mcts_max_simulations: 50,
            latency_ema_alpha: 0.3,
            telemetry_buffer_bytes: 1_048_576,
            ..EdgeConfig::default()
        }
    }

    fn make_mcts_config(action_space: u32) -> LatentMctsConfig {
        LatentMctsConfig {
            base: MctsConfig {
                num_simulations: 10,
                action_space,
                max_depth: 10,
                ..MctsConfig::default()
            },
            ..LatentMctsConfig::default()
        }
    }

    fn make_edge_agent() -> EdgeAgent<StubLatentModel> {
        let model = StubLatentModel::new(8, 64);
        let edge_cfg = make_edge_config();
        let mcts_cfg = make_mcts_config(8);
        EdgeAgent::new(model, &edge_cfg, mcts_cfg, "test-v1".to_string())
    }

    #[test]
    fn test_select_action_returns_valid_response() {
        let mut agent = make_edge_agent();
        let obs = make_test_observation();
        let resp = agent.select_action(&obs, 0);
        assert!(resp.action_id < 8);
        assert!(resp.decision_time_ms < 10_000); // sanity check
    }

    #[test]
    fn test_fallback_on_mcts_error() {
        /// A model that always fails on inference.
        #[derive(Clone)]
        struct FailingModel;

        impl LatentForwardModel for FailingModel {
            fn initial_inference(
                &self,
                _observation: &[f32],
            ) -> anyhow::Result<LatentInferenceOutput> {
                anyhow::bail!("deliberate test failure")
            }
            fn recurrent_inference(
                &self,
                _state: &LatentState,
                _action: u32,
            ) -> anyhow::Result<LatentInferenceOutput> {
                anyhow::bail!("deliberate test failure")
            }
            fn action_space_size(&self) -> u32 {
                4
            }
        }

        let edge_cfg = make_edge_config();
        let mcts_cfg = make_mcts_config(4);
        let mut agent = EdgeAgent::new(FailingModel, &edge_cfg, mcts_cfg, "fail-v1".to_string());

        let obs = make_test_observation();
        let resp = agent.select_action(&obs, 0);
        // Should fall back to Noop (action 0)
        assert_eq!(resp.action_id, 0);
    }

    #[test]
    fn test_metadata_returns_rl_type() {
        let agent = make_edge_agent();
        let meta = agent.metadata();
        assert_eq!(meta.agent_type, "rl");
        assert_eq!(meta.model_name, "test-v1");
    }

    #[test]
    fn test_name() {
        let agent = make_edge_agent();
        assert_eq!(agent.name(), "EdgeAgent");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut agent = make_edge_agent();
        let obs = make_test_observation();

        // Generate some state
        agent.select_action(&obs, 0);
        assert!(agent.mcts.estimator().total_samples() > 0);

        // Reset
        agent.reset();
        assert_eq!(agent.mcts.estimator().total_samples(), 0);
    }

    #[test]
    fn test_telemetry_accessible() {
        let agent = make_edge_agent();
        let snap = agent.telemetry().snapshot();
        assert_eq!(snap.pending_replays, 0);
        assert_eq!(snap.total_replays_recorded, 0);
    }

    #[test]
    fn test_telemetry_mut_accessible() {
        let mut agent = make_edge_agent();
        agent.telemetry_mut().clear();
        assert_eq!(agent.telemetry().pending_count(), 0);
    }

    #[test]
    fn test_model_version() {
        let agent = make_edge_agent();
        assert_eq!(agent.model_version(), "test-v1");
    }

    #[test]
    fn test_flatten_observation() {
        let obs = make_test_observation();
        let flat = flatten_observation(&obs);
        // Should contain at least grid + inventory + scalars + day_phase + drone fields
        assert!(!flat.is_empty());
        // Health should be somewhere in the flat vec
        assert!(flat.contains(&1.0));
    }

    #[test]
    fn test_agent_interface_trait_object() {
        // Verify EdgeAgent can be used as a trait object
        let agent = make_edge_agent();
        let boxed: Box<dyn AgentInterface> = Box::new(agent);
        assert_eq!(boxed.name(), "EdgeAgent");
    }
}
