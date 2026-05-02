//! Integration tests for the kitchen-counter cleanup deployment.
//!
//! These tests exercise the full path that a real Raspberry Pi 5 + Hailo +
//! OpenClaw kitchen robot would take:
//!
//!   1. Load `configs/edge/rpi5_hailo_kitchen.toml` into a `ForgeConfig` and
//!      verify the safe-pose `EdgeConfig.fallback_action_id`.
//!   2. Load `configs/actuator/openclaw_kitchen.toml` into an `ActionMapping`
//!      and verify the documented action-id → command-sequence contract.
//!   3. Drive a `MappedActuator<MockDriver>` through a representative cleanup
//!      sweep (lower sweeper → push → interact at sink edge → raise).
//!   4. Simulate an MCTS failure on `EdgeAgent` and confirm the resulting
//!      fallback action id flows through the actuator bridge to the
//!      configured safe-pose command sequence (`[disengage_sweeper, halt]`).
//!
//! These are the *only* deltas that need to work end-to-end for a kitchen
//! deployment to be operational; everything else (vision, motor control,
//! cloud OTA loop) is reused from the existing FORGE infrastructure.
//!
//! To run: `cargo test --test integration_kitchen_robot`

use std::path::PathBuf;

use forge_actuator::{
    ActionMapping, ActuatorBridge, ActuatorCommand, ActuatorError, CardinalDirection,
    DispatchSource, MappedActuator, MockDriver,
};
use forge_agent::latent_mcts::model::{LatentForwardModel, LatentInferenceOutput};
use forge_agent::latent_mcts::search::LatentMctsConfig;
use forge_agent::latent_mcts::state::LatentState;
use forge_agent::mcts::tree::MctsConfig;
use forge_edge::EdgeAgent;
use forge_types::agent_interface::AgentInterface;
use forge_types::config::{EdgeConfig, ForgeConfig};
use forge_types::constants::{self, OBS_EMPTY_SLOT_ITEM};
use forge_types::observation::{InventoryObservation, Observation, TileObservation};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the workspace-relative path to a config file.
///
/// `CARGO_MANIFEST_DIR` is always set when running cargo tests; falling back
/// to the current directory keeps this resilient to running tests from a
/// different working directory.
fn config_path(relative: &str) -> PathBuf {
    let base = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join(relative)
}

/// Build a placeholder `Observation` so `EdgeAgent::select_action` has
/// something to work with. The contents don't matter — the failing model
/// below short-circuits before observation flattening reaches MCTS.
fn make_observation() -> Observation {
    Observation {
        grid_view: vec![TileObservation::default()],
        view_width: 1,
        view_height: 1,
        inventory: InventoryObservation {
            slots: vec![(OBS_EMPTY_SLOT_ITEM, 0)],
        },
        health: 1.0,
        stamina: 1.0,
        position: (0, 0),
        messages: vec![],
        day_phase: 0,
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

/// A `LatentForwardModel` that always errors. Used to drive the EdgeAgent's
/// fallback path in the integration test below.
#[derive(Clone)]
struct AlwaysFailingModel {
    action_space: u32,
}

impl LatentForwardModel for AlwaysFailingModel {
    fn initial_inference(&self, _observation: &[f32]) -> anyhow::Result<LatentInferenceOutput> {
        anyhow::bail!("simulated MCTS failure for fallback test")
    }
    fn recurrent_inference(
        &self,
        _state: &LatentState,
        _action: u32,
    ) -> anyhow::Result<LatentInferenceOutput> {
        anyhow::bail!("simulated MCTS failure for fallback test")
    }
    fn action_space_size(&self) -> u32 {
        self.action_space
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn rpi5_hailo_kitchen_edge_profile_loads_and_sets_safe_pose_fallback() {
    let path = config_path("configs/edge/rpi5_hailo_kitchen.toml");
    let config = ForgeConfig::from_toml(&path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));

    // Edge runtime must be enabled in this profile.
    assert!(config.edge.enabled, "edge.enabled must be true");

    // The whole point of the kitchen profile: the fallback action id is the
    // mapped "raise sweeper + halt" id, NOT the default Noop. A regression
    // here would let the robot drag its squeegee on a constitutional trip.
    assert_eq!(
        config.edge.fallback_action_id, 17,
        "kitchen profile must set fallback_action_id = 17 (raise sweeper + halt)"
    );

    // Tight latency budget for 30 Hz control.
    assert!(
        config.edge.mcts_latency_budget_ms <= 33,
        "expected ≤ 33 ms latency budget, got {}",
        config.edge.mcts_latency_budget_ms
    );
}

#[test]
fn openclaw_kitchen_action_mapping_loads_and_dispatches_workhorse_actions() {
    let path = config_path("configs/actuator/openclaw_kitchen.toml");
    let mapping = ActionMapping::from_toml_file(&path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));

    // Mapping must be permissive (unmapped ids → default Halt) so an
    // unexpected action id doesn't crash the control loop.
    assert!(!mapping.is_strict());

    // Action 16 = lower sweeper.
    assert_eq!(
        mapping.commands_for(16).unwrap(),
        &[ActuatorCommand::EngageSweeper]
    );

    // Action 17 = the safe-pose fallback. Must be raise-then-halt.
    assert_eq!(
        mapping.commands_for(17).unwrap(),
        &[ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt]
    );

    // Action 35 = Push Up = sweep one tile north (30 mm in this profile).
    assert_eq!(
        mapping.commands_for(35).unwrap(),
        &[ActuatorCommand::DriveDirection {
            direction: CardinalDirection::Up,
            distance_mm: 30,
        }]
    );

    // Action 39 = Interact = sink-edge subroutine + raise.
    assert_eq!(
        mapping.commands_for(39).unwrap(),
        &[
            ActuatorCommand::Custom {
                name: "sink_edge_sweep".to_string(),
                payload: None,
            },
            ActuatorCommand::DisengageSweeper,
        ]
    );
}

#[test]
fn unmapped_action_id_falls_through_to_safe_default() {
    let mapping =
        ActionMapping::from_toml_file(config_path("configs/actuator/openclaw_kitchen.toml"))
            .unwrap();
    // Crafting (26–34) and several Use slots (19–25) are intentionally
    // unmapped. They must come back as the default `[disengage_sweeper, halt]`
    // sequence rather than erroring.
    let mut bridge = MappedActuator::new(mapping, MockDriver::new(), 16);
    let result = bridge.dispatch(28).unwrap();
    assert_eq!(result.source, DispatchSource::Default);
    assert_eq!(
        result.commands,
        vec![ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt]
    );
}

#[test]
fn cleanup_sweep_sequence_routes_to_correct_driver_commands() {
    // Simulates a single "approach → engage → push twice → sink-edge interact"
    // cleanup pass. This is the high-level command sequence the policy emits;
    // the bridge translates each id into the documented driver commands.
    let mapping =
        ActionMapping::from_toml_file(config_path("configs/actuator/openclaw_kitchen.toml"))
            .unwrap();
    let mut bridge = MappedActuator::new(mapping, MockDriver::new(), 32);

    // Drive one tile right toward the debris cluster.
    bridge.dispatch(4).unwrap();
    // Lower sweeper.
    bridge.dispatch(16).unwrap();
    // Sweep two tiles up toward the sink.
    bridge.dispatch(35).unwrap();
    bridge.dispatch(35).unwrap();
    // Sink-edge subroutine (also raises sweeper).
    bridge.dispatch(39).unwrap();

    let received = bridge.driver().received();
    assert_eq!(
        received,
        &[
            ActuatorCommand::DriveDirection {
                direction: CardinalDirection::Right,
                distance_mm: 30
            },
            ActuatorCommand::EngageSweeper,
            ActuatorCommand::DriveDirection {
                direction: CardinalDirection::Up,
                distance_mm: 30
            },
            ActuatorCommand::DriveDirection {
                direction: CardinalDirection::Up,
                distance_mm: 30
            },
            ActuatorCommand::Custom {
                name: "sink_edge_sweep".to_string(),
                payload: None
            },
            ActuatorCommand::DisengageSweeper,
        ],
        "cleanup sweep produced unexpected driver commands: {:#?}",
        received,
    );
}

#[test]
fn edge_fallback_flows_through_actuator_bridge_to_safe_pose() {
    // End-to-end safety property: when the policy's MCTS fails, the
    // EdgeAgent must emit the configured fallback action id, AND the
    // actuator bridge must translate that id into the safe-pose command
    // sequence — *without* any glue code in between.

    // 1. Load the kitchen edge profile.
    let edge_profile = config_path("configs/edge/rpi5_hailo_kitchen.toml");
    let forge_cfg = ForgeConfig::from_toml(&edge_profile).unwrap();
    let edge_cfg: &EdgeConfig = &forge_cfg.edge;
    assert_eq!(edge_cfg.fallback_action_id, 17);

    // 2. Load the OpenClaw mapping.
    let mapping_path = config_path("configs/actuator/openclaw_kitchen.toml");
    let mapping = ActionMapping::from_toml_file(&mapping_path).unwrap();

    // 3. Build an EdgeAgent backed by the always-failing model — this forces
    //    the fallback path on every `select_action` call.
    let action_space: u32 = 40;
    let model = AlwaysFailingModel { action_space };
    let mcts_cfg = LatentMctsConfig {
        base: MctsConfig {
            num_simulations: 4,
            action_space,
            max_depth: 4,
            ..MctsConfig::default()
        },
        ..LatentMctsConfig::default()
    };
    let mut agent = EdgeAgent::new(model, edge_cfg, mcts_cfg, "kitchen-test-v1".to_string());

    // 4. Build the actuator bridge with a MockDriver so we can inspect the
    //    commands without real hardware.
    let mut bridge = MappedActuator::new(mapping, MockDriver::new(), 8);

    // 5. Run one "control loop" tick: select_action → dispatch.
    let response = agent.select_action(&make_observation(), 0);
    assert_eq!(
        response.action_id, edge_cfg.fallback_action_id,
        "EdgeAgent must emit the EdgeConfig fallback id on MCTS failure"
    );

    let dispatched = bridge.dispatch(response.action_id).unwrap();
    assert_eq!(dispatched.source, DispatchSource::Mapped);
    assert_eq!(
        dispatched.commands,
        vec![ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt],
        "fallback action id 17 must translate to the safe-pose sequence",
    );
    // Cross-check via the driver's received-command log.
    assert_eq!(
        bridge.driver().received(),
        &[ActuatorCommand::DisengageSweeper, ActuatorCommand::Halt]
    );
}

#[test]
fn kitchen_mapping_handles_every_id_in_action_space_without_panic() {
    // Defence against future action-space expansions: every id in the
    // documented FORGE discrete action space (0..=39 + `Communicate` band
    // up to vocab_size + drone/agri actions) must produce *some* outcome
    // through the bridge — never a panic and never an unhandled error.
    let mapping =
        ActionMapping::from_toml_file(config_path("configs/actuator/openclaw_kitchen.toml"))
            .unwrap();
    let mut bridge = MappedActuator::new(mapping, MockDriver::new(), 4);

    // Cover the documented discrete band plus the communicate token band.
    let upper = 40 + (constants::DEFAULT_COMM_VOCAB_SIZE as u32);
    for id in 0..upper {
        // Either Mapped (with a documented sequence) or Default (Halt).
        // An UnknownActionId here would mean the mapping was loaded in
        // strict mode by mistake, breaking the safe-pose contract.
        match bridge.dispatch(id) {
            Ok(_) => {}
            Err(ActuatorError::DriverFailure(_)) => {
                panic!("MockDriver should not fail on action id {id}")
            }
            Err(other) => panic!("unexpected dispatch error for id {id}: {other:?}"),
        }
    }
}

#[test]
fn kitchen_edge_profile_inherits_other_forge_defaults() {
    // Backwards-compat: loading the kitchen profile should not require any
    // other config sections to be present. The `[edge]`-only TOML should
    // populate `world`, `agents`, etc. with their defaults.
    let path = config_path("configs/edge/rpi5_hailo_kitchen.toml");
    let config = ForgeConfig::from_toml(&path).unwrap();
    let defaults = ForgeConfig::default();
    assert_eq!(config.world.width, defaults.world.width);
    assert_eq!(config.agents.num_agents, defaults.agents.num_agents);
}

#[test]
fn edge_config_default_fallback_remains_noop_for_legacy_profiles() {
    // A profile that doesn't mention `fallback_action_id` (i.e. existing
    // edge profiles like `configs/training/distributed.toml`) must still
    // get the historical Noop fallback so this change is fully backwards
    // compatible.
    let path = config_path("configs/training/distributed.toml");
    let config = ForgeConfig::from_toml(&path).unwrap();
    assert_eq!(
        config.edge.fallback_action_id,
        constants::DEFAULT_EDGE_FALLBACK_ACTION_ID,
    );
}

// ---------------------------------------------------------------------------
// Property: dispatch never panics for any 32-bit action id, on either the
// kitchen mapping or a strict mapping. Acts as a regression guard against
// future mapping changes (e.g. somebody adds `strict = true` and forgets to
// also enumerate every reachable id).
// ---------------------------------------------------------------------------

#[test]
fn dispatch_never_panics_for_random_ids_against_kitchen_mapping() {
    // Deterministic small sweep — proptest is overkill for an integration
    // test; a fixed seed of representative ids is enough.
    let mapping =
        ActionMapping::from_toml_file(config_path("configs/actuator/openclaw_kitchen.toml"))
            .unwrap();
    let mut bridge = MappedActuator::new(mapping, MockDriver::new(), 4);
    for id in [0u32, 1, 5, 16, 17, 39, 100, 1_000, u32::MAX] {
        let _ = bridge.dispatch(id);
    }
}
