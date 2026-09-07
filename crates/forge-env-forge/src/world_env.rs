//! [`WorldEnv`] — single-agent [`Env`] wrapper around FORGE's
//! [`forge_core::WorldState`].
//!
//! Mirrors the surface area of [`forge_python::ForgeEnv`] (single-agent
//! Gymnasium binding) in native Rust so downstream crates can drive
//! `WorldState` through the generic [`Env`] trait.

use std::borrow::Cow;

use forge_env::{ActionSpec, Env, ObsSpec, StepOutput};
use forge_types::action::Action;
use forge_types::config::{ForgeConfig, GridType};
use forge_types::observation::{Observation, StepInfo, StepResult};
use tracing::{instrument, warn};

use crate::error::ForgeEnvError;

/// Bound the observation values pass through unmodified — bounds here
/// describe the *typed* range of fields after f32 casting, not the
/// post-normalisation range. Consumers needing tight normalisation
/// should use the [`crate::FlatForgeEnv`] adapter with `normalize=true`.
const OBS_BOUND_LOW: f32 = -1.0e6;
const OBS_BOUND_HIGH: f32 = 1.0e6;

/// Single-agent env wrapper around [`forge_core::WorldState`].
///
/// `step` takes a typed [`Action`]; for discrete-index input, use
/// [`Self::step_from_discrete`] (used internally by [`crate::FlatForgeEnv`]).
pub struct WorldEnv {
    state: forge_core::WorldState,
    config: ForgeConfig,
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    /// Reusable buffer for `WorldState::step_into` — keeps the hot path
    /// allocation-free after warmup. See the zero-allocation contract
    /// documented in [`forge_types::observation::StepResult`].
    step_result: StepResult,
}

impl WorldEnv {
    /// Construct from a [`ForgeConfig`].
    ///
    /// Returns [`ForgeEnvError::WorldInit`] if `WorldState::new` rejects
    /// the config, or [`ForgeEnvError::NoAgents`] if the resulting world
    /// has zero agents (single-agent semantics require >= 1).
    pub fn new(config: ForgeConfig) -> Result<Self, ForgeEnvError> {
        let state = forge_core::WorldState::new(config.clone())
            .map_err(|e| ForgeEnvError::WorldInit(e.to_string()))?;
        if state.agents.is_empty() {
            return Err(ForgeEnvError::NoAgents);
        }

        let action_space_n = Action::space_size_full(
            config.agents.comm_vocab_size,
            config.drone.enabled,
            config.agri.enabled && config.drone.enabled,
            config.world.grid_type == GridType::Hex,
        );
        let action_spec = ActionSpec::discrete(action_space_n);
        let obs_spec = build_obs_spec(&config);

        Ok(Self {
            state,
            config,
            obs_spec,
            action_spec,
            step_result: StepResult::default(),
        })
    }

    /// Access the underlying config (read-only).
    pub fn config(&self) -> &ForgeConfig {
        &self.config
    }

    /// Step using a discrete action index, mirroring `forge-python::ForgeEnv`.
    ///
    /// Decodes via [`Action::from_discrete_full`] with the same flags
    /// the constructor used to size the action space.
    #[instrument(skip(self), fields(action_id))]
    pub fn step_from_discrete(
        &mut self,
        action_id: u32,
    ) -> Result<StepOutput<Observation, StepInfo>, ForgeEnvError> {
        let action = self.decode_action(action_id)?;
        self.step(action)
    }

    pub(crate) fn decode_action(&self, action_id: u32) -> Result<Action, ForgeEnvError> {
        let hex_enabled = self.config.world.grid_type == GridType::Hex;
        Action::from_discrete_full(
            action_id,
            self.config.agents.comm_vocab_size,
            self.config.drone.enabled,
            self.config.agri.enabled && self.config.drone.enabled,
            hex_enabled,
        )
        .ok_or_else(|| ForgeEnvError::InvalidDiscreteAction {
            action_id,
            space_n: self.action_spec.discrete_n().unwrap_or(0),
        })
    }
}

/// Build the [`ObsSpec`] describing the single-agent `Observation`.
/// Shape uses the per-tile feature count plus scalar fields so consumers
/// know the upper bound on the flat dim without instantiating a flattener.
fn build_obs_spec(config: &ForgeConfig) -> ObsSpec {
    let vr = config.agents.default_vision_radius as usize;
    let view_side = 2 * vr + 1;
    let grid_elems = view_side * view_side * (crate::config::TILE_FEATURE_COUNT);
    let cap = config.agents.default_carry_capacity as usize;
    let n_pred = config.task.max_predicates as usize;
    let scalars = 2 /* vitals */ + 2 /* position */ + 1 /* day_phase */;
    let total = grid_elems + cap * (crate::config::INVENTORY_FEATURES_PER_SLOT) + scalars + n_pred;
    ObsSpec {
        shape: vec![total],
        low: OBS_BOUND_LOW,
        high: OBS_BOUND_HIGH,
        dtype: forge_env::DType::F32,
        name: "forge_observation".to_string(),
    }
}

impl Env for WorldEnv {
    type Obs = Observation;
    type Action = Action;
    type Info = StepInfo;
    type Error = ForgeEnvError;

    #[instrument(skip_all, fields(env = "forge-world"))]
    fn reset_into(&mut self, seed: Option<u64>, out: &mut Self::Obs) -> Result<(), Self::Error> {
        let result = self.state.reset(seed);
        *out = result.observations.into_iter().next().unwrap_or_else(|| {
            warn!("WorldState reset produced no observations; using default");
            Observation::default()
        });
        Ok(())
    }

    #[instrument(skip_all, fields(env = "forge-world"))]
    fn step_into(
        &mut self,
        action: Self::Action,
        out: &mut StepOutput<Self::Obs, Self::Info>,
    ) -> Result<(), Self::Error> {
        let actions = [action];
        // Reuse the cached StepResult buffer to keep the hot path
        // allocation-free after warmup (see zero-allocation contract in
        // CLAUDE.md and `crates/forge-bench/src/bin/allocation_audit.rs`).
        self.state.step_into(&actions, &mut self.step_result);
        if let Some(first_obs) = self.step_result.observations.first() {
            // `Observation::copy_from` reuses Vec capacity. Derived
            // `clone_from` is `*self = src.clone()` and was allocating
            // ~4 heap blocks per step (CI `EnvTrait_WorldEnv_Move_Up@n=1`).
            out.obs.copy_from(first_obs);
        } else {
            warn!("WorldState step produced no observations; using default");
            out.obs = Observation::default();
        }
        out.reward = self.step_result.rewards.first().copied().unwrap_or(0.0);
        out.terminated = self.step_result.terminated;
        out.truncated = self.step_result.truncated;
        out.info.copy_from(&self.step_result.info);
        Ok(())
    }

    fn obs_spec(&self) -> &ObsSpec {
        &self.obs_spec
    }

    fn action_spec(&self) -> &ActionSpec {
        &self.action_spec
    }

    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "forge-world-{}x{}",
            self.config.world.width, self.config.world.height
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ForgeConfig {
        let mut cfg = ForgeConfig::default();
        cfg.world.width = 16;
        cfg.world.height = 16;
        cfg
    }

    #[test]
    fn new_constructs_with_default_config() {
        let env = WorldEnv::new(tiny_config()).expect("construct");
        assert!(env.action_spec.discrete_n().unwrap() > 0);
        assert!(!env.obs_spec.shape.is_empty());
    }

    #[test]
    fn reset_and_step_produce_observation() {
        let mut env = WorldEnv::new(tiny_config()).unwrap();
        let obs = env.reset(Some(42)).unwrap();
        assert!(!obs.grid_view.is_empty());
        let s = env.step(Action::Noop).unwrap();
        assert!(!s.obs.grid_view.is_empty());
    }

    #[test]
    fn invalid_discrete_action_returns_error() {
        let mut env = WorldEnv::new(tiny_config()).unwrap();
        let _ = env.reset(None).unwrap();
        let n = env.action_spec.discrete_n().unwrap();
        let err = env.step_from_discrete(n + 100).unwrap_err();
        assert!(matches!(err, ForgeEnvError::InvalidDiscreteAction { .. }));
    }

    #[test]
    fn action_space_size_matches_space_size_full() {
        let cfg = tiny_config();
        let env = WorldEnv::new(cfg.clone()).unwrap();
        let expected = Action::space_size_full(
            cfg.agents.comm_vocab_size,
            cfg.drone.enabled,
            cfg.agri.enabled && cfg.drone.enabled,
            cfg.world.grid_type == GridType::Hex,
        );
        assert_eq!(env.action_spec.discrete_n().unwrap(), expected);
    }

    #[test]
    fn name_is_dynamic_per_config() {
        let env = WorldEnv::new(tiny_config()).unwrap();
        let n = env.name();
        assert!(n.contains("forge-world"));
    }

    #[test]
    fn config_accessor_returns_underlying_config() {
        let env = WorldEnv::new(tiny_config()).unwrap();
        assert_eq!(env.config().world.width, 16);
    }

    #[test]
    fn obs_spec_and_action_spec_accessors_match_construction() {
        let env = WorldEnv::new(tiny_config()).unwrap();
        // obs_spec().shape was set in build_obs_spec
        assert!(!env.obs_spec().shape.is_empty());
        // action_spec is always Discrete
        assert!(env.action_spec().discrete_n().is_some());
    }

    #[test]
    fn step_into_reuses_step_result_capacity_after_warmup() {
        // Drives the zero-allocation contract: after warmup, repeated
        // `step_into` calls must not grow the cached `StepResult` vectors.
        // This is the regression guard for PR #53 review feedback.
        let mut env = WorldEnv::new(tiny_config()).unwrap();
        let _ = env.reset(Some(7)).unwrap();
        let mut out: StepOutput<Observation, StepInfo> = StepOutput::default();
        // Warm-up: let internal buffers reach steady-state capacity.
        for _ in 0..16 {
            env.step_into(Action::Noop, &mut out).unwrap();
        }
        let obs_cap = env.step_result.observations.capacity();
        let rew_cap = env.step_result.rewards.capacity();
        let agents_alive_cap = env.step_result.info.agents_alive.capacity();
        let out_grid_cap = out.obs.grid_view.capacity();
        for _ in 0..64 {
            env.step_into(Action::Noop, &mut out).unwrap();
            assert_eq!(env.step_result.observations.capacity(), obs_cap);
            assert_eq!(env.step_result.rewards.capacity(), rew_cap);
            assert_eq!(
                env.step_result.info.agents_alive.capacity(),
                agents_alive_cap
            );
            assert_eq!(out.obs.grid_view.capacity(), out_grid_cap);
        }
    }
}
