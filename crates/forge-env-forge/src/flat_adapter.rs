//! [`FlatForgeEnv`] — wraps [`crate::WorldEnv`] to satisfy
//! [`forge_env::FlatObsEnv`].
//!
//! The flattening layout is deterministic and matches
//! [`crate::FlatObsConfig::flat_dim`]:
//!
//! 1. grid view (row-major over tiles, then per-tile feature order)
//! 2. inventory slots (`item_type`, `count` per slot)
//! 3. vitals (`health`, `stamina`)
//! 4. position (`x`, `y`)
//! 5. day phase
//! 6. task progress vector
//!
//! Each toggle in [`crate::FlatObsConfig`] elides the corresponding
//! block when `false`. Block boundaries are stable across calls so
//! downstream consumers (NN inputs, hash-based schema_id computation)
//! can rely on them.

use std::borrow::Cow;

use forge_env::{ActionSpec, Env, FlatObsEnv, ObsSpec, StepOutput};
use forge_types::config::ForgeConfig;
use forge_types::observation::{Observation, StepInfo};
use tracing::instrument;

use crate::config::{
    task_predicate_count, FlatObsConfig, INVENTORY_FEATURES_PER_SLOT, TILE_FEATURE_COUNT,
};
use crate::error::ForgeEnvError;
use crate::world_env::WorldEnv;

/// Pure-function flattener used by [`FlatForgeEnv`]. Owning it as a
/// separate type lets tests verify the layout without spinning up a
/// full [`WorldEnv`].
#[derive(Debug, Clone)]
pub struct ObsFlattener {
    cfg: FlatObsConfig,
    forge_cfg: ForgeConfig,
    flat_dim: usize,
}

impl ObsFlattener {
    /// Build a flattener for the given env config.
    pub fn new(cfg: FlatObsConfig, forge_cfg: ForgeConfig) -> Self {
        let flat_dim = cfg.flat_dim(&forge_cfg);
        Self {
            cfg,
            forge_cfg,
            flat_dim,
        }
    }

    /// Returns the output dimension (== `cfg.flat_dim(&forge_cfg)`).
    pub fn flat_dim(&self) -> usize {
        self.flat_dim
    }

    /// Flatten an observation into a freshly-allocated `Vec<f32>`.
    pub fn flatten(&self, obs: &Observation) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.flat_dim);
        self.flatten_into(obs, &mut out);
        out
    }

    /// Flatten into a caller-owned buffer. The buffer is `clear()`'d
    /// then extended; capacity is reused.
    pub fn flatten_into(&self, obs: &Observation, out: &mut Vec<f32>) {
        out.clear();
        if self.cfg.include_grid {
            for tile in &obs.grid_view {
                out.push(tile.terrain as f32);
                out.push(if tile.has_agent { 1.0 } else { 0.0 });
                out.push(if tile.has_object { 1.0 } else { 0.0 });
                out.push(if tile.has_resource { 1.0 } else { 0.0 });
                out.push(tile.elevation as f32);
                out.push(tile.object_type as f32);
                out.push(tile.resource_type as f32);
            }
            // Pad/truncate to the expected grid block size — defensive
            // against `Observation`s built with mismatched view radii.
            let vr = self.forge_cfg.agents.default_vision_radius as usize;
            let view_side = 2 * vr + 1;
            let expected = view_side * view_side * TILE_FEATURE_COUNT;
            out.resize(expected, 0.0);
        }
        if self.cfg.include_inventory {
            let cap = self.forge_cfg.agents.default_carry_capacity as usize;
            for slot in obs.inventory.slots.iter().take(cap) {
                out.push(slot.0 as f32);
                out.push(slot.1 as f32);
            }
            let prefix = if self.cfg.include_grid {
                let vr = self.forge_cfg.agents.default_vision_radius as usize;
                let view_side = 2 * vr + 1;
                view_side * view_side * TILE_FEATURE_COUNT
            } else {
                0
            };
            let expected = prefix + cap * INVENTORY_FEATURES_PER_SLOT;
            out.resize(expected, 0.0);
        }
        if self.cfg.include_vitals {
            out.push(obs.health);
            out.push(obs.stamina);
        }
        if self.cfg.include_position {
            let (x, y) = obs.position;
            if self.cfg.normalize {
                let w = self.forge_cfg.world.width.max(1) as f32;
                let h = self.forge_cfg.world.height.max(1) as f32;
                out.push(x as f32 / w);
                out.push(y as f32 / h);
            } else {
                out.push(x as f32);
                out.push(y as f32);
            }
        }
        if self.cfg.include_day_phase {
            if self.cfg.normalize {
                out.push(obs.day_phase as f32 / 4.0);
            } else {
                out.push(obs.day_phase as f32);
            }
        }
        if self.cfg.include_task_progress {
            let n = task_predicate_count(&self.forge_cfg);
            for v in obs.task_progress.iter().take(n) {
                out.push(*v);
            }
            // Pad if the observation has fewer entries than configured.
            let target = self.flat_dim;
            out.resize(target, 0.0);
        }
        debug_assert_eq!(
            out.len(),
            self.flat_dim,
            "flatten produced {} elements, expected {}",
            out.len(),
            self.flat_dim
        );
    }
}

/// Flat-observation wrapper around [`WorldEnv`].
///
/// Satisfies [`forge_env::FlatObsEnv`] so it can drive `latent_mcts`
/// and any other consumer expecting `(Vec<f32>, u32)` IO.
pub struct FlatForgeEnv {
    inner: WorldEnv,
    flattener: ObsFlattener,
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    /// Cached typed-obs buffer for `reset_into` — avoids allocating a
    /// fresh `Observation` each episode.
    inner_obs_buf: forge_types::observation::Observation,
    /// Cached inner step-output buffer for `step_into` — avoids
    /// allocating a fresh `StepOutput<Observation, StepInfo>` each step.
    inner_step_buf: StepOutput<forge_types::observation::Observation, forge_types::observation::StepInfo>,
}

impl FlatForgeEnv {
    /// Build a [`FlatForgeEnv`] from a [`ForgeConfig`] and a
    /// [`FlatObsConfig`]. Constructs the underlying [`WorldEnv`]
    /// internally.
    pub fn new(forge_cfg: ForgeConfig, flat_cfg: FlatObsConfig) -> Result<Self, ForgeEnvError> {
        let inner = WorldEnv::new(forge_cfg.clone())?;
        let flattener = ObsFlattener::new(flat_cfg, forge_cfg.clone());
        let obs_spec = ObsSpec::flat_f32("forge_flat", flattener.flat_dim(), 0.0, 1.0);
        // Re-use the action-space sizing the inner env already computed.
        let action_spec = inner.action_spec().clone();
        Ok(Self {
            inner,
            flattener,
            obs_spec,
            action_spec,
            inner_obs_buf: forge_types::observation::Observation::default(),
            inner_step_buf: StepOutput::default(),
        })
    }
}

impl Env for FlatForgeEnv {
    type Obs = Vec<f32>;
    type Action = u32;
    type Info = StepInfo;
    type Error = ForgeEnvError;

    #[instrument(skip_all, fields(env = "forge-flat"))]
    fn reset_into(
        &mut self,
        seed: Option<u64>,
        out: &mut Vec<f32>,
    ) -> Result<(), Self::Error> {
        self.inner.reset_into(seed, &mut self.inner_obs_buf)?;
        self.flattener.flatten_into(&self.inner_obs_buf, out);
        Ok(())
    }

    #[instrument(skip_all, fields(env = "forge-flat", action_id = action))]
    fn step_into(
        &mut self,
        action: u32,
        out: &mut StepOutput<Vec<f32>, Self::Info>,
    ) -> Result<(), Self::Error> {
        // Decode action before mutably borrowing inner_step_buf.
        let typed_action = self.inner.decode_action(action)?;
        self.inner.step_into(typed_action, &mut self.inner_step_buf)?;
        self.flattener
            .flatten_into(&self.inner_step_buf.obs, &mut out.obs);
        out.reward = self.inner_step_buf.reward;
        out.terminated = self.inner_step_buf.terminated;
        out.truncated = self.inner_step_buf.truncated;
        out.info = self.inner_step_buf.info.clone();
        Ok(())
    }

    fn obs_spec(&self) -> &ObsSpec {
        &self.obs_spec
    }

    fn action_spec(&self) -> &ActionSpec {
        &self.action_spec
    }

    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("forge-flat")
    }
}

impl FlatObsEnv for FlatForgeEnv {
    fn obs_dim(&self) -> usize {
        self.flattener.flat_dim()
    }

    /// FlatForgeEnv constructs its `action_spec` as `Discrete` in
    /// [`FlatForgeEnv::new`]; the invariant cannot be violated by
    /// public API. We document the assumption and fall back to 0 if
    /// the invariant ever breaks (e.g. via future code changes) so a
    /// caller's argmax doesn't panic. The `flat_forge_env_action_spec_is_discrete`
    /// test in `tests/forge_env_parity.rs` gates this invariant in CI.
    fn num_actions(&self) -> u32 {
        self.action_spec.discrete_n().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::observation::Observation;

    fn small_config() -> ForgeConfig {
        let mut cfg = ForgeConfig::default();
        cfg.world.width = 16;
        cfg.world.height = 16;
        cfg
    }

    #[test]
    fn flattener_dim_matches_spec() {
        let cfg = small_config();
        let flat_cfg = FlatObsConfig::default();
        let f = ObsFlattener::new(flat_cfg.clone(), cfg.clone());
        assert_eq!(f.flat_dim(), flat_cfg.flat_dim(&cfg));
    }

    #[test]
    fn flatten_empty_observation_produces_correct_length() {
        let cfg = small_config();
        let flat_cfg = FlatObsConfig::default();
        let f = ObsFlattener::new(flat_cfg, cfg);
        let obs = Observation::default();
        let v = f.flatten(&obs);
        assert_eq!(v.len(), f.flat_dim());
    }

    #[test]
    fn toggling_blocks_changes_dim() {
        let cfg = small_config();
        let with = FlatObsConfig::default();
        let without = FlatObsConfig {
            include_task_progress: false,
            ..FlatObsConfig::default()
        };
        let dim_with = ObsFlattener::new(with, cfg.clone()).flat_dim();
        let dim_without = ObsFlattener::new(without, cfg.clone()).flat_dim();
        assert!(dim_with >= dim_without);
    }

    #[test]
    fn flat_forge_env_accessors_match_construction() {
        let cfg = small_config();
        let env = FlatForgeEnv::new(cfg, FlatObsConfig::default()).unwrap();
        use forge_env::Env;
        assert!(!env.obs_spec().shape.is_empty());
        assert!(env.action_spec().discrete_n().is_some());
        assert_eq!(env.name().as_ref(), "forge-flat");
    }

    #[test]
    fn flatten_with_non_normalize_emits_raw_scalars() {
        let cfg = small_config();
        let flat_cfg = FlatObsConfig {
            normalize: false,
            include_grid: false,
            include_inventory: false,
            include_vitals: false,
            include_position: true,
            include_day_phase: true,
            include_task_progress: false,
        };
        let f = ObsFlattener::new(flat_cfg, cfg);
        let obs = Observation {
            position: (5, 7),
            day_phase: 2,
            ..Observation::default()
        };
        let v = f.flatten(&obs);
        // [x, y, day_phase] = [5.0, 7.0, 2.0]
        assert_eq!(v, vec![5.0, 7.0, 2.0]);
    }

    #[test]
    fn flatten_into_reuses_buffer() {
        let cfg = small_config();
        let f = ObsFlattener::new(FlatObsConfig::default(), cfg);
        let obs = Observation::default();
        let mut buf = Vec::with_capacity(f.flat_dim());
        let initial_cap = buf.capacity();
        for _ in 0..10 {
            f.flatten_into(&obs, &mut buf);
        }
        assert_eq!(buf.capacity(), initial_cap, "buffer must be reused");
        assert_eq!(buf.len(), f.flat_dim());
    }
}
