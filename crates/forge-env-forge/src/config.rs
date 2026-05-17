//! Configuration for the flat observation adapter.

use serde::{Deserialize, Serialize};

use forge_types::config::ForgeConfig;
use forge_types::observation::Observation;

/// Toggles controlling which observation fields appear in the flat
/// `Vec<f32>` output.
///
/// Every option is `bool`-toggleable; the resulting `obs_dim` is
/// derived from these flags + the underlying [`ForgeConfig`] (vision
/// radius, carry capacity, etc.), never hardcoded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatObsConfig {
    /// Include the ego-centric grid view (most of the observation).
    #[serde(default = "default_true")]
    pub include_grid: bool,
    /// Include the inventory tensor.
    #[serde(default = "default_true")]
    pub include_inventory: bool,
    /// Include health and stamina scalars.
    #[serde(default = "default_true")]
    pub include_vitals: bool,
    /// Include absolute position scalars.
    #[serde(default = "default_true")]
    pub include_position: bool,
    /// Include the day-phase scalar.
    #[serde(default = "default_true")]
    pub include_day_phase: bool,
    /// Include per-task-predicate progress vector.
    #[serde(default = "default_true")]
    pub include_task_progress: bool,
    /// Normalise scalar fields to `[0, 1]`. When `false`, raw u8/u16 values
    /// are cast to `f32` directly.
    #[serde(default = "default_true")]
    pub normalize: bool,
}

fn default_true() -> bool {
    true
}

impl Default for FlatObsConfig {
    fn default() -> Self {
        Self {
            include_grid: true,
            include_inventory: true,
            include_vitals: true,
            include_position: true,
            include_day_phase: true,
            include_task_progress: true,
            normalize: true,
        }
    }
}

impl FlatObsConfig {
    /// Compute the flat dimension this config implies, given a
    /// [`ForgeConfig`]. Mirrors the layout used by
    /// [`crate::ObsFlattener::flatten`].
    ///
    /// Layout (each block contributes its element count only when the
    /// corresponding toggle is `true`):
    /// 1. grid: `(2*vision_radius+1)^2 * 7` tile features
    /// 2. inventory: `carry_capacity * 2` (item type + count)
    /// 3. vitals: `2` (health, stamina)
    /// 4. position: `2` (x, y)
    /// 5. day phase: `1`
    /// 6. task progress: `num_predicates`
    pub fn flat_dim(&self, forge_cfg: &ForgeConfig) -> usize {
        let vr = forge_cfg.agents.default_vision_radius as usize;
        let cap = forge_cfg.agents.default_carry_capacity as usize;
        let n_pred = task_predicate_count(forge_cfg);

        let mut total = 0;
        if self.include_grid {
            let view_side = 2 * vr + 1;
            total += view_side * view_side * TILE_FEATURE_COUNT;
        }
        if self.include_inventory {
            total += cap * INVENTORY_FEATURES_PER_SLOT;
        }
        if self.include_vitals {
            total += 2;
        }
        if self.include_position {
            total += 2;
        }
        if self.include_day_phase {
            total += 1;
        }
        if self.include_task_progress {
            total += n_pred;
        }
        total
    }
}

/// Number of float features emitted per tile by the flattener.
///
/// Layout: `[terrain, has_agent, has_object, has_resource, elevation,
/// object_type, resource_type]`. The booleans are emitted as `0.0`/`1.0`.
pub(crate) const TILE_FEATURE_COUNT: usize = 7;

/// Number of float features emitted per inventory slot. `(item_type, count)`.
pub(crate) const INVENTORY_FEATURES_PER_SLOT: usize = 2;

/// Look up the number of task predicates from the config.
///
/// Falls back to a value derived from the first observation if the
/// config doesn't expose it directly. Centralised here so the flattener
/// and `flat_dim` stay in sync.
pub(crate) fn task_predicate_count(forge_cfg: &ForgeConfig) -> usize {
    // FORGE doesn't currently expose num_predicates on ForgeConfig
    // directly; tasks are loaded separately. Use the configured task
    // count or default 0. This is intentionally conservative — callers
    // wanting non-zero task progress should pass `num_predicates` via
    // a future config field.
    forge_cfg.task.max_predicates as usize
}

/// Helper to derive task predicate count from a sample [`Observation`]
/// when the config doesn't carry it explicitly.
pub fn task_predicate_count_from_obs(obs: &Observation) -> usize {
    obs.task_progress.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::observation::Observation;

    #[test]
    fn task_predicate_count_from_obs_returns_progress_len() {
        let obs = Observation {
            task_progress: vec![0.1, 0.2, 0.3],
            ..Observation::default()
        };
        assert_eq!(task_predicate_count_from_obs(&obs), 3);
    }

    #[test]
    fn flat_dim_responds_to_toggles() {
        let cfg = ForgeConfig::default();
        let all_on = FlatObsConfig::default().flat_dim(&cfg);
        let all_off = FlatObsConfig {
            include_grid: false,
            include_inventory: false,
            include_vitals: false,
            include_position: false,
            include_day_phase: false,
            include_task_progress: false,
            normalize: true,
        }
        .flat_dim(&cfg);
        assert!(all_on > all_off);
        assert_eq!(all_off, 0);
    }
}
