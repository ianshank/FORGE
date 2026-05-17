//! Space description types used by both observations and actions.
//!
//! These mirror the Gymnasium space taxonomy (`Box`, `Discrete`,
//! `MultiDiscrete`) at a serialisable level so the same spec can travel
//! across language boundaries (Rust → Python → Node) without re-encoding.

use serde::{Deserialize, Serialize};

/// Numeric element type of an observation tensor.
///
/// `forge-env` does not encode tensors directly; consumers use this hint
/// to choose appropriate buffer types (e.g. `Vec<f32>` for `F32`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    /// 32-bit IEEE-754 float.
    #[default]
    F32,
    /// 32-bit signed integer.
    I32,
    /// 8-bit unsigned integer (e.g. RGB pixels).
    U8,
}

/// Description of an environment's observation space.
///
/// `shape` is the logical tensor shape (e.g. `[height, width, channels]`
/// or `[flat_dim]` for flattened observations). `low`/`high` give scalar
/// bounds that apply to every element; per-element bounds are out of
/// scope for v1 and can be modelled by a future `ObsSpec::Box` variant
/// if required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObsSpec {
    /// Logical tensor shape.
    pub shape: Vec<usize>,
    /// Inclusive lower bound applied to every element.
    pub low: f32,
    /// Inclusive upper bound applied to every element.
    pub high: f32,
    /// Element dtype.
    pub dtype: DType,
    /// Human-readable name (e.g. "minecraft_symbolic_obs", "forge_flat").
    pub name: String,
}

impl ObsSpec {
    /// Build a flat F32 spec with the given length and bounds.
    pub fn flat_f32(name: impl Into<String>, len: usize, low: f32, high: f32) -> Self {
        Self {
            shape: vec![len],
            low,
            high,
            dtype: DType::F32,
            name: name.into(),
        }
    }

    /// Product of the shape dimensions — the number of scalar elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().copied().product()
    }
}

/// Description of an environment's action space.
///
/// `Discrete` covers FORGE and Minecraft action indices; `MultiDiscrete`
/// is reserved for envs that expose multiple independent discrete
/// dimensions (e.g. one head per limb); `Box` is reserved for continuous
/// control. v1 implementations of FORGE/MC use `Discrete`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionSpec {
    /// A single discrete action with `n` choices.
    Discrete {
        /// Number of discrete actions.
        n: u32,
        /// Optional human-readable labels indexed by action id.
        labels: Option<Vec<String>>,
    },
    /// `nvec.len()` independent discrete dimensions, each with its own size.
    MultiDiscrete {
        /// Per-dimension cardinality.
        nvec: Vec<u32>,
    },
    /// Continuous action vector with per-element bounds.
    Box {
        /// Per-element lower bounds.
        low: Vec<f32>,
        /// Per-element upper bounds.
        high: Vec<f32>,
    },
}

impl ActionSpec {
    /// Build a discrete spec with `n` actions and no labels.
    pub fn discrete(n: u32) -> Self {
        Self::Discrete { n, labels: None }
    }

    /// Build a discrete spec with `n` actions and the given labels.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `labels.len() == n as usize`.
    pub fn discrete_with_labels(n: u32, labels: Vec<String>) -> Self {
        debug_assert_eq!(labels.len(), n as usize, "labels.len() must equal n");
        Self::Discrete {
            n,
            labels: Some(labels),
        }
    }

    /// Returns the total number of distinct actions if this is a
    /// `Discrete` space; `None` otherwise.
    pub fn discrete_n(&self) -> Option<u32> {
        match self {
            Self::Discrete { n, .. } => Some(*n),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obs_spec_num_elements_matches_product() {
        let s = ObsSpec {
            shape: vec![3, 4, 5],
            low: 0.0,
            high: 1.0,
            dtype: DType::F32,
            name: "test".into(),
        };
        assert_eq!(s.num_elements(), 60);
    }

    #[test]
    fn obs_spec_flat_helper_builds_one_dimensional_shape() {
        let s = ObsSpec::flat_f32("forge_flat", 128, -1.0, 1.0);
        assert_eq!(s.shape, vec![128]);
        assert_eq!(s.num_elements(), 128);
        assert_eq!(s.dtype, DType::F32);
    }

    #[test]
    fn action_spec_discrete_n_round_trips() {
        let s = ActionSpec::discrete(40);
        assert_eq!(s.discrete_n(), Some(40));
        let m = ActionSpec::MultiDiscrete {
            nvec: vec![3, 4, 5],
        };
        assert_eq!(m.discrete_n(), None);
    }

    #[test]
    fn obs_spec_serde_roundtrip() {
        let s = ObsSpec::flat_f32("forge_flat", 64, 0.0, 1.0);
        let json = serde_json::to_string(&s).expect("serialize");
        let back: ObsSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s, back);
    }

    #[test]
    fn action_spec_serde_roundtrip() {
        let labels = vec!["noop".to_string(), "fire".to_string()];
        let s = ActionSpec::discrete_with_labels(2, labels);
        let json = serde_json::to_string(&s).expect("serialize");
        let back: ActionSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s, back);
    }

    #[test]
    fn action_spec_serde_tag_is_kind() {
        let s = ActionSpec::discrete(4);
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"kind\":\"discrete\""), "got: {json}");
    }

    #[test]
    fn dtype_defaults_to_f32() {
        assert_eq!(DType::default(), DType::F32);
    }

    proptest::proptest! {
        #[test]
        fn obs_shape_product_matches_for_arbitrary_shapes(
            dims in proptest::collection::vec(1usize..8, 1..5),
        ) {
            let expected: usize = dims.iter().copied().product();
            let s = ObsSpec {
                shape: dims,
                low: 0.0,
                high: 1.0,
                dtype: DType::F32,
                name: "p".into(),
            };
            proptest::prop_assert_eq!(s.num_elements(), expected);
        }

        #[test]
        fn discrete_n_round_trips_arbitrary_counts(n in 1u32..10_000) {
            let s = ActionSpec::discrete(n);
            proptest::prop_assert_eq!(s.discrete_n(), Some(n));
        }
    }
}
