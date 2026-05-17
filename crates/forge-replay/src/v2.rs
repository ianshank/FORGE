//! Env-agnostic trajectory format **v2**.
//!
//! Unlike [`crate::trajectory::Trajectory`] (v1), which stores typed
//! [`forge_types::observation::Observation`]s, v2 stores flat
//! `Vec<f32>` observations plus the MCTS policy/value targets needed
//! by a MuZero-style trainer. This lets the same format carry
//! trajectories from any env (Minecraft, FORGE, future backends) as
//! long as observations are flattened to a float vector.
//!
//! ## Versioning
//!
//! [`TRAJECTORY_FORMAT_VERSION`] is pinned. Any breaking change MUST
//! bump it; readers fail fast on mismatch. v1 trajectories remain
//! untouched — a one-way converter [`from_v1`] is provided for
//! migration.
//!
//! ## Schema id
//!
//! Every v2 trajectory carries a `schema_id` (sha256 of canonical
//! `(env_id, obs_dim, action_count, action_map, reward_config)`).
//! The runner that writes trajectories and the trainer that consumes
//! them MUST compute the same schema_id; mismatch is a hard error so
//! action-space or reward drift never silently poisons a replay
//! buffer.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::trajectory::Trajectory;

/// Trajectory schema version. Bumping this requires a migrating
/// converter; existing replays must not load under a higher version.
pub const TRAJECTORY_FORMAT_VERSION: u32 = 2;

/// One transition in an env-agnostic trajectory.
///
/// `policy_target` and `value_target` are emitted by the planner
/// (e.g. MCTS visit-count distribution + bootstrapped n-step return)
/// and consumed by the trainer's loss heads. Both must have stable
/// dimensionality across a trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepV2 {
    /// Env tick at which this transition was observed.
    pub tick: u64,
    /// Flat observation vector. `obs.len() == header.obs_dim`.
    pub obs: Vec<f32>,
    /// Discrete action id taken. `action_id < header.action_count`.
    pub action_id: u32,
    /// Policy improvement target (length == `action_count`). Typically
    /// the visit-count distribution from MCTS.
    pub policy_target: Vec<f32>,
    /// Value target (scalar) — bootstrapped n-step return or terminal.
    pub value_target: f32,
    /// Scalar reward observed for this transition.
    pub reward: f32,
    /// Natural terminal.
    pub terminated: bool,
    /// Artificial truncation.
    pub truncated: bool,
}

/// Full v2 trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryV2 {
    /// MUST equal [`TRAJECTORY_FORMAT_VERSION`].
    pub format_version: u32,
    /// Free-form env id, e.g. "minecraft", "forge".
    pub env_id: String,
    /// sha256 of canonical (env_id, obs_dim, action_count, action_map,
    /// reward_config). Cross-checked at trainer load time.
    pub schema_id: String,
    /// Unique episode id (caller-supplied; e.g. ULID).
    pub episode_id: String,
    /// Seed used at reset, if any.
    pub seed: Option<u64>,
    /// Flat observation dimensionality (must match every step's `obs.len()`).
    pub obs_dim: usize,
    /// Discrete action count.
    pub action_count: u32,
    /// All steps in temporal order.
    pub steps: Vec<StepV2>,
    /// Sum of rewards. Stored as a cache; trainer may recompute.
    pub final_reward: f32,
    /// RFC3339 start timestamp (caller-supplied).
    pub started_at: String,
    /// RFC3339 end timestamp (caller-supplied).
    pub ended_at: String,
}

impl TrajectoryV2 {
    /// Build an empty trajectory header.
    pub fn empty(
        env_id: impl Into<String>,
        schema_id: impl Into<String>,
        episode_id: impl Into<String>,
        obs_dim: usize,
        action_count: u32,
        seed: Option<u64>,
        started_at: impl Into<String>,
    ) -> Self {
        Self {
            format_version: TRAJECTORY_FORMAT_VERSION,
            env_id: env_id.into(),
            schema_id: schema_id.into(),
            episode_id: episode_id.into(),
            seed,
            obs_dim,
            action_count,
            steps: Vec::new(),
            final_reward: 0.0,
            started_at: started_at.into(),
            ended_at: String::new(),
        }
    }

    /// Append a step. Validates `obs.len()` and `policy_target.len()`
    /// against the header. Returns the offending lengths on mismatch.
    pub fn push(&mut self, step: StepV2) -> Result<(), TrajectoryError> {
        if step.obs.len() != self.obs_dim {
            return Err(TrajectoryError::ObsDimMismatch {
                expected: self.obs_dim,
                got: step.obs.len(),
            });
        }
        if step.policy_target.len() as u32 != self.action_count {
            return Err(TrajectoryError::PolicyDimMismatch {
                expected: self.action_count as usize,
                got: step.policy_target.len(),
            });
        }
        if step.action_id >= self.action_count {
            return Err(TrajectoryError::ActionOutOfRange {
                action_id: step.action_id,
                action_count: self.action_count,
            });
        }
        self.final_reward += step.reward;
        self.steps.push(step);
        Ok(())
    }

    /// Number of steps recorded.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// True iff no steps have been recorded.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Mark the trajectory complete with the supplied end timestamp.
    pub fn finalize(&mut self, ended_at: impl Into<String>) {
        self.ended_at = ended_at.into();
    }

    /// Validate the header's invariants. Cheap; call before save.
    pub fn validate(&self) -> Result<(), TrajectoryError> {
        if self.format_version != TRAJECTORY_FORMAT_VERSION {
            return Err(TrajectoryError::VersionMismatch {
                expected: TRAJECTORY_FORMAT_VERSION,
                got: self.format_version,
            });
        }
        if self.action_count == 0 {
            return Err(TrajectoryError::InvalidHeader(
                "action_count must be > 0".into(),
            ));
        }
        if self.obs_dim == 0 {
            return Err(TrajectoryError::InvalidHeader("obs_dim must be > 0".into()));
        }
        for (i, s) in self.steps.iter().enumerate() {
            if s.obs.len() != self.obs_dim {
                return Err(TrajectoryError::ObsDimMismatch {
                    expected: self.obs_dim,
                    got: s.obs.len(),
                })
                .map_err(|_e| TrajectoryError::InvalidStep {
                    index: i,
                    reason: "obs.len() != header.obs_dim".into(),
                });
            }
        }
        Ok(())
    }

    /// Serialise to JSON (UTF-8) and write to `path` atomically
    /// (tmp file in same dir + rename). Caller picks the filename.
    #[instrument(skip(self), fields(path = %path.as_ref().display(), len = self.steps.len()))]
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), TrajectoryError> {
        self.validate()?;
        let path = path.as_ref();
        let dir = path
            .parent()
            .ok_or_else(|| TrajectoryError::Io(format!("no parent dir for {}", path.display())))?;
        let tmp = dir.join(format!(
            ".{}.tmp",
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "trajectory".into())
        ));
        let bytes =
            serde_json::to_vec(self).map_err(|e| TrajectoryError::Io(format!("serialise: {e}")))?;
        std::fs::write(&tmp, &bytes)
            .map_err(|e| TrajectoryError::Io(format!("write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| TrajectoryError::Io(format!("rename to {}: {e}", path.display())))?;
        debug!("wrote trajectory");
        Ok(())
    }

    /// Load from a JSON file written by [`Self::save_json`].
    /// Fails on `format_version` mismatch.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, TrajectoryError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| TrajectoryError::Io(format!("read {}: {e}", path.display())))?;
        let t: TrajectoryV2 = serde_json::from_slice(&bytes)
            .map_err(|e| TrajectoryError::Io(format!("deserialise: {e}")))?;
        t.validate()?;
        Ok(t)
    }
}

/// Errors emitted by v2 validation and IO.
#[derive(Debug, thiserror::Error)]
pub enum TrajectoryError {
    /// `format_version` in the file disagreed with the compiled constant.
    #[error("trajectory format version mismatch: expected {expected}, got {got}")]
    VersionMismatch {
        /// Compiled version.
        expected: u32,
        /// Loaded version.
        got: u32,
    },
    /// Step `obs.len()` disagreed with the header's `obs_dim`.
    #[error("obs dim mismatch: header={expected}, step={got}")]
    ObsDimMismatch {
        /// Header value.
        expected: usize,
        /// Step value.
        got: usize,
    },
    /// `policy_target.len()` disagreed with `action_count`.
    #[error("policy target dim mismatch: expected {expected}, got {got}")]
    PolicyDimMismatch {
        /// Expected length (= action_count).
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// A step's `action_id` was outside `[0, action_count)`.
    #[error("action_id {action_id} out of range (action_count={action_count})")]
    ActionOutOfRange {
        /// The offending id.
        action_id: u32,
        /// The configured count.
        action_count: u32,
    },
    /// Header field constraint violated (action_count > 0, obs_dim > 0, ...).
    #[error("invalid header: {0}")]
    InvalidHeader(String),
    /// Step-level invariant violated; includes the offending index.
    #[error("invalid step #{index}: {reason}")]
    InvalidStep {
        /// Step index.
        index: usize,
        /// Human reason.
        reason: String,
    },
    /// File IO or JSON IO failure.
    #[error("io error: {0}")]
    Io(String),
}

/// Conversion options for [`from_v1`].
///
/// Bundled as a struct so clippy doesn't trip on a long argument list
/// and so call sites read more clearly.
pub struct FromV1Options<'a> {
    /// Env id to embed in the resulting v2 header.
    pub env_id: String,
    /// schema_id the trainer is expected to validate against.
    pub schema_id: String,
    /// Unique episode id.
    pub episode_id: String,
    /// Flat observation dimensionality the flattener will produce.
    pub obs_dim: usize,
    /// Target action-space size.
    pub action_count: u32,
    /// RFC3339 start timestamp.
    pub started_at: String,
    /// RFC3339 end timestamp.
    pub ended_at: String,
    /// Flattens a step's per-agent observations into a flat vector.
    pub flatten_step: &'a dyn Fn(&[forge_types::observation::Observation]) -> Vec<f32>,
}

/// Convert a v1 [`Trajectory`] into v2.
///
/// Multi-agent v1 trajectories carry `Vec<Observation>` per step; the
/// caller-supplied `opts.flatten_step` decides how to project that slice
/// into a flat vector (single-agent shim usually picks `[0]`).
pub fn from_v1(v1: &Trajectory, opts: FromV1Options<'_>) -> Result<TrajectoryV2, TrajectoryError> {
    let mut out = TrajectoryV2::empty(
        opts.env_id,
        opts.schema_id,
        opts.episode_id,
        opts.obs_dim,
        opts.action_count,
        Some(v1.metadata.seed),
        opts.started_at,
    );
    let action_count = opts.action_count;
    for (i, step) in v1.steps.iter().enumerate() {
        let obs = (opts.flatten_step)(&step.observations);
        // v1 stores per-agent action ids; use agent 0 to match the
        // single-agent shim convention in `forge-env-forge`.
        let action_id = step.actions.first().copied().unwrap_or(0);
        // Uniform-ish policy target — caller should overwrite with real
        // MCTS distributions in the streaming runner; this just keeps
        // the file structurally valid for converted v1 data.
        let policy_target = vec![1.0 / action_count as f32; action_count as usize];
        let reward = step.rewards.first().copied().unwrap_or(0.0);
        out.push(StepV2 {
            tick: if step.tick > 0 { step.tick } else { i as u64 },
            obs,
            action_id,
            policy_target,
            value_target: 0.0,
            reward,
            terminated: step.terminated,
            truncated: step.truncated,
        })?;
    }
    out.finalize(opts.ended_at);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_step(obs_dim: usize, action_count: u32, action_id: u32, reward: f32) -> StepV2 {
        StepV2 {
            tick: 0,
            obs: vec![0.0; obs_dim],
            action_id,
            policy_target: vec![1.0 / action_count as f32; action_count as usize],
            value_target: 0.0,
            reward,
            terminated: false,
            truncated: false,
        }
    }

    fn empty_traj(obs_dim: usize, action_count: u32) -> TrajectoryV2 {
        TrajectoryV2::empty(
            "test",
            "schema-id-1",
            "ep-1",
            obs_dim,
            action_count,
            Some(0),
            "2026-01-01T00:00:00Z",
        )
    }

    #[test]
    fn format_version_is_pinned() {
        assert_eq!(TRAJECTORY_FORMAT_VERSION, 2);
        let t = empty_traj(8, 4);
        assert_eq!(t.format_version, 2);
    }

    #[test]
    fn push_validates_obs_dim() {
        let mut t = empty_traj(8, 4);
        let bad = StepV2 {
            obs: vec![0.0; 9],
            ..make_step(9, 4, 0, 0.0)
        };
        assert!(matches!(
            t.push(bad),
            Err(TrajectoryError::ObsDimMismatch { .. })
        ));
    }

    #[test]
    fn push_validates_policy_dim() {
        let mut t = empty_traj(8, 4);
        let bad = StepV2 {
            policy_target: vec![0.25; 3],
            ..make_step(8, 4, 0, 0.0)
        };
        assert!(matches!(
            t.push(bad),
            Err(TrajectoryError::PolicyDimMismatch { .. })
        ));
    }

    #[test]
    fn push_rejects_action_out_of_range() {
        let mut t = empty_traj(8, 4);
        let bad = make_step(8, 4, 99, 0.0);
        assert!(matches!(
            t.push(bad),
            Err(TrajectoryError::ActionOutOfRange { .. })
        ));
    }

    #[test]
    fn final_reward_accumulates_across_pushes() {
        let mut t = empty_traj(4, 2);
        for r in [1.0_f32, -0.5, 2.5] {
            t.push(make_step(4, 2, 0, r)).unwrap();
        }
        assert!((t.final_reward - 3.0).abs() < 1e-6);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn validate_rejects_zero_action_count() {
        let t = TrajectoryV2 {
            action_count: 0,
            ..empty_traj(4, 1)
        };
        assert!(matches!(
            t.validate(),
            Err(TrajectoryError::InvalidHeader(_))
        ));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ep-1.json");
        let mut t = empty_traj(4, 2);
        t.push(make_step(4, 2, 0, 1.0)).unwrap();
        t.push(make_step(4, 2, 1, -1.0)).unwrap();
        t.finalize("2026-01-01T00:00:05Z");
        t.save_json(&path).unwrap();
        let back = TrajectoryV2::load_json(&path).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn load_rejects_version_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ep.json");
        let mut t = empty_traj(4, 2);
        t.format_version = 99;
        t.push(make_step(4, 2, 0, 0.0)).unwrap();
        // Bypass save's validate() by writing raw JSON.
        std::fs::write(&path, serde_json::to_vec(&t).unwrap()).unwrap();
        let err = TrajectoryV2::load_json(&path).unwrap_err();
        assert!(matches!(err, TrajectoryError::VersionMismatch { .. }));
    }

    #[test]
    fn from_v1_preserves_rewards_and_terminal_flags() {
        use crate::trajectory::{Trajectory, TrajectoryStep};
        use forge_types::observation::Observation;
        let mut v1 = Trajectory::new();
        v1.metadata.seed = 7;
        v1.steps.push(TrajectoryStep {
            tick: 0,
            observations: vec![Observation::default()],
            actions: vec![3],
            rewards: vec![0.5],
            terminated: false,
            truncated: false,
            reasoning: vec![None],
            confidences: vec![0.0],
            decision_times_ms: vec![0],
        });
        let v2 = from_v1(
            &v1,
            FromV1Options {
                env_id: "forge".into(),
                schema_id: "schema-1".into(),
                episode_id: "ep-1".into(),
                obs_dim: 8,
                action_count: 4,
                started_at: "start".into(),
                ended_at: "end".into(),
                flatten_step: &|_obs: &[Observation]| vec![0.0; 8],
            },
        )
        .unwrap();
        assert_eq!(v2.steps.len(), 1);
        assert!((v2.final_reward - 0.5).abs() < 1e-6);
        // Action is clipped/decoded: v1 stored 3, action_count=4, so it's valid.
        assert_eq!(v2.steps[0].action_id, 3);
        assert_eq!(v2.obs_dim, 8);
        assert_eq!(v2.action_count, 4);
    }
}
