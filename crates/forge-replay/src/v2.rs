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

/// On-disk file extension for plain JSON trajectories.
pub const JSON_EXT: &str = "json";

/// On-disk file extension for gzip-compressed JSON trajectories.
/// (Path API returns just `"gz"` for `.json.gz` — we keep the
/// combined form here for documentation / for callers building
/// filenames.)
pub const JSON_GZ_EXT: &str = "json.gz";

/// Suffix returned by `Path::extension()` for gzip trajectory files.
/// Used by [`TrajectoryV2::load_json`] to auto-detect compression.
pub const GZ_SUFFIX: &str = "gz";

/// Hard cap on decompressed bytes accepted by [`TrajectoryV2::load_json`]
/// when the file extension is `.gz`.
///
/// Sized at ~125x a realistic 4 MB JSON episode, so legitimate
/// trajectories never hit the cap. Pathological gzip-bomb input that
/// would decompress past this limit produces an `UnexpectedEof` from
/// `serde_json` after the inner reader is closed by `take()` — no
/// OOM. Security defence-in-depth; the runner trusts its own
/// trajectory dir, but the cap protects against a hostile actor that
/// can drop a file there.
pub const MAX_DECOMPRESSED_TRAJECTORY_BYTES: usize = 512 * 1024 * 1024;

/// Gzip compression level for [`TrajectoryV2::save_json_gz`].
///
/// Accepts both the named variants and an explicit `Custom(0..=9)`
/// integer for TOML configs. Maps onto [`flate2::Compression`] at the
/// codec boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", untagged)]
pub enum TrajectoryGzipLevel {
    /// `flate2::Compression::default()` (level 6).
    #[serde(rename = "default")]
    Named(NamedGzipLevel),
    /// Caller-supplied level 0..=9. Validation lives in
    /// [`TrajectoryGzipLevel::resolve`] — values outside that range
    /// produce a `TrajectoryError::Io` rather than panicking.
    Custom(u32),
}

/// Named gzip presets. Kept as a separate enum so the TOML
/// representation can use lowercase string variants (`"fastest"`,
/// `"default"`, `"best"`) without conflicting with the
/// `Custom(u32)` integer form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NamedGzipLevel {
    /// `flate2::Compression::fast()` (level 1).
    Fastest,
    /// `flate2::Compression::default()` (level 6).
    Default,
    /// `flate2::Compression::best()` (level 9).
    Best,
}

impl Default for TrajectoryGzipLevel {
    fn default() -> Self {
        Self::Named(NamedGzipLevel::Default)
    }
}

impl TrajectoryGzipLevel {
    /// Resolve to a `flate2::Compression`, validating the
    /// `Custom(level)` integer against `0..=9`.
    pub fn resolve(&self) -> Result<flate2::Compression, TrajectoryError> {
        Ok(match self {
            Self::Named(NamedGzipLevel::Fastest) => flate2::Compression::fast(),
            Self::Named(NamedGzipLevel::Default) => flate2::Compression::default(),
            Self::Named(NamedGzipLevel::Best) => flate2::Compression::best(),
            Self::Custom(level) => {
                if *level > 9 {
                    return Err(TrajectoryError::Io(format!(
                        "gzip level must be in 0..=9, got {level}"
                    )));
                }
                flate2::Compression::new(*level)
            }
        })
    }
}

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
    /// RFC3339 end timestamp. `None` until [`TrajectoryV2::finalize`] is
    /// called. `#[serde(default)]` keeps older v2 JSONL (which encoded
    /// this as an empty string) loadable — see migration test below.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<String>,
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
            ended_at: None,
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
        self.ended_at = Some(ended_at.into());
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
                return Err(TrajectoryError::InvalidStep {
                    index: i,
                    reason: format!(
                        "obs.len() = {}, header.obs_dim = {}",
                        s.obs.len(),
                        self.obs_dim
                    ),
                });
            }
            if s.policy_target.len() as u32 != self.action_count {
                return Err(TrajectoryError::InvalidStep {
                    index: i,
                    reason: format!(
                        "policy_target.len() = {}, action_count = {}",
                        s.policy_target.len(),
                        self.action_count
                    ),
                });
            }
            if s.action_id >= self.action_count {
                return Err(TrajectoryError::InvalidStep {
                    index: i,
                    reason: format!(
                        "action_id = {}, action_count = {}",
                        s.action_id, self.action_count
                    ),
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

    /// Gzip-compressed sibling of [`Self::save_json`]. Atomic write
    /// (`.tmp` sibling + rename). Caller picks the filename — by
    /// convention `<episode_id>.json.gz`.
    ///
    /// Compression level flows from the caller; `level.resolve()`
    /// validates `Custom(>9)` and surfaces a [`TrajectoryError::Io`]
    /// rather than panicking.
    #[instrument(skip(self, level), fields(path = %path.as_ref().display(), len = self.steps.len()))]
    pub fn save_json_gz(
        &self,
        path: impl AsRef<Path>,
        level: TrajectoryGzipLevel,
    ) -> Result<(), TrajectoryError> {
        use std::io::Write;

        self.validate()?;
        let compression = level.resolve()?;
        let path = path.as_ref();
        let dir = path
            .parent()
            .ok_or_else(|| TrajectoryError::Io(format!("no parent dir for {}", path.display())))?;
        let tmp = dir.join(format!(
            ".{}.tmp",
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "trajectory.json.gz".into())
        ));
        let raw =
            serde_json::to_vec(self).map_err(|e| TrajectoryError::Io(format!("serialise: {e}")))?;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), compression);
        encoder
            .write_all(&raw)
            .map_err(|e| TrajectoryError::Io(format!("gzip encode: {e}")))?;
        let compressed = encoder
            .finish()
            .map_err(|e| TrajectoryError::Io(format!("gzip finish: {e}")))?;
        std::fs::write(&tmp, &compressed)
            .map_err(|e| TrajectoryError::Io(format!("write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| TrajectoryError::Io(format!("rename to {}: {e}", path.display())))?;
        debug!(
            raw_bytes = raw.len(),
            compressed_bytes = compressed.len(),
            "wrote gzip trajectory"
        );
        Ok(())
    }

    /// Load from a trajectory file. Auto-detects compression by
    /// extension: paths ending in `.gz` decompress through
    /// `flate2::read::GzDecoder` (capped at
    /// [`MAX_DECOMPRESSED_TRAJECTORY_BYTES`]); other paths are read
    /// as plain JSON via the existing path.
    ///
    /// Fails on `format_version` mismatch, on decompressed payloads
    /// exceeding the cap, and on the usual serde / IO errors.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, TrajectoryError> {
        let path = path.as_ref();
        let is_gz = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case(GZ_SUFFIX))
            .unwrap_or(false);
        let t: TrajectoryV2 = if is_gz {
            use std::io::Read;
            let file = std::fs::File::open(path)
                .map_err(|e| TrajectoryError::Io(format!("read {}: {e}", path.display())))?;
            let decoder = flate2::read::GzDecoder::new(file);
            // `.take()` caps decompressed bytes at MAX_DECOMPRESSED_*.
            // Pathological gzip-bomb input has its inner stream
            // truncated at the cap, after which `serde_json` sees an
            // `UnexpectedEof` and surfaces TrajectoryError::Io — no
            // OOM.
            let mut capped = decoder.take(MAX_DECOMPRESSED_TRAJECTORY_BYTES as u64 + 1);
            let mut buf = Vec::new();
            capped
                .read_to_end(&mut buf)
                .map_err(|e| TrajectoryError::Io(format!("gzip decode {}: {e}", path.display())))?;
            if buf.len() > MAX_DECOMPRESSED_TRAJECTORY_BYTES {
                return Err(TrajectoryError::Io(format!(
                    "decompressed trajectory exceeds {MAX_DECOMPRESSED_TRAJECTORY_BYTES}-byte cap (got >{} bytes)",
                    MAX_DECOMPRESSED_TRAJECTORY_BYTES
                )));
            }
            serde_json::from_slice(&buf)
                .map_err(|e| TrajectoryError::Io(format!("deserialise: {e}")))?
        } else {
            let bytes = std::fs::read(path)
                .map_err(|e| TrajectoryError::Io(format!("read {}: {e}", path.display())))?;
            serde_json::from_slice(&bytes)
                .map_err(|e| TrajectoryError::Io(format!("deserialise: {e}")))?
        };
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
    fn ended_at_is_none_until_finalize() {
        let t = empty_traj(4, 2);
        assert!(t.ended_at.is_none());
    }

    #[test]
    fn ended_at_optional_roundtrip_with_and_without_finalize() {
        let tmp = tempfile::tempdir().unwrap();
        // Unfinalised: ended_at omitted from JSON (skip_serializing_if).
        let path_no_end = tmp.path().join("no-end.json");
        let mut t_no = empty_traj(4, 2);
        t_no.push(make_step(4, 2, 0, 0.0)).unwrap();
        t_no.save_json(&path_no_end).unwrap();
        let raw = std::fs::read_to_string(&path_no_end).unwrap();
        assert!(
            !raw.contains("ended_at"),
            "skip_serializing_if should omit the field when None: {raw}"
        );
        let back = TrajectoryV2::load_json(&path_no_end).unwrap();
        assert_eq!(back.ended_at, None);

        // Finalised: ended_at present and round-trips as Some(_).
        let path_end = tmp.path().join("end.json");
        let mut t_end = empty_traj(4, 2);
        t_end.push(make_step(4, 2, 0, 0.0)).unwrap();
        t_end.finalize("2026-01-01T00:00:05Z");
        t_end.save_json(&path_end).unwrap();
        let back = TrajectoryV2::load_json(&path_end).unwrap();
        assert_eq!(back.ended_at.as_deref(), Some("2026-01-01T00:00:05Z"));
    }

    #[test]
    fn legacy_ended_at_string_payload_is_loadable() {
        // Older writers emitted `"ended_at": ""` or a literal RFC3339 string.
        // `#[serde(default)]` plus `Option<String>` must accept both shapes.
        let tmp = tempfile::tempdir().unwrap();
        let path_empty = tmp.path().join("legacy-empty.json");
        let path_str = tmp.path().join("legacy-string.json");
        let mut t = empty_traj(4, 2);
        t.push(make_step(4, 2, 0, 0.0)).unwrap();
        let mut value = serde_json::to_value(&t).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("ended_at".into(), serde_json::Value::String(String::new()));
        std::fs::write(&path_empty, serde_json::to_vec(&value).unwrap()).unwrap();
        let back = TrajectoryV2::load_json(&path_empty).unwrap();
        // Empty string is preserved verbatim — callers can treat empty as
        // "unfinalised" if they care; the schema is liberal in what it accepts.
        assert_eq!(back.ended_at.as_deref(), Some(""));

        value.as_object_mut().unwrap().insert(
            "ended_at".into(),
            serde_json::Value::String("2025-12-31T23:59:59Z".into()),
        );
        std::fs::write(&path_str, serde_json::to_vec(&value).unwrap()).unwrap();
        let back = TrajectoryV2::load_json(&path_str).unwrap();
        assert_eq!(back.ended_at.as_deref(), Some("2025-12-31T23:59:59Z"));
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
    fn empty_predicate_returns_true_when_no_steps() {
        let t = empty_traj(4, 2);
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn validate_rejects_zero_obs_dim() {
        let t = TrajectoryV2 {
            obs_dim: 0,
            ..empty_traj(1, 1)
        };
        assert!(matches!(
            t.validate(),
            Err(TrajectoryError::InvalidHeader(_))
        ));
    }

    #[test]
    fn validate_walks_each_step_and_reports_index() {
        let mut t = empty_traj(4, 2);
        t.push(make_step(4, 2, 0, 0.0)).unwrap();
        t.push(make_step(4, 2, 1, 0.0)).unwrap();
        // Mutate a step's obs to violate the invariant.
        t.steps[1].obs = vec![0.0; 99];
        let err = t.validate().unwrap_err();
        match err {
            TrajectoryError::InvalidStep { index, reason } => {
                assert_eq!(index, 1);
                assert!(reason.contains("obs.len()"));
            }
            other => panic!("expected InvalidStep, got {other:?}"),
        }
    }

    #[test]
    fn validate_catches_policy_dim_drift_per_step() {
        let mut t = empty_traj(4, 2);
        t.push(make_step(4, 2, 0, 0.0)).unwrap();
        t.steps[0].policy_target = vec![0.5; 99];
        let err = t.validate().unwrap_err();
        assert!(matches!(err, TrajectoryError::InvalidStep { .. }));
    }

    #[test]
    fn validate_catches_action_id_drift_per_step() {
        let mut t = empty_traj(4, 2);
        t.push(make_step(4, 2, 0, 0.0)).unwrap();
        t.steps[0].action_id = 99;
        let err = t.validate().unwrap_err();
        assert!(matches!(err, TrajectoryError::InvalidStep { .. }));
    }

    #[test]
    fn save_json_fails_validate_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        let mut t = empty_traj(4, 2);
        t.push(make_step(4, 2, 0, 0.0)).unwrap();
        t.action_count = 0;
        let err = t.save_json(&path).unwrap_err();
        assert!(matches!(err, TrajectoryError::InvalidHeader(_)));
    }

    #[test]
    fn from_v1_preserves_seed() {
        use crate::trajectory::Trajectory;
        let mut v1 = Trajectory::new();
        v1.metadata.seed = 12345;
        let v2 = from_v1(
            &v1,
            FromV1Options {
                env_id: "forge".into(),
                schema_id: "s".into(),
                episode_id: "e".into(),
                obs_dim: 1,
                action_count: 1,
                started_at: "a".into(),
                ended_at: "b".into(),
                flatten_step: &|_| vec![0.0; 1],
            },
        )
        .unwrap();
        assert_eq!(v2.seed, Some(12345));
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

    // ---------------- gzip compression tests (Track 4) ----------------

    fn populated_traj(obs_dim: usize, action_count: u32, num_steps: usize) -> TrajectoryV2 {
        let mut t = empty_traj(obs_dim, action_count);
        for i in 0..num_steps {
            t.push(make_step(obs_dim, action_count, 0, i as f32))
                .unwrap();
        }
        t.finalize("2026-01-01T00:00:01Z");
        t
    }

    #[test]
    fn save_json_gz_then_load_json_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ep-1.json.gz");
        let original = populated_traj(4, 2, 5);
        original
            .save_json_gz(&path, TrajectoryGzipLevel::default())
            .unwrap();
        let back = TrajectoryV2::load_json(&path).unwrap();
        assert_eq!(back.steps.len(), original.steps.len());
        assert_eq!(back.episode_id, original.episode_id);
        assert_eq!(back.obs_dim, original.obs_dim);
        assert_eq!(back.action_count, original.action_count);
    }

    #[test]
    fn load_json_auto_detects_gzip_by_extension() {
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("ep-a.json");
        let gz_path = dir.path().join("ep-a.json.gz");
        let traj = populated_traj(2, 2, 3);
        traj.save_json(&json_path).unwrap();
        traj.save_json_gz(&gz_path, TrajectoryGzipLevel::default())
            .unwrap();

        let from_json = TrajectoryV2::load_json(&json_path).unwrap();
        let from_gz = TrajectoryV2::load_json(&gz_path).unwrap();
        assert_eq!(from_json.steps.len(), from_gz.steps.len());
        assert_eq!(from_json.episode_id, from_gz.episode_id);
        for (a, b) in from_json.steps.iter().zip(from_gz.steps.iter()) {
            assert_eq!(a.obs, b.obs);
            assert_eq!(a.action_id, b.action_id);
            assert!((a.reward - b.reward).abs() < 1e-6);
        }
    }

    #[test]
    fn load_json_gz_with_corrupt_bytes_returns_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.json.gz");
        // Random non-gzip bytes.
        std::fs::write(&path, b"this is not a gzip stream").unwrap();
        let err = TrajectoryV2::load_json(&path).unwrap_err();
        assert!(matches!(err, TrajectoryError::Io(_)), "got {err:?}");
    }

    #[test]
    fn load_json_rejects_decompressed_payload_past_cap() {
        // Build a gzip stream that decompresses to MAX_+1 bytes of
        // zeros. flate2 compresses zeros aggressively so the on-disk
        // file is tiny while the cap blocks the decompressed payload.
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bomb.json.gz");
        let mut encoder = flate2::write::GzEncoder::new(
            std::fs::File::create(&path).unwrap(),
            flate2::Compression::default(),
        );
        // Write MAX_+1 zero bytes in 1 MB chunks.
        let chunk = vec![0u8; 1024 * 1024];
        let mut written = 0usize;
        while written <= MAX_DECOMPRESSED_TRAJECTORY_BYTES {
            let n = chunk
                .len()
                .min(MAX_DECOMPRESSED_TRAJECTORY_BYTES + 1 - written);
            encoder.write_all(&chunk[..n]).unwrap();
            written += n;
        }
        encoder.finish().unwrap();

        let err = TrajectoryV2::load_json(&path).unwrap_err();
        match err {
            TrajectoryError::Io(msg) => assert!(
                msg.contains("byte cap") || msg.contains("exceeds"),
                "got: {msg}"
            ),
            other => panic!("expected TrajectoryError::Io, got {other:?}"),
        }
    }

    #[test]
    fn gzip_level_custom_rejects_out_of_range() {
        let bad = TrajectoryGzipLevel::Custom(10);
        let err = bad.resolve().unwrap_err();
        assert!(matches!(err, TrajectoryError::Io(_)));
    }

    #[test]
    fn gzip_level_named_resolves_to_flate2_compression() {
        TrajectoryGzipLevel::Named(NamedGzipLevel::Fastest)
            .resolve()
            .unwrap();
        TrajectoryGzipLevel::Named(NamedGzipLevel::Default)
            .resolve()
            .unwrap();
        TrajectoryGzipLevel::Named(NamedGzipLevel::Best)
            .resolve()
            .unwrap();
        TrajectoryGzipLevel::Custom(0).resolve().unwrap();
        TrajectoryGzipLevel::Custom(9).resolve().unwrap();
    }
}
