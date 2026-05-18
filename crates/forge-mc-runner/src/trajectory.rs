//! [`TrajectoryWriter`] — episode-scoped wrapper over `TrajectoryV2`.
//!
//! Lifecycle:
//!
//! 1. `TrajectoryWriter::new(dir, env_id, schema_id, obs_dim, action_count)`
//!    creates a writer ready to begin its first episode.
//! 2. `start_episode(episode_id, seed, started_at)` opens a new
//!    in-memory `TrajectoryV2` buffer.
//! 3. `record_step(step)` appends one transition (validated by
//!    `TrajectoryV2::push`).
//! 4. `finalize_and_save(ended_at)` stamps `ended_at`, validates the
//!    full trajectory, and writes `<dir>/<episode_id>.json` atomically.
//!
//! The writer never re-enters a finalised episode — `record_step`
//! between `finalize_and_save` and the next `start_episode` returns
//! [`RunnerError::WriterState`].
//!
//! The trajectory directory is created on first save if missing.

use std::path::{Path, PathBuf};

use forge_replay::v2::{StepV2, TrajectoryV2};
use tracing::{debug, instrument};

use crate::error::RunnerError;

/// Per-episode trajectory accumulator + atomic JSON writer.
pub struct TrajectoryWriter {
    dir: PathBuf,
    env_id: String,
    schema_id: String,
    obs_dim: usize,
    action_count: u32,
    current: Option<TrajectoryV2>,
}

impl TrajectoryWriter {
    /// Build a writer rooted at `dir`. The directory is **not** created
    /// here — it's created on first `finalize_and_save` if missing, so
    /// constructing a writer in a test never touches the FS.
    pub fn new(
        dir: impl Into<PathBuf>,
        env_id: impl Into<String>,
        schema_id: impl Into<String>,
        obs_dim: usize,
        action_count: u32,
    ) -> Self {
        Self {
            dir: dir.into(),
            env_id: env_id.into(),
            schema_id: schema_id.into(),
            obs_dim,
            action_count,
            current: None,
        }
    }

    /// Directory the writer is configured for.
    pub fn directory(&self) -> &Path {
        &self.dir
    }

    /// `obs_dim` the writer pins on every trajectory it opens.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// `action_count` the writer pins on every trajectory it opens.
    pub fn action_count(&self) -> u32 {
        self.action_count
    }

    /// Whether an episode is currently in progress (between
    /// `start_episode` and `finalize_and_save`).
    pub fn has_episode_in_progress(&self) -> bool {
        self.current.is_some()
    }

    /// Number of steps recorded in the in-progress episode, or 0 if
    /// no episode is open.
    pub fn current_len(&self) -> usize {
        self.current.as_ref().map(|t| t.len()).unwrap_or(0)
    }

    /// Begin a fresh episode. Errors if one is already in progress.
    #[instrument(skip_all, fields(episode_id))]
    pub fn start_episode(
        &mut self,
        episode_id: impl Into<String>,
        seed: Option<u64>,
        started_at: impl Into<String>,
    ) -> Result<(), RunnerError> {
        if self.current.is_some() {
            return Err(RunnerError::WriterState(
                "start_episode called while another episode was in progress".into(),
            ));
        }
        let id = episode_id.into();
        tracing::Span::current().record("episode_id", id.as_str());
        self.current = Some(TrajectoryV2::empty(
            self.env_id.clone(),
            self.schema_id.clone(),
            id,
            self.obs_dim,
            self.action_count,
            seed,
            started_at,
        ));
        Ok(())
    }

    /// Append a step. Forwards validation errors from `TrajectoryV2::push`
    /// (obs dim, policy dim, action range).
    pub fn record_step(&mut self, step: StepV2) -> Result<(), RunnerError> {
        let t = self.current.as_mut().ok_or_else(|| {
            RunnerError::WriterState(
                "record_step called without an active episode (call start_episode first)".into(),
            )
        })?;
        t.push(step)?;
        Ok(())
    }

    /// Stamp `ended_at`, validate, and write the episode JSON to
    /// `<dir>/<episode_id>.json`. Returns the final path.
    #[instrument(skip_all, fields(episode_len))]
    pub fn finalize_and_save(
        &mut self,
        ended_at: impl Into<String>,
    ) -> Result<PathBuf, RunnerError> {
        let mut t = self.current.take().ok_or_else(|| {
            RunnerError::WriterState("finalize_and_save called without an active episode".into())
        })?;
        t.finalize(ended_at);
        t.validate()?;
        tracing::Span::current().record("episode_len", t.len());

        if !self.dir.exists() {
            std::fs::create_dir_all(&self.dir).map_err(|e| RunnerError::io(&self.dir, e))?;
        }
        let filename = format!("{}.json", t.episode_id);
        let path = self.dir.join(filename);
        t.save_json(&path)?;
        debug!(path = %path.display(), len = t.len(), "trajectory written");
        Ok(path)
    }

    /// Drop the in-progress episode without writing. Useful for tests
    /// and for aborting a partial episode the planner couldn't finish.
    /// No-op if no episode is active.
    pub fn discard_current(&mut self) {
        self.current = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a writer + a usable StepV2 factory parameterised on dims.
    struct Fixture {
        _dir: tempfile::TempDir,
        writer: TrajectoryWriter,
    }

    impl Fixture {
        fn new(obs_dim: usize, action_count: u32) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("traj");
            let writer =
                TrajectoryWriter::new(path, "minecraft", "schema-x", obs_dim, action_count);
            Self { _dir: dir, writer }
        }

        fn make_step(&self, tick: u64, action_id: u32) -> StepV2 {
            StepV2 {
                tick,
                obs: vec![0.0; self.writer.obs_dim()],
                action_id,
                policy_target: vec![
                    1.0 / self.writer.action_count() as f32;
                    self.writer.action_count() as usize
                ],
                value_target: 0.0,
                reward: 1.0,
                terminated: false,
                truncated: false,
            }
        }
    }

    #[test]
    fn new_does_not_touch_filesystem() {
        let fx = Fixture::new(4, 2);
        assert!(!fx.writer.directory().exists());
        assert!(!fx.writer.has_episode_in_progress());
    }

    #[test]
    fn record_step_before_start_episode_errors() {
        let mut fx = Fixture::new(4, 2);
        let step = fx.make_step(0, 0);
        let err = fx.writer.record_step(step).unwrap_err();
        assert!(matches!(err, RunnerError::WriterState(_)));
    }

    #[test]
    fn finalize_without_start_errors() {
        let mut fx = Fixture::new(4, 2);
        let err = fx.writer.finalize_and_save("now").unwrap_err();
        assert!(matches!(err, RunnerError::WriterState(_)));
    }

    #[test]
    fn start_record_finalize_writes_file_and_clears_state() {
        let mut fx = Fixture::new(4, 3);
        fx.writer
            .start_episode("ep-1", Some(42), "2026-05-17T00:00:00Z")
            .unwrap();
        for t in 0..5u64 {
            fx.writer.record_step(fx.make_step(t, 0)).unwrap();
        }
        assert_eq!(fx.writer.current_len(), 5);
        let path = fx.writer.finalize_and_save("2026-05-17T00:00:05Z").unwrap();
        assert!(path.exists(), "file must be on disk");
        assert!(path.ends_with("ep-1.json"));
        assert!(
            !fx.writer.has_episode_in_progress(),
            "writer must reset after finalize"
        );
        // Re-loadable round-trip — invariant check.
        let back = TrajectoryV2::load_json(&path).unwrap();
        assert_eq!(back.steps.len(), 5);
        assert_eq!(back.episode_id, "ep-1");
        assert_eq!(back.env_id, "minecraft");
    }

    #[test]
    fn start_episode_while_already_in_progress_errors() {
        let mut fx = Fixture::new(2, 2);
        fx.writer.start_episode("ep-a", None, "now").unwrap();
        let err = fx.writer.start_episode("ep-b", None, "now").unwrap_err();
        assert!(matches!(err, RunnerError::WriterState(_)));
    }

    #[test]
    fn record_step_with_wrong_obs_dim_errors() {
        let mut fx = Fixture::new(4, 2);
        fx.writer.start_episode("ep-bad-obs", None, "now").unwrap();
        let bad = StepV2 {
            tick: 0,
            obs: vec![0.0; 99], // wrong
            action_id: 0,
            policy_target: vec![0.5, 0.5],
            value_target: 0.0,
            reward: 0.0,
            terminated: false,
            truncated: false,
        };
        let err = fx.writer.record_step(bad).unwrap_err();
        assert!(matches!(err, RunnerError::Trajectory(_)));
    }

    #[test]
    fn record_step_with_action_out_of_range_errors() {
        let mut fx = Fixture::new(2, 2);
        fx.writer.start_episode("ep-bad-act", None, "now").unwrap();
        let bad = StepV2 {
            tick: 0,
            obs: vec![0.0; 2],
            action_id: 99, // out of range
            policy_target: vec![0.5, 0.5],
            value_target: 0.0,
            reward: 0.0,
            terminated: false,
            truncated: false,
        };
        let err = fx.writer.record_step(bad).unwrap_err();
        assert!(matches!(err, RunnerError::Trajectory(_)));
    }

    /// Saving an empty episode is allowed (the planner may decide to
    /// record a 0-step trajectory for, e.g., immediate-failure cases).
    /// The header invariants still must hold.
    #[test]
    fn empty_episode_can_be_finalized_and_saved() {
        let mut fx = Fixture::new(2, 2);
        fx.writer.start_episode("ep-empty", None, "now").unwrap();
        let path = fx.writer.finalize_and_save("now").unwrap();
        let back = TrajectoryV2::load_json(&path).unwrap();
        assert!(back.steps.is_empty());
        assert_eq!(back.episode_id, "ep-empty");
    }

    #[test]
    fn discard_current_clears_state_no_file() {
        let mut fx = Fixture::new(2, 2);
        fx.writer.start_episode("ep-x", None, "now").unwrap();
        fx.writer.record_step(fx.make_step(0, 0)).unwrap();
        fx.writer.discard_current();
        assert!(!fx.writer.has_episode_in_progress());
        // No file should exist for ep-x.
        let path = fx.writer.directory().join("ep-x.json");
        assert!(!path.exists());
        // Discard is idempotent.
        fx.writer.discard_current();
    }

    #[test]
    fn directory_is_created_on_first_save() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("deeply").join("nested").join("trajs");
        assert!(!nested.exists());
        let mut w = TrajectoryWriter::new(&nested, "minecraft", "schema-x", 2, 2);
        w.start_episode("ep1", None, "now").unwrap();
        w.record_step(StepV2 {
            tick: 0,
            obs: vec![0.0, 0.0],
            action_id: 0,
            policy_target: vec![0.5, 0.5],
            value_target: 0.0,
            reward: 0.0,
            terminated: false,
            truncated: false,
        })
        .unwrap();
        w.finalize_and_save("now").unwrap();
        assert!(nested.exists());
        assert!(nested.join("ep1.json").exists());
    }

    #[test]
    fn accessors_report_constructor_args() {
        let w = TrajectoryWriter::new("/tmp/x", "forge", "sid", 7, 11);
        assert_eq!(w.obs_dim(), 7);
        assert_eq!(w.action_count(), 11);
        assert_eq!(w.directory(), std::path::Path::new("/tmp/x"));
        assert!(!w.has_episode_in_progress());
        assert_eq!(w.current_len(), 0);
    }
}
