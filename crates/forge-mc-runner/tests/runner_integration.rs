//! End-to-end integration test for [`forge_mc_runner::Runner`].
//!
//! Exercises the runner's full public API through the same path a
//! production caller would:
//!
//! 1. Bootstrap a `model_manifest.json` at `version=1` on disk.
//! 2. Construct the runner with a `FlatObsEnv` stub + `LatentMctsSearch`
//!    over a `StubLatentModel`, a real [`TrajectoryWriter`], and a
//!    [`HotReloadWatcher`].
//! 3. Drive two episodes via [`Runner::run`].
//! 4. Between iterations of the test, bump the manifest to v2 →
//!    confirm the reload callback fires and the runner advances its
//!    `last_model_version`.
//! 5. Read each trajectory file back from disk and assert the
//!    `TrajectoryV2` invariants (obs_dim, action_count, policy-target
//!    sum ≈ 1, step count matches env termination).
//!
//! All env / model state lives in this file so the test exercises the
//! re-exported public surface of `forge-mc-runner`, not its `#[cfg(test)]`
//! internals. The stub env mirrors the shape of `runner.rs::tests`'
//! `StubFlatEnv` but with `terminate_at = Some(3)` for fast turnaround.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use forge_agent::latent_mcts::model::{LatentForwardModel, StubLatentModel};
use forge_agent::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch};
use forge_env::spec::{ActionSpec, ObsSpec};
use forge_env::{Env, EnvError, FlatObsEnv, StepOutput};
use forge_mc_runner::{
    HotReloadWatcher, ModelFileEntry, ModelManifest, ModelManifestFiles, Runner, RunnerConfig,
    RunnerError, TrajectoryWriter,
};
use forge_replay::v2::TrajectoryV2;

const SCHEMA_ID: &str = "stub-schema-7f";

fn write_manifest(path: &std::path::Path, version: u64) {
    let m = ModelManifest::new(
        version,
        SCHEMA_ID,
        "2026-05-20T00:00:00Z",
        ModelManifestFiles {
            representation: ModelFileEntry {
                path: "r.onnx".into(),
                sha256: "a".repeat(64),
            },
            dynamics: ModelFileEntry {
                path: "d.onnx".into(),
                sha256: "b".repeat(64),
            },
            prediction: ModelFileEntry {
                path: "p.onnx".into(),
                sha256: "c".repeat(64),
            },
        },
    );
    m.save_json(path).unwrap();
}

struct StubFlatEnv {
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    obs_dim: usize,
    action_count: u32,
    tick: u64,
    terminate_at: u64,
}

impl StubFlatEnv {
    fn new(obs_dim: usize, action_count: u32, terminate_at: u64) -> Self {
        Self {
            obs_spec: ObsSpec::flat_f32("integration-stub", obs_dim, 0.0, 1.0),
            action_spec: ActionSpec::discrete(action_count),
            obs_dim,
            action_count,
            tick: 0,
            terminate_at,
        }
    }
}

impl Env for StubFlatEnv {
    type Obs = Vec<f32>;
    type Action = u32;
    type Info = ();
    type Error = EnvError;

    fn reset_into(&mut self, _seed: Option<u64>, out: &mut Vec<f32>) -> Result<(), Self::Error> {
        self.tick = 0;
        out.clear();
        out.resize(self.obs_dim, 0.0);
        Ok(())
    }

    fn step_into(
        &mut self,
        action: u32,
        out: &mut StepOutput<Vec<f32>, ()>,
    ) -> Result<(), Self::Error> {
        if action >= self.action_count {
            return Err(EnvError::InvalidAction {
                action_id: action,
                space_n: self.action_count,
            });
        }
        self.tick += 1;
        out.obs.clear();
        out.obs.resize(self.obs_dim, self.tick as f32);
        out.reward = 1.0;
        out.terminated = self.tick >= self.terminate_at;
        out.truncated = false;
        Ok(())
    }

    fn obs_spec(&self) -> &ObsSpec {
        &self.obs_spec
    }
    fn action_spec(&self) -> &ActionSpec {
        &self.action_spec
    }
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("integration-stub")
    }
}

impl FlatObsEnv for StubFlatEnv {
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }
    fn num_actions(&self) -> u32 {
        self.action_count
    }
}

fn build_runner(
    config: RunnerConfig,
    obs_dim: usize,
    action_count: u32,
    terminate_at: u64,
) -> Runner<StubFlatEnv, StubLatentModel> {
    let env = StubFlatEnv::new(obs_dim, action_count, terminate_at);

    let mut mcts_cfg = LatentMctsConfig::default();
    mcts_cfg.base.num_simulations = 4;
    mcts_cfg.add_exploration_noise = false;
    let search = LatentMctsSearch::new(StubLatentModel::new(action_count, 8), mcts_cfg);

    let writer = TrajectoryWriter::new(
        &config.trajectory_dir,
        &config.env_id,
        &config.schema_id,
        obs_dim,
        action_count,
    );
    let watcher = HotReloadWatcher::new(&config.manifest_path);
    Runner::new(config, env, search, writer, watcher)
}

#[test]
fn end_to_end_two_episodes_with_manifest_bump() {
    let dir = tempfile::tempdir().unwrap();
    let traj_dir = dir.path().join("traj");
    let manifest_path = dir.path().join("model_manifest.json");
    write_manifest(&manifest_path, 1);

    let cfg = RunnerConfig {
        episodes: 2,
        max_steps_per_episode: 16,
        trajectory_dir: traj_dir.clone(),
        manifest_path: manifest_path.clone(),
        env_id: "integration-stub".into(),
        schema_id: SCHEMA_ID.into(),
        planning_sims: 4,
        action_repeat: 1,
        base_seed: Some(7),
        metrics_port: 0,
    };
    cfg.validate().unwrap();

    // Track reload calls + the manifest versions seen.
    let reload_count = Arc::new(AtomicU64::new(0));
    let observed_versions: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let reload_count_clone = Arc::clone(&reload_count);
    let observed_versions_clone = Arc::clone(&observed_versions);

    let mut runner = build_runner(cfg, 3, 2, 3).with_reload_fn(Box::new(
        move |_m: &mut StubLatentModel, manifest: &ModelManifest| {
            reload_count_clone.fetch_add(1, Ordering::SeqCst);
            observed_versions_clone
                .lock()
                .unwrap()
                .push(manifest.version);
            Ok(())
        },
    ));

    // Run episode 1 (a reload to v1 fires before the episode begins).
    let outcome_phase1 = runner.run(Some(1)).unwrap();
    assert_eq!(outcome_phase1.episodes_completed, 1);
    assert_eq!(outcome_phase1.terminated_count, 1);
    assert_eq!(outcome_phase1.total_steps, 3);
    assert_eq!(outcome_phase1.reloads_applied, 1);
    assert_eq!(outcome_phase1.last_model_version, Some(1));

    // Bump the manifest to v2 → next run picks it up between episodes.
    write_manifest(&manifest_path, 2);

    // Run episode 2.
    let outcome_phase2 = runner.run(Some(1)).unwrap();
    assert_eq!(outcome_phase2.episodes_completed, 1);
    assert_eq!(outcome_phase2.terminated_count, 1);
    assert_eq!(outcome_phase2.total_steps, 3);
    assert_eq!(outcome_phase2.reloads_applied, 2);
    assert_eq!(outcome_phase2.last_model_version, Some(2));

    assert_eq!(reload_count.load(Ordering::SeqCst), 2);
    assert_eq!(observed_versions.lock().unwrap().clone(), vec![1, 2]);

    // Inspect on-disk trajectories.
    let ep1_path = traj_dir.join("ep-000001.json");
    let ep2_path = traj_dir.join("ep-000002.json");
    assert!(ep1_path.exists(), "{} missing", ep1_path.display());
    assert!(ep2_path.exists(), "{} missing", ep2_path.display());

    for path in [&ep1_path, &ep2_path] {
        let t = TrajectoryV2::load_json(path).unwrap();
        assert_eq!(t.env_id, "integration-stub");
        assert_eq!(t.schema_id, SCHEMA_ID);
        assert_eq!(t.obs_dim, 3);
        assert_eq!(t.action_count, 2);
        assert_eq!(t.steps.len(), 3, "stub terminates after 3 steps");
        for step in &t.steps {
            assert_eq!(step.obs.len(), 3);
            assert_eq!(step.policy_target.len(), 2);
            let sum: f32 = step.policy_target.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "policy_target sum = {sum}");
        }
        assert!(t.steps.last().unwrap().terminated);
    }

    // No .tmp siblings should linger after atomic save.
    for entry in std::fs::read_dir(&traj_dir).unwrap() {
        let e = entry.unwrap();
        let n = e.file_name();
        let n = n.to_string_lossy();
        assert!(!n.ends_with(".tmp"), "leftover temp file: {n}");
    }
}

#[test]
fn run_propagates_reload_callback_errors() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("model_manifest.json");
    write_manifest(&manifest_path, 1);

    let cfg = RunnerConfig {
        episodes: 1,
        max_steps_per_episode: 8,
        trajectory_dir: dir.path().join("traj"),
        manifest_path: manifest_path.clone(),
        env_id: "integration-stub".into(),
        schema_id: SCHEMA_ID.into(),
        planning_sims: 2,
        action_repeat: 1,
        base_seed: Some(7),
        metrics_port: 0,
    };

    let mut runner = build_runner(cfg, 2, 2, 2).with_reload_fn(Box::new(
        |_m: &mut StubLatentModel, _manifest: &ModelManifest| {
            Err(RunnerError::Reload("simulated reload failure".into()))
        },
    ));

    let err = runner.run(None).unwrap_err();
    assert!(
        matches!(err, RunnerError::Reload(ref msg) if msg.contains("simulated reload failure")),
        "got: {err:?}"
    );
}

/// Sanity check that the stub model + 0-sim search degenerates gracefully:
/// the runner still records a valid trajectory because `normalize_visits`
/// falls back to a uniform policy distribution when all visit counts are 0.
#[test]
fn zero_simulation_search_yields_uniform_policy_target() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("manifest.json"); // missing on purpose

    let cfg = RunnerConfig {
        episodes: 1,
        max_steps_per_episode: 4,
        trajectory_dir: dir.path().join("traj"),
        manifest_path: manifest_path.clone(),
        env_id: "integration-stub".into(),
        schema_id: SCHEMA_ID.into(),
        planning_sims: 0, // recorded in trajectory; planner uses MctsConfig
        action_repeat: 1,
        base_seed: Some(0),
        metrics_port: 0,
    };

    let env = StubFlatEnv::new(2, 4, 2);

    let mut mcts_cfg = LatentMctsConfig::default();
    mcts_cfg.base.num_simulations = 0; // force zero-sim search
    mcts_cfg.add_exploration_noise = false;
    let search = LatentMctsSearch::new(StubLatentModel::new(4, 4), mcts_cfg);
    let writer = TrajectoryWriter::new(&cfg.trajectory_dir, &cfg.env_id, &cfg.schema_id, 2, 4);
    let watcher = HotReloadWatcher::new(&cfg.manifest_path);
    let mut runner = Runner::new(cfg, env, search, writer, watcher);

    let outcome = runner.run(None).unwrap();
    assert_eq!(outcome.episodes_completed, 1);
    assert_eq!(outcome.total_steps, 2);
    let p = runner.config().trajectory_dir.join("ep-000001.json");
    let t = TrajectoryV2::load_json(&p).unwrap();
    for step in &t.steps {
        // 4 actions × 0.25 = 1.0
        for v in &step.policy_target {
            assert!((v - 0.25).abs() < 1e-4, "expected uniform, got {v}");
        }
    }
}

/// Underscore-prefix the trait import to silence unused-warning when this
/// file is checked standalone (the impl is in `model.rs` but the trait
/// must be in scope for `StubLatentModel::action_space_size` etc).
#[allow(dead_code)]
fn _trait_in_scope_check() {
    let m = StubLatentModel::new(2, 4);
    let _ = m.action_space_size();
}
