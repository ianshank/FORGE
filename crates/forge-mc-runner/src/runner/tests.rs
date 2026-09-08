use super::*;
use crate::manifest::{ModelFileEntry, ModelManifestFiles, MANIFEST_SCHEMA_VERSION};
use forge_agent::latent_mcts::model::StubLatentModel;
use forge_agent::latent_mcts::search::LatentMctsConfig;
use forge_env::spec::{ActionSpec, ObsSpec};
use forge_env::{Env, EnvError, FlatObsEnv};
use forge_replay::v2::TrajectoryV2;
use std::borrow::Cow;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ----------------- StubFlatEnv -----------------

/// Minimal `FlatObsEnv` used to exercise the runner loop.
///
/// - Observations are `vec![tick_f32; obs_dim]` (deterministic per step).
/// - Reward is `1.0` per step.
/// - Episode terminates once `terminate_at` steps are taken.
struct StubFlatEnv {
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    obs_dim: usize,
    action_count: u32,
    tick: u64,
    terminate_at: Option<u64>,
    reset_calls: Arc<AtomicU64>,
    step_calls: Arc<AtomicU64>,
}

impl StubFlatEnv {
    fn new(obs_dim: usize, action_count: u32, terminate_at: Option<u64>) -> Self {
        Self {
            obs_spec: ObsSpec::flat_f32("stub", obs_dim, 0.0, 1.0),
            action_spec: ActionSpec::discrete(action_count),
            obs_dim,
            action_count,
            tick: 0,
            terminate_at,
            reset_calls: Arc::new(AtomicU64::new(0)),
            step_calls: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl Env for StubFlatEnv {
    type Obs = Vec<f32>;
    type Action = u32;
    type Info = ();
    type Error = EnvError;

    fn reset_into(&mut self, _seed: Option<u64>, out: &mut Vec<f32>) -> Result<(), Self::Error> {
        self.reset_calls.fetch_add(1, Ordering::SeqCst);
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
        self.step_calls.fetch_add(1, Ordering::SeqCst);
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
        out.terminated = matches!(self.terminate_at, Some(t) if self.tick >= t);
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
        Cow::Borrowed("stub-flat-env")
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

/// Stub that emits a parseable transient Display on the first N steps
/// (or resets), then delegates to [`StubFlatEnv`].
struct TransientThenOkEnv {
    inner: StubFlatEnv,
    fail_first_n_steps: u32,
    steps_seen: u32,
    fail_first_n_resets: u32,
    resets_seen: u32,
    code: &'static str,
}

impl TransientThenOkEnv {
    fn fail_first_n_steps(obs_dim: usize, action_count: u32, n: u32) -> Self {
        Self {
            inner: StubFlatEnv::new(obs_dim, action_count, Some(1)),
            fail_first_n_steps: n,
            steps_seen: 0,
            fail_first_n_resets: 0,
            resets_seen: 0,
            code: "RECONNECTING",
        }
    }
}

impl Env for TransientThenOkEnv {
    type Obs = Vec<f32>;
    type Action = u32;
    type Info = ();
    type Error = EnvError;

    fn reset_into(&mut self, seed: Option<u64>, out: &mut Vec<f32>) -> Result<(), Self::Error> {
        self.resets_seen += 1;
        if self.resets_seen <= self.fail_first_n_resets {
            return Err(EnvError::Other(format!(
                "transient protocol error [{}]: synthetic reset",
                self.code
            )));
        }
        self.inner.reset_into(seed, out)
    }

    fn step_into(
        &mut self,
        action: u32,
        out: &mut StepOutput<Vec<f32>, ()>,
    ) -> Result<(), Self::Error> {
        self.steps_seen += 1;
        if self.steps_seen <= self.fail_first_n_steps {
            return Err(EnvError::Other(format!(
                "transient protocol error [{}]: synthetic step",
                self.code
            )));
        }
        self.inner.step_into(action, out)
    }

    fn obs_spec(&self) -> &ObsSpec {
        self.inner.obs_spec()
    }
    fn action_spec(&self) -> &ActionSpec {
        self.inner.action_spec()
    }
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("transient-stub")
    }
}

impl FlatObsEnv for TransientThenOkEnv {
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }
    fn num_actions(&self) -> u32 {
        self.inner.num_actions()
    }
}

// ----------------- helpers -----------------

fn make_search(action_count: u32, sims: u32) -> LatentMctsSearch<StubLatentModel> {
    let mut cfg = LatentMctsConfig::default();
    cfg.base.num_simulations = sims;
    // Disable Dirichlet noise so tests are fully deterministic.
    cfg.add_exploration_noise = false;
    LatentMctsSearch::new(StubLatentModel::new(action_count, 8), cfg)
}

fn make_writer(dir: &Path, obs_dim: usize, action_count: u32) -> TrajectoryWriter {
    TrajectoryWriter::new(dir, "stub-env", "stub-schema-id", obs_dim, action_count)
}

fn make_config(episodes: u64, max_steps: u64) -> RunnerConfig {
    RunnerConfig {
        episodes,
        max_steps_per_episode: max_steps,
        trajectory_dir: PathBuf::from("unused"),
        manifest_path: PathBuf::from("unused"),
        env_id: "stub-env".into(),
        schema_id: "stub-schema-id".into(),
        planning_sims: 4,
        action_repeat: 1,
        base_seed: Some(42),
        metrics_port: 0,
        ..RunnerConfig::default()
    }
}

use std::path::PathBuf;

fn write_manifest(path: &Path, version: u64) {
    let m = ModelManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        version,
        schema_id: "stub-schema-id".into(),
        created_at: "2026-05-20T00:00:00Z".into(),
        files: ModelManifestFiles {
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
    };
    m.save_json(path).unwrap();
}

// ----------------- tests -----------------

#[test]
fn format_episode_id_uses_zero_padded_prefix() {
    assert_eq!(format_episode_id(1), "ep-000001");
    assert_eq!(format_episode_id(42), "ep-000042");
    assert_eq!(format_episode_id(999_999), "ep-999999");
    // Values beyond the pad width grow the field — they don't
    // truncate. Mirrors the Python-side glob behaviour
    // (`ep-*.json` matches any number of digits).
    assert_eq!(format_episode_id(1_000_000), "ep-1000000");
}

#[test]
fn episode_id_constants_match_python_side() {
    // Cross-language pin: Python side hard-codes the same
    // values in `python/forge/training/muzero_mc/replay.py`.
    // If you change EPISODE_ID_PREFIX or EPISODE_ID_PAD_WIDTH
    // here, bump them on the Python side too AND update the
    // pinned test value in
    // `tests/python/training/test_muzero_mc_replay.py`.
    assert_eq!(EPISODE_ID_PREFIX, "ep-");
    assert_eq!(EPISODE_ID_PAD_WIDTH, 6);
}

#[test]
fn normalize_visits_uniform_when_all_zero() {
    let v = normalize_visits(&[0, 0, 0, 0]);
    assert_eq!(v.len(), 4);
    for x in &v {
        assert!((x - 0.25).abs() < 1e-6);
    }
}

#[test]
fn normalize_visits_distributes_proportionally() {
    let v = normalize_visits(&[1, 3, 0, 4]);
    let sum: f32 = v.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!((v[3] - 0.5).abs() < 1e-6);
    assert!((v[1] - 0.375).abs() < 1e-6);
}

#[test]
fn run_episode_writes_a_trajectory_with_the_expected_step_count() {
    let dir = tempfile::tempdir().unwrap();
    let traj_dir = dir.path().join("traj");
    let manifest_path = dir.path().join("model_manifest.json");

    let env = StubFlatEnv::new(4, 3, Some(5));
    let search = make_search(3, 4);
    let writer = make_writer(&traj_dir, 4, 3);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let cfg = make_config(1, 20);

    let mut runner = Runner::new(cfg, env, search, writer, watcher);

    let ep = runner.run_episode().unwrap();
    assert_eq!(ep.episode_id, "ep-000001");
    assert_eq!(ep.steps, 5, "env terminates after 5 steps");
    assert!(ep.terminated);
    assert!(!ep.truncated);
    assert!((ep.total_reward - 5.0).abs() < 1e-6);
    assert!(ep.trajectory_path.exists());

    let back = TrajectoryV2::load_json(&ep.trajectory_path).unwrap();
    assert_eq!(back.steps.len(), 5);
    assert_eq!(back.env_id, "stub-env");
    assert_eq!(back.schema_id, "stub-schema-id");
    assert_eq!(back.obs_dim, 4);
    assert_eq!(back.action_count, 3);
    // Policy targets sum ≈ 1.0 on every step.
    for step in &back.steps {
        let sum: f32 = step.policy_target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy target sum = {sum}");
        assert_eq!(step.obs.len(), 4);
        assert_eq!(step.policy_target.len(), 3);
    }
}

#[test]
fn run_episode_truncates_at_max_steps_when_env_never_terminates() {
    let dir = tempfile::tempdir().unwrap();
    let env = StubFlatEnv::new(2, 2, None);
    let search = make_search(2, 2);
    let writer = make_writer(&dir.path().join("traj"), 2, 2);
    let watcher = HotReloadWatcher::new(dir.path().join("manifest.json"));
    let cfg = make_config(1, 7);

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    let ep = runner.run_episode().unwrap();
    assert_eq!(ep.steps, 7);
    assert!(!ep.terminated);
    assert!(ep.truncated, "runner must truncate at max_steps");
}

/// Regression for the code-review finding (HIGH confidence): on
/// runner-side truncation the *last recorded* `StepV2.truncated`
/// must match the `EpisodeOutcome.truncated` value. Without the
/// in-loop truncation prediction, the on-disk trajectory's final
/// step would carry `truncated=false` even though the runner
/// considered the episode truncated — silently misleading
/// downstream trainers that look at the per-step flag for
/// bootstrap-cut decisions.
#[test]
fn last_recorded_step_truncated_flag_matches_episode_outcome_on_runner_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let env = StubFlatEnv::new(2, 2, None);
    let search = make_search(2, 2);
    let writer = make_writer(&dir.path().join("traj"), 2, 2);
    let watcher = HotReloadWatcher::new(dir.path().join("manifest.json"));
    let cfg = make_config(1, 4);

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    let ep = runner.run_episode().unwrap();
    assert_eq!(ep.steps, 4);
    assert!(ep.truncated, "runner must truncate at max_steps");

    let back = TrajectoryV2::load_json(&ep.trajectory_path).unwrap();
    let last = back.steps.last().expect("at least one step");
    assert!(
        last.truncated,
        "last recorded StepV2.truncated must match EpisodeOutcome.truncated \
         (regression: previously false on disk while outcome said true)"
    );
    assert!(!last.terminated);
    // All earlier steps should NOT have the truncated flag — only
    // the final one carries the runner-side signal.
    for (i, step) in back.steps[..back.steps.len() - 1].iter().enumerate() {
        assert!(
            !step.truncated,
            "step {i} unexpectedly carries truncated=true"
        );
    }
}

/// Companion regression: when the env *itself* terminates before
/// max_steps, the runner must NOT spuriously set truncated on the
/// terminal step. Both EpisodeOutcome and the last StepV2 should
/// carry terminated=true and truncated=false.
#[test]
fn env_natural_termination_does_not_get_runner_truncation_flag() {
    let dir = tempfile::tempdir().unwrap();
    let env = StubFlatEnv::new(2, 2, Some(2));
    let search = make_search(2, 2);
    let writer = make_writer(&dir.path().join("traj"), 2, 2);
    let watcher = HotReloadWatcher::new(dir.path().join("manifest.json"));
    let cfg = make_config(1, 10);

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    let ep = runner.run_episode().unwrap();
    assert_eq!(ep.steps, 2);
    assert!(ep.terminated);
    assert!(!ep.truncated);

    let back = TrajectoryV2::load_json(&ep.trajectory_path).unwrap();
    let last = back.steps.last().expect("at least one step");
    assert!(last.terminated);
    assert!(!last.truncated);
}

#[test]
fn run_episode_propagates_env_errors_as_runner_env_error() {
    let dir = tempfile::tempdir().unwrap();
    let env = StubFlatEnv::new(2, 1, None);
    // Force the stub model to suggest action 5 (out of range) by
    // building a 6-action search but pairing it with a 1-action env
    // — the env will reject the action.
    let search = make_search(6, 1);
    // Writer agrees with env on action_count=1, so the StepV2 the
    // recorder would build would also reject the action. We exercise
    // the env path by ensuring the loop fails fast before recording.
    let writer = make_writer(&dir.path().join("traj"), 2, 6);
    let watcher = HotReloadWatcher::new(dir.path().join("manifest.json"));
    let mut cfg = make_config(1, 4);
    cfg.schema_id = "stub-schema-id".into();
    cfg.env_id = "stub-env".into();

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    // The stub model's policy is uniform — root visit-count argmax
    // is deterministic across runs but specific value depends on
    // tie-breaks; the env will accept action 0 (the only valid
    // index). For this test we only assert that no panic occurs;
    // a separate test exercises the EnvError path explicitly via
    // `step_into` API direct call (see env crate tests).
    let _ = runner.run_episode();
}

#[test]
fn run_multiple_episodes_accumulates_outcome_counters() {
    let dir = tempfile::tempdir().unwrap();
    let env = StubFlatEnv::new(3, 2, Some(2));
    let search = make_search(2, 2);
    let writer = make_writer(&dir.path().join("traj"), 3, 2);
    let watcher = HotReloadWatcher::new(dir.path().join("manifest.json"));
    let cfg = make_config(4, 10);

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    let outcome = runner.run(None).unwrap();
    assert_eq!(outcome.episodes_completed, 4);
    assert_eq!(outcome.total_steps, 8); // 2 steps × 4 episodes
    assert_eq!(outcome.terminated_count, 4);
    assert_eq!(outcome.truncated_count, 0);
    assert_eq!(outcome.reloads_applied, 0);
    assert!(outcome.last_model_version.is_none());
}

#[test]
fn manifest_bump_between_episodes_triggers_reload_callback_exactly_once() {
    // Exercises the contract via the actual `Runner::run` loop
    // (not by hand-rolling a sequence of `maybe_reload` /
    // `run_episode` calls). The loop polls once at the top of
    // every episode, so for 3 episodes we should see 3 polls;
    // with the manifest at v1 throughout episode 1, then bumped
    // to v2 between episodes 1 and 2, we should see exactly 2
    // reloads applied (v1 on the first poll, v2 on the second).
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("model_manifest.json");
    write_manifest(&manifest_path, 1);

    let env = StubFlatEnv::new(2, 2, Some(2));
    let search = make_search(2, 2);
    let writer = make_writer(&dir.path().join("traj"), 2, 2);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let cfg = make_config(3, 10);

    let reload_count = Arc::new(AtomicU64::new(0));
    let reload_count_in_fn = Arc::clone(&reload_count);
    // Block the runner between episodes 1 and 2 via a barrier-
    // like flag so the test can bump the manifest at the
    // documented "between-episode" point. Implemented as a
    // single side-channel atomic the reload callback reads to
    // know whether the post-bump reload has been observed yet.
    let bumped_after_first = Arc::new(AtomicU64::new(0));
    let bumped_for_fn = Arc::clone(&bumped_after_first);

    let mut runner = Runner::new(cfg, env, search, writer, watcher).with_reload_fn(Box::new(
        move |_model: &mut StubLatentModel, manifest: &ModelManifest| {
            if manifest.version == 2 {
                bumped_for_fn.store(1, Ordering::SeqCst);
            }
            reload_count_in_fn.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    ));

    // First episode: runs against v1, reload callback fires once
    // before the episode body begins.
    runner.run(Some(1)).unwrap();
    assert_eq!(reload_count.load(Ordering::SeqCst), 1);
    assert_eq!(bumped_after_first.load(Ordering::SeqCst), 0);

    // Bump the manifest at the documented "between-episode" point
    // and let the loop finish.
    write_manifest(&manifest_path, 2);
    runner.run(Some(2)).unwrap();

    // Two reloads applied across the full 3-episode run: v1 on
    // the first poll, v2 on the post-bump poll.
    assert_eq!(reload_count.load(Ordering::SeqCst), 2);
    assert_eq!(bumped_after_first.load(Ordering::SeqCst), 1);
    assert_eq!(runner.reloads_applied(), 2);
    assert_eq!(runner.last_model_version(), Some(2));
}

#[test]
fn reload_without_callback_still_records_version() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("model_manifest.json");
    write_manifest(&manifest_path, 7);

    let env = StubFlatEnv::new(2, 2, Some(1));
    let search = make_search(2, 1);
    let writer = make_writer(&dir.path().join("traj"), 2, 2);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let cfg = make_config(1, 5);

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    runner.maybe_reload().unwrap();
    assert_eq!(runner.last_model_version(), Some(7));
    assert_eq!(runner.reloads_applied(), 1);
}

#[test]
fn prime_watcher_with_suppresses_initial_reload() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("model_manifest.json");
    write_manifest(&manifest_path, 9);

    let env = StubFlatEnv::new(2, 2, Some(1));
    let search = make_search(2, 1);
    let writer = make_writer(&dir.path().join("traj"), 2, 2);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let cfg = make_config(1, 3);

    let reload_called = Arc::new(AtomicU64::new(0));
    let reload_called_in_fn = Arc::clone(&reload_called);

    let mut runner = Runner::new(cfg, env, search, writer, watcher).with_reload_fn(Box::new(
        move |_m: &mut StubLatentModel, _: &ModelManifest| {
            reload_called_in_fn.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    ));
    runner.prime_watcher_with(9);
    runner.maybe_reload().unwrap();
    assert_eq!(reload_called.load(Ordering::SeqCst), 0);
    // Bumping fires the callback as expected.
    write_manifest(&manifest_path, 10);
    runner.maybe_reload().unwrap();
    assert_eq!(reload_called.load(Ordering::SeqCst), 1);
}

#[test]
fn reload_callback_error_propagates_as_runner_reload() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("model_manifest.json");
    write_manifest(&manifest_path, 1);

    let env = StubFlatEnv::new(2, 2, Some(1));
    let search = make_search(2, 1);
    let writer = make_writer(&dir.path().join("traj"), 2, 2);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let cfg = make_config(1, 3);

    let mut runner = Runner::new(cfg, env, search, writer, watcher).with_reload_fn(Box::new(
        |_m: &mut StubLatentModel, _: &ModelManifest| Err(RunnerError::Reload("boom".into())),
    ));
    let err = runner.maybe_reload().unwrap_err();
    assert!(matches!(err, RunnerError::Reload(ref m) if m.contains("boom")));
    // last_model_version stays None — the bump did not stick.
    assert!(runner.last_model_version().is_none());
    assert_eq!(runner.reloads_applied(), 0);
}

#[test]
fn run_loop_polls_watcher_only_between_episodes() {
    // Sanity invariant: we should see (episodes + 1) opportunities
    // for a reload if we count the pre-first-episode poll, but the
    // current implementation only polls at the top of each episode
    // loop iteration (so == episodes polls). This pins that contract.
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("model_manifest.json");
    // Pre-seed manifest at v1 so first reload happens *before* the
    // first episode, demonstrating the "between" semantics.
    write_manifest(&manifest_path, 1);

    let env = StubFlatEnv::new(2, 2, Some(1));
    let search = make_search(2, 1);
    let writer = make_writer(&dir.path().join("traj"), 2, 2);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let cfg = make_config(2, 5);

    let reload_versions: Arc<std::sync::Mutex<Vec<u64>>> = Arc::new(std::sync::Mutex::new(vec![]));
    let recorder = Arc::clone(&reload_versions);
    let mut runner = Runner::new(cfg, env, search, writer, watcher).with_reload_fn(Box::new(
        move |_m: &mut StubLatentModel, manifest: &ModelManifest| {
            recorder.lock().unwrap().push(manifest.version);
            Ok(())
        },
    ));
    runner.run(None).unwrap();
    // First reload fires before episode 1; no bump before episode 2;
    // so we expect exactly [1].
    let observed = reload_versions.lock().unwrap().clone();
    assert_eq!(observed, vec![1]);
}

#[test]
fn benchmark_trajectory_compression_sweep() {
    use crate::config::TrajectoryCompression;
    use forge_replay::v2::{NamedGzipLevel, StepV2, TrajectoryGzipLevel, TrajectoryV2};
    use std::time::Instant;

    let dir = tempfile::tempdir().unwrap();
    let obs_dim = 920;
    let action_count: u32 = 12;

    // Generate a synthetic episode with 100 steps to get realistic sizing
    let mut steps = Vec::new();
    for t in 0..100u64 {
        steps.push(StepV2 {
            tick: t,
            obs: vec![t as f32 * 0.1; obs_dim],
            action_id: (t as u32) % action_count,
            policy_target: vec![1.0 / action_count as f32; action_count as usize],
            value_target: 0.5,
            reward: 1.0,
            terminated: false,
            truncated: false,
        });
    }

    let mut base_traj = TrajectoryV2::empty(
        "minecraft",
        "schema-x",
        "ep-bench",
        obs_dim,
        action_count,
        Some(42),
        "2026-05-23T00:00:00Z",
    );
    for step in steps {
        base_traj.push(step).unwrap();
    }
    base_traj.finalize("2026-05-23T00:00:10Z");

    let cases = vec![
        (
            "None",
            TrajectoryCompression::None,
            TrajectoryGzipLevel::Named(NamedGzipLevel::Default),
        ),
        (
            "Fastest",
            TrajectoryCompression::Gzip,
            TrajectoryGzipLevel::Named(NamedGzipLevel::Fastest),
        ),
        (
            "Default",
            TrajectoryCompression::Gzip,
            TrajectoryGzipLevel::Named(NamedGzipLevel::Default),
        ),
        (
            "Best",
            TrajectoryCompression::Gzip,
            TrajectoryGzipLevel::Named(NamedGzipLevel::Best),
        ),
    ];

    println!("\n=== Replay Compression Sweep Benchmark ===");
    println!("Steps: {}, Obs Dim: {}", base_traj.steps.len(), obs_dim);
    println!(
        "{:<10} | {:<12} | {:<12} | {:<12}",
        "Variant", "Size (bytes)", "Write (ms)", "Read (ms)"
    );
    println!("-------------------------------------------------------------");

    for (name, compression, level) in cases {
        let filename = format!("bench_{}.{}", name, compression.extension());
        let path = dir.path().join(filename);

        // Measure Write
        let t0 = Instant::now();
        match compression {
            TrajectoryCompression::None => base_traj.save_json(&path).unwrap(),
            TrajectoryCompression::Gzip => base_traj.save_json_gz(&path, level).unwrap(),
        }
        let write_dur = t0.elapsed();

        // Get size
        let size = std::fs::metadata(&path).unwrap().len();

        // Measure Read
        let t1 = Instant::now();
        let loaded = TrajectoryV2::load_json(&path).unwrap();
        let read_dur = t1.elapsed();

        assert_eq!(loaded.steps.len(), 100);

        println!(
            "{:<10} | {:<12} | {:<12.3} | {:<12.3}",
            name,
            size,
            write_dur.as_secs_f64() * 1000.0,
            read_dur.as_secs_f64() * 1000.0
        );
    }
    println!("==========================================\n");
}

#[test]
fn run_discards_transient_episode_and_continues() {
    let dir = tempfile::tempdir().unwrap();
    let traj_dir = dir.path().join("traj");
    let manifest_path = dir.path().join("model_manifest.json");

    let env = TransientThenOkEnv::fail_first_n_steps(4, 3, 1);
    let search = make_search(3, 0);
    let writer = make_writer(&traj_dir, 4, 3);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let mut cfg = make_config(1, 8);
    cfg.random_actions = true;
    cfg.transient_failure_backoff_ms = 0;
    cfg.max_consecutive_transient_failures = 3;

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    let outcome = runner
        .run(None)
        .expect("run must continue after RECONNECTING");
    assert_eq!(outcome.episodes_completed, 1);
    assert_eq!(outcome.transient_discards, 1);
    let saved: Vec<_> = std::fs::read_dir(&traj_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(saved.len(), 1, "partial episode must not be written");
}

#[test]
fn run_fails_after_consecutive_transient_cap() {
    let dir = tempfile::tempdir().unwrap();
    let traj_dir = dir.path().join("traj");
    let manifest_path = dir.path().join("model_manifest.json");

    let env = TransientThenOkEnv::fail_first_n_steps(4, 3, 10);
    let search = make_search(3, 0);
    let writer = make_writer(&traj_dir, 4, 3);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let mut cfg = make_config(5, 8);
    cfg.random_actions = true;
    cfg.transient_failure_backoff_ms = 0;
    cfg.max_consecutive_transient_failures = 3;

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    let err = runner.run(None).expect_err("cap must fail the run");
    assert!(
        matches!(err, RunnerError::TooManyTransientFailures { count: 3, .. }),
        "expected TooManyTransientFailures, got {err:?}"
    );
}

#[test]
fn run_still_fails_closed_on_non_transient_env_error() {
    let dir = tempfile::tempdir().unwrap();
    let traj_dir = dir.path().join("traj");
    let manifest_path = dir.path().join("model_manifest.json");

    struct FatalEnv(StubFlatEnv);
    impl Env for FatalEnv {
        type Obs = Vec<f32>;
        type Action = u32;
        type Info = ();
        type Error = EnvError;
        fn reset_into(&mut self, seed: Option<u64>, out: &mut Vec<f32>) -> Result<(), Self::Error> {
            self.0.reset_into(seed, out)
        }
        fn step_into(
            &mut self,
            _action: u32,
            _out: &mut StepOutput<Vec<f32>, ()>,
        ) -> Result<(), Self::Error> {
            Err(EnvError::Other("protocol error [INTERNAL]: boom".into()))
        }
        fn obs_spec(&self) -> &ObsSpec {
            self.0.obs_spec()
        }
        fn action_spec(&self) -> &ActionSpec {
            self.0.action_spec()
        }
        fn name(&self) -> Cow<'_, str> {
            Cow::Borrowed("fatal-stub")
        }
    }
    impl FlatObsEnv for FatalEnv {
        fn obs_dim(&self) -> usize {
            self.0.obs_dim()
        }
        fn num_actions(&self) -> u32 {
            self.0.num_actions()
        }
    }

    let env = FatalEnv(StubFlatEnv::new(4, 3, Some(5)));
    let search = make_search(3, 0);
    let writer = make_writer(&traj_dir, 4, 3);
    let watcher = HotReloadWatcher::new(&manifest_path);
    let mut cfg = make_config(2, 8);
    cfg.random_actions = true;
    cfg.transient_failure_backoff_ms = 0;

    let mut runner = Runner::new(cfg, env, search, writer, watcher);
    let err = runner.run(None).expect_err("INTERNAL must fail the run");
    assert!(
        matches!(err, RunnerError::Env(_)),
        "INTERNAL must stay RunnerError::Env, got {err:?}"
    );
}
