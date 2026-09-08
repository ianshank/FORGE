//! Drives [`forge_mc_runner::Runner`] against the **real Minecraft wire
//! protocol**.
//!
//! `runner_integration.rs` exercises the same loop against an in-process
//! `StubFlatEnv`, which returns pre-baked observations from a struct
//! field. That covers the loop's control flow but proves nothing about
//! the protocol: no JSON is serialised, no handshake is validated, no
//! action id ever reaches a wire.
//!
//! This test substitutes a real [`MinecraftEnv`] talking to
//! [`MockBot`] — the same scripted server `forge-env-mc`'s own tests
//! use. Everything between the runner and the socket is production
//! code: `ClientMsg`/`ServerMsg` serde, the `Hello` handshake and its
//! mismatch checks, action-map lookup, and the observation buffer
//! contract.
//!
//! Because the mock records what it receives, the assertions can cover
//! ground a stub env structurally cannot: **which action ids the
//! planner actually chose**, and that they arrived in order.

use forge_agent::latent_mcts::model::StubLatentModel;
use forge_agent::latent_mcts::search::{LatentMctsConfig, LatentMctsSearch};
use forge_env::FlatObsEnv;
use forge_env_mc::action_map::{ActionEntry, ActionKind, ActionMap};
use forge_env_mc::config::MinecraftEnvConfig;
use forge_env_mc::protocol::{ClientMsg, ServerMsg, SCHEMA_VERSION};
use forge_env_mc::testing::MockBot;
use forge_env_mc::{McEnvError, MinecraftEnv};
use forge_mc_runner::{HotReloadWatcher, Runner, RunnerConfig, RunnerError, TrajectoryWriter};
use forge_replay::v2::TrajectoryV2;

/// Observation width the mock reports and the runner buffers against.
const OBS_DIM: usize = 4;

/// Latent width for the stub planner. Small: this test is about the
/// wire, not about search quality.
const LATENT_DIM: usize = 8;

/// Planning simulations per step. Enough for the search to produce a
/// non-degenerate visit distribution, small enough to stay fast.
const PLANNING_SIMS: u32 = 4;

fn sample_map() -> ActionMap {
    ActionMap {
        schema_version: 1,
        entries: vec![
            ActionEntry {
                id: 0,
                kind: ActionKind::Noop { ticks: 1 },
            },
            ActionEntry {
                id: 1,
                kind: ActionKind::Jump,
            },
            ActionEntry {
                id: 2,
                kind: ActionKind::Attack,
            },
        ],
    }
}

fn hello_for(map: &ActionMap) -> ServerMsg {
    ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        action_count: map.action_count(),
        obs_dim: OBS_DIM,
        schema_id: map.canonical_sha256(),
        grid_shape: None,
    }
}

/// One `Observation` reply. `terminated` ends the episode.
fn obs(tick: u64, reward: f32, terminated: bool) -> ServerMsg {
    ServerMsg::Observation {
        tick,
        // Distinct per tick so a trajectory assertion can prove the
        // bytes came off the wire rather than from a zeroed buffer.
        obs: vec![tick as f32; OBS_DIM],
        reward,
        terminated,
        truncated: false,
        info: serde_json::json!({}),
    }
}

/// Reply script for one episode of `steps` actions, terminating on the
/// last. Index 0 answers `Reset`; the rest answer `Step`s.
fn episode_script(steps: u64) -> Vec<ServerMsg> {
    let mut replies = vec![obs(0, 0.0, false)];
    for tick in 1..=steps {
        replies.push(obs(tick, 1.0, tick == steps));
    }
    replies
}

fn runner_config(dir: &std::path::Path, schema_id: &str, episodes: u64) -> RunnerConfig {
    RunnerConfig {
        episodes,
        max_steps_per_episode: 32,
        trajectory_dir: dir.join("trajectories"),
        // Deliberately absent: `HotReloadWatcher::poll` treats a missing
        // manifest as "nothing to reload", which is what this test
        // wants. Hot-reload has its own coverage in
        // `runner_integration.rs`.
        manifest_path: dir.join("models").join("model_manifest.json"),
        env_id: "minecraft-mock".into(),
        schema_id: schema_id.into(),
        planning_sims: PLANNING_SIMS,
        metrics_port: 0,
        ..RunnerConfig::default()
    }
}

fn build_runner(config: RunnerConfig, env: MinecraftEnv) -> Runner<MinecraftEnv, StubLatentModel> {
    let action_count = env.num_actions();
    let obs_dim = env.obs_dim();

    let mut mcts_cfg = LatentMctsConfig::default();
    mcts_cfg.base.num_simulations = config.planning_sims;
    mcts_cfg.add_exploration_noise = false;
    let search = LatentMctsSearch::new(StubLatentModel::new(action_count, LATENT_DIM), mcts_cfg);

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

/// The full loop over a real socket: handshake, reset, stepped
/// episode, trajectory on disk — and a recorded client-message
/// sequence proving what the planner actually sent.
#[test]
fn runner_drives_a_full_episode_over_the_real_protocol() {
    const STEPS: u64 = 3;

    let tmp = tempfile::tempdir().expect("tempdir");
    let map = sample_map();
    let schema_id = map.canonical_sha256();

    let bot = MockBot::bind(hello_for(&map), episode_script(STEPS));
    let cfg = MinecraftEnvConfig {
        ws_url: bot.ws_url(),
        ..MinecraftEnvConfig::default()
    };
    let server = bot.run();

    // Handshake happens here, against a real socket.
    let env = MinecraftEnv::connect(cfg, map.clone()).expect("handshake");
    assert_eq!(env.obs_dim(), OBS_DIM);
    assert_eq!(env.num_actions(), map.action_count());

    let config = runner_config(tmp.path(), &schema_id, 1);
    let trajectory_dir = config.trajectory_dir.clone();
    let mut runner = build_runner(config, env);
    let outcome = runner.run(None).expect("run");

    assert_eq!(outcome.episodes_completed, 1);
    assert_eq!(outcome.total_steps, STEPS);
    assert_eq!(
        outcome.terminated_count, 1,
        "the mock's final observation set terminated=true"
    );

    // What the planner actually sent. A stub env cannot show this.
    let sent = server.received();
    assert_eq!(
        sent.len(),
        (STEPS + 1) as usize,
        "one Reset plus one Step per action: {sent:?}"
    );
    assert!(
        matches!(sent[0], ClientMsg::Reset { .. }),
        "episode must open with Reset, got {:?}",
        sent[0]
    );
    let wire_actions: Vec<u32> = sent[1..]
        .iter()
        .enumerate()
        .map(|(i, msg)| match msg {
            ClientMsg::Step { action_id } => *action_id,
            other => panic!("expected Step at index {}, got {other:?}", i + 1),
        })
        .collect();
    server.join();

    // The trajectory must carry the observations the mock actually
    // sent, round-tripped through JSON.
    let files: Vec<_> = std::fs::read_dir(&trajectory_dir)
        .expect("trajectory dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    assert_eq!(files.len(), 1, "one episode, one trajectory: {files:?}");

    let raw = std::fs::read_to_string(&files[0]).expect("read trajectory");
    let traj: TrajectoryV2 = serde_json::from_str(&raw).expect("parse trajectory");
    assert_eq!(traj.obs_dim, OBS_DIM);
    assert_eq!(traj.action_count, map.action_count());
    assert_eq!(traj.schema_id, schema_id);
    assert_eq!(traj.steps.len(), STEPS as usize);

    // The observations must be the ones the mock put on the wire, tick
    // by tick. Asserting only step 0 would prove nothing: its obs is
    // the Reset reply at tick 0, i.e. all zeroes, which is exactly what
    // a freshly-zeroed buffer holds — a runner that never copied the
    // payload would pass. The later ticks are non-zero and distinct.
    let expected_obs: Vec<Vec<f32>> = (0..STEPS).map(|tick| vec![tick as f32; OBS_DIM]).collect();
    let actual_obs: Vec<Vec<f32>> = traj.steps.iter().map(|s| s.obs.clone()).collect();
    assert_eq!(
        actual_obs, expected_obs,
        "trajectory observations must be the floats the mock sent, in order"
    );
    assert!(
        actual_obs.iter().any(|o| o.iter().any(|f| *f != 0.0)),
        "at least one observation must be non-zero, or a zeroed buffer would satisfy this test"
    );

    // The action ids the planner put on the WIRE must be the ones it
    // recorded in the trajectory. Range-checking the wire values alone
    // re-proves what `MinecraftEnv::step_into` already enforces: a
    // regression that serialised a different in-range action would
    // still pass. Only this comparison ties the two sides together.
    let recorded_actions: Vec<u32> = traj.steps.iter().map(|s| s.action_id).collect();
    assert_eq!(
        wire_actions, recorded_actions,
        "each action the planner chose must reach the socket unchanged, in order"
    );
    assert!(
        wire_actions.iter().all(|a| *a < map.action_count()),
        "every action id must fall inside the {} declared actions: {wire_actions:?}",
        map.action_count()
    );
}

/// A `Hello` that disagrees with the action map fails the handshake
/// before any episode runs — so a mismatched bot cannot silently
/// produce a trajectory against the wrong action space.
#[test]
fn handshake_mismatch_fails_before_any_episode() {
    let map = sample_map();
    let wrong_hello = ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        // One more action than the map declares.
        action_count: map.action_count() + 1,
        obs_dim: OBS_DIM,
        schema_id: map.canonical_sha256(),
        grid_shape: None,
    };

    let bot = MockBot::bind(wrong_hello, vec![]);
    let cfg = MinecraftEnvConfig {
        ws_url: bot.ws_url(),
        ..MinecraftEnvConfig::default()
    };
    let server = bot.run();

    let Err(err) = MinecraftEnv::connect(cfg, map) else {
        panic!("action_count mismatch must fail the handshake");
    };
    assert!(
        matches!(err, McEnvError::HandshakeMismatch { .. }),
        "expected HandshakeMismatch, got {err:?}"
    );
    server.join();
}

/// A protocol-level error mid-episode surfaces as
/// [`RunnerError::Env`] rather than panicking the loop.
#[test]
fn mid_episode_protocol_error_surfaces_as_runner_env_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let map = sample_map();
    let schema_id = map.canonical_sha256();

    // Reset succeeds; the first Step is answered with an Error frame.
    let script = vec![
        obs(0, 0.0, false),
        ServerMsg::Error {
            code: "INTERNAL".into(),
            message: "bot lost its connection to the server".into(),
        },
    ];
    let bot = MockBot::bind(hello_for(&map), script);
    let cfg = MinecraftEnvConfig {
        ws_url: bot.ws_url(),
        ..MinecraftEnvConfig::default()
    };
    let server = bot.run();

    let env = MinecraftEnv::connect(cfg, map).expect("handshake");
    let config = runner_config(tmp.path(), &schema_id, 1);
    let mut runner = build_runner(config, env);

    let Err(err) = runner.run(None) else {
        panic!("a protocol error mid-episode must fail the run");
    };
    assert!(
        matches!(err, RunnerError::Env(_)),
        "protocol errors must surface as RunnerError::Env, got {err:?}"
    );
    server.join();
}

/// Mid-episode `RECONNECTING` discards the partial trajectory and
/// continues the run. The next Reset starts a fresh MDP.
#[test]
fn mid_episode_reconnecting_discards_and_continues_run() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let map = sample_map();
    let schema_id = map.canonical_sha256();

    let script = vec![
        obs(0, 0.0, false),
        ServerMsg::Error {
            code: "RECONNECTING".into(),
            message: "mineflayer reconnecting; discard this episode".into(),
        },
        obs(0, 0.0, false),
        obs(1, 1.0, true),
    ];
    let bot = MockBot::bind(hello_for(&map), script);
    let cfg = MinecraftEnvConfig {
        ws_url: bot.ws_url(),
        ..MinecraftEnvConfig::default()
    };
    let server = bot.run();

    let env = MinecraftEnv::connect(cfg, map).expect("handshake");
    let mut config = runner_config(tmp.path(), &schema_id, 1);
    config.random_actions = true;
    config.action_repeat = 1;
    config.transient_failure_backoff_ms = 0;
    config.max_consecutive_transient_failures = 3;
    let mut runner = build_runner(config, env);

    let outcome = runner
        .run(None)
        .expect("RECONNECTING must not fail the run");
    assert!(
        outcome.episodes_completed >= 1,
        "next episode after discard must complete, got {outcome:?}"
    );
    assert_eq!(outcome.transient_discards, 1);
    server.join();
}

/// Three consecutive `RECONNECTING` frames hit the cap and fail the run.
#[test]
fn consecutive_reconnecting_hits_cap_and_fails_run() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let map = sample_map();
    let schema_id = map.canonical_sha256();

    let script = vec![
        obs(0, 0.0, false),
        ServerMsg::Error {
            code: "RECONNECTING".into(),
            message: "1".into(),
        },
        obs(0, 0.0, false),
        ServerMsg::Error {
            code: "BUSY".into(),
            message: "2".into(),
        },
        obs(0, 0.0, false),
        ServerMsg::Error {
            code: "RECONNECTING".into(),
            message: "3".into(),
        },
    ];
    let bot = MockBot::bind(hello_for(&map), script);
    let cfg = MinecraftEnvConfig {
        ws_url: bot.ws_url(),
        ..MinecraftEnvConfig::default()
    };
    let server = bot.run();

    let env = MinecraftEnv::connect(cfg, map).expect("handshake");
    let mut config = runner_config(tmp.path(), &schema_id, 10);
    config.random_actions = true;
    config.action_repeat = 1;
    config.transient_failure_backoff_ms = 0;
    config.max_consecutive_transient_failures = 3;
    let mut runner = build_runner(config, env);

    let err = runner.run(None).expect_err("cap must fail the run");
    assert!(
        matches!(err, RunnerError::TooManyTransientFailures { count: 3, .. }),
        "expected TooManyTransientFailures, got {err:?}"
    );
    server.join();
}
