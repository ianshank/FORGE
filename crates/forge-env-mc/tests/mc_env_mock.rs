//! Integration test against a mock WebSocket server.
//!
//! Drives [`MinecraftEnv`] against [`forge_env_mc::testing::MockBot`],
//! the scripted stand-in for the Node `mc-bot` bridge. The mock lives
//! in the library (behind the `testing` feature) rather than here so
//! downstream crates -- notably `forge-mc-runner` -- can exercise the
//! same real wire protocol without duplicating it.

use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use forge_env::{Env, FlatObsEnv};
use forge_env_mc::action_map::{ActionEntry, ActionKind, ActionMap};
use forge_env_mc::config::MinecraftEnvConfig;
use forge_env_mc::protocol::{ClientMsg, GridShape, ServerMsg, SCHEMA_VERSION};
use forge_env_mc::testing::{MockBot, MockBotHandle};
use forge_env_mc::{McEnvError, MinecraftEnv};
use tungstenite::Message;

const OBS_DIM: usize = 4;

/// Helper — builds a `ServerMsg::Hello` with `grid_shape: None`. Keeps
/// the test surface tight after the v0.5 `grid_shape` field landed
/// on the wire (block-grid cross-check tests have their own builders
/// further down).
fn build_hello(action_count: u32, obs_dim: usize, schema_id: impl Into<String>) -> ServerMsg {
    ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        action_count,
        obs_dim,
        schema_id: schema_id.into(),
        grid_shape: None,
    }
}

/// Join a mock server that is expected to panic, reporting whether it
/// did — **without** disturbing the process-global panic hook.
///
/// `catch_unwind` alone captures the panic. Swapping the hook to
/// silence the expected backtrace would suppress panic diagnostics for
/// every other test libtest runs concurrently in this process, trading
/// one tidy backtrace for the message that would explain an unrelated
/// failure. It would not even silence this one reliably: a short server
/// timeout can fire before the swap executes. The expected backtraces
/// are noise worth living with.
///
/// Every test here that provokes a server-side panic must go through
/// this helper, so the rule has exactly one place to be broken.
fn join_expecting_panic(server: MockBotHandle) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| server.join())).is_err()
}

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

fn cfg_with_url(url: String) -> MinecraftEnvConfig {
    MinecraftEnvConfig {
        ws_url: url,
        ..MinecraftEnvConfig::default()
    }
}

fn cfg_with_url_and_expected(url: String, expected_dim: usize) -> MinecraftEnvConfig {
    let mut c = cfg_with_url(url);
    c.observation.expected_dim = Some(expected_dim);
    c
}

fn cfg_with_expected_schema(
    url: String,
    expected_schema_id: impl Into<String>,
) -> MinecraftEnvConfig {
    let mut c = cfg_with_url(url);
    c.expected_schema_id = Some(expected_schema_id.into());
    c
}

fn obs_msg(tick: u64) -> ServerMsg {
    ServerMsg::Observation {
        tick,
        obs: vec![0.0; OBS_DIM],
        reward: 0.0,
        terminated: false,
        truncated: false,
        info: serde_json::json!({}),
    }
}

#[test]
fn handshake_validates_action_count_and_obs_dim() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url_and_expected(url, OBS_DIM);

    let env = MinecraftEnv::connect(cfg, map.clone()).expect("connect");
    assert_eq!(env.obs_dim(), OBS_DIM);
    assert_eq!(env.num_actions(), map.action_count());
    drop(env);
    server.join();
}

#[test]
fn handshake_rejects_action_count_mismatch() {
    let map = sample_map();
    let hello = build_hello(999, OBS_DIM, "x"); // wrong action_count on purpose
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);

    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected error");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    server.join();
}

#[test]
fn handshake_rejects_schema_version_mismatch() {
    let map = sample_map();
    // Hand-rolled here (not via build_hello) so we can flip the schema
    // version on purpose.
    let hello = ServerMsg::Hello {
        schema_version: SCHEMA_VERSION + 100,
        action_count: map.action_count(),
        obs_dim: OBS_DIM,
        schema_id: "x".into(),
        grid_shape: None,
    };
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected error");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    server.join();
}

#[test]
fn handshake_rejects_schema_id_mismatch_when_expected_set() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, "server-schema");
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_expected_schema(url, "client-schema");
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected error");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    server.join();
}

#[test]
fn reset_and_steps_drive_a_full_short_episode() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let mut replies = vec![obs_msg(0)]; // reply to Reset
    for t in 1..=5u64 {
        replies.push(obs_msg(t));
    }
    let bot = MockBot::bind(hello, replies);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    let r = env.reset(Some(42)).unwrap();
    assert_eq!(r.len(), OBS_DIM);
    for expected_tick in 1..=5u64 {
        let s = env.step(0).unwrap();
        assert_eq!(s.info.tick, expected_tick);
    }
    let _ = env.close();
    server.join();
}

/// The mock records what the client SENT, not just what it did with
/// the replies. That is the capability downstream crates need: a
/// `forge-mc-runner` test can assert which action ids its planner
/// actually chose, which no stub env can show.
#[test]
fn mock_records_the_client_message_sequence() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    // One reply per client message: Reset, then two Steps.
    let bot = MockBot::bind(hello, vec![obs_msg(0), obs_msg(1), obs_msg(2)]);
    let url = bot.ws_url();
    let server = bot.run();

    let mut env = MinecraftEnv::connect(cfg_with_url(url), map).unwrap();
    env.reset(Some(7)).unwrap();
    env.step(2).unwrap();
    env.step(1).unwrap();
    let _ = env.close();

    let sent = server.received();
    assert_eq!(
        sent,
        vec![
            ClientMsg::Reset { seed: Some(7) },
            ClientMsg::Step { action_id: 2 },
            ClientMsg::Step { action_id: 1 },
        ],
        "mock should record the exact ClientMsg sequence, in order"
    );
    server.join();
}

/// Two mocks bound at once must not collide. Guards the bind-and-hold
/// contract: the old helper picked a port, dropped the listener, then
/// re-bound it later, leaving a window for another test to claim it.
#[test]
fn concurrently_bound_mocks_get_distinct_ports() {
    let map = sample_map();
    let hello = || build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let first = MockBot::bind(hello(), vec![]);
    let second = MockBot::bind(hello(), vec![]);
    assert_ne!(
        first.local_addr().port(),
        second.local_addr().port(),
        "each bound mock must hold its own port"
    );
    assert!(first.ws_url().starts_with("ws://127.0.0.1:"));

    // The distinct-ports assertion above would hold even if `bind()`
    // released the port, because the kernel hands out a fresh one each
    // time. This is what actually pins "held": re-binding a live mock's
    // address must fail.
    assert!(
        TcpListener::bind(first.local_addr()).is_err(),
        "bind() must HOLD its port; re-binding it must fail with AddrInUse"
    );
}

/// A genuine server-side fault must reach the test thread. The mock
/// used to swallow every panic from its serving thread, which could
/// let a wire-protocol test pass while the protocol was broken —
/// exactly the failure mode these tests exist to catch.
///
/// Drives a real fault (a read timeout, not a disconnect) and asserts
/// `join()` re-raises it.
#[test]
fn join_surfaces_a_real_server_fault() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    // A reply is scripted, so the mock will wait for a client message
    // that never comes and time out.
    let bot = MockBot::bind(hello, vec![obs_msg(0)]).with_io_timeout(Duration::from_millis(50));
    let url = bot.ws_url();
    let server = bot.run();

    // Complete the handshake, then sit idle. Holding the socket open
    // means the mock sees a timeout rather than a disconnect.
    let (_client, _response) = tungstenite::connect(&url).expect("client connect");

    assert!(
        join_expecting_panic(server),
        "join() must re-raise a real server fault, not swallow it"
    );
}

/// The counterpart: a client that disconnects before the reply script
/// is exhausted is an orderly end to the session, so `join()` must
/// stay quiet. Guards against over-correcting the fix above into
/// spurious failures.
#[test]
fn join_stays_quiet_on_an_orderly_client_disconnect() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    // Two replies scripted, but the client leaves after the handshake.
    let bot = MockBot::bind(hello, vec![obs_msg(0), obs_msg(1)]);
    let url = bot.ws_url();
    let server = bot.run();

    let env = MinecraftEnv::connect(cfg_with_url(url), map).expect("connect");
    drop(env);

    // No panic: an early client exit is not a fault.
    server.join();
}

/// A client that sends a text frame which is not a valid `ClientMsg`
/// is committing a protocol violation. The mock must surface it rather
/// than dropping the frame and replying anyway — silently ignoring it
/// would let exactly the wire regressions this mock exists to catch
/// slip through green.
#[test]
fn malformed_client_frame_is_a_fault_not_a_silent_drop() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let bot = MockBot::bind(hello, vec![obs_msg(0)]);
    let url = bot.ws_url();
    let server = bot.run();

    // Hand-rolled client: `MinecraftEnv` cannot send malformed frames,
    // which is precisely why this needs a raw one.
    let (mut client, _response) = tungstenite::connect(&url).expect("client connect");
    client
        .send(Message::Text("{\"type\":\"not-a-real-variant\"}".into()))
        .expect("send malformed frame");

    assert!(
        join_expecting_panic(server),
        "a text frame that is not a valid ClientMsg must fail the test, not be dropped"
    );
}

/// The same for a binary frame: the protocol is text-only, so a binary
/// frame from the client is a fault.
#[test]
fn binary_client_frame_is_a_fault_not_a_silent_drop() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let bot = MockBot::bind(hello, vec![obs_msg(0)]);
    let url = bot.ws_url();
    let server = bot.run();

    let (mut client, _response) = tungstenite::connect(&url).expect("client connect");
    client
        .send(Message::Binary(vec![0x00, 0x01]))
        .expect("send binary frame");

    assert!(
        join_expecting_panic(server),
        "a binary frame from the client must fail the test, not be dropped"
    );
}

#[test]
fn step_with_action_out_of_range_returns_invalid_action() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    // Only need the obs for reset; the bad step won't reach the server.
    let bot = MockBot::bind(hello, vec![obs_msg(0)]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map.clone()).unwrap();
    let _ = env.reset(None).unwrap();
    let err = env.step(map.action_count() + 5).unwrap_err();
    assert!(matches!(err, McEnvError::InvalidAction { .. }));
    let _ = env.close();
    server.join();
}

#[test]
fn accessors_expose_schema_id_action_map_config_and_specs() {
    let map = sample_map();
    let known_id = "deadbeef";
    let hello = build_hello(map.action_count(), OBS_DIM, known_id);
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);
    let env = MinecraftEnv::connect(cfg, map.clone()).unwrap();
    assert_eq!(env.schema_id(), known_id);
    assert_eq!(env.action_map().action_count(), map.action_count());
    assert!(env.config().ws_url.contains("127.0.0.1"));
    use forge_env::Env;
    assert_eq!(env.obs_spec().num_elements(), OBS_DIM);
    assert!(env.action_spec().discrete_n().is_some());
    let name = env.name();
    assert!(name.contains("minecraft-"));
    drop(env);
    server.join();
}

#[test]
fn close_then_step_returns_closed_error() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);
    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    use forge_env::Env;
    env.close().unwrap();
    // Second close is a no-op
    env.close().unwrap();
    let err = env.step(0).unwrap_err();
    assert!(matches!(err, McEnvError::Closed));
    let err2 = env.reset(None).unwrap_err();
    assert!(matches!(err2, McEnvError::Closed));
    server.join();
}

#[test]
fn duplicate_hello_mid_episode_is_unexpected() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let mid_hello = build_hello(map.action_count(), OBS_DIM, "x");
    let bot = MockBot::bind(hello, vec![mid_hello]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);
    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    use forge_env::Env;
    let err = env.reset(None).unwrap_err();
    assert!(matches!(err, McEnvError::Unexpected(_)));
    server.join();
}

#[test]
fn obs_dim_mismatch_in_observation_returns_obs_dim_error() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let bad_obs = ServerMsg::Observation {
        tick: 0,
        obs: vec![0.0; OBS_DIM + 5], // wrong length
        reward: 0.0,
        terminated: false,
        truncated: false,
        info: serde_json::json!({}),
    };
    let bot = MockBot::bind(hello, vec![bad_obs]);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);
    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    use forge_env::Env;
    let err = env.reset(None).unwrap_err();
    assert!(matches!(err, McEnvError::ObsDimMismatch { .. }));
    server.join();
}

#[test]
fn handshake_rejects_obs_dim_mismatch_when_expected_set() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    // Expected_dim != OBS_DIM the bot reports.
    let cfg = cfg_with_url_and_expected(url, OBS_DIM + 7);
    let err = MinecraftEnv::connect(cfg, map).err().expect("expected err");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    server.join();
}

#[test]
fn binary_frame_after_handshake_returns_unexpected() {
    // Mock that emits a raw Binary frame instead of an Observation.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let server = thread::spawn(move || {
        let (stream, _peer) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut ws = tungstenite::accept(stream).unwrap();
        let hello_json = serde_json::to_string(&hello).unwrap();
        ws.send(Message::Text(hello_json)).unwrap();
        // Wait for client's Reset/Step then reply with a Binary frame.
        let _ = ws.read();
        ws.send(Message::Binary(vec![1, 2, 3, 4])).unwrap();
        let _ = ws.close(None);
    });
    let cfg = cfg_with_url(url);
    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    use forge_env::Env;
    let err = env.reset(None).unwrap_err();
    assert!(matches!(err, McEnvError::Unexpected(_)));
    let _ = server.join();
}

#[test]
fn close_frame_from_server_returns_websocket_error() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let server = thread::spawn(move || {
        let (stream, _peer) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut ws = tungstenite::accept(stream).unwrap();
        let hello_json = serde_json::to_string(&hello).unwrap();
        ws.send(Message::Text(hello_json)).unwrap();
        // Wait for client's Reset, then close without sending observation.
        let _ = ws.read();
        let _ = ws.close(None);
    });
    let cfg = cfg_with_url(url);
    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    use forge_env::Env;
    let err = env.reset(None).unwrap_err();
    assert!(matches!(err, McEnvError::WebSocket(_)));
    let _ = server.join();
}

fn hello_with_grid(
    action_count: u32,
    obs_dim: usize,
    schema_id: impl Into<String>,
    grid: GridShape,
) -> ServerMsg {
    ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        action_count,
        obs_dim,
        schema_id: schema_id.into(),
        grid_shape: Some(grid),
    }
}

#[test]
fn handshake_accepts_matching_grid_shape() {
    let map = sample_map();
    let grid = GridShape {
        height: 11,
        width: 11,
        depth: 1,
        channels: 7,
        vector_dim: 73,
    };
    let obs_dim = (grid.height as usize)
        * (grid.width as usize)
        * (grid.depth as usize)
        * (grid.channels as usize)
        + (grid.vector_dim as usize);
    let hello = hello_with_grid(map.action_count(), obs_dim, map.canonical_sha256(), grid);
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();

    let mut cfg = cfg_with_url(url);
    cfg.observation.expected_dim = Some(obs_dim);
    cfg.observation.expected_grid_shape = Some(grid);
    let env = MinecraftEnv::connect(cfg, map).expect("connect");
    assert_eq!(env.obs_dim(), obs_dim);
    drop(env);
    server.join();
}

#[test]
fn handshake_rejects_grid_shape_mismatch() {
    let map = sample_map();
    // Bot advertises channels=6, client expects channels=7. Total
    // obs_dim happens to match — without the grid_shape cross-check
    // this would silently train on scrambled feature axes.
    let server_grid = GridShape {
        height: 7,
        width: 7,
        depth: 1,
        channels: 6,
        vector_dim: 0,
    };
    let client_grid = GridShape {
        height: 7,
        width: 7,
        depth: 1,
        channels: 7,
        vector_dim: 0,
    };
    let obs_dim = (server_grid.height as usize)
        * (server_grid.width as usize)
        * (server_grid.channels as usize);
    let hello = hello_with_grid(
        map.action_count(),
        obs_dim,
        map.canonical_sha256(),
        server_grid,
    );
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();

    let mut cfg = cfg_with_url(url);
    cfg.observation.expected_grid_shape = Some(client_grid);
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected handshake mismatch");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    server.join();
}

#[test]
fn handshake_rejects_missing_grid_shape_when_expected_set() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256()); // grid_shape: None
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let mut cfg = cfg_with_url(url);
    cfg.observation.expected_grid_shape = Some(GridShape {
        height: 7,
        width: 7,
        depth: 1,
        channels: 7,
        vector_dim: 0,
    });
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected handshake mismatch");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    server.join();
}

#[test]
fn handshake_accepts_any_grid_shape_when_expected_is_none() {
    // Backwards-compat pin (peer-review S5): when the runner config
    // does NOT pin `expected_grid_shape`, the bot's `Hello` may
    // advertise any grid_shape (or none) and the handshake must
    // accept it.  Catches a regression where the v0.5 grid_shape
    // gate accidentally becomes mandatory.
    let map = sample_map();
    let server_grid = GridShape {
        height: 3,
        width: 3,
        depth: 1,
        channels: 7,
        vector_dim: 0,
    };
    // 3 (h) * 3 (w) * 1 (d) * 7 (ch) = 63 floats — single Y-layer with the
    // documented per-tile channel count.
    let obs_dim: usize = 3 * 3 * 7;
    let hello = hello_with_grid(
        map.action_count(),
        obs_dim,
        map.canonical_sha256(),
        server_grid,
    );
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let mut cfg = cfg_with_url(url);
    cfg.observation.expected_dim = Some(obs_dim);
    cfg.observation.expected_grid_shape = None;
    let env = MinecraftEnv::connect(cfg, map).expect("connect (no expected grid_shape)");
    assert_eq!(env.obs_dim(), obs_dim);
    drop(env);
    server.join();
}

#[test]
fn handshake_rejects_grid_dims_that_dont_sum_to_obs_dim() {
    let map = sample_map();
    let grid = GridShape {
        height: 3,
        width: 3,
        depth: 1,
        channels: 7,
        vector_dim: 0,
    };
    // Bot lies: advertises obs_dim != grid.height*width*depth*channels.
    let obs_dim_advertised = 100;
    let hello = hello_with_grid(
        map.action_count(),
        obs_dim_advertised,
        map.canonical_sha256(),
        grid,
    );
    let bot = MockBot::bind(hello, vec![]);
    let url = bot.ws_url();
    let server = bot.run();
    let mut cfg = cfg_with_url(url);
    cfg.observation.expected_grid_shape = Some(grid);
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected handshake mismatch");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    server.join();
}

#[test]
fn server_error_message_propagates() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let replies = vec![
        obs_msg(0),
        ServerMsg::Error {
            code: "MID_EPISODE_FAULT".into(),
            message: "synthetic".into(),
        },
    ];
    let bot = MockBot::bind(hello, replies);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    let _ = env.reset(None).unwrap();
    let err = env.step(0).unwrap_err();
    assert!(matches!(err, McEnvError::Protocol { .. }));
    let _ = env.close();
    server.join();
}

#[test]
fn reconnecting_error_maps_to_transient() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let replies = vec![
        obs_msg(0),
        ServerMsg::Error {
            code: forge_env_mc::ERROR_CODE_RECONNECTING.into(),
            message: "synthetic".into(),
        },
    ];
    let bot = MockBot::bind(hello, replies);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    let _ = env.reset(None).unwrap();
    let err = env.step(0).unwrap_err();
    assert!(
        matches!(err, McEnvError::Transient { ref code, .. } if code == forge_env_mc::ERROR_CODE_RECONNECTING),
        "expected Transient RECONNECTING, got {err:?}"
    );
    let _ = env.close();
    server.join();
}

#[test]
fn busy_error_maps_to_transient() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let replies = vec![
        obs_msg(0),
        ServerMsg::Error {
            code: forge_env_mc::ERROR_CODE_BUSY.into(),
            message: "synthetic".into(),
        },
    ];
    let bot = MockBot::bind(hello, replies);
    let url = bot.ws_url();
    let server = bot.run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    let _ = env.reset(None).unwrap();
    let err = env.step(0).unwrap_err();
    assert!(
        matches!(err, McEnvError::Transient { ref code, .. } if code == forge_env_mc::ERROR_CODE_BUSY),
        "expected Transient BUSY, got {err:?}"
    );
    let _ = env.close();
    server.join();
}

/// A client that closes gracefully mid-script must end the session, not
/// fail the test.
///
/// `ws.read()` returns `Ok(Message::Close(_))` for a proper close — not
/// an `Err` — so the disconnect classifier never sees it. tungstenite
/// has already moved to `ClosedByPeer`, so replying fails with
/// `SendAfterClosing`. Before the fix, every test escaped only because
/// its reply script length exactly equalled the client's message count:
/// one spare reply turned a textbook-correct shutdown into a panic
/// blaming the mock's internals.
#[test]
fn graceful_client_close_mid_script_is_not_a_fault() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    // Deliberately MORE replies than the client will consume.
    let bot = MockBot::bind(hello, vec![obs_msg(0), obs_msg(1), obs_msg(2)]);
    let url = bot.ws_url();
    let server = bot.run();

    let mut env = MinecraftEnv::connect(cfg_with_url(url), map).expect("connect");
    env.reset(Some(1)).expect("reset");
    // `Env::close` sends a WebSocket Close frame.
    env.close().expect("close");

    // No panic: this is how a well-behaved client leaves.
    server.join();
}

/// A control frame must not consume a scripted reply.
///
/// The script advances per `ClientMsg`, not per frame. If a Ping ate a
/// slot, every later observation would shift by one — and `received()`
/// would still look correct, because control frames are not recorded.
#[test]
fn control_frames_do_not_consume_scripted_replies() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    // Exactly one reply, reserved for the one real ClientMsg.
    let bot = MockBot::bind(hello, vec![obs_msg(7)]);
    let url = bot.ws_url();
    let server = bot.run();

    // Raw client: `MinecraftEnv` never emits pings, which is why this
    // needs a hand-rolled one.
    let (mut client, _response) = tungstenite::connect(&url).expect("client connect");
    let _hello_frame = client.read().expect("read hello");
    client.send(Message::Ping(Vec::new())).expect("send ping");
    client
        .send(Message::Text(
            serde_json::to_string(&ClientMsg::Reset { seed: Some(1) }).unwrap(),
        ))
        .expect("send reset");

    // The single reply must answer the Reset, not the Ping. tungstenite
    // auto-replies Pong, so skip any control frames on the way.
    let observation = loop {
        match client.read().expect("read reply") {
            Message::Text(text) => break text,
            _ => continue,
        }
    };
    let parsed: ServerMsg = serde_json::from_str(&observation).expect("parse reply");
    assert!(
        matches!(parsed, ServerMsg::Observation { tick: 7, .. }),
        "the scripted reply must go to the Reset, not be eaten by the Ping: {parsed:?}"
    );
    assert_eq!(
        server.received(),
        vec![ClientMsg::Reset { seed: Some(1) }],
        "control frames must not be recorded as ClientMsgs"
    );
    server.join();
}

/// A client that never connects must fail fast, not hang the harness.
///
/// `TcpListener::accept` blocks with no timeout, and the io timeout
/// applies only once a stream exists. libtest has no per-test timeout,
/// so before the fix this parked the whole `cargo test` process.
#[test]
fn absent_client_fails_fast_instead_of_hanging() {
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let bot = MockBot::bind(hello, vec![]).with_accept_timeout(Duration::from_millis(50));
    let server = bot.run();

    // Never connect.
    let started = std::time::Instant::now();
    let panicked = join_expecting_panic(server);

    assert!(panicked, "an absent client must fail, not hang");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "must fail within the accept budget, not the io timeout: took {:?}",
        started.elapsed()
    );
}

/// Every expected-panic join must go through [`join_expecting_panic`].
///
/// This file once swapped the **process-global** panic hook at four
/// sites to silence expected backtraces. The hook is shared by every
/// test libtest runs concurrently in this binary, so each swap could
/// suppress the diagnostics that explain an unrelated failure — and it
/// did not even silence its own target reliably, since a 50 ms read or
/// accept timeout can fire before `set_hook` executes.
///
/// The first fix removed one of the four and the other three survived
/// unnoticed, which is why this is a test and not a comment: reviewing
/// four near-identical blocks is exactly the task humans and bots both
/// did badly here. `catch_unwind` alone is sufficient, and confining it
/// to the helper leaves one place for the rule to be broken.
///
/// The needles are assembled at runtime so this test's own source does
/// not contain the very patterns it forbids — a self-match would make
/// the assertions unfixable rather than merely failing.
#[test]
fn expected_panic_joins_do_not_touch_the_global_panic_hook() {
    const SOURCE: &str = include_str!("mc_env_mock.rs");

    // Guard the guard: if `include_str!` ever resolved to something
    // else, every assertion below would pass vacuously.
    assert!(
        SOURCE.contains(&format!("fn {}", "join_expecting_panic")),
        "the embedded source is not this file; this guard has drifted"
    );

    for suffix in ["set_hook", "take_hook"] {
        let needle = format!("std::panic::{suffix}");
        assert_eq!(
            SOURCE.matches(&needle).count(),
            0,
            "{needle} mutates process-global state every concurrent test \
             shares; use join_expecting_panic, which asserts the panic \
             without it"
        );
    }

    // One unwind catch, inside the helper. A second means a call site
    // has grown its own copy again. Counts *calls* rather than the bare
    // name, so the prose in these doc comments does not inflate it.
    let call = format!("catch_{}(", "unwind");
    assert_eq!(
        SOURCE.matches(&call).count(),
        1,
        "expected exactly one `{call}` call, the one inside \
         join_expecting_panic; a second means a call site grew its own"
    );
}
