//! Integration test against a mock WebSocket server.
//!
//! Spins up a `std::thread` running `tungstenite::accept` and a scripted
//! sequence of `ServerMsg` replies, then exercises [`MinecraftEnv`]
//! against it.

use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use forge_env::{Env, FlatObsEnv};
use forge_env_mc::action_map::{ActionEntry, ActionKind, ActionMap};
use forge_env_mc::config::MinecraftEnvConfig;
use forge_env_mc::protocol::{GridShape, ServerMsg, SCHEMA_VERSION};
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

fn pick_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// Mock bot. `replies` is a sequence of `ServerMsg`s emitted after each
/// `ClientMsg` is read. The first message sent (before any client
/// message is read) is always a `Hello` with the provided params.
struct Mock {
    listener: TcpListener,
    hello: ServerMsg,
    replies: Vec<ServerMsg>,
}

impl Mock {
    fn spawn(addr: &str, hello: ServerMsg, replies: Vec<ServerMsg>) -> Self {
        let listener = TcpListener::bind(addr).unwrap();
        Self {
            listener,
            hello,
            replies,
        }
    }

    fn run(self) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let (stream, _peer) = self.listener.accept().expect("accept");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut ws = tungstenite::accept(stream).expect("ws handshake");
            // Send Hello first.
            let hello = serde_json::to_string(&self.hello).unwrap();
            ws.send(Message::Text(hello)).unwrap();
            // Then for each subsequent client message, send a scripted reply.
            for reply in self.replies {
                let _msg = ws.read().expect("read client msg");
                let s = serde_json::to_string(&reply).unwrap();
                ws.send(Message::Text(s)).unwrap();
            }
            let _ = ws.close(None);
        })
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
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let server = Mock::spawn(&addr, hello, vec![]).run();
    let cfg = cfg_with_url_and_expected(url, OBS_DIM);

    let env = MinecraftEnv::connect(cfg, map.clone()).expect("connect");
    assert_eq!(env.obs_dim(), OBS_DIM);
    assert_eq!(env.num_actions(), map.action_count());
    drop(env);
    let _ = server.join();
}

#[test]
fn handshake_rejects_action_count_mismatch() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(999, OBS_DIM, "x"); // wrong action_count on purpose
    let server = Mock::spawn(&addr, hello, vec![]).run();
    let cfg = cfg_with_url(url);

    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected error");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    let _ = server.join();
}

#[test]
fn handshake_rejects_schema_version_mismatch() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
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
    let server = Mock::spawn(&addr, hello, vec![]).run();
    let cfg = cfg_with_url(url);
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected error");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    let _ = server.join();
}

#[test]
fn handshake_rejects_schema_id_mismatch_when_expected_set() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, "server-schema");
    let server = Mock::spawn(&addr, hello, vec![]).run();
    let cfg = cfg_with_expected_schema(url, "client-schema");
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected error");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    let _ = server.join();
}

#[test]
fn reset_and_steps_drive_a_full_short_episode() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let mut replies = vec![obs_msg(0)]; // reply to Reset
    for t in 1..=5u64 {
        replies.push(obs_msg(t));
    }
    let server = Mock::spawn(&addr, hello, replies).run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    let r = env.reset(Some(42)).unwrap();
    assert_eq!(r.len(), OBS_DIM);
    for expected_tick in 1..=5u64 {
        let s = env.step(0).unwrap();
        assert_eq!(s.info.tick, expected_tick);
    }
    let _ = env.close();
    let _ = server.join();
}

#[test]
fn step_with_action_out_of_range_returns_invalid_action() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    // Only need the obs for reset; the bad step won't reach the server.
    let server = Mock::spawn(&addr, hello, vec![obs_msg(0)]).run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map.clone()).unwrap();
    let _ = env.reset(None).unwrap();
    let err = env.step(map.action_count() + 5).unwrap_err();
    assert!(matches!(err, McEnvError::InvalidAction { .. }));
    let _ = env.close();
    let _ = server.join();
}

#[test]
fn accessors_expose_schema_id_action_map_config_and_specs() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let known_id = "deadbeef";
    let hello = build_hello(map.action_count(), OBS_DIM, known_id);
    let server = Mock::spawn(&addr, hello, vec![]).run();
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
    let _ = server.join();
}

#[test]
fn close_then_step_returns_closed_error() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let server = Mock::spawn(&addr, hello, vec![]).run();
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
    let _ = server.join();
}

#[test]
fn duplicate_hello_mid_episode_is_unexpected() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let mid_hello = build_hello(map.action_count(), OBS_DIM, "x");
    let server = Mock::spawn(&addr, hello, vec![mid_hello]).run();
    let cfg = cfg_with_url(url);
    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    use forge_env::Env;
    let err = env.reset(None).unwrap_err();
    assert!(matches!(err, McEnvError::Unexpected(_)));
    let _ = server.join();
}

#[test]
fn obs_dim_mismatch_in_observation_returns_obs_dim_error() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
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
    let server = Mock::spawn(&addr, hello, vec![bad_obs]).run();
    let cfg = cfg_with_url(url);
    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    use forge_env::Env;
    let err = env.reset(None).unwrap_err();
    assert!(matches!(err, McEnvError::ObsDimMismatch { .. }));
    let _ = server.join();
}

#[test]
fn handshake_rejects_obs_dim_mismatch_when_expected_set() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let server = Mock::spawn(&addr, hello, vec![]).run();
    // Expected_dim != OBS_DIM the bot reports.
    let cfg = cfg_with_url_and_expected(url, OBS_DIM + 7);
    let err = MinecraftEnv::connect(cfg, map).err().expect("expected err");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    let _ = server.join();
}

#[test]
fn binary_frame_after_handshake_returns_unexpected() {
    // Mock that emits a raw Binary frame instead of an Observation.
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).unwrap();
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
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).unwrap();
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
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let grid = GridShape {
        height: 11,
        width: 11,
        depth: 1,
        channels: 7,
        vector_dim: 73,
    };
    let obs_dim =
        (grid.height as usize) * (grid.width as usize) * (grid.depth as usize) * (grid.channels as usize)
            + (grid.vector_dim as usize);
    let hello = hello_with_grid(map.action_count(), obs_dim, map.canonical_sha256(), grid);
    let server = Mock::spawn(&addr, hello, vec![]).run();

    let mut cfg = cfg_with_url(url);
    cfg.observation.expected_dim = Some(obs_dim);
    cfg.observation.expected_grid_shape = Some(grid);
    let env = MinecraftEnv::connect(cfg, map).expect("connect");
    assert_eq!(env.obs_dim(), obs_dim);
    drop(env);
    let _ = server.join();
}

#[test]
fn handshake_rejects_grid_shape_mismatch() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
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
    let hello = hello_with_grid(map.action_count(), obs_dim, map.canonical_sha256(), server_grid);
    let server = Mock::spawn(&addr, hello, vec![]).run();

    let mut cfg = cfg_with_url(url);
    cfg.observation.expected_grid_shape = Some(client_grid);
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected handshake mismatch");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    let _ = server.join();
}

#[test]
fn handshake_rejects_missing_grid_shape_when_expected_set() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256()); // grid_shape: None
    let server = Mock::spawn(&addr, hello, vec![]).run();
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
    let _ = server.join();
}

#[test]
fn handshake_rejects_grid_dims_that_dont_sum_to_obs_dim() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
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
    let server = Mock::spawn(&addr, hello, vec![]).run();
    let mut cfg = cfg_with_url(url);
    cfg.observation.expected_grid_shape = Some(grid);
    let err = MinecraftEnv::connect(cfg, map)
        .err()
        .expect("expected handshake mismatch");
    assert!(matches!(err, McEnvError::HandshakeMismatch { .. }));
    let _ = server.join();
}

#[test]
fn server_error_message_propagates() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = build_hello(map.action_count(), OBS_DIM, map.canonical_sha256());
    let replies = vec![
        obs_msg(0),
        ServerMsg::Error {
            code: "MID_EPISODE_FAULT".into(),
            message: "synthetic".into(),
        },
    ];
    let server = Mock::spawn(&addr, hello, replies).run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    let _ = env.reset(None).unwrap();
    let err = env.step(0).unwrap_err();
    assert!(matches!(err, McEnvError::Protocol { .. }));
    let _ = env.close();
    let _ = server.join();
}
