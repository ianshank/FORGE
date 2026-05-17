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
use forge_env_mc::protocol::{ServerMsg, SCHEMA_VERSION};
use forge_env_mc::{McEnvError, MinecraftEnv};
use tungstenite::Message;

const OBS_DIM: usize = 4;

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
    let hello = ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        action_count: map.action_count(),
        obs_dim: OBS_DIM,
        schema_id: map.canonical_sha256(),
    };
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
    let hello = ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        action_count: 999, // wrong on purpose
        obs_dim: OBS_DIM,
        schema_id: "x".into(),
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
fn handshake_rejects_schema_version_mismatch() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = ServerMsg::Hello {
        schema_version: SCHEMA_VERSION + 100,
        action_count: map.action_count(),
        obs_dim: OBS_DIM,
        schema_id: "x".into(),
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
fn reset_and_steps_drive_a_full_short_episode() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        action_count: map.action_count(),
        obs_dim: OBS_DIM,
        schema_id: map.canonical_sha256(),
    };
    let mut replies = vec![obs_msg(0)]; // reply to Reset
    for t in 1..=5u64 {
        replies.push(obs_msg(t));
    }
    let server = Mock::spawn(&addr, hello, replies).run();
    let cfg = cfg_with_url(url);

    let mut env = MinecraftEnv::connect(cfg, map).unwrap();
    let r = env.reset(Some(42)).unwrap();
    assert_eq!(r.obs.len(), OBS_DIM);
    assert_eq!(r.info.tick, 0);
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
    let hello = ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        action_count: map.action_count(),
        obs_dim: OBS_DIM,
        schema_id: map.canonical_sha256(),
    };
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
fn server_error_message_propagates() {
    let port = pick_port();
    let url = format!("ws://127.0.0.1:{port}");
    let addr = format!("127.0.0.1:{port}");
    let map = sample_map();
    let hello = ServerMsg::Hello {
        schema_version: SCHEMA_VERSION,
        action_count: map.action_count(),
        obs_dim: OBS_DIM,
        schema_id: map.canonical_sha256(),
    };
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
