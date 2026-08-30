//! Scripted mock of the Node `mc-bot` WebSocket server, for tests.
//!
//! [`MockBot`] stands in for the real bridge: it completes a WebSocket
//! handshake, sends a [`ServerMsg::Hello`], then replies to each client
//! message with the next entry from a scripted list. That is enough to
//! drive [`crate::MinecraftEnv`] — and anything layered on top of it,
//! such as `forge-mc-runner`'s episode loop — over the *real* wire
//! protocol without a Minecraft server, a JVM, or Docker.
//!
//! It also **records every [`ClientMsg`] it receives**, so a test can
//! assert what the client actually sent (which action ids a planner
//! chose, whether `Close` was sent on drop) rather than only what it
//! did with the replies.
//!
//! Gated behind the `testing` feature so the mock never ships in a
//! release binary. Consumers enable it as a dev-dependency:
//!
//! ```toml
//! [dev-dependencies]
//! forge-env-mc = { workspace = true, features = ["testing"] }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use forge_env_mc::protocol::{ServerMsg, SCHEMA_VERSION};
//! use forge_env_mc::testing::MockBot;
//!
//! let hello = ServerMsg::Hello {
//!     schema_version: SCHEMA_VERSION,
//!     action_count: 3,
//!     obs_dim: 4,
//!     schema_id: "deadbeef".into(),
//!     grid_shape: None,
//! };
//! let bot = MockBot::bind(hello, vec![]);
//! let url = bot.ws_url(); // feed this to MinecraftEnvConfig::ws_url
//! let handle = bot.run();
//! // ... drive MinecraftEnv against `url` ...
//! handle.join();
//! ```

use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tungstenite::Message;

use crate::protocol::{ClientMsg, ServerMsg};

/// Default read/write timeout applied to the accepted socket.
///
/// Bounds a stuck test rather than letting it hang until the harness
/// timeout: if the client never sends the next message, the mock
/// thread errors out instead of parking forever.
pub const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(5);

/// Loopback address the mock binds to. Port `0` asks the OS for a free
/// port, which [`MockBot::bind`] then reads back from the bound
/// listener — the listener is held for the mock's lifetime, so no
/// other process can claim the port in between.
const BIND_ADDR: &str = "127.0.0.1:0";

/// True for errors that mean "the client went away", as opposed to a
/// protocol or timeout failure.
///
/// Several tests deliberately drop the env before the reply script is
/// exhausted; that is an orderly end to the session, not a fault. Every
/// other error — a timeout, a malformed frame, a capacity violation —
/// stays fatal so [`MockBotHandle::join`] can surface it.
fn is_orderly_disconnect(err: &tungstenite::Error) -> bool {
    match err {
        tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed => true,
        // A client dropped without sending a Close frame — which is
        // what `drop(env)` does — surfaces here, not as an `Io` error.
        // Only this one protocol error is orderly; a malformed frame or
        // a capacity violation still panics.
        tungstenite::Error::Protocol(
            tungstenite::error::ProtocolError::ResetWithoutClosingHandshake,
        ) => true,
        // An abrupt drop can also surface at the socket layer. A read
        // timeout is `WouldBlock`/`TimedOut` and is deliberately NOT
        // matched here, so a hung client still fails the test.
        tungstenite::Error::Io(io) => matches!(
            io.kind(),
            std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::UnexpectedEof
        ),
        _ => false,
    }
}

/// A scripted stand-in for the Node `mc-bot` WebSocket server.
///
/// Construct with [`MockBot::bind`], read the address off it, then
/// [`MockBot::run`] to serve on a background thread.
pub struct MockBot {
    listener: TcpListener,
    hello: ServerMsg,
    replies: Vec<ServerMsg>,
    io_timeout: Duration,
}

impl MockBot {
    /// Bind a listener on an OS-assigned loopback port.
    ///
    /// `hello` is sent unconditionally once the WebSocket handshake
    /// completes, before any client message is read. `replies` is
    /// consumed in order: one entry per client message received.
    /// Once exhausted the mock closes the connection.
    ///
    /// The listener is bound here and **held**, so the port cannot be
    /// taken by another test between selection and use.
    ///
    /// # Panics
    ///
    /// Panics if the loopback listener cannot be bound, which in a
    /// test environment means the harness itself is broken.
    #[must_use]
    pub fn bind(hello: ServerMsg, replies: Vec<ServerMsg>) -> Self {
        let listener = TcpListener::bind(BIND_ADDR).expect("bind mock bot listener");
        Self {
            listener,
            hello,
            replies,
            io_timeout: DEFAULT_IO_TIMEOUT,
        }
    }

    /// Override the socket read/write timeout.
    ///
    /// Raise it for a test that deliberately keeps the client idle;
    /// lower it to assert on timeout behaviour quickly.
    #[must_use]
    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }

    /// The bound address, including the OS-assigned port.
    ///
    /// # Panics
    ///
    /// Panics if the bound listener has no local address.
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.listener.local_addr().expect("mock bot local_addr")
    }

    /// The `ws://` URL a [`crate::config::MinecraftEnvConfig`] should
    /// point at to reach this mock.
    #[must_use]
    pub fn ws_url(&self) -> String {
        format!("ws://{}", self.local_addr())
    }

    /// Serve the scripted session on a background thread.
    ///
    /// Accepts one connection, completes the WebSocket handshake,
    /// sends the `Hello`, then alternates read-then-reply until the
    /// reply script is exhausted, recording each received
    /// [`ClientMsg`] as it goes.
    ///
    /// A client that disconnects before the script is exhausted ends
    /// the session cleanly. Any other failure — a timeout, a malformed
    /// frame — panics the serving thread, and
    /// [`MockBotHandle::join`] re-raises it in the test thread.
    #[must_use]
    pub fn run(self) -> MockBotHandle {
        let received = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&received);
        let handle = thread::spawn(move || {
            let (stream, _peer) = self.listener.accept().expect("accept");
            stream
                .set_read_timeout(Some(self.io_timeout))
                .expect("set read timeout");
            stream
                .set_write_timeout(Some(self.io_timeout))
                .expect("set write timeout");
            let mut ws = tungstenite::accept(stream).expect("ws handshake");

            let hello = serde_json::to_string(&self.hello).expect("serialize hello");
            ws.send(Message::Text(hello)).expect("send hello");

            for reply in self.replies {
                let msg = match ws.read() {
                    Ok(msg) => msg,
                    Err(err) if is_orderly_disconnect(&err) => break,
                    Err(err) => panic!("mock bot failed reading a client message: {err}"),
                };
                // Text frames are the only thing the real client sends;
                // record them so tests can assert on what was sent.
                // Anything else is left unrecorded rather than
                // panicking, so a test can script a non-text frame
                // without the recorder deciding the outcome.
                if let Message::Text(text) = &msg {
                    if let Ok(parsed) = serde_json::from_str::<ClientMsg>(text) {
                        recorder.lock().expect("recorder mutex").push(parsed);
                    }
                }
                let encoded = serde_json::to_string(&reply).expect("serialize reply");
                match ws.send(Message::Text(encoded)) {
                    Ok(()) => {}
                    Err(err) if is_orderly_disconnect(&err) => break,
                    Err(err) => panic!("mock bot failed sending a reply: {err}"),
                }
            }
            let _ = ws.close(None);
        });
        MockBotHandle { handle, received }
    }
}

/// Handle to a running [`MockBot`].
///
/// Exposes what the client sent and lets the test join the serving
/// thread.
pub struct MockBotHandle {
    handle: thread::JoinHandle<()>,
    received: Arc<Mutex<Vec<ClientMsg>>>,
}

impl MockBotHandle {
    /// Snapshot of the [`ClientMsg`]s received so far, in order.
    ///
    /// Safe to call while the mock is still serving; call it after
    /// [`MockBotHandle::join`] for the complete sequence.
    ///
    /// # Panics
    ///
    /// Panics if the recorder mutex was poisoned by a panic on the
    /// serving thread — in which case the test has already failed.
    #[must_use]
    pub fn received(&self) -> Vec<ClientMsg> {
        self.received.lock().expect("recorder mutex").clone()
    }

    /// Wait for the serving thread to finish, re-raising any panic.
    ///
    /// A mock that quietly swallowed its own failures could let a test
    /// pass while the protocol was broken, which would defeat the
    /// point of testing against a real wire. Orderly client
    /// disconnects are handled inside [`MockBot::run`] and do not
    /// panic, so anything that reaches here is a genuine fault and is
    /// re-raised in the calling thread with its original message.
    ///
    /// # Panics
    ///
    /// Panics if the serving thread panicked.
    pub fn join(self) {
        if let Err(payload) = self.handle.join() {
            std::panic::resume_unwind(payload);
        }
    }
}
