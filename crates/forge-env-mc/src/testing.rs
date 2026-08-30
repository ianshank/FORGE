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

use std::io::ErrorKind;
use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tungstenite::Message;

use crate::protocol::{ClientMsg, ServerMsg};

/// Default read/write timeout applied to the accepted socket.
///
/// Bounds a stuck test rather than letting it hang until the harness
/// timeout: if the client never sends the next message, the mock
/// thread errors out instead of parking forever.
pub const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(5);

/// Default budget for a client to connect.
///
/// `TcpListener::accept` blocks with no timeout of its own, and
/// [`DEFAULT_IO_TIMEOUT`] applies only to the *stream*, i.e. after a
/// connection exists. Without this bound a test whose client never
/// connects — because `MinecraftEnv::connect` validates the action map
/// before opening a socket, so any client-side precondition failure
/// returns early — would park in [`MockBotHandle::join`] forever, and
/// libtest has no per-test timeout to rescue it.
pub const DEFAULT_ACCEPT_TIMEOUT: Duration = Duration::from_secs(10);

/// Poll interval while waiting for a connection.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(5);

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
pub(crate) fn is_orderly_disconnect(err: &tungstenite::Error) -> bool {
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
    accept_timeout: Duration,
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
            accept_timeout: DEFAULT_ACCEPT_TIMEOUT,
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

    /// Override how long the mock waits for a client to connect.
    ///
    /// Lower it to assert quickly on a client that is expected never to
    /// arrive; raise it for a deliberately slow starter.
    #[must_use]
    pub fn with_accept_timeout(mut self, timeout: Duration) -> Self {
        self.accept_timeout = timeout;
        self
    }

    /// The socket read/write timeout this mock will apply.
    #[must_use]
    pub fn io_timeout(&self) -> Duration {
        self.io_timeout
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
            // `accept()` blocks with no timeout of its own, so poll it
            // against a deadline. Otherwise a test whose client never
            // connects parks here forever and takes the whole libtest
            // process with it -- a hang, not a failure.
            self.listener
                .set_nonblocking(true)
                .expect("set listener non-blocking");
            let deadline = Instant::now() + self.accept_timeout;
            let (stream, _peer) = loop {
                match self.listener.accept() {
                    Ok(accepted) => break accepted,
                    Err(err) if err.kind() == ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "mock bot: no client connected within {:?}",
                            self.accept_timeout
                        );
                        thread::sleep(ACCEPT_POLL_INTERVAL);
                    }
                    Err(err) => panic!("mock bot failed to accept a connection: {err}"),
                }
            };
            stream
                .set_nonblocking(false)
                .expect("restore blocking mode");
            stream
                .set_read_timeout(Some(self.io_timeout))
                .expect("set read timeout");
            stream
                .set_write_timeout(Some(self.io_timeout))
                .expect("set write timeout");
            let mut ws = tungstenite::accept(stream).expect("ws handshake");

            let hello = serde_json::to_string(&self.hello).expect("serialize hello");
            ws.send(Message::Text(hello)).expect("send hello");

            // Peek rather than consume: the script must advance once per
            // `ClientMsg`, NOT once per frame. A control frame that ate a
            // reply slot would silently shift every later observation by
            // one, and `received()` would look correct because control
            // frames are not recorded. Peeking also preserves the
            // no-replies case: an empty script reads nothing and closes
            // immediately, rather than blocking until the read timeout.
            let mut replies = self.replies.into_iter().peekable();
            while replies.peek().is_some() {
                let msg = match ws.read() {
                    Ok(msg) => msg,
                    Err(err) if is_orderly_disconnect(&err) => break,
                    Err(err) => panic!("mock bot failed reading a client message: {err}"),
                };
                // A conforming client sends `ClientMsg` as text and
                // nothing else. Silently ignoring anything other than a
                // well-formed text frame would let exactly the
                // regressions this mock exists to catch -- a client
                // emitting binary, or malformed JSON -- pass while the
                // mock cheerfully sent its scripted reply.
                match &msg {
                    Message::Text(text) => match serde_json::from_str::<ClientMsg>(text) {
                        Ok(parsed) => recorder.lock().expect("recorder mutex").push(parsed),
                        Err(err) => panic!(
                            "client sent a text frame that is not a valid ClientMsg: {err}\n\
                             frame: {text}"
                        ),
                    },
                    // A graceful close ends the session. tungstenite has
                    // already moved to `ClosedByPeer`, so replying would
                    // fail with `SendAfterClosing` -- reading this as a
                    // fault would make a textbook-correct client shutdown
                    // fail the test.
                    Message::Close(_) => break,
                    // Control frames carry no `ClientMsg`; skip without
                    // consuming a scripted reply.
                    Message::Ping(_) | Message::Pong(_) => continue,
                    other => panic!(
                        "client sent a non-text frame, which the protocol does not permit: \
                         {other:?}"
                    ),
                }
                let reply = replies.next().expect("peeked reply must exist");
                let encoded = serde_json::to_string(&reply).expect("serialize reply");
                match ws.send(Message::Text(encoded)) {
                    Ok(()) => {}
                    // Defensive, and deliberately unexercised by the
                    // suite: `read()` above always runs first, so every
                    // reachable disconnect is classified there. This arm
                    // only fires in a narrow race (the read succeeds from
                    // the socket buffer, the peer's RST lands before the
                    // write). Constructing that deterministically needs
                    // SO_LINGER, which is unstable, so it is left
                    // untested rather than pinned by a flaky test.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error as IoError, ErrorKind};
    use tungstenite::error::ProtocolError;

    /// Pins every arm of [`is_orderly_disconnect`] directly.
    ///
    /// The behavioural tests reach only one arm: a dropped
    /// `MinecraftEnv` yields `Protocol(ResetWithoutClosingHandshake)`,
    /// so deleting the `ConnectionClosed`/`AlreadyClosed` arm or the
    /// whole `Io` arm was invisible to the suite — both mutations left
    /// every test green, because the fallthrough preserved the outcome
    /// the one covered test asserts.
    #[test]
    fn orderly_disconnect_classifies_every_arm() {
        // Socket-level ways a client can vanish.
        for kind in [
            ErrorKind::BrokenPipe,
            ErrorKind::ConnectionReset,
            ErrorKind::ConnectionAborted,
            ErrorKind::UnexpectedEof,
        ] {
            assert!(
                is_orderly_disconnect(&tungstenite::Error::Io(IoError::from(kind))),
                "{kind:?} is an abrupt client exit and must be orderly"
            );
        }

        // A stalled client must stay fatal, or a hung test looks clean.
        for kind in [
            ErrorKind::WouldBlock,
            ErrorKind::TimedOut,
            ErrorKind::PermissionDenied,
        ] {
            assert!(
                !is_orderly_disconnect(&tungstenite::Error::Io(IoError::from(kind))),
                "{kind:?} is a fault, not a disconnect"
            );
        }

        assert!(is_orderly_disconnect(&tungstenite::Error::ConnectionClosed));
        assert!(is_orderly_disconnect(&tungstenite::Error::AlreadyClosed));
        assert!(is_orderly_disconnect(&tungstenite::Error::Protocol(
            ProtocolError::ResetWithoutClosingHandshake
        )));

        // Only that ONE protocol error is orderly; the rest are real
        // violations and must not be swallowed.
        assert!(!is_orderly_disconnect(&tungstenite::Error::Protocol(
            ProtocolError::SendAfterClosing
        )));
        assert!(!is_orderly_disconnect(&tungstenite::Error::Utf8));
    }
}
