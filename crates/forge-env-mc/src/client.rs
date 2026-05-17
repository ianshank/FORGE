//! Synchronous WebSocket client for the mc-bot protocol.
//!
//! Sync (not async) because `Env::step` is a blocking call. The
//! underlying transport is `tungstenite` 0.24.

use std::io::Read;
use std::net::TcpStream;
use std::time::Duration;

use tracing::{debug, instrument};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

use crate::error::McEnvError;
use crate::protocol::{ClientMsg, ServerMsg};

/// Owns the WebSocket and (de)serialises messages.
pub struct ProtocolClient {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
}

impl ProtocolClient {
    /// Connect to the bot at `url`. Applies an OS-level read timeout
    /// derived from `heartbeat_ms` so a dead bot doesn't hang forever.
    #[instrument(skip_all, fields(url = url))]
    pub fn connect(url: &str, heartbeat_ms: u64) -> Result<Self, McEnvError> {
        let (socket, _resp) = tungstenite::connect(url)?;
        if let MaybeTlsStream::Plain(s) = socket.get_ref() {
            // Best-effort timeouts. Errors here are non-fatal; the
            // heartbeat watcher in the caller is the real safety net.
            let _ = s.set_read_timeout(Some(Duration::from_millis(heartbeat_ms)));
            let _ = s.set_write_timeout(Some(Duration::from_millis(heartbeat_ms)));
        }
        debug!("connected");
        Ok(Self { socket })
    }

    /// Send a typed [`ClientMsg`].
    #[instrument(skip(self), fields(kind))]
    pub fn send(&mut self, msg: &ClientMsg) -> Result<(), McEnvError> {
        let s = serde_json::to_string(msg)?;
        tracing::Span::current().record("kind", message_kind(msg));
        self.socket.send(Message::Text(s))?;
        Ok(())
    }

    /// Read a typed [`ServerMsg`]. Blocks until a message arrives or
    /// the OS-level timeout fires (mapped to [`McEnvError::WebSocket`]).
    #[instrument(skip(self))]
    pub fn recv(&mut self) -> Result<ServerMsg, McEnvError> {
        loop {
            let msg = self.socket.read()?;
            match msg {
                Message::Text(t) => {
                    let parsed: ServerMsg = serde_json::from_str(&t)?;
                    return Ok(parsed);
                }
                Message::Binary(_) => {
                    return Err(McEnvError::Unexpected(
                        "binary frames are not supported in protocol v1".into(),
                    ));
                }
                Message::Ping(payload) => {
                    self.socket.send(Message::Pong(payload))?;
                    continue;
                }
                Message::Pong(_) | Message::Frame(_) => continue,
                Message::Close(_) => {
                    return Err(McEnvError::WebSocket("connection closed".into()));
                }
            }
        }
    }

    /// Politely close the connection.
    pub fn close(&mut self) -> Result<(), McEnvError> {
        let _ = self.socket.send(Message::Close(None));
        Ok(())
    }
}

fn message_kind(m: &ClientMsg) -> &'static str {
    match m {
        ClientMsg::Reset { .. } => "reset",
        ClientMsg::Step { .. } => "step",
        ClientMsg::Close => "close",
    }
}

// Avoid unused-import warning when `tungstenite::stream` not otherwise used.
#[allow(dead_code)]
fn _force_unused_read_stays_available<R: Read>(_: &mut R) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_kind_covers_all_variants() {
        assert_eq!(message_kind(&ClientMsg::Reset { seed: None }), "reset");
        assert_eq!(message_kind(&ClientMsg::Step { action_id: 0 }), "step");
        assert_eq!(message_kind(&ClientMsg::Close), "close");
    }

    #[test]
    fn connect_returns_websocket_error_on_unreachable_url() {
        // Port 1 is privileged; nothing listens. Any quick failure works.
        let res = ProtocolClient::connect("ws://127.0.0.1:1", 100);
        match res {
            Err(McEnvError::WebSocket(_)) => {}
            Err(other) => panic!("expected WebSocket error, got {other:?}"),
            Ok(_) => panic!("connect should have failed"),
        }
    }
}
