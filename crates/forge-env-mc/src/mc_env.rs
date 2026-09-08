//! [`MinecraftEnv`] — the [`Env`] impl. Drives a Node mc-bot over
//! WebSocket using the [`crate::protocol`] message types.

use std::borrow::Cow;

use forge_env::{ActionSpec, Env, FlatObsEnv, ObsSpec, StepOutput};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, instrument, warn};

use crate::action_map::ActionMap;
use crate::client::ProtocolClient;
use crate::config::MinecraftEnvConfig;
use crate::error::McEnvError;
use crate::protocol::{ClientMsg, GridShape, ServerMsg, SCHEMA_VERSION};

/// Diagnostic info attached to each [`StepOutput`] from `MinecraftEnv`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinecraftStepInfo {
    /// Tick reported by the bot.
    pub tick: u64,
    /// Free-form JSON value from the bot's `info` field.
    pub raw: serde_json::Value,
}

/// Minecraft env. See crate docs.
pub struct MinecraftEnv {
    client: ProtocolClient,
    config: MinecraftEnvConfig,
    action_map: ActionMap,
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    /// `schema_id` the bot reported in `Hello`. Cached for diagnostics.
    schema_id: String,
    closed: bool,
}

impl MinecraftEnv {
    /// Connect to the bot, perform the handshake, and validate that
    /// the bot's reported schema matches what the client expects.
    pub fn connect(config: MinecraftEnvConfig, action_map: ActionMap) -> Result<Self, McEnvError> {
        action_map.validate()?;
        let mut client = ProtocolClient::connect(&config.ws_url, config.heartbeat_ms)?;
        let hello = client.recv()?;
        let (schema_version, action_count, obs_dim, schema_id, grid_shape) = match hello {
            ServerMsg::Hello {
                schema_version,
                action_count,
                obs_dim,
                schema_id,
                grid_shape,
            } => (schema_version, action_count, obs_dim, schema_id, grid_shape),
            other => {
                return Err(McEnvError::Unexpected(format!(
                    "expected Hello, got {other:?}"
                )));
            }
        };

        if schema_version != SCHEMA_VERSION {
            return Err(McEnvError::HandshakeMismatch {
                client: format!("schema_version={SCHEMA_VERSION}"),
                server: format!("schema_version={schema_version}"),
            });
        }
        let our_action_count = action_map.action_count();
        if action_count != our_action_count {
            return Err(McEnvError::HandshakeMismatch {
                client: format!("action_count={our_action_count}"),
                server: format!("action_count={action_count}"),
            });
        }
        if let Some(expected) = config.observation.expected_dim {
            if expected != obs_dim {
                return Err(McEnvError::HandshakeMismatch {
                    client: format!("obs_dim={expected}"),
                    server: format!("obs_dim={obs_dim}"),
                });
            }
        }
        if let Some(expected) = &config.expected_schema_id {
            if expected != &schema_id {
                return Err(McEnvError::HandshakeMismatch {
                    client: format!("schema_id={expected}"),
                    server: format!("schema_id={schema_id}"),
                });
            }
        }
        // grid_shape cross-check. We require the bot to advertise a
        // grid_shape whenever the operator pinned `expected_grid_shape`
        // (silent absence would mean the bot ran with the legacy
        // flat-only encoder against a trainer expecting spatial input).
        if let Some(expected) = config.observation.expected_grid_shape {
            let Some(server) = grid_shape else {
                warn!(
                    expected = ?expected,
                    "grid_shape mismatch: client expected Some(_), server reported None"
                );
                return Err(McEnvError::HandshakeMismatch {
                    client: format!("grid_shape={expected:?}"),
                    server: "grid_shape=None".to_string(),
                });
            };
            if !grid_shape_matches(&expected, &server) {
                warn!(
                    expected = ?expected,
                    server = ?server,
                    "grid_shape mismatch: field-wise comparison failed"
                );
                return Err(McEnvError::HandshakeMismatch {
                    client: format!("grid_shape={expected:?}"),
                    server: format!("grid_shape={server:?}"),
                });
            }
            // Defensive: the grid + flat dims must add up to obs_dim.
            let derived = (server.height as usize)
                * (server.width as usize)
                * (server.depth as usize)
                * (server.channels as usize)
                + (server.vector_dim as usize);
            if derived != obs_dim {
                warn!(
                    derived,
                    obs_dim,
                    server = ?server,
                    "grid_shape derived obs_dim != advertised obs_dim — bot is lying about its shape"
                );
                return Err(McEnvError::HandshakeMismatch {
                    client: format!("derived_obs_dim={derived}"),
                    server: format!("obs_dim={obs_dim}"),
                });
            }
        }

        let obs_spec = ObsSpec::flat_f32("minecraft_symbolic", obs_dim, -1.0e6, 1.0e6);
        let action_spec = ActionSpec::discrete(action_count);

        debug!(action_count, obs_dim, schema_id, "minecraft env connected");

        Ok(Self {
            client,
            config,
            action_map,
            obs_spec,
            action_spec,
            schema_id,
            closed: false,
        })
    }

    /// The bot-reported schema id (sha256 of canonical config form).
    pub fn schema_id(&self) -> &str {
        &self.schema_id
    }

    /// The loaded action map.
    pub fn action_map(&self) -> &ActionMap {
        &self.action_map
    }

    /// Read config (read-only).
    pub fn config(&self) -> &MinecraftEnvConfig {
        &self.config
    }

    fn ensure_open(&self) -> Result<(), McEnvError> {
        if self.closed {
            Err(McEnvError::Closed)
        } else {
            Ok(())
        }
    }

    /// Fill `out` from a server message. Used by `step_into`.
    fn fill_step_output(
        &self,
        msg: ServerMsg,
        out: &mut StepOutput<Vec<f32>, MinecraftStepInfo>,
    ) -> Result<(), McEnvError> {
        match msg {
            ServerMsg::Observation {
                tick,
                obs,
                reward,
                terminated,
                truncated,
                info,
            } => {
                let expected = self.obs_spec.num_elements();
                if obs.len() != expected {
                    error!(
                        got = obs.len(),
                        expected, "bot returned obs of wrong length"
                    );
                    return Err(McEnvError::ObsDimMismatch {
                        expected,
                        got: obs.len(),
                    });
                }
                // Reuse the caller's obs buffer.
                out.obs.clear();
                out.obs.extend_from_slice(&obs);
                out.reward = reward;
                out.terminated = terminated;
                out.truncated = truncated;
                out.info = MinecraftStepInfo { tick, raw: info };
                Ok(())
            }
            ServerMsg::Error { code, message } => {
                warn!(code, message, "bot reported protocol error");
                Err(McEnvError::from_protocol_error(code, message))
            }
            ServerMsg::Hello { .. } => {
                Err(McEnvError::Unexpected("duplicate Hello mid-episode".into()))
            }
        }
    }
}

impl Env for MinecraftEnv {
    type Obs = Vec<f32>;
    type Action = u32;
    type Info = MinecraftStepInfo;
    type Error = McEnvError;

    #[instrument(skip_all, fields(env = "minecraft", seed))]
    fn reset_into(&mut self, seed: Option<u64>, out: &mut Vec<f32>) -> Result<(), Self::Error> {
        self.ensure_open()?;
        tracing::Span::current().record("seed", seed.unwrap_or(0));
        self.client.send(&ClientMsg::Reset { seed })?;
        let msg = self.client.recv()?;
        // For reset we only capture the observation; reward/flags are
        // not meaningful at episode start.
        match msg {
            ServerMsg::Observation { obs, .. } => {
                let expected = self.obs_spec.num_elements();
                if obs.len() != expected {
                    error!(
                        got = obs.len(),
                        expected, "bot returned obs of wrong length on reset"
                    );
                    return Err(McEnvError::ObsDimMismatch {
                        expected,
                        got: obs.len(),
                    });
                }
                out.clear();
                out.extend_from_slice(&obs);
                Ok(())
            }
            ServerMsg::Error { code, message } => {
                warn!(code, message, "bot reported protocol error on reset");
                Err(McEnvError::from_protocol_error(code, message))
            }
            ServerMsg::Hello { .. } => {
                Err(McEnvError::Unexpected("duplicate Hello mid-episode".into()))
            }
        }
    }

    #[instrument(skip_all, fields(env = "minecraft", action_id = action))]
    fn step_into(
        &mut self,
        action: u32,
        out: &mut StepOutput<Vec<f32>, Self::Info>,
    ) -> Result<(), Self::Error> {
        self.ensure_open()?;
        let n = self.action_spec.discrete_n().ok_or_else(|| {
            McEnvError::Unexpected("minecraft action_spec is not discrete".into())
        })?;
        if action >= n {
            return Err(McEnvError::InvalidAction {
                action_id: action,
                space_n: n,
            });
        }
        self.client.send(&ClientMsg::Step { action_id: action })?;
        let msg = self.client.recv()?;
        self.fill_step_output(msg, out)
    }

    fn obs_spec(&self) -> &ObsSpec {
        &self.obs_spec
    }

    fn action_spec(&self) -> &ActionSpec {
        &self.action_spec
    }

    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(format!("minecraft-{}", self.schema_id))
    }

    fn close(&mut self) -> Result<(), Self::Error> {
        if !self.closed {
            let _ = self.client.send(&ClientMsg::Close);
            let _ = self.client.close();
            self.closed = true;
        }
        Ok(())
    }
}

impl FlatObsEnv for MinecraftEnv {
    fn obs_dim(&self) -> usize {
        self.obs_spec.num_elements()
    }
    fn num_actions(&self) -> u32 {
        self.action_spec.discrete_n().unwrap_or(0)
    }
}

// MinecraftEnv's `step_into` reuses the caller's obs buffer for the
// observation bytes; internal network I/O deserialization still allocates,
// but that is unavoidable and not governed by the zero-alloc CI gate.

fn grid_shape_matches(expected: &GridShape, server: &GridShape) -> bool {
    expected.height == server.height
        && expected.width == server.width
        && expected.depth == server.depth
        && expected.channels == server.channels
        && expected.vector_dim == server.vector_dim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_shape_matches_is_field_wise() {
        let a = GridShape {
            height: 11,
            width: 11,
            depth: 11,
            channels: 7,
            vector_dim: 73,
        };
        let mut b = a;
        assert!(grid_shape_matches(&a, &b));
        b.channels = 6;
        assert!(!grid_shape_matches(&a, &b));
        b = a;
        b.vector_dim = 31;
        assert!(!grid_shape_matches(&a, &b));
    }
}
