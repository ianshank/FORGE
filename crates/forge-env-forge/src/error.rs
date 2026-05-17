//! Error type for the FORGE env shim.

use thiserror::Error;

/// Error returned by [`crate::WorldEnv`] and [`crate::FlatForgeEnv`].
#[derive(Debug, Error)]
pub enum ForgeEnvError {
    /// `WorldState::new` rejected the supplied config.
    #[error("world initialisation failed: {0}")]
    WorldInit(String),
    /// `Action::from_discrete_full` returned `None` for the given index.
    #[error("invalid discrete action: id={action_id}, space_n={space_n}")]
    InvalidDiscreteAction {
        /// The offending action id.
        action_id: u32,
        /// The configured action-space cardinality.
        space_n: u32,
    },
    /// The world had no agents when single-agent semantics required exactly one.
    #[error("expected at least one agent in WorldState; configure agents.count >= 1")]
    NoAgents,
    /// Env was used after `close` was called.
    #[error("env is closed")]
    Closed,
}
