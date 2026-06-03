//! Multi-agent swarm coordination.
//!
//! Provides the [`SwarmProtocol`] coordination interface and a cooperative
//! Centralized-Training / Decentralized-Execution (CTDE) MCTS implementation
//! ([`CooperativeMctsProtocol`]) that reuses FORGE's single-agent PUCT search.
//! [`IndependentProtocol`](protocol::IndependentProtocol) remains as the
//! no-coordination baseline.

pub mod centralized_critic;
pub mod cooperative_mcts;
pub mod joint_search;
pub mod protocol;

pub use centralized_critic::{CentralizedCritic, Critic, IndependentCritic, JointPolicyValue};
pub use cooperative_mcts::{CooperativeMctsConfig, CooperativeMctsProtocol, JointStrategy};
pub use joint_search::JointMctsPlanner;
pub use protocol::{IndependentProtocol, SwarmProtocol};
