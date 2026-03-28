//! Cooperative MCTS for multi-agent planning (Phase 6 — future stub).
//!
//! Will implement Centralized Training, Decentralized Execution (CTDE)
//! MCTS for drone swarm coordination. Currently a placeholder.

/// Placeholder for cooperative MCTS configuration.
///
/// Will be implemented in Phase 6 when multi-agent coordination
/// is developed using FORGE's PettingZoo API.
#[derive(Debug, Clone)]
pub struct CooperativeMctsConfig {
    /// Number of agents to coordinate.
    pub num_agents: usize,
    /// Whether to use centralized value estimation.
    pub centralized_critic: bool,
}

impl Default for CooperativeMctsConfig {
    fn default() -> Self {
        Self {
            num_agents: 2,
            centralized_critic: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CooperativeMctsConfig::default();
        assert_eq!(config.num_agents, 2);
        assert!(config.centralized_critic);
    }
}
