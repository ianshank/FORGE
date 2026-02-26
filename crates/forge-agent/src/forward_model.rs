//! Forward model API for planning agents.
//!
//! Provides a trait for simulating future states without modifying
//! the real simulation. Used by MCTS and other search-based planners.

use forge_core::WorldState;
use forge_types::observation::StepResult;
use forge_types::Action;

/// A forward model that can simulate actions from a given state.
///
/// This trait abstracts the simulation so planning algorithms
/// don't depend directly on WorldState internals.
pub trait ForwardModel: Send + Sync {
    /// Returns a snapshot (clone) of the current state.
    fn snapshot(&self) -> WorldState;

    /// Simulates a single step from the given state.
    /// Returns the resulting state and step result without modifying the original.
    fn simulate(&self, state: &WorldState, actions: &[Action]) -> (WorldState, StepResult);

    /// Returns true if the state is terminal (episode ended).
    fn is_terminal(&self, state: &WorldState) -> bool;

    /// Returns the number of agents.
    fn num_agents(&self, state: &WorldState) -> usize;

    /// Returns the action space size for discrete actions.
    fn action_space_size(&self) -> u32;
}

/// Default forward model that clones the state and calls step().
#[derive(Debug, Clone)]
pub struct DefaultForwardModel {
    /// Action space size (cached).
    action_space: u32,
}

impl DefaultForwardModel {
    /// Creates a new forward model.
    pub fn new(comm_vocab_size: u16) -> Self {
        Self {
            action_space: Action::space_size(comm_vocab_size),
        }
    }
}

impl ForwardModel for DefaultForwardModel {
    fn snapshot(&self) -> WorldState {
        // This is typically called from a state reference, but the trait
        // doesn't hold state. Callers should clone the state directly.
        unreachable!("Use state.clone() instead")
    }

    fn simulate(&self, state: &WorldState, actions: &[Action]) -> (WorldState, StepResult) {
        let mut next_state = state.clone();
        let result = next_state.step(actions);
        (next_state, result)
    }

    fn is_terminal(&self, state: &WorldState) -> bool {
        state.terminated || state.truncated
    }

    fn num_agents(&self, state: &WorldState) -> usize {
        state.agents.len()
    }

    fn action_space_size(&self) -> u32 {
        self.action_space
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;
    use forge_types::grid::Direction;

    fn make_test_config() -> ForgeConfig {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config
    }

    #[test]
    fn test_simulate_produces_new_state() {
        let config = make_test_config();
        let state = WorldState::new(config);
        let model = DefaultForwardModel::new(0);

        let (next_state, result) = model.simulate(&state, &[Action::Move(Direction::Right)]);

        // Original state unchanged
        assert_eq!(state.tick, 0);
        // New state advanced
        assert_eq!(next_state.tick, 1);
        assert!(!result.terminated);
    }

    #[test]
    fn test_simulate_determinism() {
        let config = make_test_config();
        let state = WorldState::new(config);
        let model = DefaultForwardModel::new(0);

        let actions = vec![Action::Move(Direction::Right)];
        let (s1, r1) = model.simulate(&state, &actions);
        let (s2, r2) = model.simulate(&state, &actions);

        assert_eq!(s1.tick, s2.tick);
        assert_eq!(s1.agents[0].position, s2.agents[0].position);
        assert_eq!(r1.terminated, r2.terminated);
    }

    #[test]
    fn test_is_terminal() {
        let config = make_test_config();
        let state = WorldState::new(config);
        let model = DefaultForwardModel::new(0);

        assert!(!model.is_terminal(&state));
    }

    #[test]
    fn test_num_agents() {
        let config = make_test_config();
        let state = WorldState::new(config);
        let model = DefaultForwardModel::new(0);

        assert_eq!(model.num_agents(&state), 1);
    }

    #[test]
    fn test_action_space_size() {
        let model = DefaultForwardModel::new(16);
        assert_eq!(model.action_space_size(), 48);

        let model = DefaultForwardModel::new(0);
        assert_eq!(model.action_space_size(), 32);
    }
}
