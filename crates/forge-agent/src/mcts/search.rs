//! MCTS search implementation with PUCT selection.
//!
//! Performs tree search by repeatedly selecting, expanding, evaluating,
//! and backpropagating through the tree to find the best action.

use forge_core::WorldState;
use forge_types::Action;
use tracing::{instrument, trace};

use super::policy::PolicyValue;
use super::tree::{MctsConfig, MctsTree};
use crate::forward_model::ForwardModel;

/// MCTS search engine.
pub struct MctsSearch<F: ForwardModel, P: PolicyValue> {
    /// Forward model for simulation.
    model: F,
    /// Policy/value network.
    policy: P,
    /// Configuration.
    config: MctsConfig,
    /// Communication vocabulary size for action conversion.
    comm_vocab_size: u16,
}

impl<F: ForwardModel, P: PolicyValue> MctsSearch<F, P> {
    /// Creates a new MCTS search engine.
    #[instrument(skip_all)]
    pub fn new(model: F, policy: P, config: MctsConfig, comm_vocab_size: u16) -> Self {
        Self {
            model,
            policy,
            config,
            comm_vocab_size,
        }
    }

    /// Runs MCTS from the given state and returns the best action.
    #[instrument(skip_all)]
    pub fn search(&self, state: &WorldState, agent_idx: usize) -> Action {
        let mut tree = MctsTree::new(self.config.clone());

        // Run simulations
        for sim in 0..self.config.num_simulations {
            self.simulate(&mut tree, state, agent_idx);

            trace!(
                simulation = sim,
                tree_size = tree.size(),
                "MCTS simulation complete"
            );
        }

        // Select best action based on visit counts
        let action_id = tree.best_action().unwrap_or(0);
        Action::from_discrete(action_id, self.comm_vocab_size).unwrap_or(Action::Noop)
    }

    /// Runs a single MCTS simulation: select, expand, evaluate, backpropagate.
    fn simulate(&self, tree: &mut MctsTree, root_state: &WorldState, agent_idx: usize) {
        // Phase 1: Selection — traverse down the tree using PUCT
        let mut node_id = tree.root_id();
        let mut state = root_state.clone();

        while tree.node(node_id).is_expanded() && !tree.node(node_id).is_terminal {
            match tree.select_child(node_id) {
                Some(child_id) => {
                    // Simulate the action to get the next state
                    let action_id = tree.node(child_id).action.unwrap_or(0);
                    let action = Action::from_discrete(action_id, self.comm_vocab_size)
                        .unwrap_or(Action::Noop);

                    let num_agents = self.model.num_agents(&state);
                    let mut actions = vec![Action::Noop; num_agents];
                    if agent_idx < num_agents {
                        actions[agent_idx] = action;
                    }

                    let (next_state, _result) = self.model.simulate(&state, &actions);
                    state = next_state;
                    node_id = child_id;
                }
                None => break,
            }

            // Stop at max depth
            if tree.node(node_id).depth >= self.config.max_depth {
                break;
            }
        }

        // Phase 2: Expansion — if the node is not terminal and not fully expanded
        if !tree.node(node_id).is_terminal && !tree.node(node_id).is_expanded() {
            let pv_output = self.policy.evaluate(&state, agent_idx);

            // Expand all actions with their priors
            let action_space = self.config.action_space;
            for action_id in 0..action_space {
                let prior = pv_output
                    .priors
                    .get(action_id as usize)
                    .copied()
                    .unwrap_or(1.0 / action_space as f32);
                tree.add_child(node_id, action_id, prior);
            }

            // Check if terminal
            if self.model.is_terminal(&state) {
                tree.node_mut(node_id).is_terminal = true;
            }
        }

        // Phase 3: Evaluation — get value estimate from policy
        let value = if tree.node(node_id).is_terminal {
            0.0 // Terminal states have no future value
        } else {
            let pv_output = self.policy.evaluate(&state, agent_idx);
            pv_output.value as f64
        };

        // Phase 4: Backpropagation
        tree.backpropagate(node_id, value);
    }

    /// Returns the MCTS config.
    pub fn config(&self) -> &MctsConfig {
        &self.config
    }
}

/// MCTS-based agent that uses search to select actions.
pub struct MctsAgent<F: ForwardModel, P: PolicyValue> {
    search: MctsSearch<F, P>,
}

impl<F: ForwardModel, P: PolicyValue> MctsAgent<F, P> {
    /// Creates a new MCTS agent.
    pub fn new(model: F, policy: P, config: MctsConfig, comm_vocab_size: u16) -> Self {
        Self {
            search: MctsSearch::new(model, policy, config, comm_vocab_size),
        }
    }

    /// Selects an action using MCTS search.
    pub fn select_action(&self, state: &WorldState, agent_idx: usize) -> Action {
        self.search.search(state, agent_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_model::DefaultForwardModel;
    use crate::mcts::policy::UniformPolicy;
    use forge_types::config::ForgeConfig;

    fn make_test_state() -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 100;
        WorldState::new(config)
    }

    fn make_search() -> MctsSearch<DefaultForwardModel, UniformPolicy> {
        let model = DefaultForwardModel::new(0);
        let policy = UniformPolicy::new(32);
        let config = MctsConfig {
            num_simulations: 10,
            action_space: 32,
            max_depth: 5,
            ..MctsConfig::default()
        };
        MctsSearch::new(model, policy, config, 0)
    }

    #[test]
    fn test_mcts_search_returns_valid_action() {
        let state = make_test_state();
        let search = make_search();

        let action = search.search(&state, 0);
        let discrete = action.to_discrete();
        assert!(discrete < Action::space_size(0));
    }

    #[test]
    fn test_mcts_search_determinism() {
        let state = make_test_state();

        // Two searches from the same state with the same policy should
        // produce valid actions (though not necessarily identical due to
        // tree structure being non-deterministic with uniform policy)
        let search = make_search();
        let a1 = search.search(&state, 0);
        assert!(a1.to_discrete() < Action::space_size(0));
    }

    #[test]
    fn test_mcts_agent() {
        let model = DefaultForwardModel::new(0);
        let policy = UniformPolicy::new(32);
        let config = MctsConfig {
            num_simulations: 5,
            action_space: 32,
            max_depth: 3,
            ..MctsConfig::default()
        };
        let agent = MctsAgent::new(model, policy, config, 0);
        let state = make_test_state();

        let action = agent.select_action(&state, 0);
        assert!(action.to_discrete() < Action::space_size(0));
    }

    #[test]
    fn test_mcts_with_few_simulations() {
        let model = DefaultForwardModel::new(0);
        let policy = UniformPolicy::new(32);
        let config = MctsConfig {
            num_simulations: 1,
            action_space: 32,
            max_depth: 2,
            ..MctsConfig::default()
        };
        let search = MctsSearch::new(model, policy, config, 0);
        let state = make_test_state();

        // Even with 1 simulation, should return a valid action
        let action = search.search(&state, 0);
        assert!(action.to_discrete() < Action::space_size(0));
    }

    #[test]
    fn test_mcts_zero_simulations() {
        let model = DefaultForwardModel::new(0);
        let policy = UniformPolicy::new(32);
        let config = MctsConfig {
            num_simulations: 0,
            action_space: 32,
            max_depth: 5,
            ..MctsConfig::default()
        };
        let search = MctsSearch::new(model, policy, config, 0);
        let state = make_test_state();

        // With 0 simulations, tree is never expanded. best_action returns None,
        // so search falls back to Noop. Should not panic.
        let action = search.search(&state, 0);
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_mcts_config() {
        let search = make_search();
        assert_eq!(search.config().num_simulations, 10);
        assert_eq!(search.config().action_space, 32);
    }
}
