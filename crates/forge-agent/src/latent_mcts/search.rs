//! Latent-space MCTS search for MuZero planning.
//!
//! Implements PUCT-based tree search that operates entirely on latent state
//! vectors from the learned world model. This module parallels
//! [`crate::mcts::search`] but does not depend on [`WorldState`].

use tracing::{instrument, trace};

use super::model::LatentForwardModel;
use super::state::LatentState;
use crate::mcts::tree::MctsConfig;

/// Min-max statistics for Q-value normalization within the search tree.
///
/// Tracks the range of observed Q-values and normalizes them to [0, 1]
/// for scale-invariant PUCT exploration.
#[derive(Debug)]
struct MinMaxStats {
    min: f32,
    max: f32,
}

impl MinMaxStats {
    fn new() -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
        }
    }

    fn update(&mut self, value: f32) {
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn normalize(&self, value: f32) -> f32 {
        if self.max > self.min {
            (value - self.min) / (self.max - self.min)
        } else {
            0.0
        }
    }
}

/// A node in the latent MCTS tree.
#[derive(Debug)]
struct LatentNode {
    /// Latent state at this node (populated on expansion).
    latent_state: Option<LatentState>,
    /// Predicted reward received when transitioning to this node.
    reward: f32,
    /// Prior probability from the prediction network.
    prior: f32,
    /// Number of times this node has been visited.
    visit_count: u32,
    /// Sum of backed-up values.
    value_sum: f64,
    /// Children indexed by action.
    children: Vec<Option<usize>>,
    /// Whether this node has been expanded.
    is_expanded: bool,
}

impl LatentNode {
    fn new(prior: f32, action_space: u32) -> Self {
        Self {
            latent_state: None,
            reward: 0.0,
            prior,
            visit_count: 0,
            value_sum: 0.0,
            children: vec![None; action_space as usize],
            is_expanded: false,
        }
    }

    fn mean_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f64
        }
    }
}

/// Result of a latent MCTS search.
#[derive(Debug, Clone)]
pub struct LatentSearchResult {
    /// The selected action.
    pub action: u32,
    /// Visit count for each action at the root.
    pub visit_counts: Vec<u32>,
    /// Estimated value at the root.
    pub root_value: f32,
}

/// Configuration for latent MCTS search.
///
/// Extends [`MctsConfig`] with MuZero-specific parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatentMctsConfig {
    /// Base MCTS configuration (c_puct, num_simulations, etc.).
    #[serde(flatten)]
    pub base: MctsConfig,
    /// Dirichlet noise alpha for root exploration.
    pub dirichlet_alpha: f32,
    /// Dirichlet noise mixing weight (0 = no noise, 1 = all noise).
    pub dirichlet_epsilon: f32,
    /// Whether to add Dirichlet exploration noise at the root.
    pub add_exploration_noise: bool,
}

impl Default for LatentMctsConfig {
    fn default() -> Self {
        Self {
            base: MctsConfig::default(),
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            add_exploration_noise: true,
        }
    }
}

/// Latent-space MCTS search engine for MuZero planning.
///
/// Performs Monte Carlo Tree Search using a learned world model
/// ([`LatentForwardModel`]) instead of the simulation engine. The search
/// builds a tree of latent states and selects actions based on
/// visit count statistics.
///
/// # Type Parameters
///
/// * `M`: A [`LatentForwardModel`] implementation providing neural inference.
pub struct LatentMctsSearch<M: LatentForwardModel> {
    model: M,
    config: LatentMctsConfig,
}

impl<M: LatentForwardModel> LatentMctsSearch<M> {
    /// Creates a new latent MCTS search engine.
    #[instrument(skip_all)]
    pub fn new(model: M, config: LatentMctsConfig) -> Self {
        Self { model, config }
    }

    /// Borrow the underlying model immutably (for introspection — number
    /// of actions, etc.).
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Borrow the underlying model mutably.
    ///
    /// Intended use is between-episode hot-reload by the runner. The
    /// `&mut self` requirement here is what enforces "not during a
    /// search": [`search`](Self::search) takes `&self`, so callers
    /// cannot hold a live search borrow while also asking for a mutable
    /// model borrow.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Run MCTS search from an observation and return the best action.
    ///
    /// # Arguments
    ///
    /// * `observation` - Flat observation vector.
    ///
    /// # Returns
    ///
    /// A [`LatentSearchResult`] containing the selected action,
    /// visit counts, and root value estimate.
    #[instrument(skip_all)]
    pub fn search(&self, observation: &[f32]) -> anyhow::Result<LatentSearchResult> {
        let action_space = self.model.action_space_size();
        let mut nodes: Vec<LatentNode> = Vec::new();
        let mut min_max = MinMaxStats::new();

        // Create root node
        let output = self.model.initial_inference(observation)?;
        let mut root = LatentNode::new(1.0, action_space);
        root.latent_state = Some(output.latent_state);
        root.is_expanded = true;

        // Expand root children with initial policy
        let policy_probs = softmax(&output.policy_logits);
        for (action_id, _prior) in policy_probs.iter().enumerate() {
            let child_idx = nodes.len() + 1 + action_id; // +1 for root at index 0
            root.children[action_id] = Some(child_idx);
        }

        // Push root
        nodes.push(root);

        // Push root's children
        for &prior in &policy_probs {
            nodes.push(LatentNode::new(prior, action_space));
        }

        // Run simulations
        for sim in 0..self.config.base.num_simulations {
            self.simulate(&mut nodes, &mut min_max, action_space)?;
            trace!(
                simulation = sim,
                tree_size = nodes.len(),
                "Latent MCTS simulation"
            );
        }

        // Collect visit counts from root children
        let mut visit_counts = vec![0u32; action_space as usize];
        for (action_id, child_idx_opt) in nodes[0].children.iter().enumerate() {
            if let Some(&child_idx) = child_idx_opt.as_ref() {
                if child_idx < nodes.len() {
                    visit_counts[action_id] = nodes[child_idx].visit_count;
                }
            }
        }

        // Select action with highest visit count
        let action = visit_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        Ok(LatentSearchResult {
            action,
            visit_counts,
            root_value: output.value,
        })
    }

    /// Run a single MCTS simulation: select → expand → backpropagate.
    #[instrument(skip_all)]
    fn simulate(
        &self,
        nodes: &mut Vec<LatentNode>,
        min_max: &mut MinMaxStats,
        action_space: u32,
    ) -> anyhow::Result<()> {
        let mut path: Vec<usize> = vec![0]; // Start at root
        let mut node_idx = 0;
        let mut depth = 0;

        // Selection: traverse tree using PUCT
        while nodes[node_idx].is_expanded && depth < self.config.base.max_depth {
            match self.select_child(nodes, node_idx, min_max) {
                Some((child_idx, _action)) => {
                    path.push(child_idx);
                    node_idx = child_idx;
                    depth += 1;
                }
                None => break,
            }
        }

        // Expansion: if the leaf is not yet expanded, run recurrent inference
        let value = if !nodes[node_idx].is_expanded {
            // Find the action that led to this node
            let parent_idx = if path.len() >= 2 {
                path[path.len() - 2]
            } else {
                0
            };

            let action = nodes[parent_idx]
                .children
                .iter()
                .enumerate()
                .find(|(_, child)| child.as_ref() == Some(&node_idx))
                .map(|(a, _)| a as u32)
                .unwrap_or(0);

            if let Some(parent_latent) = &nodes[parent_idx].latent_state {
                let output = self.model.recurrent_inference(parent_latent, action)?;
                nodes[node_idx].latent_state = Some(output.latent_state);
                nodes[node_idx].reward = output.reward;
                nodes[node_idx].is_expanded = true;

                // Expand children
                let policy_probs = softmax(&output.policy_logits);
                for (a, &prior) in policy_probs.iter().enumerate() {
                    let child = LatentNode::new(prior, action_space);
                    let child_idx = nodes.len();
                    nodes.push(child);
                    nodes[node_idx].children[a] = Some(child_idx);
                }

                output.value as f64
            } else {
                0.0
            }
        } else {
            nodes[node_idx].mean_value()
        };

        // Backpropagation
        let discount = self.config.base.discount as f64;
        let mut current_value = value;

        for &idx in path.iter().rev() {
            nodes[idx].visit_count += 1;
            nodes[idx].value_sum += current_value;
            min_max.update(nodes[idx].mean_value() as f32);
            current_value = nodes[idx].reward as f64 + discount * current_value;
        }

        Ok(())
    }

    /// Select the best child using PUCT with min-max Q normalization.
    fn select_child(
        &self,
        nodes: &[LatentNode],
        node_idx: usize,
        min_max: &MinMaxStats,
    ) -> Option<(usize, u32)> {
        let node = &nodes[node_idx];
        let parent_visits_sqrt = (node.visit_count as f64).sqrt();
        let c_puct = self.config.base.c_puct as f64;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_child = None;

        for (action, child_idx_opt) in node.children.iter().enumerate() {
            if let Some(&child_idx) = child_idx_opt.as_ref() {
                if child_idx >= nodes.len() {
                    continue;
                }
                let child = &nodes[child_idx];
                let q_value = if child.visit_count > 0 {
                    min_max.normalize(child.mean_value() as f32) as f64
                } else {
                    0.0
                };
                let exploration = c_puct * child.prior as f64 * parent_visits_sqrt
                    / (1.0 + child.visit_count as f64);
                let score = q_value + exploration;

                if score > best_score {
                    best_score = score;
                    best_child = Some((child_idx, action as u32));
                }
            }
        }

        trace!(
            node_idx,
            best = ?best_child,
            "select_child"
        );

        best_child
    }

    /// Returns the search configuration.
    pub fn config(&self) -> &LatentMctsConfig {
        &self.config
    }
}

/// Compute softmax over a slice of floats.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    if sum > 0.0 {
        exp_vals.iter().map(|x| x / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latent_mcts::model::StubLatentModel;

    fn make_search(sims: u32, action_space: u32) -> LatentMctsSearch<StubLatentModel> {
        let model = StubLatentModel::new(action_space, 64);
        let config = LatentMctsConfig {
            base: MctsConfig {
                num_simulations: sims,
                action_space,
                max_depth: 10,
                ..MctsConfig::default()
            },
            ..LatentMctsConfig::default()
        };
        LatentMctsSearch::new(model, config)
    }

    #[test]
    fn test_search_returns_valid_action() {
        let search = make_search(10, 8);
        let obs = vec![0.0; 100];
        let result = search.search(&obs).unwrap();

        assert!(result.action < 8);
        assert_eq!(result.visit_counts.len(), 8);
    }

    #[test]
    fn test_visit_counts_sum_equals_simulations() {
        let search = make_search(20, 4);
        let obs = vec![0.0; 50];
        let result = search.search(&obs).unwrap();

        // Total visits across children should approximately equal num_simulations
        let total_visits: u32 = result.visit_counts.iter().sum();
        assert!(total_visits > 0);
        assert!(total_visits <= 20);
    }

    #[test]
    fn test_zero_simulations_returns_action() {
        let search = make_search(0, 4);
        let obs = vec![0.0; 50];
        let result = search.search(&obs).unwrap();

        // Should still return a valid action (action 0 as fallback)
        assert!(result.action < 4);
    }

    #[test]
    fn test_single_action_space() {
        let search = make_search(5, 1);
        let obs = vec![0.0; 50];
        let result = search.search(&obs).unwrap();

        assert_eq!(result.action, 0);
        assert_eq!(result.visit_counts.len(), 1);
    }

    #[test]
    fn test_large_action_space() {
        let search = make_search(10, 75);
        let obs = vec![0.0; 920];
        let result = search.search(&obs).unwrap();

        assert!(result.action < 75);
        assert_eq!(result.visit_counts.len(), 75);
    }

    #[test]
    fn test_search_config() {
        let search = make_search(50, 8);
        assert_eq!(search.config().base.num_simulations, 50);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![0.0; 4];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_peaked() {
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let probs = softmax(&logits);
        assert!(probs[2] > 0.99);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_min_max_stats() {
        let mut stats = MinMaxStats::new();
        stats.update(1.0);
        stats.update(5.0);
        assert!((stats.normalize(1.0) - 0.0).abs() < 1e-6);
        assert!((stats.normalize(5.0) - 1.0).abs() < 1e-6);
        assert!((stats.normalize(3.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_min_max_stats_single_value() {
        let mut stats = MinMaxStats::new();
        stats.update(3.0);
        assert!((stats.normalize(3.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_latent_mcts_config_defaults() {
        let config = LatentMctsConfig::default();
        assert!(config.dirichlet_alpha > 0.0);
        assert!(config.dirichlet_epsilon >= 0.0 && config.dirichlet_epsilon <= 1.0);
        assert!(config.base.c_puct > 0.0);
    }

    #[test]
    fn test_min_max_stats_negative_values() {
        let mut stats = MinMaxStats::new();
        stats.update(-10.0);
        stats.update(-1.0);
        assert!((stats.normalize(-5.5) - 0.5).abs() < 1e-6);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn visit_counts_bounded(sims in 1u32..50, actions in 2u32..20) {
                let search = make_search(sims, actions);
                let obs = vec![0.0; 50];
                let result = search.search(&obs).unwrap();
                let total: u32 = result.visit_counts.iter().sum();
                prop_assert!(total <= sims);
            }

            #[test]
            fn action_in_range(sims in 1u32..20, actions in 1u32..50) {
                let search = make_search(sims, actions);
                let obs = vec![0.0; 100];
                let result = search.search(&obs).unwrap();
                prop_assert!(result.action < actions);
            }
        }
    }
}
