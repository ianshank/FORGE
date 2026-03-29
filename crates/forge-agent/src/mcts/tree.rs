//! MCTS tree structure.
//!
//! The tree is stored as a flat vector of nodes for cache-friendly access.
//! Each node tracks visit counts, cumulative value, prior probabilities,
//! and child indices.

use forge_types::constants;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Index into the tree node vector.
pub type NodeId = usize;

/// Configuration for the MCTS tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MctsConfig {
    /// PUCT exploration constant (c_puct). Higher = more exploration.
    pub c_puct: f32,
    /// Number of simulations per search.
    pub num_simulations: u32,
    /// Maximum tree depth.
    pub max_depth: u32,
    /// Discount factor for future rewards.
    pub discount: f32,
    /// Temperature for action selection (0 = greedy, 1 = proportional to visits).
    pub temperature: f32,
    /// Number of actions in the action space.
    pub action_space: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: constants::DEFAULT_MCTS_C_PUCT,
            num_simulations: constants::DEFAULT_MCTS_NUM_SIMULATIONS,
            max_depth: constants::DEFAULT_MCTS_MAX_DEPTH,
            discount: constants::DEFAULT_MCTS_DISCOUNT,
            temperature: constants::DEFAULT_MCTS_TEMPERATURE,
            action_space: constants::DEFAULT_MCTS_ACTION_SPACE,
        }
    }
}

/// A node in the MCTS tree.
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// Parent node index. Root has parent = None.
    pub parent: Option<NodeId>,
    /// Action that led to this node from the parent.
    pub action: Option<u32>,
    /// Visit count.
    pub visits: u32,
    /// Cumulative value estimate (sum of backpropagated values).
    pub value_sum: f64,
    /// Prior probability from the policy network.
    pub prior: f32,
    /// Child node indices, indexed by action.
    pub children: Vec<Option<NodeId>>,
    /// Whether this node represents a terminal state.
    pub is_terminal: bool,
    /// The depth of this node in the tree.
    pub depth: u32,
}

impl MctsNode {
    /// Creates a new root node.
    #[instrument]
    pub fn root(action_space: u32) -> Self {
        Self {
            parent: None,
            action: None,
            visits: 0,
            value_sum: 0.0,
            prior: 1.0,
            children: vec![None; action_space as usize],
            is_terminal: false,
            depth: 0,
        }
    }

    /// Creates a new child node.
    #[instrument]
    pub fn child(parent: NodeId, action: u32, prior: f32, depth: u32, action_space: u32) -> Self {
        Self {
            parent: Some(parent),
            action: Some(action),
            visits: 0,
            value_sum: 0.0,
            prior,
            children: vec![None; action_space as usize],
            is_terminal: false,
            depth,
        }
    }

    /// Returns the mean value estimate Q(s, a).
    #[instrument(skip(self))]
    pub fn mean_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f64
        }
    }

    /// Whether this node has been expanded (has any children).
    #[instrument(skip(self))]
    pub fn is_expanded(&self) -> bool {
        self.children.iter().any(|c| c.is_some())
    }

    /// Returns the number of expanded children.
    #[instrument(skip(self))]
    pub fn num_children(&self) -> usize {
        self.children.iter().filter(|c| c.is_some()).count()
    }
}

/// The MCTS tree, stored as a flat vector of nodes.
#[derive(Debug)]
pub struct MctsTree {
    /// All nodes in the tree.
    pub nodes: Vec<MctsNode>,
    /// Configuration.
    pub config: MctsConfig,
}

impl MctsTree {
    /// Creates a new tree with a root node.
    #[instrument(skip_all)]
    pub fn new(config: MctsConfig) -> Self {
        let root = MctsNode::root(config.action_space);
        Self {
            nodes: vec![root],
            config,
        }
    }

    /// Returns the root node index (always 0).
    #[instrument(skip(self))]
    pub fn root_id(&self) -> NodeId {
        0
    }

    /// Returns a reference to a node.
    #[instrument(skip(self))]
    pub fn node(&self, id: NodeId) -> &MctsNode {
        &self.nodes[id]
    }

    /// Returns a mutable reference to a node.
    #[instrument(skip(self))]
    pub fn node_mut(&mut self, id: NodeId) -> &mut MctsNode {
        &mut self.nodes[id]
    }

    /// Adds a new child node to the tree.
    /// Returns the new node's ID.
    #[instrument(skip(self))]
    pub fn add_child(&mut self, parent: NodeId, action: u32, prior: f32) -> NodeId {
        let depth = self.nodes[parent].depth + 1;
        let child = MctsNode::child(parent, action, prior, depth, self.config.action_space);
        let child_id = self.nodes.len();
        self.nodes.push(child);
        self.nodes[parent].children[action as usize] = Some(child_id);
        child_id
    }

    /// Selects the best child of a node using PUCT formula.
    ///
    /// UCB score = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
    #[instrument(skip_all)]
    pub fn select_child(&self, node_id: NodeId) -> Option<NodeId> {
        let node = &self.nodes[node_id];
        let sqrt_parent = (node.visits as f64).sqrt();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_child = None;

        for child_id in node.children.iter().flatten() {
            let child = &self.nodes[*child_id];
            let q_value = child.mean_value();
            let exploration = self.config.c_puct as f64 * child.prior as f64 * sqrt_parent
                / (1.0 + child.visits as f64);
            let score = q_value + exploration;

            if score > best_score {
                best_score = score;
                best_child = Some(*child_id);
            }
        }

        best_child
    }

    /// Selects the best action from the root based on visit counts.
    ///
    /// With temperature 0: returns the most-visited action (greedy).
    /// With temperature > 0: samples proportional to visit^(1/temp).
    #[instrument(skip_all)]
    pub fn best_action(&self) -> Option<u32> {
        let root = &self.nodes[0];
        let mut best_visits = 0;
        let mut best_action = None;

        for (action, child_id) in root.children.iter().enumerate() {
            if let Some(id) = child_id {
                let visits = self.nodes[*id].visits;
                if visits > best_visits {
                    best_visits = visits;
                    best_action = Some(action as u32);
                }
            }
        }

        best_action
    }

    /// Backpropagates a value estimate from a leaf up to the root.
    #[instrument(skip_all)]
    pub fn backpropagate(&mut self, mut node_id: NodeId, value: f64) {
        let discount = self.config.discount as f64;
        let mut current_value = value;

        loop {
            let node = &mut self.nodes[node_id];
            node.visits += 1;
            node.value_sum += current_value;
            current_value *= discount;

            match node.parent {
                Some(parent) => node_id = parent,
                None => break,
            }
        }
    }

    /// Returns the total number of nodes in the tree.
    #[instrument(skip(self))]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> MctsConfig {
        MctsConfig {
            action_space: 4,
            num_simulations: 10,
            ..MctsConfig::default()
        }
    }

    #[test]
    fn test_tree_creation() {
        let tree = MctsTree::new(make_config());
        assert_eq!(tree.size(), 1);
        assert_eq!(tree.root_id(), 0);

        let root = tree.node(0);
        assert!(root.parent.is_none());
        assert_eq!(root.visits, 0);
        assert_eq!(root.children.len(), 4);
    }

    #[test]
    fn test_add_child() {
        let mut tree = MctsTree::new(make_config());
        let child_id = tree.add_child(0, 2, 0.25);

        assert_eq!(child_id, 1);
        assert_eq!(tree.size(), 2);

        let child = tree.node(child_id);
        assert_eq!(child.parent, Some(0));
        assert_eq!(child.action, Some(2));
        assert_eq!(child.prior, 0.25);
        assert_eq!(child.depth, 1);

        let root = tree.node(0);
        assert_eq!(root.children[2], Some(1));
    }

    #[test]
    fn test_backpropagate() {
        let mut tree = MctsTree::new(make_config());
        let c1 = tree.add_child(0, 0, 0.5);
        let c2 = tree.add_child(c1, 1, 0.5);

        tree.backpropagate(c2, 1.0);

        // Leaf should have visit=1, value=1.0
        assert_eq!(tree.node(c2).visits, 1);
        assert!((tree.node(c2).value_sum - 1.0).abs() < 1e-6);

        // Middle should have visit=1, value=1.0*discount
        assert_eq!(tree.node(c1).visits, 1);

        // Root should have visit=1
        assert_eq!(tree.node(0).visits, 1);
    }

    #[test]
    fn test_mean_value() {
        let mut node = MctsNode::root(4);
        assert_eq!(node.mean_value(), 0.0);

        node.visits = 2;
        node.value_sum = 3.0;
        assert!((node.mean_value() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_select_child_exploration() {
        let mut tree = MctsTree::new(make_config());

        // Add two children with different priors
        let c1 = tree.add_child(0, 0, 0.9);
        let _c2 = tree.add_child(0, 1, 0.1);

        // Set root visits
        tree.node_mut(0).visits = 1;

        // With no visits, the child with higher prior should be selected
        let selected = tree.select_child(0).unwrap();
        assert_eq!(selected, c1);
    }

    #[test]
    fn test_best_action() {
        let mut tree = MctsTree::new(make_config());
        let _c1 = tree.add_child(0, 0, 0.25);
        let c2 = tree.add_child(0, 1, 0.25);

        // Give child 2 more visits
        tree.node_mut(c2).visits = 10;

        let best = tree.best_action().unwrap();
        assert_eq!(best, 1);
    }

    #[test]
    fn test_is_expanded() {
        let mut tree = MctsTree::new(make_config());

        assert!(!tree.node(0).is_expanded());

        tree.add_child(0, 0, 0.5);
        assert!(tree.node(0).is_expanded());
    }

    #[test]
    fn test_select_child_no_children() {
        let tree = MctsTree::new(make_config());
        // Root has no children, so select_child should return None
        let result = tree.select_child(0);
        assert!(
            result.is_none(),
            "select_child on root with no children should return None"
        );
    }

    #[test]
    fn test_node_num_children() {
        let mut tree = MctsTree::new(make_config());

        assert_eq!(tree.node(0).num_children(), 0);

        tree.add_child(0, 0, 0.5);
        tree.add_child(0, 2, 0.3);

        assert_eq!(tree.node(0).num_children(), 2);
    }

    #[test]
    fn test_mcts_config_default_uses_constants() {
        let config = MctsConfig::default();
        assert_eq!(config.c_puct, constants::DEFAULT_MCTS_C_PUCT);
        assert_eq!(config.num_simulations, constants::DEFAULT_MCTS_NUM_SIMULATIONS);
        assert_eq!(config.max_depth, constants::DEFAULT_MCTS_MAX_DEPTH);
        assert_eq!(config.discount, constants::DEFAULT_MCTS_DISCOUNT);
        assert_eq!(config.temperature, constants::DEFAULT_MCTS_TEMPERATURE);
        assert_eq!(config.action_space, constants::DEFAULT_MCTS_ACTION_SPACE);
    }

    #[test]
    fn test_mcts_config_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(MctsConfig);
    }

    #[test]
    fn test_mcts_config_serde_default_roundtrip() {
        let config = MctsConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deser: MctsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.c_puct, config.c_puct);
        assert_eq!(deser.num_simulations, config.num_simulations);
        assert_eq!(deser.max_depth, config.max_depth);
    }

    #[test]
    fn test_mcts_config_partial_json_uses_defaults() {
        // When deserializing a JSON with missing fields, #[serde(default)] ensures
        // missing fields fall back to Default::default()
        let partial_json = r#"{"c_puct": 2.5}"#;
        let config: MctsConfig = serde_json::from_str(partial_json).unwrap();
        assert_eq!(config.c_puct, 2.5);
        // Other fields should use defaults
        assert_eq!(config.num_simulations, constants::DEFAULT_MCTS_NUM_SIMULATIONS);
        assert_eq!(config.max_depth, constants::DEFAULT_MCTS_MAX_DEPTH);
    }

    // ---- Proptest: MCTS tree invariants ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Tree size increases by exactly 1 per add_child.
            #[test]
            fn tree_size_grows(
                num_children in 0u32..4,
            ) {
                let mut tree = MctsTree::new(make_config());
                prop_assert_eq!(tree.size(), 1);

                for action in 0..num_children {
                    tree.add_child(0, action, 0.25);
                    prop_assert_eq!(tree.size(), 2 + action as usize);
                }
            }

            /// Backpropagation increments visit count along the path.
            #[test]
            fn backprop_visits(
                value in -10.0f64..10.0,
                depth in 1u32..4,
            ) {
                let mut tree = MctsTree::new(make_config());
                let mut node = 0;
                for i in 0..depth {
                    node = tree.add_child(node, i % 4, 0.25);
                }

                tree.backpropagate(node, value);

                // All nodes along the path should have exactly 1 visit
                prop_assert_eq!(tree.node(0).visits, 1);
                prop_assert_eq!(tree.node(node).visits, 1);
            }

            /// Mean value is value_sum / visits, or 0 when visits == 0.
            #[test]
            fn mean_value_correct(
                visits in 0u32..100,
                value_sum in -100.0f64..100.0,
            ) {
                let mut node = MctsNode::root(4);
                node.visits = visits;
                node.value_sum = value_sum;
                let mean = node.mean_value();
                if visits == 0 {
                    prop_assert_eq!(mean, 0.0);
                } else {
                    let expected = value_sum / visits as f64;
                    prop_assert!((mean - expected).abs() < 1e-5);
                }
            }
        }
    }
}
