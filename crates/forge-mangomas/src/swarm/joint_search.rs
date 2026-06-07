//! Joint multi-agent MCTS planner.
//!
//! Produces a coordinated joint action (one [`Action`] per agent) for a shared
//! [`WorldState`], reusing the **existing** single-agent PUCT search
//! ([`MctsSearch`](forge_agent::mcts::search::MctsSearch)) rather than
//! reimplementing tree search. Two strategies avoid the `A^n` joint
//! action-space blow-up:
//!
//! * [`JointStrategy::SequentialFactored`] — plan every agent at the shared root
//!   world with the single-agent search (cost `n * num_simulations`, linear in
//!   agents). Deterministic: the underlying tree selection is pure PUCT argmax.
//! * [`JointStrategy::Sampled`] — sample `K` joint actions from the critic's
//!   priors with a seeded RNG, score each with the (centralized) critic's joint
//!   value, and keep the argmax. Deterministic for a fixed `seed`.

use forge_agent::forward_model::ForwardModel;
use forge_agent::mcts::search::MctsSearch;
use forge_core::WorldState;
use forge_types::Action;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use tracing::{debug, instrument, trace, warn};

use super::centralized_critic::JointPolicyValue;
use super::cooperative_mcts::{CooperativeMctsConfig, JointStrategy};
use crate::error::{MangoMasError, MangoMasResult};

/// Joint MCTS planner over a shared world.
///
/// Generic over the forward model `F` and the joint critic `C` so a learned
/// model can be injected later (see the `onnx` sub-track); the default protocol
/// uses [`DefaultForwardModel`](forge_agent::forward_model::DefaultForwardModel)
/// with a [`Critic`](super::centralized_critic::Critic) over a
/// [`UniformPolicy`](forge_agent::mcts::policy::UniformPolicy).
#[derive(Debug, Clone)]
pub struct JointMctsPlanner<F, C>
where
    F: ForwardModel + Clone,
    C: JointPolicyValue + Clone,
{
    model: F,
    critic: C,
    config: CooperativeMctsConfig,
}

impl<F, C> JointMctsPlanner<F, C>
where
    F: ForwardModel + Clone,
    C: JointPolicyValue + Clone,
{
    /// Builds a planner from a forward model, joint critic, and config.
    pub fn new(model: F, critic: C, config: CooperativeMctsConfig) -> Self {
        Self {
            model,
            critic,
            config,
        }
    }

    /// Plans a joint action (one per agent in `world`).
    ///
    /// The world's agent count is authoritative so the result is always
    /// compatible with [`WorldState::step`]. A mismatch against the configured
    /// `num_agents` is logged but not fatal; an empty world is an error.
    #[instrument(skip(self, world), fields(strategy = ?self.config.joint_strategy))]
    pub fn plan(&self, world: &WorldState) -> MangoMasResult<Vec<Action>> {
        let n = self.model.num_agents(world);
        if n == 0 {
            return Err(MangoMasError::SwarmCoordination(
                "cannot plan a joint action for a world with no agents".to_string(),
            ));
        }
        if n != self.config.num_agents {
            warn!(
                world_agents = n,
                config_agents = self.config.num_agents,
                "world agent count differs from configured num_agents; using world count"
            );
        }
        debug!(num_agents = n, seed = self.config.seed, "joint plan start");

        let actions = match self.config.joint_strategy {
            JointStrategy::SequentialFactored => self.plan_factored(world, n),
            JointStrategy::Sampled => self.plan_sampled(world, n),
        };
        debug!(num_actions = actions.len(), "joint plan complete");
        Ok(actions)
    }

    /// Per-agent single-agent MCTS at the shared root world. Deterministic.
    ///
    /// Each agent plans against the same root state (others assumed idle), i.e.
    /// simultaneous best-response. This is intentionally *not* full
    /// coordinate-ascent conditioning: committing an earlier agent's action into
    /// the world before planning the next would require advancing a tick (via
    /// the forward model), which corrupts single-timestep joint semantics. Richer
    /// cross-agent conditioning would need a joint forward model that holds
    /// partial commitments without stepping — a future enhancement once such a
    /// model exists. The [`JointStrategy::Sampled`] path does score full joint
    /// actions against the centralized critic.
    fn plan_factored(&self, world: &WorldState, n: usize) -> Vec<Action> {
        // Align the tree's action space with the real (comm-vocab-derived)
        // action space so expansion covers every valid action — no hard-coded
        // width.
        let mut mcts_cfg = self.config.mcts.clone();
        mcts_cfg.action_space = self.model.action_space_size();

        let search = MctsSearch::new(
            self.model.clone(),
            self.critic.clone(),
            mcts_cfg,
            self.config.comm_vocab_size,
        );

        let mut actions = Vec::with_capacity(n);
        for agent_idx in 0..n {
            let action = search.search(world, agent_idx);
            trace!(agent_idx, action = ?action, "factored agent decision");
            actions.push(action);
        }
        actions
    }

    /// Sample `K` joint actions from the critic priors, keep the argmax by joint
    /// value. Deterministic given `config.seed`.
    fn plan_sampled(&self, world: &WorldState, n: usize) -> Vec<Action> {
        let action_space = self.model.action_space_size();
        let mut rng = Pcg64Mcg::seed_from_u64(self.config.seed);

        let mut best: Option<(f32, Vec<Action>)> = None;
        for sample in 0..self.config.sampled_joint_actions {
            let joint: Vec<Action> = (0..n)
                .map(|agent_idx| {
                    let priors = self.critic.evaluate(world, agent_idx).priors;
                    let id = sample_index(&priors, action_space, &mut rng);
                    Action::from_discrete(id, self.config.comm_vocab_size, false)
                        .unwrap_or(Action::Noop)
                })
                .collect();

            let (next_state, _) = self.model.simulate(world, &joint);
            let score = self.critic.joint_value(&next_state, n);
            trace!(sample, score, "sampled joint candidate");

            // Strictly-greater keeps the lowest sample index on ties → deterministic.
            // (`map_or` rather than `is_none_or`: MSRV is 1.75, the latter is 1.82.)
            if best.as_ref().map_or(true, |(b, _)| score > *b) {
                best = Some((score, joint));
            }
        }

        // `sampled_joint_actions == 0` would leave `best` empty; fall back to a
        // safe all-Noop joint of the right length.
        best.map(|(_, joint)| joint)
            .unwrap_or_else(|| vec![Action::Noop; n])
    }
}

/// Deterministically samples a discrete action id from `priors` using `rng`.
///
/// Falls back to a uniform draw over `[0, action_space)` when the priors are
/// empty or sum to zero, and clamps to the valid action range.
fn sample_index(priors: &[f32], action_space: u32, rng: &mut Pcg64Mcg) -> u32 {
    if action_space == 0 {
        return 0;
    }
    let total: f32 = priors.iter().filter(|p| p.is_finite() && **p > 0.0).sum();
    if priors.is_empty() || total <= 0.0 {
        return rng.gen_range(0..action_space);
    }
    let threshold = rng.gen_range(0.0..total);
    let mut cumulative = 0.0;
    for (idx, &p) in priors.iter().enumerate() {
        if p.is_finite() && p > 0.0 {
            cumulative += p;
            if cumulative >= threshold {
                return (idx as u32).min(action_space - 1);
            }
        }
    }
    action_space - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::centralized_critic::Critic;
    use forge_agent::forward_model::DefaultForwardModel;
    use forge_agent::mcts::policy::UniformPolicy;
    use forge_types::config::ForgeConfig;

    const TEST_COMM_VOCAB: u16 = 0;

    fn make_world(num_agents: usize) -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.world.seed = 42;
        config.agents.num_agents = num_agents as u32;
        config.agents.comm_vocab_size = TEST_COMM_VOCAB;
        config.agents.default_vision_radius = 3;
        WorldState::new(config).unwrap()
    }

    fn make_planner(
        strategy: JointStrategy,
        num_agents: usize,
    ) -> JointMctsPlanner<DefaultForwardModel, Critic<UniformPolicy>> {
        let mut config = CooperativeMctsConfig {
            num_agents,
            joint_strategy: strategy,
            comm_vocab_size: TEST_COMM_VOCAB,
            ..CooperativeMctsConfig::default()
        };
        // Keep tests fast.
        config.mcts.num_simulations = 8;
        let model = DefaultForwardModel::new(TEST_COMM_VOCAB);
        let action_space = model.action_space_size();
        let critic =
            Critic::from_config(UniformPolicy::new(action_space), config.centralized_critic);
        JointMctsPlanner::new(model, critic, config)
    }

    #[test]
    fn test_factored_length_matches_agents() {
        let planner = make_planner(JointStrategy::SequentialFactored, 3);
        let world = make_world(3);
        let actions = planner.plan(&world).unwrap();
        assert_eq!(actions.len(), 3);
    }

    #[test]
    fn test_sampled_length_matches_agents() {
        let planner = make_planner(JointStrategy::Sampled, 2);
        let world = make_world(2);
        let actions = planner.plan(&world).unwrap();
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_factored_is_deterministic() {
        let planner = make_planner(JointStrategy::SequentialFactored, 3);
        let world = make_world(3);
        let a = planner.plan(&world).unwrap();
        let b = planner.plan(&world).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_sampled_is_deterministic() {
        let planner = make_planner(JointStrategy::Sampled, 2);
        let world = make_world(2);
        let a = planner.plan(&world).unwrap();
        let b = planner.plan(&world).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_actions_in_valid_range() {
        let planner = make_planner(JointStrategy::SequentialFactored, 2);
        let world = make_world(2);
        let space = DefaultForwardModel::new(TEST_COMM_VOCAB).action_space_size();
        for action in planner.plan(&world).unwrap() {
            let id = action.try_to_discrete_full(TEST_COMM_VOCAB).unwrap();
            assert!(id < space);
        }
    }

    /// Forward model that reports zero agents, to cover the defensive guard in
    /// `plan` without an (impossible) zero-agent `WorldState`.
    #[derive(Debug, Clone)]
    struct ZeroAgentModel(DefaultForwardModel);

    impl ForwardModel for ZeroAgentModel {
        fn simulate(
            &self,
            state: &WorldState,
            actions: &[Action],
        ) -> (WorldState, forge_types::observation::StepResult) {
            self.0.simulate(state, actions)
        }
        fn is_terminal(&self, state: &WorldState) -> bool {
            self.0.is_terminal(state)
        }
        fn num_agents(&self, _state: &WorldState) -> usize {
            0
        }
        fn action_space_size(&self) -> u32 {
            self.0.action_space_size()
        }
    }

    #[test]
    fn test_zero_agents_is_error() {
        let config = CooperativeMctsConfig {
            num_agents: 1,
            comm_vocab_size: TEST_COMM_VOCAB,
            ..CooperativeMctsConfig::default()
        };
        let model = ZeroAgentModel(DefaultForwardModel::new(TEST_COMM_VOCAB));
        let critic = Critic::from_config(UniformPolicy::new(4), true);
        let planner = JointMctsPlanner::new(model, critic, config);
        // A valid 1-agent world, but the model reports 0 agents.
        let world = make_world(1);
        let err = planner.plan(&world).unwrap_err();
        assert!(matches!(err, MangoMasError::SwarmCoordination(_)));
    }

    #[test]
    fn test_agent_count_mismatch_uses_world_count() {
        // config says 5 agents but the world only has 2; plan must return the
        // world's count (and log a warn) so the result fits WorldState::step.
        let mut config = CooperativeMctsConfig {
            num_agents: 5,
            comm_vocab_size: TEST_COMM_VOCAB,
            ..CooperativeMctsConfig::default()
        };
        config.mcts.num_simulations = 4;
        let model = DefaultForwardModel::new(TEST_COMM_VOCAB);
        let action_space = model.action_space_size();
        let critic = Critic::from_config(UniformPolicy::new(action_space), true);
        let planner = JointMctsPlanner::new(model, critic, config);
        let world = make_world(2);
        assert_eq!(planner.plan(&world).unwrap().len(), 2);
    }

    #[test]
    fn test_sampled_zero_budget_falls_back_to_noop() {
        let mut config = CooperativeMctsConfig {
            num_agents: 2,
            joint_strategy: JointStrategy::Sampled,
            sampled_joint_actions: 0,
            comm_vocab_size: TEST_COMM_VOCAB,
            ..CooperativeMctsConfig::default()
        };
        config.mcts.num_simulations = 2;
        let model = DefaultForwardModel::new(TEST_COMM_VOCAB);
        let action_space = model.action_space_size();
        let critic = Critic::from_config(UniformPolicy::new(action_space), true);
        let planner = JointMctsPlanner::new(model, critic, config);
        let world = make_world(2);
        let actions = planner.plan(&world).unwrap();
        assert_eq!(actions, vec![Action::Noop; 2]);
    }

    #[test]
    fn test_sample_index_uniform_fallback() {
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        // Empty priors → uniform fallback in range.
        let id = sample_index(&[], 5, &mut rng);
        assert!(id < 5);
        // Zero action space → 0.
        assert_eq!(sample_index(&[1.0], 0, &mut rng), 0);
    }

    #[test]
    fn test_sample_index_weighted_picks_dominant() {
        let mut rng = Pcg64Mcg::seed_from_u64(123);
        // All weight on index 2.
        let priors = vec![0.0, 0.0, 1.0, 0.0];
        for _ in 0..10 {
            assert_eq!(sample_index(&priors, 4, &mut rng), 2);
        }
    }
}
