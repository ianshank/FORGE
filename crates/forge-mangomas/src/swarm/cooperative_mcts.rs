//! Cooperative multi-agent MCTS protocol.
//!
//! Implements Centralized-Training / Decentralized-Execution (CTDE) joint
//! planning for a drone swarm by reusing FORGE's existing single-agent PUCT
//! search. [`CooperativeMctsProtocol`] implements **both** the existing
//! [`SwarmProtocol`](super::protocol::SwarmProtocol) (so it is drop-in
//! swappable with `IndependentProtocol`) and the
//! [`ActionPolicy`](crate::batch_runner::ActionPolicy) (so it plugs into the
//! existing batch-collection loop via a cached joint plan).
//!
//! All tunables flow through [`CooperativeMctsConfig`] (no hard-coded values);
//! the planner is deterministic for a fixed seed.

use std::sync::Mutex;

use forge_agent::forward_model::{DefaultForwardModel, ForwardModel};
use forge_agent::mcts::policy::UniformPolicy;
use forge_agent::mcts::tree::MctsConfig;
use forge_core::WorldState;
use forge_types::action::Action;
use forge_types::observation::Observation;
use serde::{Deserialize, Serialize};
use tracing::{instrument, warn};

use super::centralized_critic::Critic;
use super::joint_search::JointMctsPlanner;
use super::protocol::SwarmProtocol;
use crate::batch_runner::ActionPolicy;
use crate::error::MangoMasResult;

/// Default number of agents to coordinate.
const DEFAULT_NUM_AGENTS: usize = 2;
/// Default critic mode: centralized (cooperative) value aggregation.
const DEFAULT_CENTRALIZED_CRITIC: bool = true;
/// Default joint-search strategy.
const DEFAULT_JOINT_STRATEGY: JointStrategy = JointStrategy::SequentialFactored;
/// Default number of sampled joint actions (only used by [`JointStrategy::Sampled`]).
const DEFAULT_SAMPLED_JOINT_ACTIONS: u32 = 16;
/// Default RNG seed for deterministic sampling.
const DEFAULT_COOP_SEED: u64 = 0;
/// Default communication vocabulary size for action encoding.
const DEFAULT_COMM_VOCAB_SIZE: u16 = 0;

/// Strategy for combining per-agent decisions into a joint action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointStrategy {
    /// Plan every agent at the shared root world with the single-agent MCTS.
    /// Cost is linear in the number of agents; fully deterministic.
    SequentialFactored,
    /// Sample `K` joint actions from the critic priors and keep the argmax by
    /// joint value. Deterministic for a fixed seed.
    Sampled,
}

impl Default for JointStrategy {
    fn default() -> Self {
        DEFAULT_JOINT_STRATEGY
    }
}

/// Configuration for cooperative MCTS.
///
/// Mirrors the project config convention: `#[serde(default)]`, module-level
/// `const DEFAULT_*`, and an explicit [`Default`] impl. Embeds the reusable
/// single-agent [`MctsConfig`] verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CooperativeMctsConfig {
    /// Number of agents to coordinate.
    pub num_agents: usize,
    /// Whether to use centralized (cooperative) value estimation.
    pub centralized_critic: bool,
    /// Joint-search strategy.
    pub joint_strategy: JointStrategy,
    /// Number of sampled joint actions (only used by [`JointStrategy::Sampled`]).
    pub sampled_joint_actions: u32,
    /// RNG seed for deterministic sampling.
    pub seed: u64,
    /// Communication vocabulary size used for action encoding.
    pub comm_vocab_size: u16,
    /// Per-agent single-agent MCTS knobs (reused from `forge-agent`).
    pub mcts: MctsConfig,
}

impl Default for CooperativeMctsConfig {
    fn default() -> Self {
        Self {
            num_agents: DEFAULT_NUM_AGENTS,
            centralized_critic: DEFAULT_CENTRALIZED_CRITIC,
            joint_strategy: DEFAULT_JOINT_STRATEGY,
            sampled_joint_actions: DEFAULT_SAMPLED_JOINT_ACTIONS,
            seed: DEFAULT_COOP_SEED,
            comm_vocab_size: DEFAULT_COMM_VOCAB_SIZE,
            mcts: MctsConfig::default(),
        }
    }
}

/// Cooperative MCTS swarm protocol.
///
/// Holds a [`JointMctsPlanner`] over the default zero-dependency forward model
/// and a uniform-policy critic, plus a cached joint plan for the
/// [`ActionPolicy`] face. Interior mutability ([`Mutex`]) lets the `&self`
/// trait methods refresh the cache.
#[derive(Debug)]
pub struct CooperativeMctsProtocol {
    config: CooperativeMctsConfig,
    planner: JointMctsPlanner<DefaultForwardModel, Critic<UniformPolicy>>,
    /// Cached joint plan from the last `plan`/`coordinate_stateful` call.
    cached: Mutex<Vec<Action>>,
}

impl CooperativeMctsProtocol {
    /// Builds a cooperative MCTS protocol from config (uniform-policy critic).
    pub fn new(config: CooperativeMctsConfig) -> Self {
        let model = DefaultForwardModel::new(config.comm_vocab_size);
        let action_space = model.action_space_size();
        let critic =
            Critic::from_config(UniformPolicy::new(action_space), config.centralized_critic);
        let planner = JointMctsPlanner::new(model, critic, config.clone());
        Self {
            config,
            planner,
            cached: Mutex::new(Vec::new()),
        }
    }

    /// Computes a joint plan for `world`, refreshes the cache, and returns it.
    #[instrument(skip(self, world))]
    pub fn plan(&self, world: &WorldState) -> MangoMasResult<Vec<Action>> {
        let actions = self.planner.plan(world)?;
        self.store_cache(actions.clone());
        Ok(actions)
    }

    /// Returns a read-only copy of the config.
    pub fn config(&self) -> &CooperativeMctsConfig {
        &self.config
    }

    fn store_cache(&self, actions: Vec<Action>) {
        match self.cached.lock() {
            Ok(mut guard) => *guard = actions,
            Err(poisoned) => {
                // Recover rather than propagate a poisoned-lock panic.
                warn!("cooperative MCTS cache lock poisoned; recovering");
                *poisoned.into_inner() = actions;
            }
        }
    }
}

impl SwarmProtocol for CooperativeMctsProtocol {
    fn name(&self) -> &str {
        "cooperative_mcts"
    }

    /// World-less fallback. Without a [`WorldState`] no simulation is possible,
    /// so this degrades to a deterministic all-`Noop` joint of the right length
    /// (the high-quality path is [`SwarmProtocol::coordinate_stateful`]). The
    /// cache is refreshed for [`ActionPolicy`] consistency.
    #[instrument(skip(self, observations, _comm_tokens))]
    fn coordinate(&self, observations: &[Observation], _comm_tokens: &[Vec<u16>]) -> Vec<Action> {
        let actions = vec![Action::Noop; observations.len()];
        self.store_cache(actions.clone());
        actions
    }

    #[instrument(skip(self, world, observations, _comm_tokens))]
    fn coordinate_stateful(
        &self,
        world: &WorldState,
        observations: &[Observation],
        _comm_tokens: &[Vec<u16>],
    ) -> Vec<Action> {
        let _ = observations;
        self.plan(world).unwrap_or_else(|e| {
            warn!(error = %e, "joint plan failed; falling back to Noop");
            // Size the fallback by the world's authoritative agent count (not the
            // observations slice) so the result always satisfies the
            // `WorldState::step` contract, even if a caller passes a mismatched
            // observations slice.
            vec![Action::Noop; world.agents.len()]
        })
    }

    fn swarm_size(&self) -> usize {
        self.config.num_agents
    }
}

impl ActionPolicy for CooperativeMctsProtocol {
    /// Cheap lookup face: returns the cached joint plan's discrete action id for
    /// `agent_idx`. [`Self::plan`] / [`SwarmProtocol::coordinate_stateful`] must
    /// be called once per step first; before that (or for an out-of-range
    /// index) it returns `0` (Noop) without panicking.
    fn select_action(&self, _obs: &Observation, agent_idx: usize) -> u32 {
        let guard = match self.cached.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard
            .get(agent_idx)
            .and_then(|a| a.try_to_discrete_full(self.config.comm_vocab_size).ok())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;
    use forge_types::observation::{InventoryObservation, Observation, TileObservation};

    pub(super) fn make_obs() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation { slots: vec![] },
            health: 1.0,
            stamina: 1.0,
            position: (0, 0),
            messages: vec![],
            day_phase: 0,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    pub(super) fn make_world(num_agents: usize) -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.world.seed = 42;
        config.agents.num_agents = num_agents as u32;
        config.agents.comm_vocab_size = 0;
        config.agents.default_vision_radius = 3;
        WorldState::new(config).unwrap()
    }

    fn fast_protocol(num_agents: usize) -> CooperativeMctsProtocol {
        let mut config = CooperativeMctsConfig {
            num_agents,
            ..CooperativeMctsConfig::default()
        };
        config.mcts.num_simulations = 8;
        CooperativeMctsProtocol::new(config)
    }

    #[test]
    fn test_config_default() {
        // Preserve the original Phase-6 stub assertions.
        let config = CooperativeMctsConfig::default();
        assert_eq!(config.num_agents, 2);
        assert!(config.centralized_critic);
        // New-field assertions against the DEFAULT_* consts.
        assert_eq!(config.joint_strategy, JointStrategy::SequentialFactored);
        assert_eq!(config.sampled_joint_actions, DEFAULT_SAMPLED_JOINT_ACTIONS);
        assert_eq!(config.seed, DEFAULT_COOP_SEED);
        assert_eq!(config.comm_vocab_size, DEFAULT_COMM_VOCAB_SIZE);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = CooperativeMctsConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: CooperativeMctsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.num_agents, config.num_agents);
        assert_eq!(parsed.joint_strategy, config.joint_strategy);
    }

    #[test]
    fn test_config_partial_json_uses_defaults() {
        // Only one field provided; the rest must fall back to defaults.
        let parsed: CooperativeMctsConfig = serde_json::from_str(r#"{"num_agents": 4}"#).unwrap();
        assert_eq!(parsed.num_agents, 4);
        assert_eq!(parsed.sampled_joint_actions, DEFAULT_SAMPLED_JOINT_ACTIONS);
        assert_eq!(parsed.joint_strategy, JointStrategy::SequentialFactored);
    }

    #[test]
    fn test_name_and_swarm_size() {
        let protocol = fast_protocol(3);
        assert_eq!(protocol.name(), "cooperative_mcts");
        assert_eq!(protocol.swarm_size(), 3);
    }

    #[test]
    fn test_config_accessor() {
        let protocol = fast_protocol(4);
        assert_eq!(protocol.config().num_agents, 4);
        assert_eq!(
            protocol.config().joint_strategy,
            JointStrategy::SequentialFactored
        );
    }

    #[test]
    fn test_coordinate_worldless_length() {
        let protocol = fast_protocol(3);
        let obs = vec![make_obs(); 3];
        let comm = vec![vec![]; 3];
        let actions = protocol.coordinate(&obs, &comm);
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0], Action::Noop);
    }

    #[test]
    fn test_coordinate_stateful_length() {
        let protocol = fast_protocol(3);
        let world = make_world(3);
        let obs = vec![make_obs(); 3];
        let comm = vec![vec![]; 3];
        let actions = protocol.coordinate_stateful(&world, &obs, &comm);
        assert_eq!(actions.len(), 3);
    }

    #[test]
    fn test_select_action_reads_cache() {
        let protocol = fast_protocol(2);
        let world = make_world(2);
        let plan = protocol.plan(&world).unwrap();
        let expected = plan[1].try_to_discrete_full(0).unwrap();
        assert_eq!(protocol.select_action(&make_obs(), 1), expected);
    }

    #[test]
    fn test_cache_recovers_from_poisoned_lock() {
        let protocol = fast_protocol(2);
        // Poison the cache mutex by panicking while holding the guard.
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = protocol.cached.lock().unwrap();
            panic!("intentional poison");
        }));
        assert!(poisoned.is_err());
        // store_cache must recover (Err(poisoned) branch), not propagate a panic.
        protocol.store_cache(vec![Action::Noop, Action::Noop]);
        // select_action must also read through a poisoned lock without panicking.
        assert_eq!(protocol.select_action(&make_obs(), 0), 0);
    }

    #[test]
    fn test_select_action_before_plan_is_zero() {
        let protocol = fast_protocol(2);
        // No plan computed yet → Noop id 0, no panic.
        assert_eq!(protocol.select_action(&make_obs(), 0), 0);
    }

    #[test]
    fn test_select_action_out_of_bounds_no_panic() {
        let protocol = fast_protocol(2);
        let world = make_world(2);
        protocol.plan(&world).unwrap();
        assert_eq!(protocol.select_action(&make_obs(), 999), 0);
    }

    #[test]
    fn test_swappable_with_dyn_swarm_protocol() {
        use super::super::protocol::IndependentProtocol;
        let world = make_world(2);
        let obs = vec![make_obs(); 2];
        let comm = vec![vec![]; 2];
        let protocols: Vec<Box<dyn SwarmProtocol>> = vec![
            Box::new(IndependentProtocol::new(2)),
            Box::new(fast_protocol(2)),
        ];
        for p in &protocols {
            assert_eq!(p.coordinate(&obs, &comm).len(), 2);
            assert_eq!(p.coordinate_stateful(&world, &obs, &comm).len(), 2);
        }
    }

    #[test]
    fn test_sampled_strategy_protocol() {
        let mut config = CooperativeMctsConfig {
            num_agents: 2,
            joint_strategy: JointStrategy::Sampled,
            ..CooperativeMctsConfig::default()
        };
        config.mcts.num_simulations = 4;
        let protocol = CooperativeMctsProtocol::new(config);
        let world = make_world(2);
        let a = protocol.plan(&world).unwrap();
        let b = protocol.plan(&world).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(a, b); // deterministic
    }

    #[test]
    fn test_independent_critic_protocol() {
        let config = CooperativeMctsConfig {
            num_agents: 2,
            centralized_critic: false,
            ..CooperativeMctsConfig::default()
        };
        let protocol = CooperativeMctsProtocol::new(config);
        let world = make_world(2);
        assert_eq!(protocol.plan(&world).unwrap().len(), 2);
    }

    #[test]
    fn test_multi_step_episode_deterministic() {
        // Drive N agents through real `WorldState::step` for K steps and assert
        // the joint plan plumbs through and the whole trace is reproducible.
        let n = 3;
        let run = || {
            let protocol = fast_protocol(n);
            let mut world = make_world(n);
            let obs = vec![make_obs(); n];
            let comm = vec![vec![]; n];
            let mut reward_trace = Vec::new();
            for _ in 0..15 {
                let actions = protocol.coordinate_stateful(&world, &obs, &comm);
                assert_eq!(actions.len(), n);
                let result = world.step(&actions);
                assert_eq!(result.rewards.len(), n);
                reward_trace.push(result.rewards.clone());
                if result.terminated || result.truncated {
                    break;
                }
            }
            reward_trace
        };
        let first = run();
        let second = run();
        assert!(!first.is_empty());
        assert_eq!(first, second);
    }
}

#[cfg(test)]
mod proptests {
    use super::tests::{make_obs, make_world};
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        /// Sampled planning is deterministic for a fixed seed and yields one
        /// action per agent.
        #[test]
        fn sampled_plan_is_deterministic(seed in any::<u64>(), n in 1usize..4) {
            let mut config = CooperativeMctsConfig {
                num_agents: n,
                joint_strategy: JointStrategy::Sampled,
                seed,
                ..CooperativeMctsConfig::default()
            };
            config.sampled_joint_actions = 4;
            config.mcts.num_simulations = 2;
            let protocol = CooperativeMctsProtocol::new(config);
            let world = make_world(n);
            let a = protocol.plan(&world).unwrap();
            let b = protocol.plan(&world).unwrap();
            prop_assert_eq!(a.len(), n);
            prop_assert_eq!(a, b);
        }

        /// The world-less fallback always returns one action per observation.
        #[test]
        fn worldless_coordinate_length_invariant(n in 0usize..6) {
            let protocol = {
                let mut config = CooperativeMctsConfig::default();
                config.mcts.num_simulations = 2;
                CooperativeMctsProtocol::new(config)
            };
            let obs = vec![make_obs(); n];
            let comm = vec![vec![]; n];
            prop_assert_eq!(protocol.coordinate(&obs, &comm).len(), n);
        }
    }
}
