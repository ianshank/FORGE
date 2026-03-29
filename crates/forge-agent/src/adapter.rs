//! Adapters bridging existing agent abstractions to the unified [`AgentInterface`].
//!
//! These adapters preserve full backward compatibility with the three existing
//! agent traits while enabling all of them to participate in the `forge-eval`
//! evaluation harness:
//!
//! | Existing Trait | Adapter | Direction |
//! |---------------|---------|-----------|
//! | `Agent` (baselines) | [`PrivilegedAgentAdapter`] | `Agent` → `AgentInterface` |
//! | `AgentInterface` | [`InterfaceToAgentAdapter`] | `AgentInterface` → `Agent` |
//!
//! For `ActionPolicy` and `CognitiveAgent` adapters, see the respective crates.
//! They are not included here to avoid pulling in `forge-mangomas` and
//! `forge-cognitive` as dependencies of `forge-agent`.

use std::any::Any;
use std::time::Instant;

use forge_core::WorldState;
use forge_types::agent_interface::{AgentInterface, AgentMetadata, AgentResponse};
use forge_types::observation::Observation;
use forge_types::Action;
use tracing::{debug, instrument, warn};

use crate::baselines::Agent;

/// Wraps a [`WorldState`]-based [`Agent`] as an [`AgentInterface`].
///
/// This adapter enables legacy baseline agents (which require full world state)
/// to participate in the agent-agnostic evaluation pipeline. It stores a
/// snapshot of the current [`WorldState`] that must be updated
/// each tick before calling [`select_action`](AgentInterface::select_action).
///
/// # Usage
///
/// ```rust,no_run
/// use forge_agent::adapter::PrivilegedAgentAdapter;
/// use forge_agent::baselines::NoopAgent;
///
/// let agent = NoopAgent;
/// let mut adapter = PrivilegedAgentAdapter::new(agent);
/// // Before each select_action call, update the world snapshot:
/// // adapter.update_world(&world);
/// ```
pub struct PrivilegedAgentAdapter<A: Agent> {
    inner: A,
    /// Snapshot of the current world state, updated each tick by the
    /// evaluation harness before calling `select_action`.
    world_snapshot: Option<WorldState>,
}

impl<A: Agent> PrivilegedAgentAdapter<A> {
    /// Creates a new adapter wrapping the given legacy agent.
    #[instrument(skip_all)]
    pub fn new(agent: A) -> Self {
        Self {
            inner: agent,
            world_snapshot: None,
        }
    }

    /// Updates the world snapshot. Must be called before each `select_action`.
    ///
    /// The evaluation harness calls this automatically when running privileged
    /// agents through the `AgentInterface` pipeline.
    #[instrument(skip_all)]
    pub fn update_world(&mut self, world: &WorldState) {
        self.world_snapshot = Some(world.clone());
    }

    /// Returns a reference to the inner agent.
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Returns a mutable reference to the inner agent.
    pub fn inner_mut(&mut self) -> &mut A {
        &mut self.inner
    }
}

impl<A: Agent> AgentInterface for PrivilegedAgentAdapter<A> {
    fn select_action(&mut self, _obs: &Observation, agent_idx: usize) -> AgentResponse {
        let started = Instant::now();

        let (action, comm_vocab) = match &self.world_snapshot {
            Some(world) => (
                self.inner.select_action(world, agent_idx),
                world.config.agents.comm_vocab_size,
            ),
            None => {
                warn!(agent_idx, "No world snapshot available, returning Noop");
                (Action::Noop, 0)
            }
        };

        AgentResponse::with_timing(action.to_discrete_full(comm_vocab), started)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::heuristic(self.inner.name())
    }

    fn update_context(&mut self, context: &dyn Any) {
        if let Some(world) = context.downcast_ref::<WorldState>() {
            self.update_world(world);
        }
    }
}

/// Wraps an [`AgentInterface`] as a legacy [`Agent`].
///
/// This adapter enables new `AgentInterface` implementations to be used
/// with existing code that expects the legacy `Agent` trait (e.g.,
/// `run_episode`). It generates observations from the provided
/// `WorldState` using [`WorldState::generate_observation`].
pub struct InterfaceToAgentAdapter {
    inner: Box<dyn AgentInterface>,
}

impl InterfaceToAgentAdapter {
    /// Creates a new adapter wrapping the given `AgentInterface`.
    #[instrument(skip_all)]
    pub fn new(agent: Box<dyn AgentInterface>) -> Self {
        Self { inner: agent }
    }

    /// Returns the inner agent's name.
    pub fn agent_name(&self) -> &str {
        self.inner.name()
    }

    /// Returns the inner agent's metadata.
    pub fn agent_metadata(&self) -> AgentMetadata {
        self.inner.metadata()
    }
}

impl Agent for InterfaceToAgentAdapter {
    fn select_action(&mut self, state: &WorldState, agent_idx: usize) -> Action {
        if agent_idx >= state.agents.len() {
            debug!(
                agent_idx,
                num_agents = state.agents.len(),
                "Agent index out of bounds, returning Noop"
            );
            return Action::Noop;
        }

        self.inner.update_context(state);

        let obs = state.generate_observation(&state.agents[agent_idx]);
        let response = self.inner.select_action(&obs, agent_idx);

        let comm_vocab = state.config.agents.comm_vocab_size;
        let drone_enabled = state.config.drone.enabled;

        Action::from_discrete(response.action_id, comm_vocab, drone_enabled).unwrap_or_else(|| {
            debug!(
                action_id = response.action_id,
                "Invalid action ID, falling back to Noop"
            );
            Action::Noop
        })
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

/// An [`AgentInterface`] wrapper around a closure.
///
/// Useful for quick ad-hoc agents in tests and evaluations.
///
/// # Examples
///
/// ```rust,no_run
/// use forge_agent::adapter::ClosureAgent;
/// use forge_types::agent_interface::AgentResponse;
///
/// let agent = ClosureAgent::new("always_move_up", |_obs, _idx| {
///     AgentResponse::from_action(1) // Move Up
/// });
/// ```
pub struct ClosureAgent<F>
where
    F: FnMut(&Observation, usize) -> AgentResponse + Send,
{
    name: String,
    func: F,
}

impl<F> ClosureAgent<F>
where
    F: FnMut(&Observation, usize) -> AgentResponse + Send,
{
    /// Creates a new closure-based agent.
    pub fn new(name: &str, func: F) -> Self {
        Self {
            name: name.to_string(),
            func,
        }
    }
}

impl<F> AgentInterface for ClosureAgent<F>
where
    F: FnMut(&Observation, usize) -> AgentResponse + Send,
{
    fn select_action(&mut self, obs: &Observation, agent_idx: usize) -> AgentResponse {
        (self.func)(obs, agent_idx)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// An [`AgentInterface`] that always returns Noop.
///
/// Convenience wrapper — equivalent to `NoopAgent` but implementing
/// `AgentInterface` directly without going through the adapter.
#[derive(Debug, Default)]
pub struct NoopInterface;

impl AgentInterface for NoopInterface {
    fn select_action(&mut self, _obs: &Observation, _agent_idx: usize) -> AgentResponse {
        AgentResponse::from_action(0)
    }

    fn name(&self) -> &str {
        "NoopInterface"
    }

    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::heuristic("NoopInterface")
    }
}

/// An [`AgentInterface`] that selects random actions.
///
/// Convenience wrapper — equivalent to `RandomAgent` but implementing
/// `AgentInterface` directly.
pub struct RandomInterface {
    action_space_size: u32,
    rng: rand_pcg::Pcg64Mcg,
}

impl RandomInterface {
    /// Creates a new random interface agent.
    ///
    /// # Panics
    ///
    /// Panics if `action_space_size` is 0 (no valid actions to select).
    pub fn new(action_space_size: u32, seed: u64) -> Self {
        assert!(
            action_space_size > 0,
            "action_space_size must be > 0, got {action_space_size}"
        );
        use rand::SeedableRng;
        Self {
            action_space_size,
            rng: rand_pcg::Pcg64Mcg::seed_from_u64(seed),
        }
    }
}

impl AgentInterface for RandomInterface {
    fn select_action(&mut self, _obs: &Observation, _agent_idx: usize) -> AgentResponse {
        use rand::Rng;
        let action_id = self.rng.gen_range(0..self.action_space_size);
        AgentResponse::from_action(action_id)
    }

    fn name(&self) -> &str {
        "RandomInterface"
    }

    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::heuristic("RandomInterface")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::{NoopAgent, RandomAgent};
    use forge_types::config::ForgeConfig;
    use forge_types::constants;
    use forge_types::observation::{InventoryObservation, TileObservation};
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    fn make_test_world() -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 100;
        WorldState::new(config).unwrap()
    }

    fn make_test_drone_world(comm_vocab: u16) -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = 42;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = comm_vocab;
        config.task.max_episode_length = 100;
        config.drone.enabled = true;
        config.drone.num_aerial = 1;
        WorldState::new(config).unwrap()
    }

    fn make_test_observation() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation {
                slots: vec![(constants::OBS_EMPTY_SLOT_ITEM, 0)],
            },
            health: 1.0,
            stamina: 1.0,
            position: (5, 5),
            messages: vec![],
            day_phase: 1,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
        }
    }

    // --- PrivilegedAgentAdapter tests ---

    #[test]
    fn test_privileged_adapter_noop() {
        let mut adapter = PrivilegedAgentAdapter::new(NoopAgent);
        let world = make_test_world();
        adapter.update_world(&world);

        let obs = make_test_observation();
        let resp = adapter.select_action(&obs, 0);
        assert_eq!(resp.action_id, 0); // Noop
    }

    #[test]
    fn test_privileged_adapter_name() {
        let adapter = PrivilegedAgentAdapter::new(NoopAgent);
        assert_eq!(adapter.name(), "NoopAgent");
    }

    #[test]
    fn test_privileged_adapter_metadata() {
        let adapter = PrivilegedAgentAdapter::new(NoopAgent);
        let meta = adapter.metadata();
        assert_eq!(meta.agent_type, "heuristic");
        assert_eq!(meta.model_name, "NoopAgent");
    }

    #[test]
    fn test_privileged_adapter_no_snapshot_returns_noop() {
        let mut adapter = PrivilegedAgentAdapter::new(NoopAgent);
        // Don't call update_world
        let obs = make_test_observation();
        let resp = adapter.select_action(&obs, 0);
        assert_eq!(resp.action_id, 0);
    }

    #[test]
    fn test_privileged_adapter_random_agent() {
        let rng = Pcg64Mcg::seed_from_u64(42);
        let mut adapter = PrivilegedAgentAdapter::new(RandomAgent::new(rng, 0));
        let world = make_test_world();
        adapter.update_world(&world);

        let obs = make_test_observation();
        let resp = adapter.select_action(&obs, 0);
        // Should produce some action (not necessarily 0)
        assert!(resp.action_id < Action::space_size(0, false));
    }

    #[test]
    fn test_privileged_adapter_update_context() {
        let mut adapter = PrivilegedAgentAdapter::new(NoopAgent);
        let world = make_test_world();

        adapter.update_context(&world);

        let obs = make_test_observation();
        let resp = adapter.select_action(&obs, 0);
        assert_eq!(resp.action_id, 0);
    }

    #[derive(Debug)]
    struct DroneAscendAgent;

    impl Agent for DroneAscendAgent {
        fn select_action(&mut self, _state: &WorldState, _agent_idx: usize) -> Action {
            Action::Ascend
        }

        fn name(&self) -> &str {
            "DroneAscendAgent"
        }
    }

    #[test]
    fn test_privileged_adapter_uses_full_action_encoding_for_drone_actions() {
        let comm_vocab = 4;
        let mut adapter = PrivilegedAgentAdapter::new(DroneAscendAgent);
        let world = make_test_drone_world(comm_vocab);
        adapter.update_context(&world);

        let obs = make_test_observation();
        let resp = adapter.select_action(&obs, 0);

        assert_eq!(resp.action_id, Action::Ascend.to_discrete_full(comm_vocab));
    }

    #[test]
    fn test_privileged_adapter_inner_access() {
        let adapter = PrivilegedAgentAdapter::new(NoopAgent);
        assert_eq!(adapter.inner().name(), "NoopAgent");
    }

    // --- InterfaceToAgentAdapter tests ---

    #[test]
    fn test_interface_to_agent_noop() {
        let mut adapter = InterfaceToAgentAdapter::new(Box::new(NoopInterface));
        let state = make_test_world();
        let action = adapter.select_action(&state, 0);
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_interface_to_agent_name() {
        let adapter = InterfaceToAgentAdapter::new(Box::new(NoopInterface));
        assert_eq!(adapter.name(), "NoopInterface");
        assert_eq!(adapter.agent_name(), "NoopInterface");
    }

    #[test]
    fn test_interface_to_agent_out_of_bounds() {
        let mut adapter = InterfaceToAgentAdapter::new(Box::new(NoopInterface));
        let state = make_test_world();
        let action = adapter.select_action(&state, 99);
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_interface_to_agent_metadata() {
        let adapter = InterfaceToAgentAdapter::new(Box::new(NoopInterface));
        let meta = adapter.agent_metadata();
        assert_eq!(meta.agent_type, "heuristic");
    }

    // --- ClosureAgent tests ---

    #[test]
    fn test_closure_agent() {
        let mut agent =
            ClosureAgent::new("test_closure", |_obs, _idx| AgentResponse::from_action(5));
        let obs = make_test_observation();
        let resp = agent.select_action(&obs, 0);
        assert_eq!(resp.action_id, 5);
        assert_eq!(agent.name(), "test_closure");
    }

    #[test]
    fn test_closure_agent_uses_observation() {
        let mut agent = ClosureAgent::new("pos_reader", |obs, _idx| {
            // Return action based on position
            AgentResponse::from_action(obs.position.0 as u32)
        });
        let obs = make_test_observation();
        let resp = agent.select_action(&obs, 0);
        assert_eq!(resp.action_id, 5); // position is (5, 5)
    }

    #[test]
    fn test_closure_agent_uses_agent_idx() {
        let mut agent = ClosureAgent::new("idx_echo", |_obs, idx| {
            AgentResponse::from_action(idx as u32)
        });
        let obs = make_test_observation();
        assert_eq!(agent.select_action(&obs, 0).action_id, 0);
        assert_eq!(agent.select_action(&obs, 3).action_id, 3);
    }

    // --- NoopInterface tests ---

    #[test]
    fn test_noop_interface() {
        let mut agent = NoopInterface;
        let obs = make_test_observation();
        let resp = agent.select_action(&obs, 0);
        assert_eq!(resp.action_id, 0);
        assert_eq!(agent.name(), "NoopInterface");
    }

    // --- RandomInterface tests ---

    #[test]
    fn test_random_interface() {
        let mut agent = RandomInterface::new(40, 42);
        let obs = make_test_observation();
        let resp = agent.select_action(&obs, 0);
        assert!(resp.action_id < 40);
    }

    #[test]
    fn test_random_interface_determinism() {
        let obs = make_test_observation();

        let mut agent1 = RandomInterface::new(40, 42);
        let mut agent2 = RandomInterface::new(40, 42);

        for _ in 0..20 {
            let r1 = agent1.select_action(&obs, 0);
            let r2 = agent2.select_action(&obs, 0);
            assert_eq!(
                r1.action_id, r2.action_id,
                "Same seed must produce same actions"
            );
        }
    }

    #[test]
    fn test_random_interface_name_and_metadata() {
        let agent = RandomInterface::new(40, 0);
        assert_eq!(agent.name(), "RandomInterface");
        assert_eq!(agent.metadata().agent_type, "heuristic");
    }

    // --- Roundtrip adapter tests ---

    #[test]
    fn test_roundtrip_noop_through_both_adapters() {
        // NoopAgent -> PrivilegedAgentAdapter (AgentInterface) -> InterfaceToAgentAdapter (Agent)
        let privileged = PrivilegedAgentAdapter::new(NoopAgent);
        let mut roundtrip = InterfaceToAgentAdapter::new(Box::new(privileged));

        let state = make_test_world();
        let action = roundtrip.select_action(&state, 0);
        // Context forwarding keeps the wrapped privileged agent usable.
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_interface_to_agent_in_run_episode() {
        let mut state = make_test_world();
        let interface_agent = NoopInterface;
        let adapted = InterfaceToAgentAdapter::new(Box::new(interface_agent));

        let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(adapted)];
        let rewards = crate::baselines::run_episode(&mut state, &mut agents, 10);
        assert_eq!(rewards.len(), 1);
    }

    #[test]
    fn test_random_interface_action_space_size_one() {
        let mut agent = RandomInterface::new(1, 42);
        let obs = make_test_observation();
        for _ in 0..20 {
            let resp = agent.select_action(&obs, 0);
            assert_eq!(resp.action_id, 0); // Only one valid action
        }
    }

    #[test]
    #[should_panic(expected = "action_space_size must be > 0")]
    fn test_random_interface_zero_action_space() {
        RandomInterface::new(0, 42);
    }

    #[test]
    fn test_privileged_adapter_update_world_multiple_times() {
        let mut adapter = PrivilegedAgentAdapter::new(NoopAgent);
        let world = make_test_world();

        adapter.update_world(&world);
        let obs = make_test_observation();
        let r1 = adapter.select_action(&obs, 0);

        adapter.update_world(&world);
        let r2 = adapter.select_action(&obs, 0);

        assert_eq!(r1.action_id, r2.action_id);
    }

    #[test]
    fn test_noop_interface_metadata() {
        let agent = NoopInterface;
        let meta = agent.metadata();
        assert_eq!(meta.agent_type, "heuristic");
        assert_eq!(meta.model_name, "NoopInterface");
    }

    #[test]
    fn test_closure_agent_default_metadata() {
        let agent = ClosureAgent::new("test_closure", |_obs, _idx| AgentResponse::from_action(0));
        let meta = agent.metadata();
        // Default metadata — empty fields
        assert!(meta.agent_type.is_empty());
    }
}
