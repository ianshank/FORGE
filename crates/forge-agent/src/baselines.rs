//! Baseline agents for benchmarking and testing.
//!
//! Provides simple agents that can be used as baselines for comparison
//! with learned policies or MCTS-based planners.

use forge_core::WorldState;
use forge_types::Action;
use rand::Rng;
use tracing::{instrument, trace};

/// Trait for agents that can select actions given a world state.
pub trait Agent: Send {
    /// Selects an action for the given agent index.
    fn select_action(&mut self, state: &WorldState, agent_idx: usize) -> Action;

    /// Returns the agent's name.
    fn name(&self) -> &str;
}

/// Random agent: selects uniformly random actions from the action space.
#[derive(Debug)]
pub struct RandomAgent<R: Rng> {
    rng: R,
    action_space_size: u32,
    comm_vocab_size: u16,
}

impl<R: Rng> RandomAgent<R> {
    /// Creates a new random agent.
    #[instrument(skip_all)]
    pub fn new(rng: R, comm_vocab_size: u16) -> Self {
        Self {
            rng,
            action_space_size: Action::space_size(comm_vocab_size),
            comm_vocab_size,
        }
    }
}

impl<R: Rng + Send> Agent for RandomAgent<R> {
    fn select_action(&mut self, _state: &WorldState, _agent_idx: usize) -> Action {
        let action_id = self.rng.gen_range(0..self.action_space_size);
        Action::from_discrete(action_id, self.comm_vocab_size).unwrap_or(Action::Noop)
    }

    fn name(&self) -> &str {
        "RandomAgent"
    }
}

/// Noop agent: always does nothing. Useful as an absolute baseline.
#[derive(Debug, Default)]
pub struct NoopAgent;

impl Agent for NoopAgent {
    fn select_action(&mut self, _state: &WorldState, _agent_idx: usize) -> Action {
        Action::Noop
    }

    fn name(&self) -> &str {
        "NoopAgent"
    }
}

/// Greedy navigation agent: moves toward a target position using
/// manhattan distance heuristic.
#[derive(Debug)]
pub struct GreedyNavigator {
    target_x: u16,
    target_y: u16,
}

impl GreedyNavigator {
    /// Creates a new greedy navigator targeting the given position.
    #[instrument]
    pub fn new(target_x: u16, target_y: u16) -> Self {
        Self { target_x, target_y }
    }
}

impl Agent for GreedyNavigator {
    fn select_action(&mut self, state: &WorldState, agent_idx: usize) -> Action {
        if agent_idx >= state.agents.len() {
            return Action::Noop;
        }

        let agent = &state.agents[agent_idx];
        if !agent.alive {
            return Action::Noop;
        }

        let ax = agent.position.x as i32;
        let ay = agent.position.y as i32;
        let tx = self.target_x as i32;
        let ty = self.target_y as i32;

        let dx = tx - ax;
        let dy = ty - ay;

        // Already at target
        if dx == 0 && dy == 0 {
            return Action::Noop;
        }

        // Prefer the axis with the larger delta
        let action = if dx.abs() >= dy.abs() {
            if dx > 0 {
                Action::Move(forge_types::grid::Direction::Right)
            } else {
                Action::Move(forge_types::grid::Direction::Left)
            }
        } else if dy > 0 {
            Action::Move(forge_types::grid::Direction::Down)
        } else {
            Action::Move(forge_types::grid::Direction::Up)
        };

        trace!(agent_idx, ?action, dx, dy, "greedy navigator action");

        action
    }

    fn name(&self) -> &str {
        "GreedyNavigator"
    }
}

/// Heuristic agent: uses a simple strategy combining collection and navigation.
/// Picks up items when available, otherwise moves randomly.
#[derive(Debug)]
pub struct HeuristicAgent<R: Rng> {
    rng: R,
    comm_vocab_size: u16,
}

impl<R: Rng> HeuristicAgent<R> {
    /// Creates a new heuristic agent.
    #[instrument(skip_all)]
    pub fn new(rng: R, comm_vocab_size: u16) -> Self {
        Self {
            rng,
            comm_vocab_size,
        }
    }
}

impl<R: Rng + Send> Agent for HeuristicAgent<R> {
    fn select_action(&mut self, state: &WorldState, agent_idx: usize) -> Action {
        if agent_idx >= state.agents.len() {
            return Action::Noop;
        }

        let agent = &state.agents[agent_idx];
        if !agent.alive {
            return Action::Noop;
        }

        // Check if standing on a resource tile
        if let Some(tile) = state.grid.get(agent.position.x, agent.position.y) {
            if tile.resource_id.is_some() && !agent.inventory.is_full() {
                return Action::PickUp;
            }
        }

        // Otherwise, move in a random direction
        let dir_idx = self.rng.gen_range(0..4u32);
        Action::from_discrete(1 + dir_idx, self.comm_vocab_size).unwrap_or(Action::Noop)
    }

    fn name(&self) -> &str {
        "HeuristicAgent"
    }
}

/// Runs an episode with the given agent(s), returning total per-agent rewards.
#[instrument(skip_all)]
pub fn run_episode(
    state: &mut WorldState,
    agents: &mut [Box<dyn Agent>],
    max_steps: u64,
) -> Vec<f32> {
    let num_agents = state.agents.len();
    let mut total_rewards = vec![0.0_f32; num_agents];

    for _step in 0..max_steps {
        if state.terminated || state.truncated {
            break;
        }

        let mut actions = Vec::with_capacity(num_agents);
        for (i, agent) in agents.iter_mut().enumerate() {
            if i < num_agents {
                actions.push(agent.select_action(state, i));
            }
        }

        let result = state.step(&actions);

        for (i, reward) in result.rewards.iter().enumerate() {
            if i < total_rewards.len() {
                total_rewards[i] += reward;
            }
        }
    }

    total_rewards
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;
    use forge_types::grid::{Direction, Position};
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

    #[test]
    fn test_random_agent_selects_valid_actions() {
        let rng = Pcg64Mcg::seed_from_u64(42);
        let mut agent = RandomAgent::new(rng, 0);
        let state = make_test_world();

        for _ in 0..100 {
            let action = agent.select_action(&state, 0);
            // Should always be a valid action
            let discrete = action.to_discrete();
            assert!(discrete < Action::space_size(0));
        }
    }

    #[test]
    fn test_noop_agent() {
        let mut agent = NoopAgent;
        let state = make_test_world();

        let action = agent.select_action(&state, 0);
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_greedy_navigator_moves_toward_target() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.agents.num_agents = 1;
        let mut state = WorldState::new(config).unwrap();

        // Place agent at (2, 2), target at (5, 2)
        state.agents[0].position = Position::new(2, 2);

        let mut agent = GreedyNavigator::new(5, 2);
        let action = agent.select_action(&state, 0);

        // Should move right (toward target)
        assert_eq!(action, Action::Move(Direction::Right));
    }

    #[test]
    fn test_greedy_navigator_at_target() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.agents.num_agents = 1;
        let mut state = WorldState::new(config).unwrap();

        state.agents[0].position = Position::new(5, 5);
        let mut agent = GreedyNavigator::new(5, 5);
        let action = agent.select_action(&state, 0);

        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_greedy_navigator_vertical() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.agents.num_agents = 1;
        let mut state = WorldState::new(config).unwrap();

        state.agents[0].position = Position::new(5, 2);
        let mut agent = GreedyNavigator::new(5, 8);
        let action = agent.select_action(&state, 0);

        assert_eq!(action, Action::Move(Direction::Down));
    }

    #[test]
    fn test_heuristic_agent() {
        let rng = Pcg64Mcg::seed_from_u64(42);
        let mut agent = HeuristicAgent::new(rng, 0);
        let state = make_test_world();

        // Should produce valid actions
        for _ in 0..50 {
            let action = agent.select_action(&state, 0);
            let discrete = action.to_discrete();
            assert!(discrete < Action::space_size(0) || action == Action::PickUp);
        }
    }

    #[test]
    fn test_run_episode_noop() {
        let mut state = make_test_world();
        let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(NoopAgent)];

        let rewards = run_episode(&mut state, &mut agents, 10);

        assert_eq!(rewards.len(), 1);
        // Noop agent should accumulate 0 reward (no tasks to complete)
        assert_eq!(rewards[0], 0.0);
    }

    #[test]
    fn test_run_episode_random() {
        let mut state = make_test_world();
        let rng = Pcg64Mcg::seed_from_u64(42);
        let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(RandomAgent::new(rng, 0))];

        let rewards = run_episode(&mut state, &mut agents, 50);

        assert_eq!(rewards.len(), 1);
        // Random agent should complete some steps without crashing
        assert!(state.tick > 0);
    }

    #[test]
    fn test_run_episode_terminates_on_truncation() {
        let mut config = ForgeConfig::default();
        config.world.width = 8;
        config.world.height = 8;
        config.agents.num_agents = 1;
        config.agents.default_vision_radius = 3;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 5;
        let mut state = WorldState::new(config).unwrap();

        let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(NoopAgent)];
        run_episode(&mut state, &mut agents, 1000);

        // Should have stopped at truncation
        assert!(state.truncated);
        assert_eq!(state.tick, 5);
    }

    #[test]
    fn test_run_episode_max_steps_zero() {
        let mut state = make_test_world();
        let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(NoopAgent)];
        let initial_tick = state.tick;

        let rewards = run_episode(&mut state, &mut agents, 0);

        assert_eq!(rewards.len(), 1);
        assert_eq!(rewards[0], 0.0);
        // Tick should not have advanced
        assert_eq!(state.tick, initial_tick);
    }

    #[test]
    fn test_greedy_navigator_out_of_bounds_agent_idx() {
        let state = make_test_world();
        let mut agent = GreedyNavigator::new(5, 5);
        // agent_idx 99 is out of bounds
        let action = agent.select_action(&state, 99);
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_greedy_navigator_dead_agent() {
        let mut state = make_test_world();
        state.agents[0].alive = false;
        let mut agent = GreedyNavigator::new(5, 5);
        let action = agent.select_action(&state, 0);
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_greedy_navigator_left_movement() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.agents.num_agents = 1;
        let mut state = WorldState::new(config).unwrap();

        // Agent at (8, 5), target at (2, 5) — should move left
        state.agents[0].position = Position::new(8, 5);
        let mut agent = GreedyNavigator::new(2, 5);
        let action = agent.select_action(&state, 0);
        assert_eq!(action, Action::Move(Direction::Left));
    }

    #[test]
    fn test_greedy_navigator_up_movement() {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.agents.num_agents = 1;
        let mut state = WorldState::new(config).unwrap();

        // Agent at (5, 8), target at (5, 2) — should move up
        state.agents[0].position = Position::new(5, 8);
        let mut agent = GreedyNavigator::new(5, 2);
        let action = agent.select_action(&state, 0);
        assert_eq!(action, Action::Move(Direction::Up));
    }

    #[test]
    fn test_heuristic_agent_out_of_bounds_agent_idx() {
        let state = make_test_world();
        let rng = Pcg64Mcg::seed_from_u64(42);
        let mut agent = HeuristicAgent::new(rng, 0);
        let action = agent.select_action(&state, 99);
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_heuristic_agent_dead_agent() {
        let mut state = make_test_world();
        state.agents[0].alive = false;
        let rng = Pcg64Mcg::seed_from_u64(42);
        let mut agent = HeuristicAgent::new(rng, 0);
        let action = agent.select_action(&state, 0);
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_agent_name() {
        let noop = NoopAgent;
        assert_eq!(noop.name(), "NoopAgent");

        let rng = Pcg64Mcg::seed_from_u64(42);
        let random = RandomAgent::new(rng, 0);
        assert_eq!(random.name(), "RandomAgent");

        let greedy = GreedyNavigator::new(5, 5);
        assert_eq!(greedy.name(), "GreedyNavigator");

        let rng2 = Pcg64Mcg::seed_from_u64(42);
        let heuristic = HeuristicAgent::new(rng2, 0);
        assert_eq!(heuristic.name(), "HeuristicAgent");
    }
}
