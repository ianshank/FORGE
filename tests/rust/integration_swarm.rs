//! Cross-crate regression tests for multi-agent swarm coordination.
//!
//! These exercise the cooperative Centralized-Training / Decentralized-Execution
//! (CTDE) MCTS protocol (`forge_mangomas::swarm`) end-to-end against a real
//! `forge_core::WorldState`, covering the contract the unit tests cannot: that a
//! `SwarmProtocol` produces step-compatible joint actions, that planning is
//! deterministic across runs, and that the cooperative protocol is a drop-in
//! swap for the no-coordination baseline.
//!
//! To run: `cargo test --test integration_swarm`

use forge_core::WorldState;
use forge_mangomas::swarm::{
    CooperativeMctsConfig, CooperativeMctsProtocol, IndependentProtocol, JointStrategy,
    SwarmProtocol,
};
use forge_types::config::ForgeConfig;
use forge_types::observation::Observation;
use forge_types::Action;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A small, fast multi-agent world. No hard-coded planner params leak in here —
/// the protocol derives its action space from the world's comm vocab.
fn make_world(num_agents: u32, seed: u64) -> WorldState {
    let mut config = ForgeConfig::default();
    config.world.width = 16;
    config.world.height = 16;
    config.world.seed = seed;
    config.agents.num_agents = num_agents;
    config.agents.comm_vocab_size = 0;
    config.task.max_episode_length = 10_000;
    WorldState::new(config).unwrap()
}

/// A cheap-but-real cooperative protocol: small simulation budget keeps the
/// test fast while still exercising the full PUCT search path.
fn fast_protocol(num_agents: usize, strategy: JointStrategy, seed: u64) -> CooperativeMctsProtocol {
    let mut config = CooperativeMctsConfig {
        num_agents,
        joint_strategy: strategy,
        seed,
        comm_vocab_size: 0,
        ..CooperativeMctsConfig::default()
    };
    config.mcts.num_simulations = 8;
    CooperativeMctsProtocol::new(config)
}

/// Empty comm tokens, one slot per agent.
fn empty_comm(n: usize) -> Vec<Vec<u16>> {
    vec![Vec::new(); n]
}

/// Drive `steps` ticks of `protocol`-chosen joint actions, returning the
/// per-step (actions, rewards) trace for equality comparisons.
fn run_episode(
    protocol: &dyn SwarmProtocol,
    world: &mut WorldState,
    observations: &[Observation],
    steps: usize,
) -> Vec<(Vec<Action>, Vec<f32>)> {
    let n = world.agents.len();
    let comm = empty_comm(n);
    let mut obs = observations.to_vec();
    let mut trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        let actions = protocol.coordinate_stateful(world, &obs, &comm);
        assert_eq!(
            actions.len(),
            n,
            "joint action vector must match agent count"
        );
        let result = world.step(&actions);
        trace.push((actions, result.rewards.clone()));
        obs = result.observations;
        if world.terminated || world.truncated {
            break;
        }
    }
    trace
}

// ---------------------------------------------------------------------------
// 1. Step-compatibility: a cooperative joint plan drives a real episode
// ---------------------------------------------------------------------------

#[test]
fn test_cooperative_episode_is_step_compatible() {
    const AGENTS: u32 = 3;
    let mut world = make_world(AGENTS, 7);
    let reset = world.reset(Some(7));
    assert_eq!(reset.observations.len(), AGENTS as usize);

    let protocol = fast_protocol(AGENTS as usize, JointStrategy::SequentialFactored, 7);
    let trace = run_episode(&protocol, &mut world, &reset.observations, 20);

    assert!(!trace.is_empty(), "episode produced no steps");
    for (actions, rewards) in &trace {
        assert_eq!(actions.len(), AGENTS as usize);
        assert_eq!(rewards.len(), AGENTS as usize);
    }
}

// ---------------------------------------------------------------------------
// 2. Determinism: same seed + same world ⇒ byte-identical action/reward trace
// ---------------------------------------------------------------------------

#[test]
fn test_cooperative_planning_is_deterministic() {
    const AGENTS: u32 = 3;

    let run = |strategy: JointStrategy| {
        let mut world = make_world(AGENTS, 11);
        let reset = world.reset(Some(11));
        let protocol = fast_protocol(AGENTS as usize, strategy, 99);
        run_episode(&protocol, &mut world, &reset.observations, 15)
    };

    // Both strategies must reproduce identically across independent runs.
    assert_eq!(
        run(JointStrategy::SequentialFactored),
        run(JointStrategy::SequentialFactored),
        "SequentialFactored planning is not deterministic"
    );
    assert_eq!(
        run(JointStrategy::Sampled),
        run(JointStrategy::Sampled),
        "Sampled planning is not deterministic"
    );
}

// ---------------------------------------------------------------------------
// 3. Swappability: the cooperative protocol drops in for the baseline
// ---------------------------------------------------------------------------

#[test]
fn test_cooperative_protocol_swaps_for_independent_baseline() {
    const AGENTS: u32 = 2;

    // Same call site, two protocols behind the trait object.
    let protocols: Vec<Box<dyn SwarmProtocol>> = vec![
        Box::new(IndependentProtocol::new(AGENTS as usize)),
        Box::new(fast_protocol(
            AGENTS as usize,
            JointStrategy::SequentialFactored,
            3,
        )),
    ];

    for protocol in &protocols {
        assert_eq!(protocol.swarm_size(), AGENTS as usize);
        let mut world = make_world(AGENTS, 3);
        let reset = world.reset(Some(3));
        let trace = run_episode(protocol.as_ref(), &mut world, &reset.observations, 10);
        assert!(!trace.is_empty());
        // The baseline yields all-Noop; the cooperative protocol yields a
        // valid joint plan. Both must remain step-compatible (asserted in
        // run_episode), which is the swappability contract.
    }
}

// ---------------------------------------------------------------------------
// 4. Sampled strategy is also step-compatible end-to-end
// ---------------------------------------------------------------------------

#[test]
fn test_sampled_strategy_episode() {
    const AGENTS: u32 = 2;
    let mut world = make_world(AGENTS, 21);
    let reset = world.reset(Some(21));
    let protocol = fast_protocol(AGENTS as usize, JointStrategy::Sampled, 21);
    let trace = run_episode(&protocol, &mut world, &reset.observations, 12);
    assert!(!trace.is_empty());
    for (actions, _) in &trace {
        assert_eq!(actions.len(), AGENTS as usize);
    }
}
