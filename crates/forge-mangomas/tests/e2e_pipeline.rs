//! End-to-end smoke test for the MangoMAS pipeline.
//!
//! Exercises the full data collection → adaptation → transfer pipeline:
//! `BatchRunner` → `FlatStateAdapter` → `BdiEpisodeCollector` →
//! `ConstitutionalConstraintMapper`.

use forge_mangomas::adapters::observation_adapter::ObservationAdapter;
use forge_mangomas::adapters::FlatStateAdapter;
use forge_mangomas::batch_runner::{BatchRunner, RandomActionPolicy};
use forge_mangomas::config::{BatchRunnerConfig, ObservationAdapterConfig, TransferConfig};
use forge_mangomas::swarm::protocol::IndependentProtocol;
use forge_mangomas::swarm::SwarmProtocol;
use forge_mangomas::transfer::{BdiEpisodeCollector, ConstitutionalConstraintMapper};
use forge_types::action::Action;

/// Small world config for fast tests.
fn small_batch_config() -> BatchRunnerConfig {
    let mut config = BatchRunnerConfig {
        max_episode_steps: 20,
        num_envs: 1,
        seed: 42,
        ..BatchRunnerConfig::default()
    };
    config.forge_config.world.width = 8;
    config.forge_config.world.height = 8;
    config.forge_config.agents.num_agents = 1;
    config.forge_config.agents.default_vision_radius = 2;
    config
}

/// Full pipeline: BatchRunner → FlatStateAdapter → BdiEpisodeCollector →
/// ConstitutionalConstraintMapper.
#[test]
fn test_full_pipeline_smoke() {
    let batch_config = small_batch_config();
    let comm_vocab = batch_config.forge_config.agents.comm_vocab_size;
    let drone_enabled = batch_config.forge_config.drone.enabled;

    // 1. Collect episodes via BatchRunner
    let runner = BatchRunner::new(batch_config);
    let action_space = Action::space_size(comm_vocab, drone_enabled);
    let policy = RandomActionPolicy::new_with_seed(action_space, 42);
    let batch = runner.collect_episodes(3, &policy).unwrap();

    assert_eq!(batch.episodes.len(), 3);
    assert!(
        batch.total_transitions > 0,
        "must have collected transitions"
    );
    for ep in &batch.episodes {
        assert!(ep.length > 0, "every episode must have at least one step");
        assert_eq!(ep.transitions.len(), ep.length as usize);
    }

    // 2. Adapt observations via FlatStateAdapter
    let obs_config = ObservationAdapterConfig::default();
    let adapter = FlatStateAdapter::new(obs_config);
    let expected_dim = adapter.output_dim();

    for ep in &batch.episodes {
        for transition in &ep.transitions {
            let flat = adapter.adapt(&transition.observation).unwrap();
            assert_eq!(flat.len(), expected_dim, "output dimension mismatch");
            assert!(
                flat.iter().all(|&v| v.is_finite()),
                "adapter must produce finite values"
            );
        }
    }

    // 3. Collect BDI training data
    let transfer_config = TransferConfig::default();
    let bdi_collector =
        BdiEpisodeCollector::new(transfer_config.clone(), comm_vocab, drone_enabled);
    let adapt_fn = |obs: &forge_types::observation::Observation| adapter.adapt(obs).unwrap();
    let bdi_data = bdi_collector
        .collect_from_episodes(&batch.episodes, &adapt_fn)
        .unwrap();

    assert_eq!(bdi_data.source_episodes, 3);
    assert_eq!(bdi_data.num_intentions, 8);
    assert_eq!(
        bdi_data.samples.len(),
        batch.total_transitions as usize,
        "one BDI sample per transition"
    );
    for sample in &bdi_data.samples {
        assert_eq!(sample.state.len(), expected_dim);
        assert!(
            sample.intention < 8,
            "intention {} out of range",
            sample.intention
        );
    }

    // 4. Check constitutional constraints
    let constraint_mapper = ConstitutionalConstraintMapper::new(&transfer_config);
    assert_eq!(
        constraint_mapper.constraints().len(),
        5,
        "default config has 5 constraints"
    );

    let mut any_violation_found = false;
    for ep in &batch.episodes {
        for transition in &ep.transitions {
            let violations = constraint_mapper.check_violations(&transition.observation);
            let penalty = constraint_mapper.compute_penalty(&transition.observation);
            // penalty must be non-positive (it's negated severity)
            assert!(penalty <= 0.0, "penalty must be <= 0, got {penalty}");
            if !violations.is_empty() {
                any_violation_found = true;
                for v in &violations {
                    assert!(
                        (0.0..=1.0).contains(&v.severity),
                        "severity {:.2} out of [0,1]",
                        v.severity
                    );
                }
            }
        }
    }
    // With random actions on a small world, we almost always get at least
    // one violation (e.g. stamina depletion or geofence) — but don't hard-fail
    // if not, just log it.
    if !any_violation_found {
        eprintln!("NOTE: no constraint violations found (unusual but not a failure)");
    }
}

/// Swarm stubs: IndependentProtocol produces correct-length Noop vectors.
#[test]
fn test_swarm_independent_protocol_with_batch() {
    let batch_config = small_batch_config();
    let comm_vocab = batch_config.forge_config.agents.comm_vocab_size;
    let drone_enabled = batch_config.forge_config.drone.enabled;

    let runner = BatchRunner::new(batch_config);
    let action_space = Action::space_size(comm_vocab, drone_enabled);
    let policy = RandomActionPolicy::new_with_seed(action_space, 99);
    let batch = runner.collect_episodes(1, &policy).unwrap();

    let ep = &batch.episodes[0];
    let protocol = IndependentProtocol::new(1);
    assert_eq!(protocol.name(), "independent");
    assert_eq!(protocol.swarm_size(), 1);

    // Feed real observations through the stub protocol
    for transition in &ep.transitions {
        let obs_slice = std::slice::from_ref(&transition.observation);
        let comm_tokens: Vec<Vec<u16>> = vec![vec![]];
        let actions = protocol.coordinate(obs_slice, &comm_tokens);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], Action::Noop);
    }
}

/// Determinism: same seed produces identical BDI datasets.
#[test]
fn test_pipeline_determinism() {
    let transfer_config = TransferConfig::default();
    let obs_config = ObservationAdapterConfig::default();
    let adapter = FlatStateAdapter::new(obs_config);
    let adapt_fn = |obs: &forge_types::observation::Observation| adapter.adapt(obs).unwrap();

    let collect = |seed: u64| {
        let mut config = small_batch_config();
        config.seed = seed;
        let comm_vocab = config.forge_config.agents.comm_vocab_size;
        let drone_enabled = config.forge_config.drone.enabled;

        let runner = BatchRunner::new(config);

        // Use noop for full determinism (random policy reseeds per-call)
        struct NoopPolicy;
        impl forge_mangomas::batch_runner::ActionPolicy for NoopPolicy {
            fn select_action(
                &self,
                _obs: &forge_types::observation::Observation,
                _agent_idx: usize,
            ) -> u32 {
                0
            }
        }

        let batch = runner.collect_episodes(2, &NoopPolicy).unwrap();
        let bdi = BdiEpisodeCollector::new(transfer_config.clone(), comm_vocab, drone_enabled);
        bdi.collect_from_episodes(&batch.episodes, &adapt_fn)
            .unwrap()
    };

    let data_a = collect(42);
    let data_b = collect(42);

    assert_eq!(data_a.samples.len(), data_b.samples.len());
    for (a, b) in data_a.samples.iter().zip(&data_b.samples) {
        assert_eq!(a.state, b.state);
        assert_eq!(a.intention, b.intention);
        assert_eq!(a.reward, b.reward);
    }
}
