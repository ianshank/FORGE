//! Parity tests: drive [`WorldEnv`] and a raw [`forge_core::WorldState`]
//! in lockstep and assert identical outputs.
//!
//! This is the CI gate that backwards-compat is honoured — if these
//! tests fail, the shim has drifted from the underlying engine.

use forge_env::Env;
use forge_env_forge::{FlatForgeEnv, FlatObsConfig, ObsFlattener, WorldEnv};
use forge_types::action::Action;
use forge_types::config::ForgeConfig;

fn tiny_cfg() -> ForgeConfig {
    let mut c = ForgeConfig::default();
    c.world.width = 16;
    c.world.height = 16;
    c
}

#[test]
fn world_env_reset_matches_world_state_reset() {
    let cfg = tiny_cfg();
    let mut env = WorldEnv::new(cfg.clone()).unwrap();
    let mut raw = forge_core::WorldState::new(cfg).unwrap();

    let env_out = env.reset(Some(7)).unwrap();
    let raw_out = raw.reset(Some(7));

    assert_eq!(env_out.terminated, raw_out.terminated);
    assert_eq!(env_out.truncated, raw_out.truncated);
    assert_eq!(
        env_out.obs.grid_view.len(),
        raw_out.observations[0].grid_view.len()
    );
    assert_eq!(env_out.obs.position, raw_out.observations[0].position);
}

#[test]
fn lockstep_step_parity_over_many_ticks() {
    let cfg = tiny_cfg();
    let mut env = WorldEnv::new(cfg.clone()).unwrap();
    let mut raw = forge_core::WorldState::new(cfg).unwrap();
    let _ = env.reset(Some(123)).unwrap();
    let _ = raw.reset(Some(123));

    // Deterministic action sequence — repeated Noop is sufficient to
    // exercise the step path without introducing flakiness from
    // ordering of moves vs other systems.
    let actions = vec![Action::Noop; 200];

    for (i, a) in actions.iter().enumerate() {
        let env_out = env.step(a.clone()).unwrap();
        let raw_out = raw.step(std::slice::from_ref(a));
        assert_eq!(
            env_out.terminated, raw_out.terminated,
            "terminated drift at step {i}"
        );
        assert_eq!(
            env_out.truncated, raw_out.truncated,
            "truncated drift at step {i}"
        );
        assert_eq!(
            env_out.reward,
            raw_out.rewards.first().copied().unwrap_or(0.0),
            "reward drift at step {i}"
        );
        assert_eq!(
            env_out.obs.position, raw_out.observations[0].position,
            "position drift at step {i}"
        );
        if env_out.terminated || env_out.truncated {
            break;
        }
    }
}

#[test]
fn flat_forge_env_produces_consistent_obs_dim() {
    let cfg = tiny_cfg();
    let mut env = FlatForgeEnv::new(cfg.clone(), FlatObsConfig::default()).unwrap();
    let r = env.reset(Some(0)).unwrap();
    // Underlying FlatObsEnv contract
    use forge_env::FlatObsEnv;
    assert_eq!(r.obs.len(), env.obs_dim());
    let s = env.step(0).unwrap();
    assert_eq!(s.obs.len(), env.obs_dim());
}

#[test]
fn step_into_keeps_buffer_dim_stable() {
    use forge_env::{StepInto, StepOutput};
    let cfg = tiny_cfg();
    let mut env = FlatForgeEnv::new(cfg.clone(), FlatObsConfig::default()).unwrap();
    use forge_env::FlatObsEnv;
    let dim = env.obs_dim();
    let mut out: StepOutput<Vec<f32>, forge_types::observation::StepInfo> = StepOutput {
        obs: Vec::with_capacity(dim),
        reward: 0.0,
        terminated: false,
        truncated: false,
        info: Default::default(),
    };
    let _ = env.reset(Some(0)).unwrap();
    for _ in 0..10 {
        env.step_into(0, &mut out).unwrap();
        assert_eq!(out.obs.len(), dim);
    }
}

/// Invariant gated test: `FlatForgeEnv`'s `action_spec` MUST be a
/// `Discrete` variant. `num_actions()` falls back to 0 if this is
/// ever violated; we never want that fallback to fire in practice.
#[test]
fn flat_forge_env_action_spec_is_discrete() {
    let cfg = tiny_cfg();
    let env = FlatForgeEnv::new(cfg, FlatObsConfig::default()).unwrap();
    use forge_env::FlatObsEnv;
    assert!(
        env.num_actions() > 0,
        "action_spec must be Discrete with n > 0"
    );
}

#[test]
fn flattener_layout_is_stable_across_runs() {
    let cfg = tiny_cfg();
    let f = ObsFlattener::new(FlatObsConfig::default(), cfg.clone());
    let obs = forge_types::observation::Observation::default();
    let a = f.flatten(&obs);
    let b = f.flatten(&obs);
    assert_eq!(a, b);
    assert_eq!(a.len(), f.flat_dim());
}
