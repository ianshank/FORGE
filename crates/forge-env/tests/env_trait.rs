//! Integration tests for the `forge-env` trait.
//!
//! These tests live outside `src/` so they exercise the public API
//! exactly as a downstream consumer would.

use std::borrow::Cow;

use forge_env::{ActionSpec, Env, EnvError, FlatObsEnv, ObsSpec, StepOutput};

/// Reference impl used to verify the trait contract end-to-end.
struct DeterministicFlatEnv {
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    seed: u64,
    step_count: u64,
    max_steps: u64,
}

impl DeterministicFlatEnv {
    fn new(obs_dim: usize, n_actions: u32, max_steps: u64) -> Self {
        Self {
            obs_spec: ObsSpec::flat_f32("det", obs_dim, 0.0, 1.0),
            action_spec: ActionSpec::discrete(n_actions),
            seed: 0,
            step_count: 0,
            max_steps,
        }
    }
}

impl Env for DeterministicFlatEnv {
    type Obs = Vec<f32>;
    type Action = u32;
    type Info = ();
    type Error = EnvError;

    fn reset_into(&mut self, seed: Option<u64>, out: &mut Vec<f32>) -> Result<(), Self::Error> {
        self.seed = seed.unwrap_or(0);
        self.step_count = 0;
        out.clear();
        out.resize(self.obs_spec.num_elements(), 0.0);
        Ok(())
    }

    fn step_into(
        &mut self,
        action: u32,
        out: &mut StepOutput<Vec<f32>, ()>,
    ) -> Result<(), Self::Error> {
        let n = self.action_spec.discrete_n().unwrap();
        if action >= n {
            return Err(EnvError::InvalidAction {
                action_id: action,
                space_n: n,
            });
        }
        self.step_count += 1;
        // Deterministic-but-trivial obs: every element = (seed + step) % 1.0
        let value = ((self.seed.wrapping_add(self.step_count)) as f32) * 1e-3;
        out.obs.clear();
        out.obs.resize(self.obs_spec.num_elements(), value);
        out.reward = 0.0;
        out.terminated = false;
        out.truncated = self.step_count >= self.max_steps;
        Ok(())
    }

    fn obs_spec(&self) -> &ObsSpec {
        &self.obs_spec
    }
    fn action_spec(&self) -> &ActionSpec {
        &self.action_spec
    }
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("deterministic-flat-env")
    }
}

impl FlatObsEnv for DeterministicFlatEnv {
    fn obs_dim(&self) -> usize {
        self.obs_spec.num_elements()
    }
    fn num_actions(&self) -> u32 {
        self.action_spec.discrete_n().unwrap()
    }
}

#[test]
fn same_seed_produces_same_trajectory() {
    let mut a = DeterministicFlatEnv::new(8, 4, 100);
    let mut b = DeterministicFlatEnv::new(8, 4, 100);
    let _ = a.reset(Some(123)).unwrap();
    let _ = b.reset(Some(123)).unwrap();
    for action in [0u32, 1, 2, 3, 0, 1, 2, 3, 0, 1] {
        let sa = a.step(action).unwrap();
        let sb = b.step(action).unwrap();
        assert_eq!(sa.obs, sb.obs);
        assert_eq!(sa.truncated, sb.truncated);
    }
}

#[test]
fn truncation_triggers_at_max_steps() {
    let mut env = DeterministicFlatEnv::new(4, 2, 3);
    let _ = env.reset(None).unwrap();
    let s1 = env.step(0).unwrap();
    assert!(!s1.truncated);
    let s2 = env.step(0).unwrap();
    assert!(!s2.truncated);
    let s3 = env.step(0).unwrap();
    assert!(s3.truncated);
}

#[test]
fn obs_spec_and_action_spec_match_flat_obs_env_dims() {
    let env = DeterministicFlatEnv::new(32, 5, 100);
    assert_eq!(env.obs_dim(), env.obs_spec().num_elements());
    assert_eq!(env.num_actions(), env.action_spec().discrete_n().unwrap());
}

#[test]
fn flat_obs_env_via_dyn_trait_object() {
    fn run_one(env: &mut dyn Env<Obs = Vec<f32>, Action = u32, Info = (), Error = EnvError>) {
        let r = env.reset(Some(7)).unwrap();
        assert!(!r.is_empty());
        let s = env.step(0).unwrap();
        assert_eq!(s.obs.len(), r.len());
    }
    let mut env = DeterministicFlatEnv::new(16, 3, 100);
    run_one(&mut env);
}
