//! Core [`Env`] trait and step-output type.

use std::borrow::Cow;

use crate::spec::{ActionSpec, ObsSpec};

/// Output of a single environment step.
///
/// Generic over the observation and info types directly (rather than
/// over the env type) so the trait can be used as a trait object —
/// `Box<dyn Env<Obs=..., Action=..., Info=..., Error=...>>` is
/// dyn-compatible because no method's return type references `Self`.
#[derive(Debug, Clone)]
pub struct StepOutput<Obs, Info> {
    /// The observation produced after the action was applied.
    pub obs: Obs,
    /// Scalar reward for this step.
    pub reward: f32,
    /// True iff the episode reached a natural terminal state.
    pub terminated: bool,
    /// True iff the episode was truncated artificially (max steps, etc.).
    pub truncated: bool,
    /// Auxiliary information (env-specific).
    pub info: Info,
}

impl<Obs: Default, Info: Default> Default for StepOutput<Obs, Info> {
    fn default() -> Self {
        Self {
            obs: Obs::default(),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: Info::default(),
        }
    }
}

/// Generic environment trait.
///
/// Implementors are typically wrappers around a simulation backend
/// (FORGE's `WorldState`, a Minecraft WebSocket client, etc.). The trait
/// is deliberately minimal: reset, step, and space description. Optional
/// extensions live in companion traits ([`StepInto`], [`FlatObsEnv`]).
///
/// ## Threading
///
/// `Env: Send` so single envs can move across threads (e.g. handed to a
/// dedicated rollout worker). Concurrent step calls on a shared env are
/// not supported — use one env per worker. `Sync` is deliberately not
/// required because most concrete envs hold non-thread-safe state
/// (WebSocket clients, mutable RNGs).
pub trait Env: Send {
    /// Concrete observation type produced by `reset`/`step`.
    type Obs;
    /// Concrete action type accepted by `step`.
    type Action: Send;
    /// Auxiliary per-step info. Must implement [`Default`] so
    /// [`StepOutput`] can be constructed ergonomically.
    type Info: Default + Send;
    /// Concrete error type returned by fallible operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Reset the environment to an initial state and return the first
    /// observation.
    ///
    /// `seed` is honoured by deterministic envs; envs with no notion of
    /// seeding are free to ignore it.
    fn reset(
        &mut self,
        seed: Option<u64>,
    ) -> Result<StepOutput<Self::Obs, Self::Info>, Self::Error>;

    /// Advance the environment by one step.
    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<StepOutput<Self::Obs, Self::Info>, Self::Error>;

    /// Description of the observation space.
    fn obs_spec(&self) -> &ObsSpec;

    /// Description of the action space.
    fn action_spec(&self) -> &ActionSpec;

    /// Human-readable env identifier — used in span labels and logs.
    ///
    /// Returns [`Cow`] so static defaults and dynamic per-instance names
    /// (e.g. `format!("minecraft-{server_version}")`) both work.
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("env")
    }

    /// Tear down the environment. Default impl is a no-op.
    fn close(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Opt-in zero-allocation step variant.
///
/// Implementors fill the provided `out` buffer in place rather than
/// returning a freshly allocated [`StepOutput`]. The caller is
/// responsible for reusing the buffer across calls; doing so makes the
/// hot path allocation-free.
///
/// Wire-bound envs (e.g. `forge-env-mc::MinecraftEnv`) intentionally do
/// not implement this trait — their step path performs unavoidable I/O
/// allocation. Such envs are excluded from the zero-allocation CI gate
/// by module path.
pub trait StepInto: Env {
    /// Step in place. After this call, `out` reflects the new state.
    fn step_into(
        &mut self,
        action: Self::Action,
        out: &mut StepOutput<Self::Obs, Self::Info>,
    ) -> Result<(), Self::Error>;
}

/// Marker trait for envs whose observations are flat `Vec<f32>` and
/// whose actions are discrete `u32` indices.
///
/// This is the surface that `latent_mcts` consumes: it operates on
/// `&[f32]` observations and selects an action by visit-count argmax.
/// Pinning these types here means downstream code can write
/// `Box<dyn FlatObsEnv<...>>` without naming the projection generics
/// each time.
pub trait FlatObsEnv: Env<Obs = Vec<f32>, Action = u32> {
    /// Number of elements in the flat observation vector. Must match
    /// `obs_spec().num_elements()`.
    fn obs_dim(&self) -> usize;
    /// Number of discrete actions. Must match `action_spec().discrete_n()`.
    fn num_actions(&self) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EnvError;
    use crate::spec::{ActionSpec, ObsSpec};

    /// Minimal env for trait-shape verification. Returns a zero-filled
    /// obs vector of configurable length, with rewards equal to the
    /// action id (cast to f32) so tests can assert determinism.
    struct MockFlatEnv {
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        tick: u64,
        closed: bool,
    }

    impl MockFlatEnv {
        fn new(obs_dim: usize, action_count: u32) -> Self {
            Self {
                obs_spec: ObsSpec::flat_f32("mock", obs_dim, 0.0, 1.0),
                action_spec: ActionSpec::discrete(action_count),
                tick: 0,
                closed: false,
            }
        }
    }

    #[derive(Debug, Default, Clone)]
    struct MockInfo {
        tick: u64,
    }

    impl Env for MockFlatEnv {
        type Obs = Vec<f32>;
        type Action = u32;
        type Info = MockInfo;
        type Error = EnvError;

        fn reset(
            &mut self,
            _seed: Option<u64>,
        ) -> Result<StepOutput<Self::Obs, Self::Info>, Self::Error> {
            if self.closed {
                return Err(EnvError::Closed);
            }
            self.tick = 0;
            Ok(StepOutput {
                obs: vec![0.0; self.obs_spec.num_elements()],
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: MockInfo { tick: 0 },
            })
        }

        fn step(&mut self, action: u32) -> Result<StepOutput<Self::Obs, Self::Info>, Self::Error> {
            if self.closed {
                return Err(EnvError::Closed);
            }
            let n = self.action_spec.discrete_n().unwrap();
            if action >= n {
                return Err(EnvError::InvalidAction {
                    action_id: action,
                    space_n: n,
                });
            }
            self.tick += 1;
            Ok(StepOutput {
                obs: vec![0.0; self.obs_spec.num_elements()],
                reward: action as f32,
                terminated: false,
                truncated: false,
                info: MockInfo { tick: self.tick },
            })
        }

        fn obs_spec(&self) -> &ObsSpec {
            &self.obs_spec
        }

        fn action_spec(&self) -> &ActionSpec {
            &self.action_spec
        }

        fn name(&self) -> Cow<'_, str> {
            Cow::Owned(format!("mock-flat-{}", self.obs_spec.num_elements()))
        }

        fn close(&mut self) -> Result<(), Self::Error> {
            self.closed = true;
            Ok(())
        }
    }

    impl FlatObsEnv for MockFlatEnv {
        fn obs_dim(&self) -> usize {
            self.obs_spec.num_elements()
        }
        fn num_actions(&self) -> u32 {
            self.action_spec.discrete_n().unwrap()
        }
    }

    impl StepInto for MockFlatEnv {
        fn step_into(
            &mut self,
            action: u32,
            out: &mut StepOutput<Vec<f32>, MockInfo>,
        ) -> Result<(), Self::Error> {
            if self.closed {
                return Err(EnvError::Closed);
            }
            let n = self.action_spec.discrete_n().unwrap();
            if action >= n {
                return Err(EnvError::InvalidAction {
                    action_id: action,
                    space_n: n,
                });
            }
            self.tick += 1;
            // Reuse the obs buffer — resize keeps capacity if it's already big enough.
            out.obs.clear();
            out.obs.resize(self.obs_spec.num_elements(), 0.0);
            out.reward = action as f32;
            out.terminated = false;
            out.truncated = false;
            out.info.tick = self.tick;
            Ok(())
        }
    }

    #[test]
    fn env_reset_and_step_produce_correct_obs_dim() {
        let mut env = MockFlatEnv::new(16, 4);
        let r = env.reset(Some(42)).unwrap();
        assert_eq!(r.obs.len(), 16);
        let s = env.step(2).unwrap();
        assert_eq!(s.obs.len(), 16);
        assert_eq!(s.reward, 2.0);
        assert_eq!(s.info.tick, 1);
    }

    #[test]
    fn invalid_action_returns_error() {
        let mut env = MockFlatEnv::new(8, 4);
        let _ = env.reset(None).unwrap();
        let err = env.step(99).unwrap_err();
        match err {
            EnvError::InvalidAction { action_id, space_n } => {
                assert_eq!(action_id, 99);
                assert_eq!(space_n, 4);
            }
            other => panic!("expected InvalidAction, got {other:?}"),
        }
    }

    #[test]
    fn close_blocks_subsequent_step() {
        let mut env = MockFlatEnv::new(8, 4);
        env.close().unwrap();
        assert!(matches!(env.step(0).unwrap_err(), EnvError::Closed));
    }

    #[test]
    fn dynamic_name_via_cow_owned() {
        let env = MockFlatEnv::new(32, 4);
        let n = env.name();
        assert_eq!(n.as_ref(), "mock-flat-32");
    }

    #[test]
    fn flat_obs_env_advertises_consistent_dims() {
        let env = MockFlatEnv::new(64, 5);
        assert_eq!(env.obs_dim(), env.obs_spec().num_elements());
        assert_eq!(env.num_actions(), env.action_spec().discrete_n().unwrap());
    }

    #[test]
    fn step_into_reuses_buffer_capacity() {
        let mut env = MockFlatEnv::new(128, 4);
        let _ = env.reset(None).unwrap();
        let mut out: StepOutput<Vec<f32>, MockInfo> = StepOutput {
            obs: Vec::with_capacity(128),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: MockInfo::default(),
        };
        let initial_cap = out.obs.capacity();
        for action in 0..10 {
            env.step_into(action % 4, &mut out).unwrap();
        }
        // Capacity must not have grown — buffer was reused.
        assert_eq!(
            out.obs.capacity(),
            initial_cap,
            "step_into must reuse the obs buffer"
        );
        assert_eq!(out.info.tick, 10);
    }

    #[test]
    fn flat_obs_env_is_object_safe() {
        // Compile-time check: we can hold one as a trait object.
        let env = MockFlatEnv::new(8, 2);
        let _boxed: Box<
            dyn FlatObsEnv<Obs = Vec<f32>, Action = u32, Info = MockInfo, Error = EnvError>,
        > = Box::new(env);
    }

    proptest::proptest! {
        #[test]
        fn step_with_valid_action_never_panics(
            obs_dim in 1usize..256,
            n_actions in 1u32..64,
            action in 0u32..64,
        ) {
            let mut env = MockFlatEnv::new(obs_dim, n_actions);
            let _ = env.reset(Some(0)).unwrap();
            let result = env.step(action);
            if action < n_actions {
                let r = result.unwrap();
                proptest::prop_assert_eq!(r.obs.len(), obs_dim);
            } else {
                proptest::prop_assert!(result.is_err());
            }
        }
    }
}
