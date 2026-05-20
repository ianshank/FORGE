//! Random-action baseline support for the v0.5 first-real-run capture.
//!
//! Exposes [`RandomLatentModel`] — a no-op [`LatentForwardModel`] used
//! to satisfy the runner's `Runner<E, M: LatentForwardModel>` generic
//! when `RunnerConfig.random_actions == true`. The runner's planning
//! step short-circuits MCTS in this mode and samples actions directly
//! via [`sample_random_action`], so the model's inference methods are
//! never actually called — but the type still has to satisfy the
//! trait so we get a `Runner<E, RandomLatentModel>` without forcing
//! the live path to load ONNX bundles.
//!
//! The random-action selection bypasses MCTS entirely. This is
//! deliberate — running MCTS with uniform priors does NOT produce
//! uniformly-random action selection (PUCT's `sqrt(parent_visits) /
//! (1 + child_visits)` term biases the search toward unvisited
//! actions, and the visit-count argmax breaks ties on first action).
//! Bypassing the planner is the only way to guarantee a calibrated
//! random baseline.

use anyhow::Result;
use forge_agent::latent_mcts::model::{LatentForwardModel, LatentInferenceOutput};
use forge_agent::latent_mcts::state::LatentState;
use rand::Rng;

/// No-op latent forward model. Returns zero-filled latents + uniform
/// policy priors so callers that DO inadvertently invoke inference
/// get a defined output (rather than `unimplemented!()`-style panics).
///
/// In the random-actions runner path, `Runner::run_episode` skips the
/// MCTS search call entirely — this struct exists only to satisfy
/// the generic bound `Runner<E, M: LatentForwardModel>`.
#[derive(Debug, Clone)]
pub struct RandomLatentModel {
    action_space: u32,
    latent_dim: usize,
}

impl RandomLatentModel {
    /// Construct with the bot-advertised action space + the runner's
    /// configured `onnx.latent_dim` (used only when inference is
    /// accidentally invoked).
    pub fn new(action_space: u32, latent_dim: usize) -> Self {
        Self {
            action_space,
            latent_dim,
        }
    }
}

impl LatentForwardModel for RandomLatentModel {
    fn initial_inference(&self, _observation: &[f32]) -> Result<LatentInferenceOutput> {
        Ok(LatentInferenceOutput {
            latent_state: LatentState::zeros(self.latent_dim),
            reward: 0.0,
            policy_logits: vec![1.0 / self.action_space as f32; self.action_space as usize],
            value: 0.0,
        })
    }

    fn recurrent_inference(
        &self,
        _state: &LatentState,
        _action: u32,
    ) -> Result<LatentInferenceOutput> {
        Ok(LatentInferenceOutput {
            latent_state: LatentState::zeros(self.latent_dim),
            reward: 0.0,
            policy_logits: vec![1.0 / self.action_space as f32; self.action_space as usize],
            value: 0.0,
        })
    }

    fn action_space_size(&self) -> u32 {
        self.action_space
    }
}

/// Sample a uniformly-random action from `0..action_count`.
///
/// Pulled into a free function (rather than buried inside
/// `Runner::run_episode`) so the χ²-uniformity test below can pin the
/// distribution directly without standing up a full runner.
pub fn sample_random_action(rng: &mut impl Rng, action_count: u32) -> u32 {
    assert!(action_count > 0, "action_count must be >= 1");
    rng.gen_range(0..action_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_pcg::Pcg64Mcg;

    /// Pearson's χ² test for uniform discrete sampling. With 10,000
    /// samples over 12 actions, the expected per-bucket count is
    /// 833.3 and the χ² statistic should land well under the
    /// chi-squared critical value at df=11, α=0.001 (≈ 31.26).
    #[test]
    fn random_actions_distribute_uniformly_over_action_space() {
        let mut rng = Pcg64Mcg::new(0xCAFEF00DDEADBEEF_u128);
        let action_count = 12_u32;
        let samples = 10_000_usize;
        let mut buckets = vec![0_u32; action_count as usize];
        for _ in 0..samples {
            let a = sample_random_action(&mut rng, action_count);
            buckets[a as usize] += 1;
        }
        let expected = samples as f64 / action_count as f64;
        let chi_squared: f64 = buckets
            .iter()
            .map(|&observed| {
                let diff = observed as f64 - expected;
                diff * diff / expected
            })
            .sum();
        // χ²(df=11, α=0.001) ≈ 31.26. Use a generous 35.0 ceiling so
        // a different `Pcg64Mcg` seed in CI (should the literal above
        // ever change) doesn't push us over an alpha=0.001 boundary
        // for legitimate uniform output.
        assert!(
            chi_squared < 35.0,
            "χ² = {chi_squared} too high for uniform sampling; buckets: {buckets:?}"
        );
    }

    #[test]
    fn random_latent_model_returns_zero_init_outputs() {
        let model = RandomLatentModel::new(7, 16);
        let obs = vec![0.5_f32; 64];
        let out = model.initial_inference(&obs).unwrap();
        assert_eq!(out.latent_state.dim(), 16);
        assert_eq!(out.policy_logits.len(), 7);
        assert!((out.reward - 0.0).abs() < f32::EPSILON);
        assert!((out.value - 0.0).abs() < f32::EPSILON);
        // Uniform priors sum to 1.0 ± floating-point drift.
        let sum: f32 = out.policy_logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "uniform priors must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn random_latent_model_action_space_size_passthrough() {
        let model = RandomLatentModel::new(12, 256);
        assert_eq!(model.action_space_size(), 12);
    }

    #[test]
    fn random_latent_model_recurrent_inference_matches_initial_invariant() {
        // The model is a type-stub; the runner's planning branch
        // short-circuits before either inference method is called.
        // But if a future refactor accidentally invokes
        // `recurrent_inference`, it must still satisfy the same
        // uniform-prior / zero-init contract as `initial_inference` so
        // downstream code isn't poisoned by NaN values.
        let model = RandomLatentModel::new(7, 16);
        let state = LatentState::zeros(16);
        let out = model
            .recurrent_inference(&state, 3)
            .expect("recurrent inference must not error on the stub model");
        assert_eq!(out.latent_state.dim(), 16);
        assert_eq!(out.policy_logits.len(), 7);
        assert!((out.reward - 0.0).abs() < f32::EPSILON);
        assert!((out.value - 0.0).abs() < f32::EPSILON);
        let sum: f32 = out.policy_logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "recurrent uniform priors must sum to 1.0, got {sum}"
        );
    }

    #[test]
    #[should_panic(expected = "action_count must be >= 1")]
    fn sample_random_action_panics_on_zero_action_count() {
        let mut rng = Pcg64Mcg::new(0);
        let _ = sample_random_action(&mut rng, 0);
    }
}
